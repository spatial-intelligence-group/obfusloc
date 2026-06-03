"""
RoMA dense matching — produces HDF5 feature and match files compatible with the
Hloc pipeline.
"""

import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from hloc.utils.parsers import names_to_pair
from tqdm import tqdm

from cam_utils import parse_retrieval_file


confs = {
    "roma_outdoor": {"output": "matches-roma_outdoor", "model": {"name": "roma_outdoor"}},
    "roma_indoor": {"output": "matches-roma_indoor", "model": {"name": "roma_indoor"}},
    "tiny_roma_v1_outdoor": {
        "output": "matches-tiny_roma_v1_outdoor",
        "model": {"name": "tiny_roma_v1_outdoor"},
    },
}


def _load_roma_model(conf: dict, device: torch.device):
    name = conf["model"]["name"]
    if name == "roma_outdoor":
        from romatch import roma_outdoor
        return roma_outdoor(device=device)
    if name == "roma_indoor":
        from romatch import roma_indoor
        return roma_indoor(device=device)
    if name == "tiny_roma_v1_outdoor":
        from romatch import tiny_roma_v1_outdoor
        return tiny_roma_v1_outdoor(device=device)
    raise ValueError(
        f"Unknown RoMA configuration '{name}' — expected one of: "
        "roma_outdoor, roma_indoor, tiny_roma_v1_outdoor"
    )


def _merge_keypoints(
    prev_kpts: np.ndarray, new_kpts: np.ndarray
) -> tuple[np.ndarray, list[int]]:
    """
    Merge new_kpts into prev_kpts, deduplicating at 0.01-px resolution.

    Uses a dict for O(N) lookup rather than a per-keypoint np.where search.
    Returns the merged float16 array and a list mapping each new keypoint
    to its index in the merged array.
    """
    prev_i = (100 * prev_kpts).astype(int)
    new_i = (100 * new_kpts).astype(int)

    # map (x_int, y_int) → index in the growing merged list
    kpt_to_idx: dict[tuple[int, int], int] = {
        (int(kpt[0]), int(kpt[1])): i for i, kpt in enumerate(prev_i)
    }
    merged: list[tuple[int, int]] = [(int(kpt[0]), int(kpt[1])) for kpt in prev_i]

    match_idxs: list[int] = []
    for kpt in new_i:
        key = (int(kpt[0]), int(kpt[1]))
        if key not in kpt_to_idx:
            kpt_to_idx[key] = len(merged)
            merged.append(key)
        match_idxs.append(kpt_to_idx[key])

    merged_arr = np.array(merged, dtype=np.float32) / 100.0 if merged else np.empty((0, 2), dtype=np.float32)
    # float16 can't hold values multiplied by 100 at full image resolution
    return merged_arr.astype(np.float16), match_idxs


def roma_match(
    conf: dict,
    pairs: Path,
    image_dir: Path,
    image_dir_ref: Path,
    matches: Path | None = None,
    features: Path | None = None,
    features_ref: Path | None = None,
    max_kps: int = 512,
    overwrite: bool = False,
) -> None:
    """
    Perform dense matching using RoMA for given image pairs and write HDF5
    feature / match files compatible with the Hloc pipeline.
    """
    pairs_str = str(pairs)
    image_dir_str = str(image_dir)
    image_dir_ref_str = str(image_dir_ref)
    matches_str = str(matches) if matches is not None else None
    features_str = str(features) if features is not None else None
    features_ref_str = str(features_ref) if features_ref is not None else None

    if matches_str is not None and os.path.isfile(matches_str) and not overwrite:
        return

    assert os.path.isfile(pairs_str)
    retrieval_pairs = parse_retrieval_file(pairs_str)

    if overwrite:
        for f in (matches_str, features_str, features_ref_str):
            if f is not None and os.path.isfile(f):
                os.remove(f)

    matches_tmp = matches_str + ".tmp"
    pairs_num = sum(len(v) for v in retrieval_pairs.values())

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    roma_model = _load_roma_model(conf, device)

    pbar = tqdm(total=pairs_num)
    for query_name, ret_list in retrieval_pairs.items():
        query_path = os.path.join(image_dir_str, query_name)
        assert os.path.isfile(query_path), f"Query image not found: {query_path}"
        query_h, query_w, _ = cv2.imread(query_path).shape

        for ref_name in ret_list:
            ref_path = os.path.join(image_dir_ref_str, ref_name)
            assert os.path.isfile(ref_path), f"Reference image not found: {ref_path}"
            ref_h, ref_w, _ = cv2.imread(ref_path).shape

            warp, certainty = roma_model.match(query_path, ref_path, device=device)
            matches_inter, certainty = roma_model.sample(warp, certainty, num=max_kps)
            kpts_query, kpts_ref = roma_model.to_pixel_coordinates(
                matches_inter, query_h, query_w, ref_h, ref_w
            )
            certainty = certainty.cpu().detach().numpy()
            kpts_query = kpts_query.cpu().detach().numpy()
            kpts_ref = kpts_ref.cpu().detach().numpy()

            pair = names_to_pair(query_name, ref_name)
            if os.path.isfile(matches_str):
                with h5py.File(matches_str, "r") as f_fin:
                    if pair in f_fin and not overwrite:
                        pbar.update(1)
                        continue

            with h5py.File(features_str, "a") as f_feat:
                prev_kpts = f_feat[query_name]["keypoints"][:].astype(np.float32) if query_name in f_feat else np.empty((0, 2), dtype=np.float32)
                if query_name in f_feat:
                    del f_feat[query_name]
                all_kpts, match_idxs_query = _merge_keypoints(prev_kpts, kpts_query)
                f_feat.create_group(query_name).create_dataset("keypoints", data=all_kpts)

            with h5py.File(features_ref_str, "a") as f_feat_ref:
                prev_kpts = f_feat_ref[ref_name]["keypoints"][:].astype(np.float32) if ref_name in f_feat_ref else np.empty((0, 2), dtype=np.float32)
                if ref_name in f_feat_ref:
                    del f_feat_ref[ref_name]
                all_kpts, match_idxs_ref = _merge_keypoints(prev_kpts, kpts_ref)
                f_feat_ref.create_group(ref_name).create_dataset("keypoints", data=all_kpts)

            matches0 = np.vstack((match_idxs_query, match_idxs_ref)).T
            with h5py.File(matches_tmp, "a") as f_matches:
                if pair in f_matches:
                    del f_matches[pair]
                grp = f_matches.create_group(pair)
                grp.create_dataset("matches0", data=matches0)
                grp.create_dataset("matching_scores0", data=certainty)

            pbar.update(1)
    pbar.close()

    print("Converting match file to Hloc format")
    with (
        h5py.File(matches_tmp, "r") as f_tmp,
        h5py.File(matches_str, "a") as f_fin,
        h5py.File(features_str, "r") as f_feat,
    ):
        for query_name, ret_list in retrieval_pairs.items():
            query_feat_num = f_feat[query_name]["keypoints"].shape[0]
            for ref_name in ret_list:
                pair = names_to_pair(query_name, ref_name)
                if pair in f_fin:
                    del f_fin[pair]
                grp = f_fin.create_group(pair)

                in_matches = f_tmp[pair]["matches0"][:].astype(int)
                matches0 = np.full(query_feat_num, -1)
                matches0[in_matches[:, 0]] = in_matches[:, 1]

                in_scores = f_tmp[pair]["matching_scores0"][:]
                matching_scores0 = np.zeros(query_feat_num)
                matching_scores0[in_matches[:, 0]] = in_scores

                grp.create_dataset("matches0", data=matches0)
                grp.create_dataset("matching_scores0", data=matching_scores0)

    os.remove(matches_tmp)
    print("RoMA matching done")
    print(f"  features query : {features_str} — exists: {os.path.isfile(features_str)}")
    print(f"  features ref   : {features_ref_str} — exists: {os.path.isfile(features_ref_str)}")
    print(f"  matches        : {matches_str} — exists: {os.path.isfile(matches_str)}")
