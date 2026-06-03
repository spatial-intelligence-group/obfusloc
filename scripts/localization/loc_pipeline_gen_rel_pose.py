#!/usr/bin/env python3

"""
Localize query images with known intrinsics relative to fully calibrated
reference images using a generalized relative pose estimator.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import poselib
sys.path.append(str(Path(__file__).parent.parent.parent / "third_party"))  # for SuperGluePretrainedNetwork
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import roma_utils
from cam_utils import (
    IMAGE_EXTS,
    T2qt,
    load_cameras,
    parse_retrieval_file,
    qt2T,
    setup_poselib_intrinsics,
    write_cams_txt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize query images using generalized relative pose estimation."
    )
    parser.add_argument(
        "query_images",
        type=str,
        help="Path to the directory with query images",
    )
    parser.add_argument(
        "query_intrinsics",
        type=str,
        help="Path to the .txt file with query intrinsics",
    )
    parser.add_argument(
        "ref_images",
        type=str,
        help="Path to the directory with reference images",
    )
    parser.add_argument(
        "ref_colmap",
        type=str,
        help="Path to the reference COLMAP model directory",
    )
    parser.add_argument(
        "hloc_workdir",
        type=str,
        help="Path to a directory for Hloc intermediate files",
    )
    parser.add_argument(
        "est_file",
        type=str,
        help="Path to the output .txt file with pose estimates in Visual Localization Benchmark format",
    )
    parser.add_argument(
        "--hloc_global_feature",
        type=str,
        choices=["dir", "netvlad", "openibl", "eigenplaces"],
        default="eigenplaces",
        help="Global feature type for image retrieval",
    )
    parser.add_argument(
        "--hloc_local_feature",
        type=str,
        choices=[
            "superpoint_aachen", "superpoint_max", "superpoint_inloc",
            "r2d2", "d2net-ss", "sift", "sosnet", "disk", "aliked-n16",
            "roma_outdoor", "roma_indoor",
        ],
        default="roma_outdoor",
        help="Local feature type",
    )
    parser.add_argument(
        "--hloc_matcher",
        type=str,
        choices=[
            "superpoint+lightglue", "disk+lightglue", "aliked+lightglue",
            "superglue", "superglue-fast", "NN-superpoint", "NN-ratio",
            "NN-mutual", "adalam", "roma_outdoor", "roma_indoor",
        ],
        default="roma_outdoor",
        help="Local feature matcher",
    )
    parser.add_argument(
        "--ret_top_k",
        type=int,
        default=10,
        help="Number of reference images retrieved per query",
    )
    parser.add_argument(
        "--ransac_max_epipolar_error",
        type=float,
        default=1.0,
        help="Epipolar error threshold for RANSAC",
    )
    parser.add_argument(
        "--ransac_min_iterations",
        type=int,
        default=1000,
        help="Minimum number of RANSAC iterations",
    )
    parser.add_argument(
        "--ransac_max_iterations",
        type=int,
        default=100000,
        help="Maximum number of RANSAC iterations",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.query_images):
        parser.error(f"query_images directory not found: {args.query_images}")
    if not os.path.exists(args.query_intrinsics):
        parser.error(f"query_intrinsics file not found: {args.query_intrinsics}")
    if not os.path.isdir(args.ref_images):
        parser.error(f"ref_images directory not found: {args.ref_images}")
    if not os.path.exists(args.ref_colmap):
        parser.error(f"ref_colmap path not found: {args.ref_colmap}")

    # load camera parameters
    query_list = sorted([
        os.path.relpath(os.path.join(dp, f), args.query_images)
        for dp, _, filenames in os.walk(args.query_images)
        for f in filenames
        if os.path.isfile(os.path.join(dp, f)) and os.path.splitext(f)[1] in IMAGE_EXTS
    ])
    if not query_list:
        parser.error(f"No images found in query directory: {args.query_images}")

    query_info_dict = load_cameras(args.query_intrinsics)

    if len(query_info_dict) == 0:
        print(f"Query camera info file is empty — writing empty estimates file: {args.est_file}")
        write_cams_txt(args.est_file, {})
        return

    ref_info_dict = load_cameras(args.ref_colmap)

    # define Hloc intermediate file paths
    workdir = Path(args.hloc_workdir)
    gf = args.hloc_global_feature
    lf = args.hloc_local_feature
    k = args.ret_top_k

    ref_global_feat_file = workdir / f"ref_global_feat_{gf}.h5"
    query_global_feat_file = workdir / f"query_global_feat_{gf}.h5"
    retrieval_pairs_file = workdir / f"retrieval_pairs_{gf}_top_{k}.txt"
    matches_file = workdir / f"matches_{gf}_top_{k}_{lf}_{args.hloc_matcher}.h5"
    ref_local_feat_file = workdir / f"ref_local_feat_top_{k}_{lf}.h5"
    query_local_feat_file = workdir / f"query_local_feat_top_{k}_{lf}.h5"

    # global feature extraction and image retrieval
    hloc_global_feat_conf = extract_features.confs[gf]
    extract_features.main(hloc_global_feat_conf, Path(args.ref_images), feature_path=ref_global_feat_file)
    extract_features.main(hloc_global_feat_conf, Path(args.query_images), feature_path=query_global_feat_file)
    pairs_from_retrieval.main(
        descriptors=query_global_feat_file,
        db_descriptors=ref_global_feat_file,
        output=retrieval_pairs_file,
        num_matched=k,
    )

    # local feature extraction and matching
    if "roma" in args.hloc_matcher:
        roma_match_conf = roma_utils.confs[args.hloc_matcher]
        roma_utils.roma_match(
            roma_match_conf,
            retrieval_pairs_file,
            Path(args.query_images),
            Path(args.ref_images),
            matches=matches_file,
            features=query_local_feat_file,
            features_ref=ref_local_feat_file,
            max_kps=1024,
        )
    else:
        hloc_local_feat_conf = extract_features.confs[lf]
        extract_features.main(hloc_local_feat_conf, Path(args.ref_images), feature_path=ref_local_feat_file)
        extract_features.main(hloc_local_feat_conf, Path(args.query_images), feature_path=query_local_feat_file)
        hloc_match_conf = match_features.confs[args.hloc_matcher]
        match_features.main(
            hloc_match_conf,
            retrieval_pairs_file,
            features=query_local_feat_file,
            features_ref=ref_local_feat_file,
            matches=matches_file,
        )

    # generalized relative pose estimation
    print("Estimating poses with generalized relative pose estimator")
    retrieval_pairs = parse_retrieval_file(retrieval_pairs_file)
    est_pose_dict = {}

    ransac_options = {
        "max_iterations": args.ransac_max_iterations,
        "min_iterations": args.ransac_min_iterations,
        "max_epipolar_error": args.ransac_max_epipolar_error,
        "seed": 42,
    }
    bundle_options = {"max_iterations": 100}

    for query_name in tqdm(query_list):
        query_stem = os.path.splitext(query_name)[0]
        if query_stem not in query_info_dict:
            print(f'WARN: query "{query_stem}" not in intrinsics file — skipping')
            continue
        if query_name not in retrieval_pairs:
            print(f'WARN: query "{query_name}" not in retrieval pairs file — skipping')
            continue

        query_cam_dict = setup_poselib_intrinsics(query_info_dict[query_stem])
        query_keypoints = np.array(get_keypoints(query_local_feat_file, query_name)).astype(np.float64)
        query_keypoints += 0.5

        query_cam_dict_list = [query_cam_dict]
        query_pose_list = [poselib.CameraPose()]
        ref_cam_dict_list = []
        ref_pose_list = []
        pairwise_matches_list = []

        ref_idx_wo_skipped = 0
        has_5_matches = False
        has_1_match = False

        for ref_name in retrieval_pairs[query_name]:
            ref_stem = os.path.splitext(ref_name)[0]
            if ref_stem not in ref_info_dict:
                continue
            ref_cam_dict = ref_info_dict[ref_stem]
            ref_keypoints = np.array(
                get_keypoints(ref_local_feat_file, ref_name)
            ).astype(np.float64)
            ref_keypoints += 0.5

            matches, scores = get_matches(matches_file, query_name, ref_name)
            matches = matches.astype(np.uint32)

            if len(scores) == 0:
                continue
            elif len(scores) >= 5:
                if not has_5_matches:
                    has_5_matches = True
                else:
                    has_1_match = True
            else:
                has_1_match = True

            query_points2D = query_keypoints[matches[:, 0], :]
            ref_points2D = ref_keypoints[matches[:, 1], :]

            ref_cam_dict_list.append(setup_poselib_intrinsics(ref_cam_dict))

            ref_pose = poselib.CameraPose()
            ref_qvec, ref_tvec = T2qt(ref_cam_dict["T"])
            ref_pose.q = ref_qvec.tolist()
            ref_pose.t = ref_tvec
            ref_pose_list.append(ref_pose)

            pairwise_matches = poselib.PairwiseMatches()
            pairwise_matches.cam_id1 = ref_idx_wo_skipped
            pairwise_matches.cam_id2 = 0
            pairwise_matches.x1 = ref_points2D
            pairwise_matches.x2 = query_points2D
            pairwise_matches_list.append(pairwise_matches)

            ref_idx_wo_skipped += 1

        if not has_5_matches or not has_1_match:
            print("WARN: not enough matches (need ≥5 from one ref + ≥1 from another) — skipping")
            continue

        query_pose, info = poselib.estimate_generalized_relative_pose(
            pairwise_matches_list,
            ref_pose_list,
            ref_cam_dict_list,
            query_pose_list,
            query_cam_dict_list,
            ransac_options,
            bundle_options,
        )

        est_pose_dict[query_name] = {"T": qt2T(query_pose.q, query_pose.t)}

    print(f"Writing pose estimates to: {args.est_file}")
    write_cams_txt(args.est_file, est_pose_dict)


if __name__ == "__main__":
    main()
