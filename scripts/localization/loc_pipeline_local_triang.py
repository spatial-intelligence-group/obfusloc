#!/usr/bin/env python3

"""
Localize query images with known intrinsics relative to fully calibrated
reference images using absolute pose estimation on a locally triangulated
point cloud, with fallback to generalized relative pose.
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import poselib
from hloc import extract_features, match_features, pairs_from_retrieval
from hloc.utils.io import get_keypoints, get_matches
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import roma_utils
from cam_utils import (
    IMAGE_EXTS,
    T2qt,
    load_cameras,
    match_keypoints_nn_mv,
    parse_retrieval_file,
    qt2T,
    sampson_dist,
    setup_poselib_intrinsics,
    triang_single_point_mv_ransac,
    write_cams_txt,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize query images using absolute pose on a local triangulated point cloud."
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
        "--ransac_max_reproj_error",
        type=float,
        default=12.0,
        help="Reprojection error threshold for absolute pose RANSAC",
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
    parser.add_argument(
        "--ransac_max_epipolar_error",
        type=float,
        default=1.0,
        help="Epipolar error threshold for the generalized relative pose fallback",
    )
    parser.add_argument(
        "--triang_re_threshold",
        type=float,
        default=1.0,
        help="Reprojection error threshold used in triangulation",
    )
    parser.add_argument(
        "--use_n_best_matches",
        type=int,
        default=-1,
        help="Limit matching to the N highest-scoring matches per pair (-1 = no limit)",
    )
    parser.add_argument(
        "--use_score_over",
        type=float,
        default=0.0,
        help="Discard matches with score below this threshold (0.0 = keep all)",
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

    if "roma" in args.hloc_matcher and "superpoint+" not in args.hloc_matcher:
        lf = args.hloc_matcher

    # for hypothetical superpoint+roma hybrid matcher: use a distinct feature file name
    if "superpoint+" in args.hloc_matcher and "roma" in args.hloc_matcher:
        ref_local_feat_file = workdir / f"ref_local_feat_top_{k}_{lf}_warped_by_{gf}_{args.hloc_matcher}.h5"
    else:
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
    match_mode = "matcher"
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
        match_mode = "nn"
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

    # absolute pose estimation
    print("Estimating poses with absolute pose estimator using local point cloud")
    retrieval_pairs = parse_retrieval_file(retrieval_pairs_file)
    est_pose_dict = {}

    ransac_options = {
        "max_iterations": args.ransac_max_iterations,
        "min_iterations": args.ransac_min_iterations,
        "max_reproj_error": args.ransac_max_reproj_error,
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

        retrieval_ref_valid = [
            ref for ref in retrieval_pairs[query_name]
            if os.path.splitext(ref)[0] in ref_info_dict
        ]

        T_query, inliers = get_abs_pose_robust_triang(
            query_name,
            query_info_dict,
            ref_info_dict,
            retrieval_ref_valid,
            matches_file,
            query_local_feat_file,
            ref_local_feat_file,
            ransac_options,
            bundle_options,
            use_n_best_matches=args.use_n_best_matches,
            use_score_over=args.use_score_over,
            match_mode=match_mode,
            re_threshold=args.triang_re_threshold,
        )

        if T_query is None:
            # P3P on local point cloud failed — fall back to generalized relative pose
            T_query, inliers = get_gen_rel_pose(
                query_name,
                query_info_dict,
                ref_info_dict,
                retrieval_ref_valid,
                matches_file,
                query_local_feat_file,
                ref_local_feat_file,
                ransac_options,
                bundle_options,
                use_n_best_matches=args.use_n_best_matches,
                use_score_over=args.use_score_over,
            )

        est_pose_dict[query_name] = {"T": T_query}

    write_cams_txt(args.est_file, est_pose_dict)


def get_abs_pose_robust_triang(
    query_name: str,
    query_info_dict: dict,
    ref_info_dict: dict,
    ref_cam_names: list[str],
    matches_file: Path,
    query_local_feat_file: Path,
    ref_local_feat_file: Path,
    abs_solver_ransac_opt: dict,
    abs_solver_bundle_opt: dict,
    use_n_best_matches: int = -1,
    use_score_over: float = 0.0,
    match_mode: str = "matcher",
    re_threshold: float = 1.0,
) -> tuple[np.ndarray | None, int]:
    pnt_set_list_query, pnt_set_list_ref = collect_2D_points(
        query_name,
        ref_cam_names,
        matches_file,
        query_local_feat_file,
        ref_local_feat_file,
        use_n_best_matches,
        use_score_over,
    )

    query_points2D_all, ref_points2D_list, ref_cam_idx_list = get_matches_direct_mv(
        matches_file,
        query_local_feat_file,
        ref_local_feat_file,
        query_name,
        ref_cam_names,
        use_n_best_matches=-1,
        use_score_over=0.0,
        match_mode=match_mode,
    )

    points3D_all, valid3D = triang_points_2D_mv_ransac(
        ref_info_dict,
        ref_cam_names,
        ref_points2D_list,
        ref_cam_idx_list,
        re_threshold,
    )

    query_points2D_all = query_points2D_all[valid3D, :]
    points3D_all = points3D_all[:, valid3D]

    if query_points2D_all.shape[0] < 3:
        return None, 0

    query_stem = os.path.splitext(query_name)[0]
    query_intrinsics_poselib = setup_poselib_intrinsics(query_info_dict[query_stem])

    pose, info = poselib.estimate_absolute_pose(
        query_points2D_all.astype(np.float64),
        points3D_all.T.astype(np.float64),
        query_intrinsics_poselib,
        abs_solver_ransac_opt,
        abs_solver_bundle_opt,
    )

    if np.isnan(pose.q).any():
        print("WARN: estimator returned a quaternion with NaN values")
        return None, 0
    if np.sum(pose.q**2) == 0.0:
        print("WARN: estimator returned a zero-norm quaternion")
        return None, 0

    T_query = qt2T(pose.q, pose.t)

    inliers = compute_inliers_ee(
        T_query,
        query_info_dict[query_stem]["K"],
        ref_info_dict,
        ref_cam_names,
        pnt_set_list_query,
        pnt_set_list_ref,
        abs_solver_ransac_opt["max_epipolar_error"],
    )

    return T_query, inliers


def get_gen_rel_pose(
    query_name: str,
    query_info_dict: dict,
    ref_info_dict: dict,
    ref_cam_names: list[str],
    matches_file: Path,
    query_local_feat_file: Path,
    ref_local_feat_file: Path,
    gen_rel_solver_ransac_opt: dict,
    gen_rel_solver_bundle_opt: dict,
    use_n_best_matches: int = -1,
    use_score_over: float = 0.0,
) -> tuple[np.ndarray | None, int]:
    query_stem = os.path.splitext(query_name)[0]
    query_cam_dict = setup_poselib_intrinsics(query_info_dict[query_stem])
    query_cam_dict_list = [query_cam_dict]
    query_pose_list = [poselib.CameraPose()]

    pnt_set_list_query, pnt_set_list_ref = collect_2D_points(
        query_name,
        ref_cam_names,
        matches_file,
        query_local_feat_file,
        ref_local_feat_file,
        use_n_best_matches,
        use_score_over,
    )

    has_5_matches = False
    has_1_match = False
    ref_cam_dict_list = []
    ref_pose_list = []
    pairwise_matches_list = []
    # cam_id in PoseLib counts only cameras added to ref_cam_dict_list
    poselib_cam_id = 0

    for ref_cam_idx, ref_name in enumerate(ref_cam_names):
        ref_stem = os.path.splitext(ref_name)[0]
        if ref_stem not in ref_info_dict:
            continue
        ref_cam_dict = ref_info_dict[ref_stem]

        # pnt_set_list is aligned with ref_cam_names (one entry per element)
        query_points2D = pnt_set_list_query[ref_cam_idx]
        ref_points2D = pnt_set_list_ref[ref_cam_idx]

        if ref_points2D.shape[0] == 0:
            continue
        elif ref_points2D.shape[0] >= 5:
            if not has_5_matches:
                has_5_matches = True
            else:
                has_1_match = True
        else:
            has_1_match = True

        ref_cam_dict_list.append(setup_poselib_intrinsics(ref_cam_dict))

        ref_pose = poselib.CameraPose()
        ref_qvec, ref_tvec = T2qt(ref_cam_dict["T"])
        ref_pose.q = ref_qvec.tolist()
        ref_pose.t = ref_tvec
        ref_pose_list.append(ref_pose)

        pairwise_matches = poselib.PairwiseMatches()
        pairwise_matches.cam_id1 = poselib_cam_id
        pairwise_matches.cam_id2 = 0
        pairwise_matches.x1 = ref_points2D
        pairwise_matches.x2 = query_points2D
        pairwise_matches_list.append(pairwise_matches)

        poselib_cam_id += 1

    if not has_5_matches or not has_1_match:
        print("WARN: not enough matches (need ≥5 from one ref + ≥1 from another) — skipping")
        return None, 0

    pose, info = poselib.estimate_generalized_relative_pose(
        pairwise_matches_list,
        ref_pose_list,
        ref_cam_dict_list,
        query_pose_list,
        query_cam_dict_list,
        gen_rel_solver_ransac_opt,
        gen_rel_solver_bundle_opt,
    )

    if np.isnan(pose.q).any():
        print("WARN: estimator returned a quaternion with NaN values")
        return None, 0
    if np.sum(pose.q**2) == 0.0:
        print("WARN: estimator returned a zero-norm quaternion")
        return None, 0

    return qt2T(pose.q, pose.t), info["num_inliers"]


def get_matches_direct_mv(
    matches_file: Path,
    query_local_feat_file: Path,
    ref_local_feat_file: Path,
    query_name: str,
    ref_name_list: list[str],
    use_n_best_matches: int = -1,
    use_score_over: float = 0.0,
    match_mode: str = "matcher",
) -> tuple[np.ndarray, list[np.ndarray], list[list[int]]]:
    """
    Find query keypoints with matches in ≥2 reference images.

    Returns:
        query_points2D: (M, 2) array of query keypoint coordinates.
        ref_points2D_list: list of M arrays, each with the corresponding
            reference points for one query keypoint.
        ref_cam_idx_list: list of M lists of reference camera indices.
    """
    if match_mode == "matcher":
        query_keypoints = get_keypoints(query_local_feat_file, query_name)
        ref_keypoints_list = []
        # (num_refs, num_query_kpts) — value is ref kpt index, -1 if no match
        match_array = np.empty((0, query_keypoints.shape[0]), dtype=int)

        for ref_name in ref_name_list:
            match_idxs = get_matches_filtered(
                matches_file, query_name, ref_name, use_n_best_matches, use_score_over
            )
            row = np.full((1, match_array.shape[1]), -1)
            row[0, match_idxs[:, 0]] = match_idxs[:, 1]
            match_array = np.vstack((match_array, row))
            ref_keypoints_list.append(get_keypoints(ref_local_feat_file, ref_name))

        valid_kpt_idxs = np.where(np.sum(match_array != -1, axis=0) >= 2)[0]
        query_points2D = query_keypoints[valid_kpt_idxs, :] + 0.5
        ref_points2D_list = []
        ref_cam_idx_list = []

        for kpt_idx in valid_kpt_idxs:
            ref_pts = []
            ref_cams = []
            for ref_idx, ref_kpts in enumerate(ref_keypoints_list):
                ref_match_idx = match_array[ref_idx, kpt_idx]
                if ref_match_idx != -1:
                    ref_pts.append(ref_kpts[ref_match_idx, :] + 0.5)
                    ref_cams.append(ref_idx)
            ref_points2D_list.append(np.array(ref_pts))
            ref_cam_idx_list.append(ref_cams)

    elif match_mode == "nn":
        query_points2D_all = get_keypoints(query_local_feat_file, query_name)
        query_keypoints_list = []
        ref_keypoints_list = []

        for ref_name in ref_name_list:
            match_idxs = get_matches_filtered(
                matches_file, query_name, ref_name, use_n_best_matches, use_score_over
            )
            if match_idxs.shape[0] == 0:
                query_keypoints_list.append(np.empty((0, 2)))
                ref_keypoints_list.append(np.empty((0, 2)))
                continue
            query_keypoints_list.append(query_points2D_all[match_idxs[:, 0], :] + 0.5)
            ref_keypoints_list.append(
                get_keypoints(ref_local_feat_file, ref_name)[match_idxs[:, 1], :] + 0.5
            )

        track_idxs = match_keypoints_nn_mv(query_keypoints_list, dist_thresh=5.0)
        query_points2D = np.empty((0, 2))
        ref_points2D_list = []
        ref_cam_idx_list = []

        for track in track_idxs:
            query_pts_arr = np.vstack([query_keypoints_list[img_idx][kpt_idx, :] for img_idx, kpt_idx in track])
            mean_query_pnt = np.mean(query_pts_arr, axis=0)
            query_points2D = np.vstack((query_points2D, mean_query_pnt))
            ref_points2D_list.append(
                np.array([ref_keypoints_list[img_idx][kpt_idx, :] for img_idx, kpt_idx in track])
            )
            ref_cam_idx_list.append([img_idx for img_idx, _ in track])
    else:
        raise ValueError(f"Unknown match_mode: {match_mode!r}")

    return query_points2D, ref_points2D_list, ref_cam_idx_list


def triang_points_2D_mv_ransac(
    ref_info_dict: dict,
    ref_name_list: list[str],
    ref_points2D_list: list[np.ndarray],
    ref_cam_idx_list: list[list[int]],
    re_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    points3D = np.empty((3, 0))
    for ref_points2D, ref_cam_idx_arr in zip(ref_points2D_list, ref_cam_idx_list):
        x_list = []
        ref_cam_list = []
        for ref_pnt2D, ref_cam_idx in zip(ref_points2D, ref_cam_idx_arr):
            ref_name = ref_name_list[ref_cam_idx]
            ref_stem = os.path.splitext(ref_name)[0]
            x_list.append(ref_pnt2D)
            ref_cam_list.append(ref_info_dict[ref_stem])
        pnt_3D = triang_single_point_mv_ransac(x_list, ref_cam_list, re_threshold)
        points3D = np.hstack((points3D, pnt_3D.reshape((3, 1))))

    valid = ~(np.isinf(points3D[0, :]) | np.isnan(points3D[0, :]))
    return points3D, valid


def collect_2D_points(
    query_name: str,
    ref_names: list[str],
    matches_file: Path,
    query_local_feat_file: Path,
    ref_local_feat_file: Path,
    use_n_best_matches: int = -1,
    use_score_over: float = 0.0,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    pnt_set_list_query = []
    pnt_set_list_ref = []

    for ref_name in ref_names:
        match_idxs = get_matches_filtered(
            matches_file, query_name, ref_name, use_n_best_matches, use_score_over
        )
        if match_idxs.shape[0] == 0:
            pnt_set_list_query.append(np.empty((0, 2)))
            pnt_set_list_ref.append(np.empty((0, 2)))
            continue
        pnt_set_list_query.append(
            get_keypoints(query_local_feat_file, query_name)[match_idxs[:, 0], :] + 0.5
        )
        pnt_set_list_ref.append(
            get_keypoints(ref_local_feat_file, ref_name)[match_idxs[:, 1], :] + 0.5
        )

    return pnt_set_list_query, pnt_set_list_ref


def get_matches_filtered(
    matches_file: Path,
    query_name: str,
    ref_name: str,
    use_n_best_matches: int = -1,
    use_score_over: float = 0.0,
    verbose: bool = False,
) -> np.ndarray:
    try:
        matches, scores = get_matches(matches_file, query_name, ref_name)
    except ValueError:
        return np.empty((0, 2))

    orig_match_num = len(scores)

    score_order = np.flip(np.argsort(scores))
    matches = matches[score_order, :]
    scores = scores[score_order]

    if use_score_over > 0.0:
        select_idxs = scores >= use_score_over
        matches = matches[select_idxs, :]
        scores = scores[select_idxs]
        if verbose:
            print(f"limiting to {len(scores)}/{orig_match_num} matches by score threshold")

    if use_n_best_matches > 0 and matches.shape[0] > use_n_best_matches:
        matches = matches[:use_n_best_matches, :]
        if verbose:
            print(f"using {use_n_best_matches}/{orig_match_num} best matches")

    return matches


def compute_inliers_ee(
    T_query: np.ndarray,
    K_query: np.ndarray,
    ref_cam_dict: dict,
    ref_cam_names: list[str],
    pnt_set_list_query: list[np.ndarray],
    pnt_set_list_ref: list[np.ndarray],
    sampson_err_thr: float,
) -> int:
    sampson_err_thr_sq = sampson_err_thr**2
    inliers = 0

    for ref_idx, ref_name in enumerate(ref_cam_names):
        ref_name_stem = os.path.splitext(ref_name)[0]
        ref_dict = ref_cam_dict[ref_name_stem]
        T_ref = ref_dict["T"]
        K_ref = ref_dict["K"]

        pnt_set_query = pnt_set_list_query[ref_idx]
        pnt_set_ref = pnt_set_list_ref[ref_idx]

        se_sq = np.array(
            sampson_dist(
                pnt_set_query.T,
                K_query,
                T_query,
                pnt_set_ref.T,
                K_ref,
                T_ref,
                squared=True,
            )
        )
        inliers += int(np.sum(se_sq <= sampson_err_thr_sq))

    return inliers


if __name__ == "__main__":
    main()
