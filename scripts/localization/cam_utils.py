"""
Camera calibration and localization utility functions.

Extrinsics convention: T is a 4x4 world-to-camera transform.
Intrinsics convention: K is a 3x3 pinhole matrix with fx=K[0,0], fy=K[1,1],
cx=K[0,2], cy=K[1,2], in pixels.
Quaternion convention: WXYZ by default; pass quat_order_str="XYZW" to switch.
"""

import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ── Constants ──────────────────────────────────────────────────────────────────

IMAGE_EXTS = (".jpg", ".png")


# ── Geometry ───────────────────────────────────────────────────────────────────


def q2R(q: np.ndarray, quat_order_str: str = "WXYZ") -> np.ndarray:
    assert len(q) == 4
    quat_order_str = quat_order_str.lower()
    assert quat_order_str in ("wxyz", "xyzw")
    scalar_first = quat_order_str == "wxyz"
    return Rotation.from_quat(q, scalar_first=scalar_first).as_matrix()


def R2q(R: np.ndarray, quat_order_str: str = "WXYZ") -> np.ndarray:
    assert R.shape == (3, 3)
    quat_order_str = quat_order_str.lower()
    assert quat_order_str in ("wxyz", "xyzw")
    scalar_first = quat_order_str == "wxyz"
    return Rotation.from_matrix(R).as_quat(scalar_first=scalar_first)


def qt2T(q: np.ndarray, t: np.ndarray, quat_order_str: str = "WXYZ") -> np.ndarray:
    assert len(q) == 4
    assert len(t) == 3
    R = q2R(q, quat_order_str)
    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T


def T2qt(T: np.ndarray, quat_order_str: str = "WXYZ") -> tuple[np.ndarray, np.ndarray]:
    assert T.shape == (4, 4)
    q = R2q(T[:3, :3], quat_order_str)
    t = T[0:3, 3]
    return q, t


def skew(vec: np.ndarray) -> np.ndarray:
    assert len(vec) == 3
    vec = vec.squeeze()
    return np.array([[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]])


def construct_essential_mat(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    R1, t1 = T1[0:3, 0:3], T1[0:3, None, 3]
    R2, t2 = T2[0:3, 0:3], T2[0:3, None, 3]
    R_rel = R2 @ R1.T
    t_rel = t2 - R_rel @ t1
    return skew(t_rel) @ R_rel


def construct_fundamental_mat(
    K1: np.ndarray, T1: np.ndarray, K2: np.ndarray, T2: np.ndarray
) -> np.ndarray:
    E = construct_essential_mat(T1, T2)
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)


def sampson_dist(
    pnts1: np.ndarray,
    K1: np.ndarray,
    T1: np.ndarray,
    pnts2: np.ndarray,
    K2: np.ndarray,
    T2: np.ndarray,
    squared: bool = False,
) -> list[float]:
    """
    Sampson distance for point correspondences.

    pnts1, pnts2: (2, N) arrays of 2-D image points.
    Returns list of N distances (or squared distances).
    """
    N = pnts1.shape[1]
    F = construct_fundamental_mat(K1, T1, K2, T2)
    p1h = np.vstack([pnts1, np.ones((1, N))])   # (3, N) homogeneous
    p2h = np.vstack([pnts2, np.ones((1, N))])
    Fp1 = F @ p1h       # (3, N)
    Ftp2 = F.T @ p2h    # (3, N)
    # squared Sampson distance: (x2^T F x1)^2 / (||Fx1||_xy^2 + ||F^Tx2||_xy^2)
    numer = (p2h * Fp1).sum(axis=0) ** 2
    denom = Fp1[0] ** 2 + Fp1[1] ** 2 + Ftp2[0] ** 2 + Ftp2[1] ** 2
    sd_sq = numer / denom
    return sd_sq.tolist() if squared else np.sqrt(sd_sq).tolist()


def triang_single_point_mv_ransac(
    pnts_list: list[np.ndarray],
    cam_dict_list: list[dict],
    re_threshold: float,
    max_iters: int = 400,
) -> np.ndarray:
    """
    Triangulate a single 3-D point from multiple views using RANSAC over pairs.

    pnts_list: distorted 2-D image points, one per camera.
    cam_dict_list: camera dicts with "T" and "colmap_cam" keys.
    Returns a (3,) array; NaN on failure.
    """
    re_threshold2 = re_threshold**2
    n = len(pnts_list)

    # precompute per-camera quantities once
    Ks = [get_K_from_cam_dict(cd) for cd in cam_dict_list]
    Ps = [Ks[k] @ cam_dict_list[k]["T"][0:3, :] for k in range(n)]
    dists = [get_opencv_distortion_params(cd) for cd in cam_dict_list]
    rcs = [cv2.Rodrigues(cd["T"][0:3, 0:3])[0].flatten() for cd in cam_dict_list]
    tcs = [cd["T"][0:3, 3] for cd in cam_dict_list]
    xs = [pnts_list[k].flatten() for k in range(n)]

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if len(all_pairs) > max_iters:
        sel = np.random.choice(len(all_pairs), size=max_iters, replace=False)
        all_pairs = [all_pairs[i] for i in sel]

    best_num_inliers = 0
    best_sum_re2 = float("inf")
    best_X = np.full(3, float("nan"))

    for i, j in all_pairs:
        x1_norm = cv2.undistortPoints(xs[i][None, :].astype(np.float32), np.eye(3), dists[i])[0, 0]
        x2_norm = cv2.undistortPoints(xs[j][None, :].astype(np.float32), np.eye(3), dists[j])[0, 0]

        X = cv2.triangulatePoints(Ps[i], Ps[j], x1_norm[:, None], x2_norm[:, None])
        X = (X[:3] / X[3]).flatten().astype(np.float32)

        if np.isnan(X).any() or np.isinf(X).any():
            continue

        num_inliers = 0
        sum_re2 = 0.0
        for k in range(n):
            x_proj = cv2.projectPoints(X[None, :], rcs[k], tcs[k], Ks[k], dists[k])[0].flatten()
            re2 = (x_proj[0] - xs[k][0]) ** 2 + (x_proj[1] - xs[k][1]) ** 2
            if re2 <= re_threshold2:
                num_inliers += 1
                sum_re2 += re2

        if num_inliers < 2:
            continue

        if num_inliers > best_num_inliers or (
            num_inliers == best_num_inliers and (sum_re2 / num_inliers) < best_sum_re2
        ):
            best_num_inliers = num_inliers
            best_X = X
            best_sum_re2 = sum_re2

    return best_X


# ── Camera IO ──────────────────────────────────────────────────────────────────


def complete_cam_params(cam_params: dict) -> dict:
    new = {}
    new["name"] = cam_params.get("name")
    new["w"] = cam_params.get("w", 1920)
    new["h"] = cam_params.get("h", 1080)
    if "K" in cam_params:
        new["K"] = cam_params["K"]
    else:
        new["K"] = np.array([
            [new["h"], 0, new["w"] / 2],
            [0, new["h"], new["h"] / 2],
            [0, 0, 1],
        ])
    new["T"] = cam_params.get("T", np.eye(4))
    if "colmap_cam" in cam_params:
        new["colmap_cam"] = cam_params["colmap_cam"]
    else:
        K = new["K"]
        new["colmap_cam"] = {
            "model": "PINHOLE",
            "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]],
        }
    return new


def load_cams_colmap(dir_path: str) -> dict:
    import pycolmap

    assert os.path.isdir(dir_path)
    model = pycolmap.Reconstruction(dir_path)

    cam_dict = {}
    for img in model.images.values():
        img_name = os.path.splitext(img.name)[0]
        cam = model.cameras[img.camera_id]
        cam_dict[img_name] = {
            "name": img_name,
            "w": cam.width,
            "h": cam.height,
            "K": cam.calibration_matrix(),
        }
        T = np.eye(4)
        if hasattr(img, "tvec"):
            T[0:3, 0:3] = img.rotation_matrix()
            T[0:3, 3] = img.tvec
        else:
            T[0:3, :] = img.cam_from_world().matrix()
        cam_dict[img_name]["T"] = T
        cam_dict[img_name]["colmap_cam"] = {"model": cam.model.name, "params": cam.params}

    return dict(sorted(cam_dict.items()))


def load_cams_txt(path: str) -> dict:
    """MeshLoc query list format: <name> <colmap_model> <w> <h> [<params>]"""
    assert os.path.isfile(path)
    cam_dict = {}
    with open(path, "rt") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            words = line.split()
            img_name = os.path.splitext(words[0])[0]
            cam_model = words[1]
            w, h = int(words[2]), int(words[3])
            if cam_model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = float(words[4])
                cx, cy = float(words[5]), float(words[6])
            elif cam_model in ("PINHOLE", "RADIAL", "OPENCV"):
                fx, fy = float(words[4]), float(words[5])
                cx, cy = float(words[6]), float(words[7])
            else:
                raise ValueError(f"Unsupported camera model: {cam_model}")
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            colmap_cam = {"model": cam_model, "params": [float(p) for p in words[4:]]}
            cam_dict[img_name] = {"name": img_name, "w": w, "h": h, "K": K, "colmap_cam": colmap_cam}
    return dict(sorted(cam_dict.items()))


def load_cameras(in_path: str) -> dict:
    """
    Load camera parameters from a file or directory.

    Supported formats: COLMAP sparse model directory (no extension) and TXT
    (MeshLoc query list).
    """
    ext = os.path.splitext(in_path)[1]
    if ext == "":
        assert os.path.isdir(in_path), f"Directory not found: {in_path}"
        cam_dict = load_cams_colmap(in_path)
    elif ext == ".txt":
        cam_dict = load_cams_txt(in_path)
    else:
        raise ValueError(f"Unknown camera file format: {in_path}")

    for cam_key in cam_dict:
        cam_dict[cam_key] = complete_cam_params(cam_dict[cam_key])

    return cam_dict


def parse_retrieval_file(path: str, get_score: bool = False):
    """
    Parse an Hloc-format retrieval file.

    Returns dict mapping query name → list of reference names.
    If get_score=True, also returns dict mapping query name → list of scores.
    """
    ret_dict: dict[str, list[str]] = {}
    scores: dict[str, list[float]] = {}

    with open(path, "rt") as f:
        for line in f:
            if not line:
                continue
            words = line.strip().split()
            query_name, ret_name = words[0], words[1]
            if query_name not in ret_dict:
                ret_dict[query_name] = []
                if len(words) > 2:
                    scores[query_name] = []
            ret_dict[query_name].append(ret_name)
            if get_score and len(words) > 2:
                scores[query_name].append(float(words[2]))

    if get_score:
        return (ret_dict, scores) if scores else (ret_dict, None)
    return ret_dict


def write_cams_txt(path: str, cam_dict: dict) -> None:
    """Write camera extrinsics in Visual Localization Benchmark format."""
    assert os.path.isdir(os.path.dirname(path)), (
        f"Output directory does not exist: {os.path.dirname(path)}"
    )
    with open(path, "wt") as f:
        for name, cam in cam_dict.items():
            T = cam["T"]
            if T is None:
                continue
            q, t = T2qt(T)
            f.write(
                f"{name} {q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"
                f" {t[0]:.6f} {t[1]:.6f} {t[2]:.6f}\n"
            )


# ── Localization helpers ────────────────────────────────────────────────────────


def setup_poselib_intrinsics(cam_dict: dict) -> dict:
    """Build a PoseLib intrinsics dict from a camera parameter dict."""
    assert "w" in cam_dict
    assert "h" in cam_dict
    return {
        "model": cam_dict["colmap_cam"]["model"],
        "width": cam_dict["w"],
        "height": cam_dict["h"],
        "params": cam_dict["colmap_cam"]["params"],
    }


def get_opencv_distortion_params(cam_dict: dict) -> np.ndarray:
    """Return OpenCV-style (k1, k2, p1, p2) distortion array for a camera dict."""
    assert "colmap_cam" in cam_dict
    params = cam_dict["colmap_cam"]["params"]
    model = cam_dict["colmap_cam"]["model"]

    dist = np.zeros(4, dtype=np.float32)
    if model in ("SIMPLE_PINHOLE", "PINHOLE"):
        pass
    elif model == "SIMPLE_RADIAL":
        dist[0] = params[3]
    elif model == "RADIAL":
        dist[0] = params[3]
        dist[1] = params[4]
    elif model == "OPENCV":
        dist[:] = params[4:]
    else:
        raise ValueError(f"Unknown camera model: {model}")

    return dist


def get_K_from_cam_dict(cam_dict: dict) -> np.ndarray:
    """Extract 3x3 calibration matrix from a camera dict via colmap_cam params."""
    assert "colmap_cam" in cam_dict
    model = cam_dict["colmap_cam"]["model"]
    params = cam_dict["colmap_cam"]["params"]

    if model.startswith("SIMPLE_"):
        fx = fy = params[0]
        cx, cy = params[1], params[2]
    else:
        fx, fy = params[0], params[1]
        cx, cy = params[2], params[3]

    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def add_to_track_recursive(
    img_idx1: int,
    idx1: int,
    match_idx_dict: dict,
    match_info_dict: dict,
    track: list,
    dist_thresh: float = 3.0,
) -> None:
    track.append((img_idx1, idx1))
    for img_idx2, idx2 in match_idx_dict[(img_idx1, idx1)]:
        if match_info_dict[(img_idx1, idx1, img_idx2, idx2)]["used"]:
            continue
        if img_idx2 in [img for img, _ in track]:
            continue
        if match_info_dict[(img_idx1, idx1, img_idx2, idx2)]["dist"] > dist_thresh:
            continue
        match_info_dict[(img_idx1, idx1, img_idx2, idx2)]["used"] = True
        match_info_dict[(img_idx2, idx2, img_idx1, idx1)]["used"] = True
        add_to_track_recursive(img_idx2, idx2, match_idx_dict, match_info_dict, track)


def match_keypoints_nn_mv(kpts_list: list[np.ndarray], dist_thresh: float = 3.0) -> list[list]:
    """
    Mutual nearest-neighbour matching across K sets of keypoints.

    Returns a list of tracks; each track is a list of (img_idx, kpt_idx) tuples.
    """
    import faiss

    faiss_index_list = []
    for kpts in kpts_list:
        idx = faiss.IndexFlatL2(2)
        idx.add(kpts.astype(np.float32))
        faiss_index_list.append(idx)

    match_idx_dict: dict = {}
    match_info_dict: dict = {}

    for img_idx1, fidx1 in enumerate(faiss_index_list):
        for img_idx2, fidx2 in enumerate(faiss_index_list):
            if img_idx1 == img_idx2 or (img_idx1, img_idx2) in match_idx_dict:
                continue
            dist12, idxs12 = fidx2.search(kpts_list[img_idx1].astype(np.float32), 1)
            _, idxs21 = fidx1.search(kpts_list[img_idx2].astype(np.float32), 1)
            idxs12 = idxs12.flatten()
            idxs21 = idxs21.flatten()
            if len(idxs12) == 0 or len(idxs21) == 0:
                continue
            for idx1, idx2 in enumerate(idxs12):
                if idxs21[idx2] == idx1 and dist12[idx1] < dist_thresh:
                    match_idx_dict.setdefault((img_idx1, idx1), []).append((img_idx2, idx2))
                    match_idx_dict.setdefault((img_idx2, idx2), []).append((img_idx1, idx1))
                    match_info_dict[(img_idx1, idx1, img_idx2, idx2)] = {
                        "dist": dist12[idx1], "used": False
                    }
                    match_info_dict[(img_idx2, idx2, img_idx1, idx1)] = {
                        "dist": dist12[idx1], "used": False
                    }

    track_list = []
    for img_idx1, idx1 in match_idx_dict:
        track: list = []
        add_to_track_recursive(img_idx1, idx1, match_idx_dict, match_info_dict, track, dist_thresh)
        if len(track) >= 2:
            track_list.append(track)

    return track_list
