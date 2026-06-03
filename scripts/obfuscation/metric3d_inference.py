#!/usr/bin/env python3

"""Metric3D monocular depth estimation.

Uses the PyTorch Hub API — no local clone of the Metric3D repository is required.
Model weights are downloaded automatically on first use.
Hub source: https://github.com/YvanYin/Metric3D/blob/main/hubconf.py
"""


import os
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png")

# Network input sizes per model family (H, W)
_INPUT_SIZE = {
    "convnext": (544, 1216),
    "vit": (616, 1064),
}
_COLOR_MEAN = [123.675, 116.28, 103.53]
_COLOR_STD = [58.395, 57.12, 57.375]
_CANONICAL_FOCAL = 1000.0  # focal length of the canonical training camera


def main():
    parser = argparse.ArgumentParser(description="Metric3D monocular depth estimation")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="INPUT_DIR",
        help="Directory of input images (searched recursively). Can also be passed as --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir_kw",
        type=str,
        default=None,
        metavar="INPUT_DIR",
        help="Directory of input images (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="OUTPUT_DIR",
        help="Output directory for depth maps (.npz). Can also be passed as --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_kw",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Output directory for depth maps (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "--intrinsics",
        type=str,
        default=None,
        help="Path to a COLMAP model directory or a text file with camera intrinsics. "
             "Text format (one camera per line): "
             "<image_name> <model> <width> <height> <fx> [<fy>] <cx> <cy>. "
             "If omitted, the canonical focal length is used and depth is not converted to metric scale.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_large",
        choices=["convnext_tiny", "convnext_large", "vit_small", "vit_large", "vit_giant2"],
        help="Metric3D backbone to use for depth estimation (default: vit_large)",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="Postfix to append to output file names, replacing the original extension "
             "(e.g. \"_depth.npz\" turns image.jpg into image_depth.npz). "
             "If omitted, the .npz is appended after the original extension.",
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir if args.input_dir is not None else args.input_dir_kw
    if args.input_dir is None:
        parser.error("input_dir is required")
    args.output_dir = args.output_dir if args.output_dir is not None else args.output_dir_kw
    if args.output_dir is None:
        parser.error("output_dir is required")
    if not os.path.exists(args.input_dir):
        parser.error(f"Input directory does not exist: {args.input_dir}")
    if args.intrinsics is not None and not os.path.exists(args.intrinsics):
        parser.error(f"Intrinsics path does not exist: {args.intrinsics}")

    out_ext = args.output_postfix

    if args.intrinsics is not None:
        if os.path.isdir(args.intrinsics):
            intr_dict = parse_colmap_intrinsics(args.intrinsics)
        else:
            intr_dict = parse_txt_intrinsics(args.intrinsics)
    else:
        intr_dict = None

    img_path_list = []
    for root, _dirs, files in os.walk(args.input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in IMG_EXTS:
                img_path_list.append(os.path.join(root, file))
    img_path_list.sort()

    if args.model.startswith("convnext"):
        input_size = _INPUT_SIZE["convnext"]
    elif args.model.startswith("vit"):
        input_size = _INPUT_SIZE["vit"]
    else:
        raise ValueError(f"Unknown model family for: {args.model}")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    model = torch.hub.load("yvanyin/metric3d", "metric3d_" + args.model, pretrain=True)
    model.to(device).eval()

    mean = torch.tensor(_COLOR_MEAN).float()[:, None, None]
    std = torch.tensor(_COLOR_STD).float()[:, None, None]

    for img_path in tqdm(img_path_list):
        img_relpath = os.path.relpath(img_path, args.input_dir)
        out_name = (os.path.splitext(img_relpath)[0] + out_ext) if out_ext is not None else (img_relpath + ".npz")
        out_path = os.path.join(args.output_dir, out_name)

        if intr_dict is not None:
            if img_relpath not in intr_dict:
                print(f"Warning: intrinsics not found for {img_relpath}. Skipping.")
                continue
            canonical_to_real_scale = intr_dict[img_relpath][0] / _CANONICAL_FOCAL
        else:
            canonical_to_real_scale = 1.0

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: could not read image {img_path}. Skipping.")
            continue

        # BGR to RGB
        img = img[:, :, ::-1]

        # Scale to network input size, keeping aspect ratio
        h_orig, w_orig = img.shape[:2]
        scale = min(input_size[0] / h_orig, input_size[1] / w_orig)
        img = cv2.resize(
            img,
            (int(w_orig * scale), int(h_orig * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

        # Pad to exact network input size
        h, w = img.shape[:2]
        pad_h = input_size[0] - h
        pad_w = input_size[1] - w
        pad_h_half = pad_h // 2
        pad_w_half = pad_w // 2
        img = cv2.copyMakeBorder(
            img,
            pad_h_half,
            pad_h - pad_h_half,
            pad_w_half,
            pad_w - pad_w_half,
            cv2.BORDER_CONSTANT,
            value=_COLOR_MEAN,
        )
        pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

        # Normalize and move to device
        img_tensor = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img_tensor = torch.div((img_tensor - mean), std)
        img_tensor = img_tensor[None, :, :, :].to(device)

        with torch.no_grad():
            pred_depth, _, _ = model.inference({"input": img_tensor})

        # Remove padding
        pred_depth = pred_depth.squeeze()
        pred_depth = pred_depth[
            pad_info[0] : pred_depth.shape[0] - pad_info[1],
            pad_info[2] : pred_depth.shape[1] - pad_info[3],
        ]

        # Upsample to original image size
        pred_depth = torch.nn.functional.interpolate(
            pred_depth[None, None, :, :], (h_orig, w_orig), mode="bilinear", align_corners=False
        ).squeeze()

        # Convert from canonical camera space (focal=1000) to metric depth.
        # canonical_to_real_scale is 1.0 when no intrinsics are provided.
        pred_depth = pred_depth * canonical_to_real_scale
        pred_depth = torch.clamp(pred_depth, 0, 300)
        pred_depth = pred_depth.cpu().detach().numpy().astype(np.float16)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(out_path, depth=pred_depth)


def parse_colmap_intrinsics(model_dir: str) -> dict:
    import pycolmap
    reconstruction = pycolmap.Reconstruction(model_dir)
    intr_dict = {}
    for img in reconstruction.images.values():
        cam = reconstruction.cameras[img.camera_id]
        if len(cam.focal_length_idxs()) == 1:
            fx = cam.focal_length
        else:
            fx = cam.focal_length_x
        intr_dict[img.name] = [fx, cam.principal_point_x, cam.principal_point_y]
    return intr_dict


def parse_txt_intrinsics(file: str) -> dict:
    """Parse a text file with one camera per line.

    Format: <image_name> <model> <width> <height> <fx> [<fy>] <cx> <cy>
    SIMPLE_* models have a single focal length; others have fx and fy separately.
    """
    intr_dict = {}
    with open(file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            words = line.split()
            name = words[0]
            model = words[1]
            if model.startswith("SIMPLE_"):
                fx = float(words[4])
                cx = float(words[5])
                cy = float(words[6])
            else:
                fx = float(words[4])
                cx = float(words[6])
                cy = float(words[7])
            intr_dict[name] = [fx, cx, cy]
    return intr_dict


if __name__ == "__main__":
    main()
