#!/usr/bin/env python3


""" Extract binary outlines of objects from depth maps."""


import os
import argparse
import numpy as np
from skimage import filters, feature
import cv2
from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description="Extract binary outlines of objects from depth maps.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="INPUT_DIR",
        help="Path to the input depth map or directory of depth maps in .npz format. Can also be passed as --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir_kw",
        type=str,
        default=None,
        metavar="INPUT_DIR",
        help="Path to the input depth map or directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="OUTPUT_DIR",
        help="Path to the output directory where the binary outlines will be saved. Can also be passed as --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_kw",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Path to the output directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="Postfix to append to output file names, replacing the .npz extension "
             "(e.g. \"_mask.png\" turns depth.npz into depth_mask.png). "
             "If omitted, the .npz extension is replaced with .png.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for depth difference to consider an edge.",
    )
    parser.add_argument(
        "--threshold_mode",
        type=str,
        default="abs",
        choices=["abs", "rel"],
        help="Mode for thresholding: "
             "\"abs\" for absolute difference in depth units, "
             "\"rel\" for relative difference where the maximum depth in the depth map equals to 1.",
    )
    parser.add_argument(
        "--depth_convert",
        type=str,
        default="none",
        choices=["none", "invert", "log"],
        help="Convert depth map to another format: "
             "\"none\" for no conversion, "
             "\"invert\" for inverting depth values, "
             "\"log\" for applying logarithmic transformation.",
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="canny",
        choices=["laplace", "sobel", "canny"],
        help="Edge detection method to use: \"laplace\", \"sobel\", or \"canny\".",
    )
    parser.add_argument(
        "--gauss_sigma",
        type=float,
        default=0.2,
        help="Standard deviation for Gaussian kernel used in edge detection.",
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir if args.input_dir is not None else args.input_dir_kw
    if args.input_dir is None:
        parser.error("input_dir is required")
    args.output_dir = args.output_dir if args.output_dir is not None else args.output_dir_kw
    if args.output_dir is None:
        parser.error("output_dir is required")

    if os.path.isdir(args.input_dir):
        depth_maps = sorted(os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.endswith(".npz"))
        if len(depth_maps) == 0:
            parser.error(f"No .npz files found in directory {args.input_dir}.")
        os.makedirs(args.output_dir, exist_ok=True)
        output_paths = [os.path.join(args.output_dir, _out_name_from_npz(os.path.basename(f), args.output_postfix)) for f in depth_maps]
    elif os.path.isfile(args.input_dir):
        if not args.input_dir.endswith(".npz"):
            parser.error("Input file must be a .npz file.")
        depth_maps = [args.input_dir]
        if os.path.isfile(args.output_dir):
            output_paths = [args.output_dir]
        else:
            os.makedirs(args.output_dir, exist_ok=True)
            output_paths = [os.path.join(args.output_dir, _out_name_from_npz(os.path.basename(args.input_dir), args.output_postfix))]
    else:
        raise ValueError(f"Input path {args.input_dir} is neither a file nor a directory.")

    for depth_map_path, output_path in tqdm(zip(depth_maps, output_paths), total=len(depth_maps)):
        depth_map = load_depth_map(depth_map_path)
        if args.depth_convert == "invert":
            depth_map = 1.0 / depth_map
        elif args.depth_convert == "log":
            depth_map = np.log(depth_map + 1e-6)
        edges = extract_edges(depth_map, args.threshold, args.threshold_mode, args.detector, args.gauss_sigma)
        cv2.imwrite(output_path, edges)


def _out_name_from_npz(basename, postfix):
    """Derive an output filename from a .npz depth map basename.

    Strips the .npz suffix to expose any embedded image extension
    (e.g. image.jpg.npz → image.jpg). With a postfix the image extension
    is also stripped so the postfix fully replaces it. Without a postfix
    the embedded image name is used as-is; if there is none, .png is appended.
    """
    stem = os.path.splitext(basename)[0]  # strip .npz → "image.jpg" or "image"
    inner_ext = os.path.splitext(stem)[1].lower()
    if postfix is not None:
        bare = os.path.splitext(stem)[0] if inner_ext in IMG_EXTS else stem
        return bare + postfix
    return stem if inner_ext in IMG_EXTS else stem + ".png"


def load_depth_map(depth_map_path):
    return np.load(depth_map_path)["depth"].astype(np.float32)


def extract_edges(depth_map, threshold=0.2, threshold_mode="abs", detector="canny", gauss_sigma=0.2):
    if threshold_mode == "rel":
        depth_map = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    if detector == "laplace":
        depth_map_blurred = filters.gaussian(depth_map, sigma=gauss_sigma)
        edge_map = filters.laplace(depth_map_blurred, ksize=3)
        edge_map = edge_map > threshold
    elif detector == "sobel":
        depth_map_blurred = filters.gaussian(depth_map, sigma=gauss_sigma)
        edge_map = filters.sobel(depth_map_blurred) > threshold
    elif detector == "canny":
        edge_map = feature.canny(depth_map, sigma=gauss_sigma, low_threshold=threshold, high_threshold=2 * threshold)
    else:
        raise ValueError(f"Unknown detector: {detector}")

    return (1 - edge_map).astype(np.uint8) * 255


if __name__ == "__main__":
    main()
