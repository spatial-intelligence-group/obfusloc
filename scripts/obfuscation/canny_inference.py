#!/usr/bin/env python3


import os
import numpy as np
import cv2
import argparse
from tqdm import tqdm


IMG_EXTS = (".jpg", ".jpeg", ".png")


def main():
    parser = argparse.ArgumentParser(description="Canny edge detection on images.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="INPUT_DIR",
        help="Path to the input image or directory. Can also be passed as --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir_kw",
        type=str,
        default=None,
        metavar="INPUT_DIR",
        help="Path to the input image or directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="OUTPUT_DIR",
        help="Path to the output mask or directory. Can also be passed as --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_kw",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Path to the output mask or directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="Postfix to append to output file names, replacing the original extension "
             "(e.g. \"_mask.png\" turns image.jpg into image_mask.png). "
             "If omitted, the output file keeps the same name as the input.",
    )
    parser.add_argument(
        "--threshold_low",
        type=int,
        default=180,
        help="Low threshold for Canny edge detection",
    )
    parser.add_argument(
        "--ratio_high",
        type=int,
        default=2,
        help="Multiplier applied to the low threshold to get a high threshold",
    )
    parser.add_argument(
        "--sobel_kernel_size",
        type=int,
        default=3,
        help="Size of the Sobel kernel for Canny edge detection",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=3,
        help="Size of the Gaussian blur kernel",
    )
    parser.add_argument(
        "--clahe_clip_limit",
        type=float,
        default=2.0,
        help="Clip limit for CLAHE contrast enhancement",
    )
    parser.add_argument(
        "--clahe_tile_size",
        type=int,
        default=8,
        help="Tile grid size (width and height) for CLAHE contrast enhancement",
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir if args.input_dir is not None else args.input_dir_kw
    if args.input_dir is None:
        parser.error("input_dir is required")
    args.output_dir = args.output_dir if args.output_dir is not None else args.output_dir_kw
    if args.output_dir is None:
        parser.error("output_dir is required")

    if os.path.isfile(args.input_dir):
        input_paths = [args.input_dir]
        if os.path.isdir(args.output_dir):
            if args.output_postfix is not None:
                out_name = os.path.splitext(os.path.basename(args.input_dir))[0] + args.output_postfix
            else:
                out_name = os.path.basename(args.input_dir)
            output_paths = [os.path.join(args.output_dir, out_name)]
        else:
            output_paths = [args.output_dir]
    elif os.path.isdir(args.input_dir):
        input_paths = sorted(os.path.join(args.input_dir, img_name) for img_name in os.listdir(args.input_dir) if img_name.lower().endswith(IMG_EXTS))
        if not os.path.isdir(args.output_dir):
            parser.error("output_dir must be a directory when input is a directory")
        if args.output_postfix is not None:
            output_paths = [os.path.join(args.output_dir, os.path.splitext(os.path.basename(p))[0] + args.output_postfix) for p in input_paths]
        else:
            output_paths = [os.path.join(args.output_dir, os.path.basename(p)) for p in input_paths]
    else:
        raise ValueError("Input must be a file or a directory")

    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            continue

        # apply CLAHE
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        clahe = cv2.createCLAHE(clipLimit=args.clahe_clip_limit, tileGridSize=(args.clahe_tile_size, args.clahe_tile_size))
        img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
        img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

        # detect edges with Canny edge detector
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        borders = canny(
            img_gray,
            args.threshold_low,
            args.ratio_high,
            args.sobel_kernel_size,
            args.blur_kernel_size,
        )

        # save the output
        cv2.imwrite(output_path, borders)


def canny(img_gray, threshold_low, ratio_high, sobel_kernel_size=3, blur_kernel_size=3):
    img_blur = cv2.blur(img_gray, (blur_kernel_size, blur_kernel_size))
    detected_edges = cv2.Canny(img_blur, threshold_low, threshold_low * ratio_high, apertureSize=sobel_kernel_size)
    borders = (255 * (detected_edges == 0)[:, :, None]).astype(np.uint8)
    return borders


if __name__ == "__main__":
    main()
