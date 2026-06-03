#!/usr/bin/env python3

""" Blur or pixelize given images. """


import argparse
import os
import cv2
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Blur or pixelize given images.")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="INPUT_DIR",
        help="Input image directory. Can also be passed as --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir_kw",
        type=str,
        default=None,
        metavar="INPUT_DIR",
        help="Input image directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="OUTPUT_DIR",
        help="Output image directory. Can also be passed as --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_kw",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Output image directory (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["blur", "pixelize"],
        default="blur",
        help="Mode of operation: \"blur\" or \"pixelize\"",
    )
    parser.add_argument(
        "--blur_kernel_size",
        type=int,
        default=5,
        help="Kernel size for gaussian filter (only used if mode is \"blur\") - larger kernel means more blurry output",
    )
    parser.add_argument(
        "--pixelize_factor",
        type=int,
        default=20,
        help="Pixelization factor (only used if mode is \"pixelize\") - larger factor means larger superpixels",
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="Postfix to append to output file names, replacing the original extension "
             "(e.g. \"_mask.png\" turns image.jpg into image_mask.png). "
             "If omitted, the output file keeps the same name as the input.",
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir if args.input_dir is not None else args.input_dir_kw
    if args.input_dir is None:
        parser.error("input_dir is required")
    args.output_dir = args.output_dir if args.output_dir is not None else args.output_dir_kw
    if args.output_dir is None:
        parser.error("output_dir is required")

    os.makedirs(args.output_dir, exist_ok=True)
    input_paths = sorted(os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    if args.output_postfix is not None:
        output_paths = [os.path.join(args.output_dir, os.path.splitext(os.path.basename(f))[0] + args.output_postfix) for f in input_paths]
    else:
        output_paths = [os.path.join(args.output_dir, os.path.basename(f)) for f in input_paths]

    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            continue
        if args.mode == "blur":
            img_out = cv2.GaussianBlur(img, (args.blur_kernel_size, args.blur_kernel_size), sigmaX=0, sigmaY=0)
        elif args.mode == "pixelize":
            h, w = img.shape[:2]
            hs, ws = h // args.pixelize_factor, w // args.pixelize_factor
            img_small = cv2.resize(img, (ws, hs), interpolation=cv2.INTER_LINEAR)
            img_out = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        cv2.imwrite(output_path, img_out)


if __name__ == "__main__":
    main()
