#!/usr/bin/env python3

""" Resize given masks to their original size based on the original images. """

import os
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Resize masks to their original size")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input mask or a directory with masks",
    )
    parser.add_argument(
        "orig_images_dir",
        type=str,
        help="Path to the original images directory",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output mask or a directory to save resized masks",
    )
    parser.add_argument(
        "--mask_postfix",
        type=str,
        default="",
        help="Postfix for the mask files",
    )
    parser.add_argument(
        "--invert_colors",
        action="store_true",
        help="Invert the colors of the images",
    )
    args = parser.parse_args()
    args.mask_postfix = os.path.splitext(args.mask_postfix)[0]  # Remove the file extension if present

    os.makedirs(args.output_dir, exist_ok=True)

    input_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    output_paths = [os.path.join(args.output_dir, os.path.basename(f)) for f in input_paths]

    orig_sizes = {}
    orig_paths = [os.path.join(args.orig_images_dir, f) for f in os.listdir(args.orig_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print("Reading original images to get their sizes...")
    for f in tqdm(orig_paths):
        img = cv2.imread(f)
        if img is None:
            print(f"Warning: Could not read original image {f}. Skipping.")
            continue
        h, w = img.shape[:2]
        basename = os.path.splitext(os.path.basename(f))[0]
        orig_sizes[basename] = (h, w)

    print("Resizing masks to original sizes...")
    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        if len(args.mask_postfix) > 0:
            basename = os.path.splitext(os.path.basename(input_path))[0][:-len(args.mask_postfix)]
        else:
            basename = os.path.splitext(os.path.basename(input_path))[0]
        orig_size = orig_sizes.get(basename)

        if orig_size is None:
            print(f"Warning: No original image found for mask {input_path}. Skipping.")
            continue

        img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            continue

        h, w = orig_size
        resized_img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)

        if args.invert_colors:
            resized_img = 255 - resized_img
        
        cv2.imwrite(output_path, resized_img)


if __name__ == "__main__":
    main()
