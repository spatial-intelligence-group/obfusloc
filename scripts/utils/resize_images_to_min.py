#!/usr/bin/env python3

""" Resize given images so they have a given minimum size (and keep the images which are already larger). """

import os
import cv2
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Resize images to a minimum size")
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to the input image or a directory with images",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="Path to the output image or a directory to save resized images",
    )
    parser.add_argument(
        "--min_size_w",
        type=int,
        default=320,
        help="Minimum size for the images (width)",
    )
    parser.add_argument(
        "--min_size_h",
        type=int,
        default=320,
        help="Minimum size for the images (height)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    
    input_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    output_paths = [os.path.join(args.output_dir, os.path.basename(f)) for f in input_paths]

    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        img = cv2.imread(input_path)
        if img is None:
            print(f"Warning: Could not read image {input_path}. Skipping.")
            continue

        h, w = img.shape[:2]
        ratio_w = args.min_size_w / w
        ratio_h = args.min_size_h / h
        ratio = max(ratio_w, ratio_h)
        
        if ratio > 1:
            new_w = round(w * ratio)
            new_h = round(h * ratio)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)            
        else:
            resized_img = img
        
        cv2.imwrite(output_path, resized_img)

if __name__ == "__main__":
    main()