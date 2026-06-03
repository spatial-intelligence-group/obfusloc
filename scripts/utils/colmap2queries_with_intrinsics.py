#!/usr/bin/env python3

# Script for conversion of COLMAP model to a "queries_with_intrinsics" file
# used e.g. by Aachen dataset.

import os
import argparse
import pycolmap

parser = argparse.ArgumentParser(description="Render images from 3D model")
parser.add_argument("colmap_model", type=str, help="Path to the colmap model")
parser.add_argument("output_file", type=str, help="Path to the output file")
parser.add_argument("--images_dir", type=str, help="Limits the list just to images in the given directory")
args = parser.parse_args()


def main(args):
    print("Loading the images and cameras")
    print(args.colmap_model)
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(args.colmap_model)
    cameras = reconstruction.cameras
    images = reconstruction.images
    print(f"Found {len(images)} images and {len(cameras)} cameras")

    img_set = None
    if args.images_dir is not None:
        img_set = set(os.listdir(args.images_dir))

    print("Writing the queries with intrinsics")
    with open(args.output_file, "w") as f:
        for img in images.values():
            name = img.name

            if img_set is not None and name not in img_set:
                continue

            cam = cameras[img.camera_id]
            model = cam.model_name
            width = cam.width
            height = cam.height
            params = " ".join(map(str, cam.params))

            f.write(f"{name} {model} {width} {height} {params}\n")


if __name__ == "__main__":
    main(args)
