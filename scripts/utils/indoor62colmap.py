#!/usr/bin/env python

"""
Convert Indoor-6 camera data to COLMAP model.
- Indoor-6: https://github.com/microsoft/SceneLandmarkLocalization
"""


import os
import math
import numpy as np
import argparse
import pycolmap


parser = argparse.ArgumentParser()
parser.add_argument('indoor6_data', type=str, help='Path to the Indoor-6 data')
parser.add_argument('output_path', type=str,
                    help='Path to the output text COLMAP model')
parser.add_argument("--output_type", type=str, required=False,
                    choices=["BIN", "TXT"], default="BIN",
                    help="Type of the output model")


COLOR_EXT = ".color.jpg"
POSE_EXT = ".pose.txt"
INTR_EXT = ".intrinsics.txt"


def main(args):
    assert os.path.isdir(args.indoor6_data)
    assert os.path.isdir(args.output_path)

    data_dict = parse_indoor6(args.indoor6_data)
    write_colmap(args.output_path, data_dict, args.output_type)


def parse_indoor6(i6_dir):
    file_list = os.listdir(i6_dir)
    color_list = [os.path.join(i6_dir, file) for file in file_list if file.endswith(COLOR_EXT)]
    color_list.sort()

    data_dict = {}

    for color_file in color_list:
        pose_file = color_file[:-len(COLOR_EXT)] + POSE_EXT
        intr_file = color_file[:-len(COLOR_EXT)] + INTR_EXT

        T = parse_pose_file(pose_file)
        intr = parse_intr_file(intr_file)

        cam_dict = {}
        cam_dict["qvec"] = R2quat(T[0:3, 0:3])
        cam_dict["tvec"] = T[0:3, 3]
        # cam_dict["tvec"] = -T[0:3,0:3].T @ T[0:3, 3]
        cam_dict["width"] = intr[0]
        cam_dict["height"] = intr[1]
        cam_dict["params"] = (intr[2], intr[3], intr[4], intr[5])

        data_dict[os.path.basename(color_file)] = cam_dict
    
    return data_dict


def write_colmap(output_path, data_dict, output_type):
    model = pycolmap.Reconstruction()

    for img_i, img_name in enumerate(data_dict):
        cam_dict = data_dict[img_name]

        new_cam = pycolmap.Camera(
            model="SIMPLE_RADIAL",
            width=cam_dict["width"],
            height=cam_dict["height"],
            params=cam_dict["params"],
            camera_id=img_i + 1,
        )
        model.add_camera_with_trivial_rig(new_cam)

        qvec = cam_dict["qvec"].flatten()  # [w, x, y, z]
        rotation = pycolmap.Rotation3d(np.array([qvec[1], qvec[2], qvec[3], qvec[0]]))  # xyzw
        pose = pycolmap.Rigid3d(rotation, cam_dict["tvec"].flatten())

        new_img = pycolmap.Image(name=img_name, camera_id=img_i + 1, image_id=img_i + 1)
        model.add_image_with_trivial_frame(new_img, pose)

    if output_type == "BIN":
        model.write_binary(output_path)
    elif output_type == "TXT":
        model.write_text(output_path)


def parse_pose_file(file):
    T = np.eye(4)
    with open(file, 'rt') as f:
        i = 0
        for line in f:
            words = line.strip().split()
            T[i, :] = np.array(list(map(float, words)))
            i += 1

    return T


def parse_intr_file(file):
    words = []
    with open(file, 'rt') as f:
        line = f.readline()
        words = line.strip().split()
    
    w = int(words[0])
    h = int(words[1])
    f = float(words[2])
    cx = float(words[3])
    cy = float(words[4])
    k = float(words[5])

    return (w, h, f, cx, cy, k)


def R2quat(R):
    tr = np.trace(R)

    if (tr > 0):
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif ((R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2])):
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif (R[1, 1] > R[2, 2]):
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([[w], [x], [y], [z]])


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)