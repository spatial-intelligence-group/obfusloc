#!/usr/bin/env python

"""
Evaluate the localization estimate in TXT format relative to ground-truth poses in COLMAP model.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pycolmap

parser = argparse.ArgumentParser(description='Argument parser')
parser.add_argument(
    'pose_est', 
    type=str, 
    help='Path to a text file with pose estimates ("image_name [qvec] [tvec]" on each line).',
)
parser.add_argument(
    'gt_colmap', 
    type=str,
    help='Path to COLMAP text model containing ground truth query camera poses.',
)
parser.add_argument(
    '--mult_dist', 
    type=float, 
    default=1.0,
    help='Mutliplier to the distance to get to desired units.',
)
parser.add_argument(
    '--custom_thresholds', 
    type=str, 
    nargs='+',
    help='Thresholds in "(pos1,ang1) (pos2,ang2) ..." format',
)
parser.add_argument(
    '--plot_err', 
    action='store_true',
    help='Show a figure with localization success rate',
)
parser.add_argument(
    "--ignore_missing", 
    action="store_true",
    help="Ignore missing values in the estimated poses, which are present in the ground truth",
)
parser.add_argument(
    "--error_path", 
    type=str, 
    help="Path to a file with absolute position and orientation errors, printed to CLI if equal to 'print'"
)
parser.add_argument(
    "--medians", 
    action="store_true", 
    help="Print only median errors"
)
parser.add_argument(
    "--decimal_places", 
    type=int, 
    default=3, 
    help="Number of printed decimal places"
)

def main(args):
    est_dict = parse_pose_est_file(args.pose_est)
    gt_dict = parse_colmap(args.gt_colmap)

    if not(est_dict):
        print("WARN: the pose_est file is empty")
        return 1

    gt_images = list(gt_dict.keys())

    pose_errors = np.full((len(gt_images), 2), np.inf)

    for gt_img_idx, gt_img_name in enumerate(gt_images):
        if gt_img_name not in est_dict:
            print("WARN: image {} is present in the COLMAP model, but no between the pose estimates".format(gt_img_name))
            continue

        pos_err, ang_err = compute_error(
            gt_dict[gt_img_name]["qvec"], gt_dict[gt_img_name]["tvec"], 
            est_dict[gt_img_name]["qvec"], est_dict[gt_img_name]["tvec"])
        
        pose_errors[gt_img_idx, 0] = args.mult_dist * pos_err
        pose_errors[gt_img_idx, 1] = ang_err

    if args.error_path:
        if args.error_path == "print":
            for img_name, img_err in zip(gt_images, pose_errors):
                print(f"{img_name} {img_err[0]:.{args.decimal_places}f} {img_err[1]:.{args.decimal_places}f}")
        else:
            save_errors(pose_errors, gt_images, args.error_path)

    if args.ignore_missing:
        # affects only median errors (does not count the infinite errors) - the recall should be the same
        valid_errors = np.isfinite(pose_errors[:,0])
        pose_errors = pose_errors[valid_errors,:]

    err_med = np.median(pose_errors, axis=0)
    
    print("- error medians:")
    print("  pos: {:.3f}".format(err_med[0]))
    print("  ang: {:.3f}".format(err_med[1]))


    if args.custom_thresholds is not None:
        if len(args.custom_thresholds) == 1:
            args.custom_thresholds = args.custom_thresholds[0].split()
        for thr_text in args.custom_thresholds:
            thr_text = thr_text.replace('(', '')
            thr_text = thr_text.replace(')', '')
            pos_thr, ang_thr = list(map(float, thr_text.split(',')))

            imgs_in_thr = np.sum(np.logical_and(pose_errors[:, 0] <= pos_thr,
                                 pose_errors[:, 1] <= ang_thr)) / len(gt_images)
            
            print("- threshold: ({})".format(thr_text))
            print("             {:.1f} %".format(100*imgs_in_thr))

    if args.plot_err:
        pose_errors_max = np.sort(pose_errors.max(axis=1).flatten())
        pose_errors_max = np.insert(pose_errors_max, 0, 0.0)

        imgs_perc_axis = np.linspace(0.0, 1.0, num=(len(gt_images) + 1))

        plt.plot(pose_errors_max, imgs_perc_axis)
        plt.xlim(left=0.0)
        plt.ylim(0.0, 1.0)
        plt.grid()
        plt.show()

def parse_pose_est_file(path):
    data_dict = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            #   NAME, QW, QX, QY, QZ, TX, TY, TZ
            words = line.split()
            name = words[0].replace("/", "_")
            qvec = np.array(list(map(float, words[1:5])))
            tvec = np.array(list(map(float, words[5:])))

            data_dict[name] = {"qvec": qvec, "tvec": tvec}
    return data_dict

def parse_colmap(path):
    data_dict = {}
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(path)
    for img in reconstruction.images.values():
        if not img.has_pose:
            continue
        pose = img.cam_from_world()
        quat_xyzw = pose.rotation.quat  # [x, y, z, w]
        qvec = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]
        tvec = pose.translation
        name = img.name.replace("/", "_")
        data_dict[name] = {"qvec": qvec, "tvec": tvec}
    return data_dict

def compute_error(qvec_a, tvec_a, qvec_b, tvec_b):
    R_a = qvec2rotmat(qvec_a)
    R_b = qvec2rotmat(qvec_b)

    pos_a = -R_a.T @ np.reshape(tvec_a, (3,1))
    pos_b = -R_b.T @ np.reshape(tvec_b, (3,1))

    # - compute the angular difference
    cos_val = (np.trace(R_a.T @ R_b) - 1)/2

    if cos_val > 1.0:
        cos_val = 1.0
    if cos_val < -1.0:
        cos_val = -1.0

    ang_diff = np.degrees(np.arccos(cos_val)).item()

    # - compute the position difference
    pos_diff = np.sqrt(np.sum(np.square(pos_a - pos_b), axis=0)).item()

    return pos_diff, ang_diff

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def save_errors(errors, image_names, path):
    with open(path, 'wt') as f:
        for line_i in range(errors.shape[0]):
            line = image_names[line_i] + " " + ' '.join(list(map(str, errors[line_i, :]))) + "\n"
            f.write(line)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
