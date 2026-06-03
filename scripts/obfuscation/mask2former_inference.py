#!/usr/bin/env python3

import argparse
import importlib.resources
import multiprocessing as mp
import os
import tempfile
import uuid
from functools import partial
from urllib.request import urlopen

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.engine.defaults import DefaultPredictor
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import _PanopticPrediction

from mask2former.config import add_maskformer2_config
from mask2former import maskformer_model  # noqa: F401


IMG_EXTS = (".jpg", ".jpeg", ".png")

DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/mask2former")

MODELS = {
    "ADE20k-ResNet50": {
        "config": "ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_R50_bs16_160k/model_final_500878.pkl",
    },
    "ADE20k-ResNet101": {
        "config": "ade20k/semantic-segmentation/maskformer2_R101_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_R101_bs16_90k/model_final_0d68be.pkl",
    },
    "ADE20k-Swin-T": {
        "config": "ade20k/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_160k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_tiny_bs16_160k/model_final_5274a6.pkl",
    },
    "ADE20k-Swin-S": {
        "config": "ade20k/semantic-segmentation/swin/maskformer2_swin_small_bs16_160k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_small_bs16_160k/model_final_011c6d.pkl",
    },
    "ADE20k-Swin-B": {
        "config": "ade20k/semantic-segmentation/swin/maskformer2_swin_base_384_bs16_160k_res640.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_384_bs16_160k_res640/model_final_503e96.pkl",
    },
    "ADE20k-Swin-B-IN21k": {
        "config": "ade20k/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_160k_res640.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_base_IN21k_384_bs16_160k_res640/model_final_7e47bf.pkl",
    },
    "ADE20k-Swin-L-IN21k": {
        "config": "ade20k/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k_res640.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/ade20k/semantic/maskformer2_swin_large_IN21k_384_bs16_160k_res640/model_final_6b4a3a.pkl",
    },
    "Cityscapes-ResNet50": {
        "config": "cityscapes/semantic-segmentation/maskformer2_R50_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R50_bs16_90k/model_final_cc1b1f.pkl",
    },
    "Cityscapes-ResNet101": {
        "config": "cityscapes/semantic-segmentation/maskformer2_R101_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_R101_bs16_90k/model_final_257ce8.pkl",
    },
    "Cityscapes-Swin-T": {
        "config": "cityscapes/semantic-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_tiny_bs16_90k/model_final_2d58d4.pkl",
    },
    "Cityscapes-Swin-S": {
        "config": "cityscapes/semantic-segmentation/swin/maskformer2_swin_small_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_small_bs16_90k/model_final_fa26ae.pkl",
    },
    "Cityscapes-Swin-B-IN21k": {
        "config": "cityscapes/semantic-segmentation/swin/maskformer2_swin_base_IN21k_384_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_base_IN21k_384_bs16_90k/model_final_1c6b65.pkl",
    },
    "Cityscapes-Swin-L-IN21k": {
        "config": "cityscapes/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/cityscapes/semantic/maskformer2_swin_large_IN21k_384_bs16_90k/model_final_17c1ee.pkl",
    },
    "MapillaryVistas-ResNet50": {
        "config": "mapillary-vistas/semantic-segmentation/maskformer2_R50_bs16_300k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer_R50_bs16_300k/model_final_6c66d0.pkl",
    },
    "MapillaryVistas-Swin-L-IN21k": {
        "config": "mapillary-vistas/semantic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_300k.yaml",
        "model_url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_90ee2d.pkl",
    },
}


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="Mask2Former segmentation inference")
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
        "--model",
        type=str,
        default="ADE20k-Swin-L-IN21k",
        choices=list(MODELS.keys()),
        help="Mask2Former model to use",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=DEFAULT_CACHE_DIR,
        help="Directory where model checkpoints are cached (default: ~/.cache/mask2former)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions (default: 0.5)",
    )
    parser.add_argument(
        "--color_mode",
        type=str,
        default="semantic",
        choices=["semantic", "random", "borders"],
        help="Color mode for output visualization (default: semantic)",
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
        input_paths = sorted(
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(IMG_EXTS)
        )
        if not os.path.isdir(args.output_dir):
            parser.error("output_dir must be a directory when input is a directory")
        if args.output_postfix is not None:
            output_paths = [
                os.path.join(args.output_dir, os.path.splitext(os.path.basename(p))[0] + args.output_postfix)
                for p in input_paths
            ]
        else:
            output_paths = [os.path.join(args.output_dir, os.path.basename(p)) for p in input_paths]
    else:
        raise ValueError("Input must be a file or a directory")

    checkpoint_path = os.path.join(args.checkpoint_dir, args.model + ".pkl")
    if not os.path.isfile(checkpoint_path):
        download_checkpoint(MODELS[args.model]["model_url"], checkpoint_path)

    config_file_tmp_list = create_tmp_config(MODELS[args.model]["config"], checkpoint_path=checkpoint_path)
    config_file_tmp_path = config_file_tmp_list[-1]

    cfg = setup_cfg(config_file_tmp_path)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused")

    for path in config_file_tmp_list:
        os.remove(path)

    np.random.seed(42)
    random_colors = (np.random.rand(100, 3) * 255).astype(np.uint8)

    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        img = read_image(input_path, format="BGR")
        predictions = predictor(img)

        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            output = draw_panoptic_seg_custom(
                panoptic_seg.to(torch.device("cpu")),
                segments_info,
                metadata,
                random_colors,
                args.color_mode,
            )
        elif "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"].argmax(dim=0).to(torch.device("cpu")).numpy()
            output = draw_sem_seg_custom(sem_seg, random_colors, args.color_mode)
        else:
            raise RuntimeError(
                f"Expected panoptic_seg or sem_seg output from model, got keys: {list(predictions.keys())}"
            )

        cv2.imwrite(output_path, output)


def download_checkpoint(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"Downloading checkpoint from {url} ...")
    response = urlopen(url)
    with open(dest, "wb") as f:
        for data in iter(partial(response.read, 32768), b""):
            f.write(data)
    print(f"Saved to {dest}")


def create_tmp_config(config_path, checkpoint_path=None):
    package = "mask2former.configs"
    with importlib.resources.files(package).joinpath(config_path).open("r") as f:
        yaml_data = yaml.load(f, Loader=Loader)

    tmp_yaml_path = os.path.join(tempfile.gettempdir(), f"tmp_config_{uuid.uuid4().hex}.yaml")

    if "_BASE_" in yaml_data:
        base_config_relpath = yaml_data["_BASE_"]
        base_config_path = os.path.join(os.path.dirname(config_path), base_config_relpath)
        tmp_yaml_path_list = create_tmp_config(base_config_path)
        yaml_data["_BASE_"] = tmp_yaml_path_list[-1]
    else:
        tmp_yaml_path_list = []

    if checkpoint_path is not None:
        yaml_data["MODEL"]["WEIGHTS"] = checkpoint_path

    with open(tmp_yaml_path, "w") as f:
        yaml.dump(yaml_data, f, Dumper=Dumper)

    tmp_yaml_path_list.append(tmp_yaml_path)
    return tmp_yaml_path_list


def setup_cfg(config_file):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.freeze()
    return cfg


def draw_panoptic_seg_custom(panoptic_seg, segments_info, metadata, random_colors, color_mode):
    output_canvas = np.full(
        (panoptic_seg.shape[0], panoptic_seg.shape[1], 3), 255, dtype=np.uint8
    )
    pred = _PanopticPrediction(panoptic_seg, segments_info, metadata)

    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        if category_idx == 0:
            continue
        if color_mode in ("semantic", "random"):
            if color_mode == "semantic":
                color = random_colors[category_idx % len(random_colors)]
            else:
                rand_int = np.random.randint(0, len(random_colors))
                color = random_colors[rand_int]
            color_full = np.tile(np.reshape(color, (1, 1, 3)), (panoptic_seg.shape[0], panoptic_seg.shape[1], 1))
            mask3 = np.stack([mask] * 3, axis=-1)
            output_canvas[mask3] = color_full[mask3]
        elif color_mode == "borders":
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(output_canvas, contours, -1, (0, 0, 0), thickness=1)

    for mask, sinfo in pred.instance_masks():
        category_idx = sinfo["category_id"]
        if category_idx == 0:
            continue
        if color_mode in ("semantic", "random"):
            if color_mode == "semantic":
                color = random_colors[category_idx % len(random_colors)]
            else:
                rand_int = np.random.randint(0, len(random_colors))
                color = random_colors[rand_int]
            color_full = np.tile(np.reshape(color, (1, 1, 3)), (panoptic_seg.shape[0], panoptic_seg.shape[1], 1))
            mask3 = np.stack([mask] * 3, axis=-1)
            output_canvas[mask3] = color_full[mask3]
        elif color_mode == "borders":
            mask_uint8 = mask.copy().astype(np.uint8)
            white = np.full((panoptic_seg.shape[0], panoptic_seg.shape[1], 3), 255, dtype=np.uint8)
            mask3 = np.stack([mask] * 3, axis=-1)
            output_canvas[mask3] = white[mask3]
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(output_canvas, contours, -1, (0, 0, 0), thickness=1)

    return output_canvas


def draw_sem_seg_custom(sem_seg, random_colors, color_mode):
    h, w = sem_seg.shape
    output_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
    categories = np.unique(sem_seg)

    for category_idx in categories:
        if category_idx == 0:
            continue
        mask = sem_seg == category_idx
        if color_mode in ("semantic", "random"):
            if color_mode == "semantic":
                color = random_colors[category_idx % len(random_colors)]
            else:
                rand_int = np.random.randint(0, len(random_colors))
                color = random_colors[rand_int]
            color_full = np.tile(np.reshape(color, (1, 1, 3)), (h, w, 1))
            mask3 = np.stack([mask] * 3, axis=-1)
            output_canvas[mask3] = color_full[mask3]
        elif color_mode == "borders":
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = [cv2.approxPolyDP(c, epsilon=0.01, closed=True) for c in contours]
            cv2.drawContours(output_canvas, contours, -1, (0, 0, 0), thickness=1)

    return output_canvas


if __name__ == "__main__":
    main()
