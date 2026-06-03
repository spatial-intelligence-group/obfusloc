#! /usr/bin/env python3

# Adapted from https://github.com/GuHuangAI/DiffusionEdge by https://github.com/GuHuangAI
# Original: https://github.com/GuHuangAI/DiffusionEdge/blob/main/demo.py
# Licensed under the Apache License, Version 2.0
#
# Copyright (c) 2026 Vojtech Panek
# SPDX-License-Identifier: BSD-3-Clause
#
# This file has been modified from the original.


"""A simplified Diffusion Edge inference script.

The configuration file (passed with --cfg) should be one of the YAML files from
https://github.com/GuHuangAI/DiffusionEdge/tree/main/configs. We used "default.yaml".

The model checkpoints in .pt format (passed with --pre_weight) can be downloaded from the realease site:
https://github.com/GuHuangAI/DiffusionEdge/releases/tag/v1.1. We used "nyud.pt".
"""


import sys
import argparse
import math
import urllib.request
from pathlib import Path

# DiffusionEdge ships a custom fork of denoising_diffusion_pytorch that is not
# on PyPI. It is included as a git submodule; add its root to sys.path so the
# fork takes precedence over any installed PyPI version.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "third_party" / "DiffusionEdge"))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from denoising_diffusion_pytorch.data import EdgeDatasetTest
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def load_conf(config_file, conf=None):
    if conf is None:
        conf = {}
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def main():
    parser = argparse.ArgumentParser(description="DiffusionEdge inference")
    parser.add_argument(
        "input_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="INPUT_DIR",
        help="Directory with the input images. Can also be passed as --input_dir.",
    )
    parser.add_argument(
        "--input_dir",
        dest="input_dir_kw",
        type=str,
        default=None,
        metavar="INPUT_DIR",
        help="Directory with the input images (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=None,
        type=str,
        metavar="OUTPUT_DIR",
        help="Directory for the output edge maps. Can also be passed as --output_dir.",
    )
    parser.add_argument(
        "--output_dir",
        dest="output_dir_kw",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Directory for the output edge maps (keyword form, alternative to positional).",
    )
    parser.add_argument(
        "--cfg",
        help="Path to YAML configuration file (from DiffusionEdge/configs/). "
             "Defaults to third_party/DiffusionEdge/configs/default.yaml.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--pre_weight",
        help="Path to a pretrained checkpoint file. "
             "If omitted, nyud.pt is downloaded from the DiffusionEdge GitHub releases and cached in "
             "~/.cache/obfusloc/diffusion_edge/.",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--sampling_timesteps",
        help="sampling timesteps",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--bs",
        help="batch size for inference",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--output_postfix",
        type=str,
        default=None,
        help="Postfix to append to output file names, replacing the original extension "
             "(e.g. \"_edge.png\" turns image.jpg into image_edge.png). "
             "If omitted, the output file keeps the same name as the input.",
    )
    args = parser.parse_args()

    args.input_dir = args.input_dir if args.input_dir is not None else args.input_dir_kw
    if args.input_dir is None:
        parser.error("input_dir is required")
    args.output_dir = args.output_dir if args.output_dir is not None else args.output_dir_kw
    if args.output_dir is None:
        parser.error("output_dir is required")

    _repo_root = Path(__file__).resolve().parents[2] / "third_party" / "DiffusionEdge"
    if args.cfg is None:
        args.cfg = str(_repo_root / "configs" / "default.yaml")

    if args.pre_weight is None:
        _weight_url = "https://github.com/GuHuangAI/DiffusionEdge/releases/download/v1.1/nyud.pt"
        _cache_dir = Path.home() / ".cache" / "obfusloc" / "diffusion_edge"
        _cache_dir.mkdir(parents=True, exist_ok=True)
        _cached_weight = _cache_dir / "nyud.pt"
        if not _cached_weight.exists():
            print(f"Downloading DiffusionEdge weights to {_cached_weight} ...")
            urllib.request.urlretrieve(_weight_url, _cached_weight)
        args.pre_weight = str(_cached_weight)

    args.cfg = load_conf(args.cfg)

    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)

    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == "cond_unet":
        from denoising_diffusion_pytorch.mask_cond_unet import Unet

        unet_cfg = model_cfg.unet
        unet = Unet(
            dim=unet_cfg.dim,
            channels=unet_cfg.channels,
            dim_mults=unet_cfg.dim_mults,
            learned_variance=unet_cfg.get("learned_variance", False),
            out_mul=unet_cfg.out_mul,
            cond_in_dim=unet_cfg.cond_in_dim,
            cond_dim=unet_cfg.cond_dim,
            cond_dim_mults=unet_cfg.cond_dim_mults,
            window_sizes1=unet_cfg.window_sizes1,
            window_sizes2=unet_cfg.window_sizes2,
            fourier_scale=unet_cfg.fourier_scale,
            cfg=unet_cfg,
        )
    else:
        raise NotImplementedError

    if model_cfg.model_type == "const_sde":
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f"{model_cfg.model_type} is not supported.")

    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get("default_scale", False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get("use_l1", True),
        cfg=model_cfg,
    )

    data_cfg = cfg.data
    if data_cfg["name"] == "edge":
        dataset = EdgeDatasetTest(
            data_root=args.input_dir,
            image_size=model_cfg.image_size,
        )
    else:
        raise NotImplementedError

    sampler_cfg = cfg.sampler
    sampler_cfg.save_folder = args.output_dir
    sampler_cfg.ckpt_path = args.pre_weight

    # The DataLoader must use batch_size=1: each image may have a different raw
    # size, and the sampler loop handles one image at a time.  --bs controls
    # only the internal sliding-window batch size inside the Sampler.
    dl = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=data_cfg.get("num_workers", 2),
    )

    sampler = Sampler(
        ldm,
        dl,
        batch_size=args.bs,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,
        output_postfix=args.output_postfix,
        cfg=cfg,
    )
    sampler.sample()


class Sampler:
    def __init__(
        self,
        model,
        data_loader,
        sample_num=1000,
        batch_size=16,
        results_folder="./results",
        rk45=False,
        output_postfix=None,
        *,
        cfg,
    ):
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision="no",
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45
        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)
        self.image_size = model.image_size
        self.output_postfix = output_postfix
        self.cfg = cfg

        self.dl = self.accelerator.prepare(data_loader)
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, _: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data["ema"]
            sd = {k[10:]: v for k, v in sd.items() if k.startswith("ema_model.")}
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data["model"])

        if "scale_factor" in data["model"]:
            model.scale_factor = data["model"]["scale_factor"]

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device

        with torch.no_grad():
            self.model.eval()
            for batch in tqdm(self.dl):
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                cond = batch["cond"]
                raw_w = batch["raw_size"][0].item()
                raw_h = batch["raw_size"][1].item()
                img_name = batch["img_name"][0]
                stem = Path(img_name).stem
                out_filename = stem + self.output_postfix if self.output_postfix is not None else img_name
                out_path = self.results_folder / out_filename
                if out_path.exists():
                    accelerator.print(f"File {out_path} already exists, skipping.")
                    continue

                mask = batch.get("ori_mask")
                if self.cfg.sampler.sample_type == "whole":
                    batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                elif self.cfg.sampler.sample_type == "slide":
                    batch_pred = self.slide_sample(
                        cond,
                        crop_size=self.cfg.sampler.get("crop_size", [320, 320]),
                        stride=self.cfg.sampler.stride,
                        mask=mask,
                        bs=self.batch_size,
                    )
                else:
                    raise NotImplementedError

                for img in batch_pred:
                    img = 1 - img  # Invert: white background with black edges
                    tv.utils.save_image(img, str(out_path))

        accelerator.print("sampling complete")

    def slide_sample(self, inputs, crop_size, stride, mask=None, bs=8):
        """Sliding-window inference with overlap.

        Args:
            inputs: Tensor of shape NxCxHxW.
            crop_size: (h_crop, w_crop) window size.
            stride: (h_stride, w_stride) step size.
            mask: optional mask tensor.
            bs: batch size for processing windows.

        Returns:
            Tensor: averaged predictions over all windows.
        """
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))

        crop_imgs, x1s, x2s, y1s, y2s = [], [], [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_imgs.append(inputs[:, :, y1:y2, x1:x2])
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)

        crop_imgs = torch.cat(crop_imgs, dim=0)
        num_windows = crop_imgs.shape[0]
        num_batches = math.ceil(num_windows / bs)

        crop_seg_logits_list = []
        for i in range(num_batches):
            start = bs * i
            end = num_windows if i == num_batches - 1 else bs * (i + 1)
            crop_imgs_batch = crop_imgs[start:end]

            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                logits = self.model.module.sample(
                    batch_size=crop_imgs_batch.shape[0], cond=crop_imgs_batch, mask=mask
                )
            elif isinstance(self.model, nn.Module):
                logits = self.model.sample(
                    batch_size=crop_imgs_batch.shape[0], cond=crop_imgs_batch, mask=mask
                )
            else:
                raise NotImplementedError
            crop_seg_logits_list.append(logits)

        crop_seg_logits = torch.cat(crop_seg_logits_list, dim=0)
        for crop_seg_logit, x1, x2, y1, y2 in zip(crop_seg_logits, x1s, x2s, y1s, y2s):
            preds += F.pad(
                crop_seg_logit,
                (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
            )
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        return preds / count_mat

    def whole_sample(self, inputs, raw_size, mask=None):
        inputs = F.interpolate(inputs, size=(416, 416), mode="bilinear", align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(
                batch_size=inputs.shape[0], cond=inputs, mask=mask
            )
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(
                batch_size=inputs.shape[0], cond=inputs, mask=mask
            )
        else:
            raise NotImplementedError

        return F.interpolate(seg_logits, size=raw_size, mode="bilinear", align_corners=True)


if __name__ == "__main__":
    main()
