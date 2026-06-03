#!/usr/bin/env python3


import os
import numpy as np
import torch
import cv2
from PIL import Image
import argparse
from tqdm import tqdm
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple
from torchvision.ops.boxes import batched_nms, box_area
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.amg import (
    coco_encode_rle,
    rle_to_mask,
    area_from_rle,
    box_xyxy_to_xywh,
    generate_crop_boxes,
    MaskData,
    batch_iterator,
    uncrop_boxes_xyxy,
    uncrop_points,
)


IMG_EXTS = (".jpg", ".jpeg", ".png")

SAM_CHECKPOINTS = {
    "vit_h": ("sam_vit_h_4b8939.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"),
    "vit_l": ("sam_vit_l_0b3195.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"),
    "vit_b": ("sam_vit_b_01ec64.pth", "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"),
}


def get_checkpoint(model_type: str) -> str:
    filename, url = SAM_CHECKPOINTS[model_type]
    cache_dir = Path.home() / ".cache" / "sam"
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = cache_dir / filename
    if not checkpoint.exists():
        print(f"Downloading SAM checkpoint for {model_type} to {checkpoint} ...")
        torch.hub.download_url_to_file(url, str(checkpoint))
    return str(checkpoint)


def main():
    parser = argparse.ArgumentParser(description="SAM automatic mask generation")
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
        "--model_type",
        type=str,
        choices=["vit_h", "vit_l", "vit_b"],
        default="vit_l",
        help="Type of the SAM model to use",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=32,
        help="Number of points per side for mask generation",
    )
    parser.add_argument(
        "--points_per_batch",
        type=int,
        default=64,
        help="Number of points per batch for mask generation",
    )
    parser.add_argument(
        "--show_masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show color-coded masks in the output",
    )
    parser.add_argument(
        "--show_borders",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show borders in the output masks",
    )
    parser.add_argument(
        "--crop_n_layers",
        type=int,
        default=0,
        help="Number of layers to run for crop mask generation. The cropping is not applied if set to 0.",
    )
    parser.add_argument(
        "--filter_text",
        type=int,
        default=0,
        help="Filter out point samples in areas with text (1 for True, 0 for False).",
    )
    parser.add_argument(
        "--checkpoint_text",
        type=str,
        default=None,
        help="Path to the OpenCV DB text detection model checkpoint. Required if --filter_text is set to 1."
        " Check https://opencv.org/blog/text-detection-and-removal-using-opencv for more details.",
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

    if torch.cuda.is_available():
        device = torch.device("cuda")
        # use bfloat16 - only for Ampere and newer, but older (Turing) does not complain (autocast should handle this)
        # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    np.random.seed(42)

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

    if args.filter_text:
        if args.checkpoint_text is None:
            parser.error("--checkpoint_text is required when --filter_text is set to 1")
        if not os.path.isfile(args.checkpoint_text):
            parser.error(f"Checkpoint for text detection does not exist: {args.checkpoint_text}")
        text_detector = init_text_detector(args.checkpoint_text)

    sam = sam_model_registry[args.model_type](checkpoint=get_checkpoint(args.model_type))
    sam.to(device=device)

    mask_generator = SamFilteredMaskGenerator(
        model=sam,
        points_per_side=args.points_per_side,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=0.7,
        stability_score_thresh=0.92,
        stability_score_offset=0.7,
        box_nms_thresh=0.7,
        crop_n_layers=args.crop_n_layers,  # mask prediction will be run again on crops of the image. Sets the number of layers to run, where each layer has 2**i_layer number of image crop
        crop_nms_thresh=0.7,
        crop_n_points_downscale_factor=2,  # number of points-per-side sampled in layer n is scaled down by crop_n_points_downscale_factor**n
        min_mask_region_area=25,  # remove disconnected regions and holes in masks with area smaller than min_mask_region_area
    )

    for input_path, output_path in tqdm(zip(input_paths, output_paths), total=len(input_paths)):
        img = Image.open(input_path).convert("RGB")
        img = np.array(img.convert("RGB"))

        if args.filter_text:
            filter_mask = get_text_mask(img, text_detector, input_size=(640, 640), try_rotate=True)
        else:
            filter_mask = None

        masks = mask_generator.generate(img, filter_mask=filter_mask)
        show_anns(masks, output_path, show_masks=args.show_masks, show_borders=args.show_borders)


def show_anns(anns, output_path, show_masks=True, show_borders=True):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        if show_masks:
            img[m] = color_mask
        if show_borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 0, 1.0), thickness=1)
    img = (img[:, :, :3] * 255).astype(np.uint8)
    cv2.imwrite(output_path, img)


def init_text_detector(checkpoint, conf_thresh=0.5, nms_thresh=0.4, bin_thresh=0.3, poly_thresh=0.5, mean=(122.67891434, 116.66876762, 104.00698793), input_size=(640, 640)):
    text_detector = cv2.dnn_TextDetectionModel_DB(checkpoint)  # type: ignore
    text_detector.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
    text_detector.setInputParams(1.0 / 255, input_size, mean, True)
    return text_detector


def get_text_mask(image, text_detector, input_size, try_rotate=True):
    image_res = cv2.resize(image, input_size)
    boxesDB50, _ = text_detector.detect(image_res)

    mask = np.zeros(image_res.shape[:2], dtype=np.uint8)
    for box_points in boxesDB50:
        box_points = np.array(box_points, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [box_points], 255)

    if try_rotate:
        image_rot = cv2.rotate(image_res, cv2.ROTATE_90_CLOCKWISE)
        boxesDB50_rot, _ = text_detector.detect(image_rot)

        mask_rot = np.zeros(image_rot.shape[:2], dtype=np.uint8)
        for box_points in boxesDB50_rot:
            box_points = np.array(box_points, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask_rot, [box_points], 255)

        mask_rot = cv2.rotate(mask_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        mask = cv2.bitwise_or(mask, mask_rot)

    mask = cv2.resize(mask, image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
    mask = (mask / 255) < 0.5  # Convert to binary mask and invert

    return mask


class SamFilteredMaskGenerator(SamAutomaticMaskGenerator):
    @torch.no_grad()
    def generate(self, image: np.ndarray, filter_mask: Optional[np.ndarray] = None) -> List[Dict[str, Any]]:
        """
        Generates masks for the given image.

        Arguments:
            image (np.ndarray): The image to generate masks for, in HWC uint8 format.
            filter_mask (np.ndarray, optional): A binary mask of the same size as the image, where 0 indicates the areas where the points should not be sampled.

        Returns:
            list(dict(str, any)): A list over records for masks. Each record is a dict containing the following keys:
                segmentation (dict(str, any) or np.ndarray): The mask. If output_mode='binary_mask', is an array of shape HW. Otherwise, is a dictionary containing the RLE.
                bbox (list(float)): The box around the mask, in XYWH format.
                area (int): The area in pixels of the mask.
                predicted_iou (float): The model's own prediction of the mask's quality. This is filtered by the pred_iou_thresh parameter.
                point_coords (list(list(float))): The point coordinates input to the model to generate this mask.
                stability_score (float): A measure of the mask's quality. This is filtered on using the stability_score_thresh parameter.
                crop_box (list(float)): The crop of the image used to generate the mask, given in XYWH format.
        """

        # Generate masks
        mask_data = self._generate_masks(image, filter_mask)

        # Filter small disconnected regions and holes in masks
        if self.min_mask_region_area > 0:
            mask_data = self.postprocess_small_regions(
                mask_data,
                self.min_mask_region_area,
                max(self.box_nms_thresh, self.crop_nms_thresh),
            )

        # Encode masks
        if self.output_mode == "coco_rle":
            mask_data["segmentations"] = [coco_encode_rle(rle) for rle in mask_data["rles"]]
        elif self.output_mode == "binary_mask":
            mask_data["segmentations"] = [rle_to_mask(rle) for rle in mask_data["rles"]]
        else:
            mask_data["segmentations"] = mask_data["rles"]

        # Write mask records
        curr_anns = []
        for idx in range(len(mask_data["segmentations"])):
            ann = {
                "segmentation": mask_data["segmentations"][idx],
                "area": area_from_rle(mask_data["rles"][idx]),
                "bbox": box_xyxy_to_xywh(mask_data["boxes"][idx]).tolist(),
                "predicted_iou": mask_data["iou_preds"][idx].item(),
                "point_coords": [mask_data["points"][idx].tolist()],
                "stability_score": mask_data["stability_score"][idx].item(),
                "crop_box": box_xyxy_to_xywh(mask_data["crop_boxes"][idx]).tolist(),
            }
            curr_anns.append(ann)

        return curr_anns


    def _generate_masks(self, image: np.ndarray, filter_mask: Optional[np.ndarray] = None) -> MaskData:
        orig_size = image.shape[:2]
        crop_boxes, layer_idxs = generate_crop_boxes(
            orig_size, self.crop_n_layers, self.crop_overlap_ratio
        )

        # Iterate over image crops
        data = MaskData()
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            crop_data = self._process_crop(image, crop_box, layer_idx, orig_size, filter_mask)
            data.cat(crop_data)

        # Remove duplicate masks between crops
        if len(crop_boxes) > 1:
            # Prefer masks from smaller crops
            scores = 1 / box_area(data["crop_boxes"])
            scores = scores.to(data["boxes"].device)
            keep_by_nms = batched_nms(
                data["boxes"].float(),
                scores,
                torch.zeros_like(data["boxes"][:, 0]),  # categories
                iou_threshold=self.crop_nms_thresh,
            )
            data.filter(keep_by_nms)

        data.to_numpy()
        return data


    def _process_crop(
        self,
        image: np.ndarray,
        crop_box: List[int],
        crop_layer_idx: int,
        orig_size: Tuple[int, ...],
        filter_mask: Optional[np.ndarray] = None,
    ) -> MaskData:
        # Crop the image and calculate embeddings
        x0, y0, x1, y1 = crop_box
        cropped_im = image[y0:y1, x0:x1, :]
        cropped_im_size = cropped_im.shape[:2]
        self.predictor.set_image(cropped_im)

        # Get points for this crop
        points_scale = np.array(cropped_im_size)[None, ::-1]
        points_for_image = self.point_grids[crop_layer_idx] * points_scale

        if filter_mask is not None:
            cropped_filt = filter_mask[y0:y1, x0:x1]
            points_for_image = points_for_image[
                cropped_filt[points_for_image[:, 1].astype(int), points_for_image[:, 0].astype(int)] > 0
            ]

        # Generate masks for this crop in batches
        data = MaskData()
        for (points,) in batch_iterator(self.points_per_batch, points_for_image):
            batch_data = self._process_batch(points, cropped_im_size, crop_box, orig_size)
            data.cat(batch_data)
            del batch_data
        self.predictor.reset_image()

        # Remove duplicates within this crop.
        keep_by_nms = batched_nms(
            data["boxes"].float(),
            data["iou_preds"],
            torch.zeros_like(data["boxes"][:, 0]),  # categories
            iou_threshold=self.box_nms_thresh,
        )
        data.filter(keep_by_nms)

        # Return to the original image frame
        data["boxes"] = uncrop_boxes_xyxy(data["boxes"], crop_box)
        data["points"] = uncrop_points(data["points"], crop_box)
        data["crop_boxes"] = torch.tensor([crop_box for _ in range(len(data["rles"]))])

        return data


if __name__ == "__main__":
    main()
