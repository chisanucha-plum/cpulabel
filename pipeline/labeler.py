import logging
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# Workaround for Windows WMI hanging issue in platform.machine() before torch imports
if platform.system() == "Windows":
    _orig_machine = platform.machine

    def _machine_patched():
        try:
            return _orig_machine()
        except Exception:
            return "AMD64"

    platform.machine = _machine_patched

from schemas import Config, ImageResult, ProcessingStats
from models import GroundingDINOModel, MobileSAMModel
from groundingdino.util.inference import load_image
from utils import (
    save_yolo_annotation,
    create_coco_annotation,
    save_image,
    apply_mask_overlay,
    ImageAugmenter,
    HITLViewer,
    BoundingBoxVector,
)
from pipeline.helmet import apply_helmet

logger = logging.getLogger(__name__)


class ImageProcessor:
    def __init__(self, config: Config, augment: bool = False, review: bool = False):
        self.config = config
        self.dino = GroundingDINOModel(config)
        self.sam = MobileSAMModel(config)
        self.stats = ProcessingStats()
        self.augment = augment
        self.review = review
        self.hitl_viewer = HITLViewer(config) if review else None
        self._helmet_ids = self._resolve_helmet_ids()

    def process_image(
        self, image_path: Path, image_id: int
    ) -> List[Tuple[ImageResult, np.ndarray]]:
        try:
            img, tensor = load_image(str(image_path))
            result = self._detect_and_segment(img, tensor, image_path, image_id)
            if result is None:
                return []

            results = [(result, img)]
            if self.augment and result.category != "rejected":
                results.extend(
                    self._apply_augmentations(img, result, image_path, image_id)
                )
            return results
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            return []

    def save_results(
        self,
        result: ImageResult,
        image: np.ndarray,
        annotation_id: int,
        category: str = "auto",
    ) -> Tuple[List[Dict], int]:
        try:
            name = result.file_name
            base = Path(self.config.output_folder)
            yolo_dir = (
                base / "yolo" / category
                if category in ("confident", "uncertain")
                else base / "yolo"
            )

            save_yolo_annotation(
                result.detections,
                image.shape,
                str(yolo_dir / "labels" / f"{name}.txt"),
            )
            save_image(image, str(yolo_dir / "images" / f"{name}.jpg"))
            save_image(image, str(base / "coco" / "images" / f"{name}.jpg"))

            vis = image.copy()
            for det, mask in zip(result.detections, result.masks):
                color = self.config.class_colors.get(det.class_id, [0, 255, 0])
                vis = apply_mask_overlay(vis, mask, color)
            save_image(vis, str(base / "visualizations" / f"{name}_vis.jpg"))

            annotations = []
            for det, mask in zip(result.detections, result.masks):
                annotations.append(
                    create_coco_annotation(
                        result.image_id,
                        annotation_id,
                        det,
                        mask,
                        result.width,
                        result.height,
                    )
                )
                annotation_id += 1

            return annotations, annotation_id
        except Exception as e:
            logger.error(f"Failed to save results for {result.file_name}: {e}")
            return [], annotation_id

    def _resolve_helmet_ids(self) -> Optional[Tuple[int, int, int]]:
        name_to_id = {v: k for k, v in self.config.classes.items()}
        required = {"helmet", "head", "no_helmet"}
        if not required.issubset(name_to_id):
            logger.warning(
                f"Helmet logic disabled — missing classes: {required - name_to_id.keys()}"
            )
            return None
        return name_to_id["helmet"], name_to_id["head"], name_to_id["no_helmet"]

    def _detect_and_segment(
        self,
        img: np.ndarray,
        tensor,
        image_path: Path,
        image_id: int,
    ) -> Optional[ImageResult]:
        detections = self.dino.detect_multiclass(tensor, self.config.classes)
        if self._helmet_ids:
            helmet_id, head_id, no_helmet_id = self._helmet_ids
            detections = apply_helmet(
                detections,
                helmet_id,
                head_id,
                helmet_id,
                no_helmet_id,
                self.config.helmet_iou_threshold,
            )
        if not detections:
            return None

        self.sam.set_image(img)
        masks = [
            self.sam.segment(tuple(BoundingBoxVector.to_xyxy(det, img.shape)))
            for det in detections
        ]

        result = ImageResult(
            image_id=image_id,
            file_name=image_path.stem,
            width=img.shape[1],
            height=img.shape[0],
            detections=detections,
            masks=masks,
        )

        if self.review:
            self.hitl_viewer.review(img, result)

        return result

    def _apply_augmentations(
        self,
        img: np.ndarray,
        result: ImageResult,
        image_path: Path,
        image_id: int,
    ) -> List[Tuple[ImageResult, np.ndarray]]:
        specs = [
            (
                "_rot15",
                ImageAugmenter.rotate_with_coords(
                    img, 15, result.detections, result.masks
                ),
            ),
            (
                "_noise",
                ImageAugmenter.add_noise_with_coords(
                    img, 0.1, result.detections, result.masks
                ),
            ),
            (
                "_fliph",
                ImageAugmenter.flip_h_with_coords(img, result.detections, result.masks),
            ),
            (
                "_dark",
                ImageAugmenter.darken_with_coords(
                    img, 0.5, result.detections, result.masks
                ),
            ),
        ]
        return [
            (
                ImageResult(
                    image_id=image_id + offset,
                    file_name=f"{image_path.stem}{suffix}",
                    width=aug_img.shape[1],
                    height=aug_img.shape[0],
                    detections=aug_dets,
                    masks=aug_masks,
                    category=result.category,
                ),
                aug_img,
            )
            for offset, (suffix, (aug_img, aug_dets, aug_masks)) in enumerate(
                specs, start=1
            )
        ]
