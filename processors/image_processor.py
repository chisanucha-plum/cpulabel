import os
import logging
import tempfile
import platform
import numpy as np
from pathlib import Path

# Workaround for Windows WMI hanging issue in platform.machine()
# This prevents torch from hanging during import on some Windows systems
if platform.system() == "Windows":
    original_machine = platform.machine
    def _machine_patched():
        try:
            return original_machine()
        except Exception:
            # Return AMD64 as default for Windows if WMI query fails
            return "AMD64"
    platform.machine = _machine_patched

import torch
from typing import Dict, List, Tuple, Optional
from PIL import Image

from schemas import Config, ImageResult, ProcessingStats, Detection
from models_module import GroundingDINOModel, MobileSAMModel
from groundingdino.util.inference import load_image
from utils_module import (
    save_yolo_annotation, create_coco_annotation, save_image,
    apply_mask_overlay, ImageAugmenter, HITLViewer
)
from processors.helmet import apply_helmet

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process images for auto-labeling"""
    
    def __init__(self, config: Config, augment: bool = False, review: bool = False):
        self.config = config
        self.dino = GroundingDINOModel(config)
        self.sam = MobileSAMModel(config)
        self.stats = ProcessingStats()
        self.augment = augment
        self.augmenter = ImageAugmenter() if augment else None
        self.review = review
        self.hitl_viewer = HITLViewer(config) if review else None

        # Resolve class IDs needed for helmet logic from config
        # Expected names in configuration.json classes section
        self._helmet_ids = self._resolve_helmet_ids()
    
    def process_image(self, image_path: Path, image_id: int) -> List[Tuple[ImageResult, np.ndarray]]:
        """Process single image with optional augmentation
        
        Args:
            image_path: Path to image file
            image_id: Unique image ID
            
        Returns:
            List of (result, image_source) tuples
        """
        try:
            image_source, _ = self._load_image(image_path)
            
            # Process original image first (detect + segment)
            result = self._process_single_version(
                image_source, image_path, image_id, ""
            )
            
            if result is None:
                return [None]
            
            results = [(result, image_source)]
            
            # Apply augmentations after detection with coordinate transformation
            if self.augment:
                augmented_results = self._apply_augmentations_with_coords(
                    image_source, result, image_path, image_id
                )
                results.extend(augmented_results)
            
            return results
        
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            return [None]
    
    def save_results(self, result: ImageResult, image_source: np.ndarray, 
                    annotation_id: int, category: str = "auto") -> Tuple[List[Dict], int]:
        """Save results for processed image
        
        Args:
            result: Image result with detections
            image_source: Original image
            annotation_id: COCO annotation ID
            category: Category folder ('auto', 'confident', 'uncertain')
        """
        try:
            label_name = result.file_name
            
            # Determine folder based on category
            if category in ["confident", "uncertain"]:
                base_path = f"{self.config.output_folder}/yolo/{category}"
                yolo_label_path = f"{base_path}/labels/{label_name}.txt"
                yolo_image_path = f"{base_path}/images/{label_name}.jpg"
            else:
                yolo_label_path = f"{self.config.output_folder}/yolo/labels/{label_name}.txt"
                yolo_image_path = f"{self.config.output_folder}/yolo/images/{label_name}.jpg"
            
            # Save YOLO format
            save_yolo_annotation(result.detections, image_source.shape, yolo_label_path)
            save_image(image_source, yolo_image_path)
            
            # Always save to COCO
            coco_image_path = f"{self.config.output_folder}/coco/images/{label_name}.jpg"
            save_image(image_source, coco_image_path)
            
            # Create visualization
            vis_image = self._create_visualization(image_source, result.detections, result.masks)
            vis_path = f"{self.config.output_folder}/visualizations/{label_name}_vis.jpg"
            save_image(vis_image, vis_path)
            
            # Create COCO annotations
            coco_annotations = []
            for detection, mask in zip(result.detections, result.masks):
                ann = create_coco_annotation(
                    result.image_id, annotation_id, detection, mask, result.width, result.height
                )
                coco_annotations.append(ann)
                annotation_id += 1
            
            return coco_annotations, annotation_id
        
        except Exception as e:
            logger.error(f"Failed to save results for {result.file_name}: {e}")
            return [], annotation_id

    def _resolve_helmet_ids(self) -> Optional[dict]:
        """Look up class IDs by name for helmet logic.

        Expects configuration.json classes named:
          'helmet'    — raw DINO query + output label when head wears helmet
          'head'      — raw DINO query for head objects
          'no_helmet' — output label for bare heads (detect: false)

        Returns dict with helmet/head/no_helmet IDs, or None if classes are missing.
        """
        name_to_id = {v: k for k, v in self.config.classes.items()}
        required = {"helmet", "head", "no_helmet"}
        if not required.issubset(name_to_id.keys()):
            missing = required - name_to_id.keys()
            logger.warning(
                f"Helmet logic disabled — missing classes in config: {missing}"
            )
            return None

        return {
            "helmet_id":        name_to_id["helmet"],
            "head_id":          name_to_id["head"],
            "out_helmet_id":    name_to_id["helmet"],    # reuse same ID
            "out_no_helmet_id": name_to_id["no_helmet"],
        }

    def _apply_helmet(self, detections: List[Detection]) -> List[Detection]:
        """Apply helmet/no_helmet reclassification if config supports it."""
        ids = self._helmet_ids
        if ids is None:
            return detections

        return apply_helmet(
            detections=detections,
            helmet_class_id=ids["helmet_id"],
            head_class_id=ids["head_id"],
            helmet_out_class_id=ids["out_helmet_id"],
            no_helmet_out_class_id=ids["out_no_helmet_id"],
            iou_threshold=self.config.helmet_iou_threshold,
        )

    def _apply_augmentations_with_coords(self, image_source: np.ndarray, 
                                         result: ImageResult, 
                                         image_path: Path, 
                                         image_id: int) -> List[Tuple[ImageResult, np.ndarray]]:
        """Apply augmentations with coordinate transformation after detection"""
        augmented_results = []
        
        # Apply rotation with coordinate transformation
        rotated_img, rotated_dets, rotated_masks = self.augmenter.rotate_with_coords(
            image_source, 15, result.detections, result.masks
        )
        rotated_result = ImageResult(
            image_id=image_id + 1,
            file_name=f"{image_path.stem}_rot15",
            width=rotated_img.shape[1],
            height=rotated_img.shape[0],
            detections=rotated_dets
        )
        rotated_result.masks = rotated_masks
        rotated_result.category = "auto"
        augmented_results.append((rotated_result, rotated_img))
        
        # Apply noise (no coordinate transformation needed)
        noisy_img, noisy_dets, noisy_masks = self.augmenter.add_noise_with_coords(
            image_source, 0.1, result.detections, result.masks
        )
        noisy_result = ImageResult(
            image_id=image_id + 2,
            file_name=f"{image_path.stem}_noise",
            width=noisy_img.shape[1],
            height=noisy_img.shape[0],
            detections=noisy_dets
        )
        noisy_result.masks = noisy_masks
        noisy_result.category = "auto"
        augmented_results.append((noisy_result, noisy_img))
        
        # Apply horizontal flip with coordinate transformation
        flip_h_img, flip_h_dets, flip_h_masks = self.augmenter.flip_h_with_coords(
            image_source, result.detections, result.masks
        )
        flip_h_result = ImageResult(
            image_id=image_id + 3,
            file_name=f"{image_path.stem}_fliph",
            width=flip_h_img.shape[1],
            height=flip_h_img.shape[0],
            detections=flip_h_dets
        )
        flip_h_result.masks = flip_h_masks
        flip_h_result.category = "auto"
        augmented_results.append((flip_h_result, flip_h_img))
        
        # Apply darkening (no coordinate transformation needed)
        dark_img, dark_dets, dark_masks = self.augmenter.darken_with_coords(
            image_source, 0.5, result.detections, result.masks
        )
        dark_result = ImageResult(
            image_id=image_id + 4,
            file_name=f"{image_path.stem}_dark",
            width=dark_img.shape[1],
            height=dark_img.shape[0],
            detections=dark_dets
        )
        dark_result.masks = dark_masks
        dark_result.category = "auto"
        augmented_results.append((dark_result, dark_img))
        
        return augmented_results

    def _process_single_version(self, img_source: np.ndarray, image_path: Path,
                                image_id: int, suffix: str) -> Optional[ImageResult]:
        """Process single image version"""
        temp_path = None
        try:
            # Convert to tensor using temp file
            temp_path = self._save_to_temp(img_source)
            _, image_tensor = load_image(temp_path)
            
            # Detect objects
            detections = self.dino.detect_multiclass(image_tensor, self.config.classes)

            # Apply helmet/no_helmet reclassification logic
            detections = self._apply_helmet(detections)

            if len(detections) == 0:
                return None
            
            # Create result
            result = ImageResult(
                image_id=image_id,
                file_name=f"{image_path.stem}{suffix}",
                width=img_source.shape[1],
                height=img_source.shape[0],
                detections=detections
            )
            
            # Segment objects
            self._segment_objects(result, img_source)
            
            # Human review (only for original, not augmented versions)
            if self.review and suffix == "":
                accepted, result, result.masks, category = self.hitl_viewer.review(
                    img_source, result, result.masks
                )
                result.category = category
                if not accepted:
                    return None
            else:
                result.category = "auto"
            
            return result
        
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
    
    def _save_to_temp(self, image: np.ndarray) -> str:
        """Save image to temporary file"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            img_pil = Image.fromarray(image)
            img_pil.save(tmp.name, quality=95)
            return tmp.name
    
    def _load_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and convert to numpy array"""
        image_source, image = load_image(str(image_path))
        
        if isinstance(image_source, torch.Tensor):
            image_source = image_source.cpu().numpy()
        
        return np.array(image_source), np.array(image_source)
    
    def _segment_objects(self, result: ImageResult, image_source: np.ndarray):
        """Segment detected objects using SAM"""
        h, w = image_source.shape[:2]
        self.sam.set_image(image_source)
        
        for detection in result.detections:
            cx, cy, bw, bh = detection.box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            mask = self.sam.segment((x1, y1, x2, y2))
            result.masks.append(mask)
    
    def _create_visualization(self, image_source: np.ndarray, 
                            detections: List[Detection], masks: List) -> np.ndarray:
        """Create visualization with masks"""
        vis_image = image_source.copy()
        
        for detection, mask in zip(detections, masks):
            color = self.config.class_colors.get(detection.class_id, [0, 255, 0])
            vis_image = apply_mask_overlay(vis_image, mask, color)
        
        return vis_image
