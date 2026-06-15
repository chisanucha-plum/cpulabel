import os
import logging
import tempfile
import numpy as np
import torch
from pathlib import Path
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
            
            # Generate image versions
            images_to_process = [("", image_source)]
            if self.augment:
                images_to_process.extend(self._generate_augmentations(image_source))
            
            results = []
            
            # Process each version
            for suffix, img_source in images_to_process:
                result = self._process_single_version(
                    img_source, image_path, image_id, suffix
                )
                if result:
                    results.append((result, img_source))
            
            return results if results else [None]
        
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

    def _generate_augmentations(self, image_source: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """Generate augmented versions of image."""
        return [
            ("_rot15", self.augmenter.rotate(image_source, 15)),
            ("_noise", self.augmenter.add_noise(image_source, 0.1)),
        ]

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
