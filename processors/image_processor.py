import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple

from schemas import Config, ImageResult, ProcessingStats, Detection
from models_module import GroundingDINOModel, MobileSAMModel
from groundingdino.util.inference import load_image
from utils_module import (
    save_yolo_annotation, create_coco_annotation, save_image,
    apply_mask_overlay
)

class ImageProcessor:
    """Process images for auto-labeling"""
    
    def __init__(self, config: Config):
        self.config = config
        self.dino = GroundingDINOModel(config)
        self.sam = MobileSAMModel(config)
        self.stats = ProcessingStats()
    
    def process_image(self, image_path: Path, image_id: int) -> Tuple[ImageResult, np.ndarray]:
        """Process single image: detect and segment"""
        image_source, image = self._load_image(image_path)
        h, w = image_source.shape[:2]
        
        # Detect objects
        detections = self.dino.detect_multiclass(image, self.config.classes)
        
        if len(detections) == 0:
            return None, None
        
        # Create result
        result = ImageResult(
            image_id=image_id,
            file_name=image_path.name,
            width=w,
            height=h,
            detections=detections
        )
        
        # Segment objects
        self._segment_objects(result, image_source, w, h)
        
        return result, image_source
    
    def save_results(self, result: ImageResult, image_source: np.ndarray, 
                    image_path: Path, annotation_id: int) -> Tuple[List[Dict], int]:
        """Save results for single image"""
        
        # Save YOLO format
        yolo_label_path = f"{self.config.output_folder}/yolo/labels/{image_path.stem}.txt"
        save_yolo_annotation(result.detections, image_source.shape, yolo_label_path)
        
        # Save images
        yolo_image_path = f"{self.config.output_folder}/yolo/images/{image_path.name}"
        save_image(image_source, yolo_image_path)
        
        coco_image_path = f"{self.config.output_folder}/coco/images/{image_path.name}"
        save_image(image_source, coco_image_path)
        
        # Create visualization
        vis_image = self._create_visualization(image_source, result)
        vis_path = f"{self.config.output_folder}/visualizations/{image_path.stem}_vis.jpg"
        save_image(vis_image, vis_path)
        
        # Create COCO annotations
        coco_annotations = self._create_coco_annotations(result, annotation_id)
        
        return coco_annotations, annotation_id + len(coco_annotations)
    
    def _load_image(self, image_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and convert to numpy array"""
        image_source, image = load_image(str(image_path))
        
        if isinstance(image_source, torch.Tensor):
            image_source = image_source.cpu().numpy()
        
        return np.array(image_source), image
    
    def _segment_objects(self, result: ImageResult, image_source: np.ndarray, w: int, h: int):
        """Segment detected objects using SAM"""
        self.sam.set_image(image_source)
        
        for detection in result.detections:
            cx, cy, bw, bh = detection.box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            
            mask = self.sam.segment((x1, y1, x2, y2))
            result.masks.append(mask)
    
    def _create_visualization(self, image_source: np.ndarray, result: ImageResult) -> np.ndarray:
        """Create visualization with masks"""
        vis_image = image_source.copy()
        
        for detection, mask in zip(result.detections, result.masks):
            color = self.config.class_colors.get(detection.class_id, [0, 255, 0])
            vis_image = apply_mask_overlay(vis_image, mask, color)
        
        return vis_image
    
    def _create_coco_annotations(self, result: ImageResult, annotation_id: int) -> List[Dict]:
        """Create COCO annotations for all detections"""
        annotations = []
        
        for detection, mask in zip(result.detections, result.masks):
            ann = create_coco_annotation(
                result.image_id, annotation_id, detection, mask, result.width, result.height
            )
            annotations.append(ann)
            annotation_id += 1
        
        return annotations
