import os
import json
import cv2
import numpy as np
from typing import List, Dict, Tuple
from schemas import Config, Detection

def save_yolo_annotation(detections: List[Detection], image_shape: Tuple, output_path: str) -> None:
    """Save annotations in YOLO format"""
    h, w = image_shape[:2]
    with open(output_path, 'w') as f:
        for det in detections:
            cx, cy, bw, bh = det.box
            f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

def create_coco_annotation(image_id: int, annotation_id: int, detection: Detection, 
                          mask: np.ndarray, width: int, height: int) -> Dict:
    """Create single COCO annotation"""
    cx, cy, bw, bh = detection.box
    
    x = int((cx - bw / 2) * width)
    y = int((cy - bh / 2) * height)
    box_width = int(bw * width)
    box_height = int(bh * height)
    
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": detection.class_id + 1,
        "bbox": [x, y, box_width, box_height],
        "area": box_width * box_height,
        "segmentation": segmentation,
        "iscrowd": 0
    }

def save_coco_json(coco_data: Dict, config: Config) -> None:
    """Save COCO format JSON"""
    output_path = f"{config.output_folder}/coco/annotations/instances.json"
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

def save_yolo_yaml(config: Config) -> None:
    """Save YOLO data.yaml"""
    class_names = [config.classes[i] for i in sorted(config.classes.keys())]
    yaml_content = f"""path: {os.path.abspath(config.output_folder)}/yolo
train: images
val: images

nc: {len(config.classes)}
names: {class_names}
"""
    with open(f"{config.output_folder}/yolo/data.yaml", 'w') as f:
        f.write(yaml_content)
