import json
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from schemas import Config, Detection


def save_yolo_annotation(detections: List[Detection], image_shape: Tuple, output_path: str) -> None:
    """Save bounding box annotations in YOLO format."""
    with open(output_path, "w", encoding="utf-8") as f:
        for det in detections:
            cx, cy, bw, bh = det.box
            f.write(f"{det.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")


def create_coco_annotation(
    image_id: int,
    annotation_id: int,
    detection: Detection,
    mask: np.ndarray,
    width: int,
    height: int,
) -> Dict:
    """Create a single COCO annotation dictionary."""
    cx, cy, bw, bh = detection.box
    bx = int((cx - bw / 2) * width)
    by = int((cy - bh / 2) * height)
    bw_px = int(bw * width)
    bh_px = int(bh * height)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    segmentation = [c.flatten().tolist() for c in contours if c.size >= 6]

    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": detection.class_id + 1,
        "bbox": [bx, by, bw_px, bh_px],
        "area": bw_px * bh_px,
        "segmentation": segmentation,
        "iscrowd": 0,
    }


def create_coco_categories(config: Config) -> List[Dict]:
    """Create COCO categories from config."""
    return [
        {"id": i + 1, "name": name, "supercategory": "object"}
        for i, name in config.classes.items()
    ]


def save_coco_json(coco_data: Dict, config: Config) -> None:
    """Save COCO dataset JSON."""
    out = Path(config.output_folder) / "coco" / "annotations" / "instances.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(coco_data, f, indent=2)


def save_yolo_yaml(config: Config) -> None:
    """Save YOLO dataset data.yaml."""
    names = [config.classes[i] for i in sorted(config.classes.keys())]
    base_path = Path(config.output_folder).resolve() / "yolo"
    yaml_content = f"""path: {base_path}
train: images
val: images

nc: {len(config.classes)}
names: {names}
"""
    out = Path(config.output_folder) / "yolo" / "data.yaml"
    with open(out, "w", encoding="utf-8") as f:
        f.write(yaml_content)
