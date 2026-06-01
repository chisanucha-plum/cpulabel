import numpy as np
from typing import List, Dict
from schemas import Config

def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: List[int]) -> np.ndarray:
    """Apply colored mask overlay to image"""
    color_array = np.array(color, dtype=np.uint8)
    masked_region = image[mask].astype(np.float32)
    blended = (masked_region * 0.6 + color_array * 0.4).astype(np.uint8)
    image[mask] = blended
    return image

def create_coco_categories(config: Config) -> List[Dict]:
    """Create COCO categories from config"""
    return [{"id": i + 1, "name": name, "supercategory": "object"} 
            for i, name in config.classes.items()]
