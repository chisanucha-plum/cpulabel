import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from schemas import Config

def setup_dataset_structure(config: Config) -> None:
    """Create output dataset folders"""
    folders = [
        f"{config.output_folder}/yolo/images",
        f"{config.output_folder}/yolo/labels",
        f"{config.output_folder}/coco/images",
        f"{config.output_folder}/coco/annotations",
        f"{config.output_folder}/visualizations"
    ]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def get_image_files(input_folder: str) -> List[Path]:
    """Get all image files from input folder"""
    image_files = list(Path(input_folder).glob("*.jpg")) + \
                  list(Path(input_folder).glob("*.png")) + \
                  list(Path(input_folder).glob("*.jpeg"))
    return sorted(image_files)

def save_image(image: np.ndarray, output_path: str) -> None:
    """Save image in BGR format"""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
