from pathlib import Path
from typing import List
import cv2
import numpy as np
from schemas import Config


def setup_dataset_structure(config: Config) -> None:
    """Create output dataset folders."""
    base = Path(config.output_folder)
    folders = [
        base / "yolo" / "images",
        base / "yolo" / "labels",
        base / "coco" / "images",
        base / "coco" / "annotations",
        base / "visualizations",
        base / "yolo" / "confident" / "images",
        base / "yolo" / "confident" / "labels",
        base / "yolo" / "uncertain" / "images",
        base / "yolo" / "uncertain" / "labels",
    ]
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)


def get_image_files(input_folder: str) -> List[Path]:
    """Get all image files from input folder."""
    folder = Path(input_folder)
    valid_extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in valid_extensions
    )


def save_image(image: np.ndarray, output_path: str) -> None:
    """Save RGB image in BGR format."""
    cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def apply_mask_overlay(image: np.ndarray, mask: np.ndarray, color: List[int]) -> np.ndarray:
    """Apply colored mask overlay to image."""
    color_array = np.array(color, dtype=np.uint8)
    masked_region = image[mask].astype(np.float32)
    blended = (masked_region * 0.6 + color_array * 0.4).astype(np.uint8)
    image[mask] = blended
    return image
