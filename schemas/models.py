import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple


@dataclass
class Detection:
    class_id: int
    box: Tuple[float, float, float, float]
    confidence: float = 0.0


@dataclass
class ImageResult:
    image_id: int
    file_name: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    masks: List = field(default_factory=list)
    category: str = "auto"


@dataclass
class ProcessingStats:
    processed_images: int = 0
    total_detections: int = 0
    detections_by_class: Dict[str, int] = field(default_factory=dict)


@dataclass
class Config:
    dino_config: str = ""
    dino_weights: str = ""
    sam_weights: str = ""
    input_folder: str = "images"
    output_folder: str = "dataset"
    box_threshold: float = 0.45
    text_threshold: float = 0.30
    helmet_iou_threshold: float = 0.30
    classes: Dict[int, str] = field(default_factory=dict)
    class_colors: Dict[int, List[int]] = field(default_factory=dict)
    class_settings: Dict[str, Any] = field(default_factory=dict)


class ConfigLoader:
    @staticmethod
    def load(path: str = "configuration.json") -> Config:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return ConfigLoader._parse_config(json.load(f))

    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> Config:
        models = data.get("models", {})
        paths = data.get("paths", {})
        det = data.get("detection", {})
        helmet = data.get("helmet", {})
        classes_raw = data.get("classes", {})
        colors = data.get("colors", {})

        classes = {
            int(k): (v["name"] if isinstance(v, dict) else v)
            for k, v in classes_raw.items()
        }
        class_colors = {int(k): v for k, v in colors.items()}

        return Config(
            dino_config=models.get("dino_config", ""),
            dino_weights=models.get("dino_weights", ""),
            sam_weights=models.get("sam_weights", ""),
            input_folder=paths.get("input_folder", "images"),
            output_folder=paths.get("output_folder", "dataset"),
            box_threshold=det.get("box_threshold", 0.45),
            text_threshold=det.get("text_threshold", 0.30),
            helmet_iou_threshold=helmet.get("iou_threshold", 0.30),
            classes=classes,
            class_colors=class_colors,
            class_settings=classes_raw,
        )
