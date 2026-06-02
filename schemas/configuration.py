import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any

@dataclass
class Config:
    """Configuration for auto-labeling pipeline"""
    dino_config: str = "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    dino_weights: str = "weights/groundingdino_swint_ogc.pth"
    sam_weights: str = "MobileSAM/weights/mobile_sam.pt"
    input_folder: str = "images"
    output_folder: str = "dataset"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    classes: Dict[int, str] = field(default_factory=lambda: {
        0: "helmet",
        1: "person",
        2: "safety vest"
    })
    class_colors: Dict[int, List[int]] = field(default_factory=lambda: {
        0: [0, 255, 0],      # Green
        1: [255, 0, 0],      # Blue
        2: [0, 255, 255]     # Yellow
    })

class ConfigLoader:
    """Load configuration from JSON file"""
    
    @staticmethod
    def load(config_path: str = "configuration.json") -> Config:
        """Load configuration from JSON file"""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return ConfigLoader._parse_config(data)
    
    @staticmethod
    def _parse_config(data: Dict[str, Any]) -> Config:
        """Parse JSON data into Config dataclass"""
        models = data.get('models', {})
        paths = data.get('paths', {})
        detection = data.get('detection', {})
        classes = data.get('classes', {})
        colors = data.get('colors', {})
        
        # Convert string keys to int for classes and colors
        classes_dict = {int(k): v for k, v in classes.items()}
        colors_dict = {int(k): v for k, v in colors.items()}
        
        return Config(
            dino_config=models.get('dino_config', ''),
            dino_weights=models.get('dino_weights', ''),
            sam_weights=models.get('sam_weights', ''),
            input_folder=paths.get('input_folder', 'images'),
            output_folder=paths.get('output_folder', 'dataset'),
            box_threshold=detection.get('box_threshold', 0.35),
            text_threshold=detection.get('text_threshold', 0.25),
            classes=classes_dict,
            class_colors=colors_dict
        )
