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
    helmet_iou_threshold: float = 0.30
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
        helmet = data.get('helmet', {})
        classes_raw = data.get('classes', {})
        colors = data.get('colors', {})
        
        # Parse classes - handle both old format (string) and new format (dict)
        classes_dict = {}
        for k, v in classes_raw.items():
            class_id = int(k)
            if isinstance(v, dict):
                classes_dict[class_id] = v.get('name', 'unknown')
            else:
                classes_dict[class_id] = v
        
        # Convert string keys to int for colors
        colors_dict = {int(k): v for k, v in colors.items()}
        
        return Config(
            dino_config=models.get('dino_config', ''),
            dino_weights=models.get('dino_weights', ''),
            sam_weights=models.get('sam_weights', ''),
            input_folder=paths.get('input_folder', 'images'),
            output_folder=paths.get('output_folder', 'dataset'),
            box_threshold=detection.get('box_threshold', 0.45),
            text_threshold=detection.get('text_threshold', 0.30),
            helmet_iou_threshold=helmet.get('iou_threshold', 0.30),
            classes=classes_dict,
            class_colors=colors_dict
        )
    
    @staticmethod
    def get_class_threshold(classes_raw: Dict, class_id: int, threshold_type: str) -> float:
        """Get per-class threshold (box or text)
        
        Args:
            classes_raw: Raw classes dict from JSON
            class_id: Class ID
            threshold_type: 'box' or 'text'
            
        Returns:
            Threshold value for that class
        """
        class_key = str(class_id)
        if class_key not in classes_raw:
            return 0.45 if threshold_type == 'box' else 0.30
        
        class_config = classes_raw[class_key]
        if isinstance(class_config, dict):
            threshold_key = f"{threshold_type}_threshold"
            return class_config.get(threshold_key, 0.45 if threshold_type == 'box' else 0.30)
        
        # Fallback to global threshold
        return 0.45 if threshold_type == 'box' else 0.30
