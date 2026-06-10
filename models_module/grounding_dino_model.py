from typing import List, Tuple, Dict, Any
import json
from groundingdino.util.inference import load_model, predict
from schemas import Config, Detection, ConfigLoader

class GroundingDINOModel:
    """Wrapper for GroundingDINO model"""
    
    def __init__(self, config: Config):
        self.model = load_model(config.dino_config, config.dino_weights, device="cpu")
        self.config = config
        self._load_class_thresholds()
    
    def _load_class_thresholds(self):
        """Load per-class thresholds from config file"""
        try:
            with open("configuration.json", "r") as f:
                data = json.load(f)
                self.classes_raw = data.get('classes', {})
        except Exception:
            self.classes_raw = {}
    
    def _get_threshold(self, class_id: int, threshold_type: str) -> float:
        """Get per-class threshold"""
        class_key = str(class_id)
        if class_key in self.classes_raw:
            class_config = self.classes_raw[class_key]
            if isinstance(class_config, dict):
                threshold_key = f"{threshold_type}_threshold"
                if threshold_key in class_config:
                    return class_config[threshold_key]
        
        # Fallback to global threshold
        return self.config.box_threshold if threshold_type == 'box' else self.config.text_threshold
    
    def detect(self, image, class_name: str, class_id: int) -> Tuple[List, List, List]:
        """Detect objects for a specific class with per-class thresholds"""
        box_thresh = self._get_threshold(class_id, 'box')
        text_thresh = self._get_threshold(class_id, 'text')
        
        boxes, logits, phrases = predict(
            device="cpu",
            model=self.model,
            image=image,
            caption=class_name,
            box_threshold=box_thresh,
            text_threshold=text_thresh,
        )
        return boxes, logits, phrases
    
    def detect_multiclass(self, image, classes: dict) -> List[Detection]:
        """Detect objects for multiple classes with per-class thresholds"""
        detections = []
        
        for class_id, class_name in classes.items():
            boxes, logits, _ = self.detect(image, class_name, class_id)
            
            if len(boxes) > 0:
                for box, logit in zip(boxes, logits):
                    det = Detection(
                        class_id=class_id,
                        box=tuple(box.tolist()),
                        confidence=float(logit)
                    )
                    detections.append(det)
        
        return detections
