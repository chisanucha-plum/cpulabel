from typing import List, Tuple
from groundingdino.util.inference import load_model, predict
from schemas import Config, Detection

class GroundingDINOModel:
    """Wrapper for GroundingDINO model"""
    
    def __init__(self, config: Config):
        self.model = load_model(config.dino_config, config.dino_weights, device="cpu")
        self.config = config
    
    def detect(self, image, class_name: str) -> Tuple[List, List, List]:
        """Detect objects for a specific class"""
        boxes, logits, phrases = predict(
            device="cpu",
            model=self.model,
            image=image,
            caption=class_name,
            box_threshold=self.config.box_threshold,
            text_threshold=self.config.text_threshold,
        )
        return boxes, logits, phrases
    
    def detect_multiclass(self, image, classes: dict) -> List[Detection]:
        """Detect objects for multiple classes"""
        detections = []
        
        for class_id, class_name in classes.items():
            boxes, _, _ = self.detect(image, class_name)
            
            if len(boxes) > 0:
                for box in boxes:
                    det = Detection(
                        class_id=class_id,
                        box=tuple(box.tolist())
                    )
                    detections.append(det)
        
        return detections
