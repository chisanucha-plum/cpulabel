from typing import List
from groundingdino.util.inference import load_model, predict
from schemas import Config, Detection


class GroundingDINOModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = load_model(config.dino_config, config.dino_weights, device="cpu")

    def _get_threshold(self, class_id: int, kind: str) -> float:
        cfg = self.config.class_settings.get(str(class_id))
        if isinstance(cfg, dict) and f"{kind}_threshold" in cfg:
            return float(cfg[f"{kind}_threshold"])
        return (
            self.config.box_threshold if kind == "box" else self.config.text_threshold
        )

    def detect_multiclass(self, image, classes: dict) -> List[Detection]:
        detections = []
        for class_id, name in classes.items():
            cfg = self.config.class_settings.get(str(class_id))
            if isinstance(cfg, dict) and cfg.get("detect") is False:
                continue

            boxes, logits, _ = predict(
                device="cpu",
                model=self.model,
                image=image,
                caption=name,
                box_threshold=self._get_threshold(class_id, "box"),
                text_threshold=self._get_threshold(class_id, "text"),
            )
            for box, logit in zip(boxes, logits):
                detections.append(
                    Detection(
                        class_id=class_id,
                        box=tuple(box.tolist()),
                        confidence=float(logit),
                    )
                )
        return detections
