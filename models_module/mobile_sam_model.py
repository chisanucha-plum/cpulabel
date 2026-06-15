import sys
import numpy as np
from typing import Tuple

sys.path.append("MobileSAM")
from mobile_sam import sam_model_registry, SamPredictor
from schemas import Config


class MobileSAMModel:
    """Wrapper for MobileSAM model"""

    def __init__(self, config: Config):
        sam = sam_model_registry["vit_t"](checkpoint=config.sam_weights)
        sam.to("cpu")
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        """Set image for prediction"""
        self.predictor.set_image(image)

    def segment(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        """Segment object from bounding box"""
        box_np = np.array(box, dtype=np.float32)[None, :]
        masks, _, _ = self.predictor.predict(box=box_np, multimask_output=False)
        return masks[0]
