import sys
from typing import Tuple
import numpy as np

sys.path.append("MobileSAM")
from mobile_sam import sam_model_registry, SamPredictor
from schemas import Config


class MobileSAMModel:
    def __init__(self, config: Config):
        sam = sam_model_registry["vit_t"](checkpoint=config.sam_weights)
        sam.to("cpu")
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray) -> None:
        self.predictor.set_image(image)

    def segment(self, box: Tuple[int, int, int, int]) -> np.ndarray:
        box_np = np.array(box, dtype=np.float32)[None, :]
        masks, _, _ = self.predictor.predict(box=box_np, multimask_output=False)
        return masks[0]
