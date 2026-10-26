from typing import Tuple
import numpy as np
from schemas import Detection


class BoundingBoxVector:
    @staticmethod
    def to_xyxy(detection: Detection, image_shape: Tuple) -> np.ndarray:
        h, w = image_shape[:2]
        cx, cy, bw, bh = detection.box
        return np.array(
            [
                int((cx - bw / 2) * w),
                int((cy - bh / 2) * h),
                int((cx + bw / 2) * w),
                int((cy + bh / 2) * h),
            ],
            dtype=np.int32,
        )

    @staticmethod
    def iou(box1: Detection, box2: Detection) -> float:
        cx1, cy1, w1, h1 = box1.box
        cx2, cy2, w2, h2 = box2.box

        x1_min, x1_max = cx1 - w1 / 2, cx1 + w1 / 2
        y1_min, y1_max = cy1 - h1 / 2, cy1 + h1 / 2
        x2_min, x2_max = cx2 - w2 / 2, cx2 + w2 / 2
        y2_min, y2_max = cy2 - h2 / 2, cy2 + h2 / 2

        inter_w = min(x1_max, x2_max) - max(x1_min, x2_min)
        inter_h = min(y1_max, y2_max) - max(y1_min, y2_min)
        if inter_w <= 0 or inter_h <= 0:
            return 0.0

        inter = inter_w * inter_h
        union = (w1 * h1) + (w2 * h2) - inter
        return inter / union if union > 0 else 0.0
