import numpy as np
from typing import List, Tuple
from schemas import Detection

class BoundingBoxVector:
    """Convert bounding boxes to vectors"""
    
    @staticmethod
    def to_vector(detection: Detection) -> np.ndarray:
        """Convert detection to vector [cx, cy, bw, bh, class_id]"""
        cx, cy, bw, bh = detection.box
        return np.array([cx, cy, bw, bh, detection.class_id], dtype=np.float32)
    
    @staticmethod
    def to_vectors(detections: List[Detection]) -> np.ndarray:
        """Convert all detections to vector matrix (N, 5)"""
        if not detections:
            return np.empty((0, 5), dtype=np.float32)
        return np.array([BoundingBoxVector.to_vector(det) for det in detections], dtype=np.float32)
    
    @staticmethod
    def to_xyxy(detection: Detection, image_shape: Tuple) -> np.ndarray:
        """Convert to xyxy format [x1, y1, x2, y2] in pixels"""
        h, w = image_shape[:2]
        cx, cy, bw, bh = detection.box
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        return np.array([x1, y1, x2, y2], dtype=np.int32)
    
    @staticmethod
    def iou(box1: Detection, box2: Detection) -> float:
        """Calculate IoU between two boxes"""
        cx1, cy1, bw1, bh1 = box1.box
        cx2, cy2, bw2, bh2 = box2.box
        
        x1_min, y1_min = cx1 - bw1/2, cy1 - bh1/2
        x1_max, y1_max = cx1 + bw1/2, cy1 + bh1/2
        x2_min, y2_min = cx2 - bw2/2, cy2 - bh2/2
        x2_max, y2_max = cx2 + bw2/2, cy2 + bh2/2
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = bw1 * bh1
        box2_area = bw2 * bh2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
