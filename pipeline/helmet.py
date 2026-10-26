from typing import List

from schemas import Detection
from utils.boxes import BoundingBoxVector


def apply_helmet(
    detections: List[Detection],
    helmet_class_id: int,
    head_class_id: int,
    helmet_out_class_id: int,
    no_helmet_out_class_id: int,
    iou_threshold: float = 0.30,
) -> List[Detection]:
    """Reclassify head detections based on helmet overlap."""
    head_dets = [d for d in detections if d.class_id == head_class_id]
    if not head_dets:
        return detections

    helmet_dets = [d for d in detections if d.class_id == helmet_class_id]
    results: List[Detection] = []

    for head in head_dets:
        best_iou = 0.0
        best_conf = 0.0

        for helmet in helmet_dets:
            iou = BoundingBoxVector.iou(head, helmet)
            if iou > best_iou:
                best_iou = iou
                best_conf = helmet.confidence

        if best_iou >= iou_threshold:
            conf = (head.confidence + best_conf) / 2.0
            cls_id = helmet_out_class_id
        else:
            conf = head.confidence
            cls_id = no_helmet_out_class_id

        results.append(Detection(class_id=cls_id, box=head.box, confidence=conf))

    return results
