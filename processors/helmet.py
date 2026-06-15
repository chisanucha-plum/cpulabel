from typing import List

from schemas import Detection
from utils_module.box_vector import BoundingBoxVector


def apply_helmet(
    detections: List[Detection],
    helmet_class_id: int,
    head_class_id: int,
    helmet_out_class_id: int,
    no_helmet_out_class_id: int,
    iou_threshold: float = 0.30,
) -> List[Detection]:
    """Replace raw detections with helmet/no_helmet logic.

    Rules:
    - Use every head box as the final bounding box.
    - If a head box overlaps any helmet box (IoU >= iou_threshold)
      → label as helmet_out_class_id  (e.g. "helmet")
    - If a head box has no overlapping helmet
      → label as no_helmet_out_class_id  (e.g. "no_helmet")
    - Raw helmet detections that are NOT matched to any head are dropped
      (they are used only as evidence, not as standalone output boxes).

    Args:
        detections:            Raw detections from GroundingDINO.
        helmet_class_id:       Class ID used by DINO for "helmet" query.
        head_class_id:         Class ID used by DINO for "head" query.
        helmet_out_class_id:   Output class ID to assign when head wears helmet.
        no_helmet_out_class_id: Output class ID to assign when head has no helmet.
        iou_threshold:         Minimum IoU to consider a helmet overlapping a head.

    Returns:
        New list of Detection objects using head boxes, relabelled.
    """
    helmet_boxes = [d for d in detections if d.class_id == helmet_class_id]
    head_boxes = [d for d in detections if d.class_id == head_class_id]

    if not head_boxes:
        return detections

    results: List[Detection] = []

    for head_det in head_boxes:
        best_iou = 0.0
        best_helmet_conf = 0.0

        for helmet_det in helmet_boxes:
            iou = BoundingBoxVector.iou(head_det, helmet_det)
            if iou > best_iou:
                best_iou = iou
                best_helmet_conf = helmet_det.confidence

        if best_iou >= iou_threshold:
            out_conf = (head_det.confidence + best_helmet_conf) / 2.0
            out_class = helmet_out_class_id
        else:
            out_conf = head_det.confidence
            out_class = no_helmet_out_class_id

        results.append(
            Detection(
                class_id=out_class,
                box=head_det.box,
                confidence=out_conf,
            )
        )

    return results