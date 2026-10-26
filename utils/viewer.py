import logging
import cv2
import numpy as np
from schemas import ImageResult
from .boxes import BoundingBoxVector

logger = logging.getLogger(__name__)


class HITLViewer:
    def __init__(self, config, confidence_threshold: float = 0.8):
        self.config = config
        self.window_name = "CPULabel - Click to Remove"
        self.confidence_threshold = confidence_threshold

    def review(self, image: np.ndarray, result: ImageResult) -> bool:
        if not result.detections:
            result.category = "rejected"
            return False

        if all(d.confidence >= self.confidence_threshold for d in result.detections):
            logger.info(f"Auto-confident: {result.file_name}")
            result.category = "confident"
            return True

        logger.info(f"Reviewing: {result.file_name}")
        dets = list(result.detections)
        masks = list(result.masks)
        active = [True] * len(dets)

        def on_mouse(event, x, y, _flags, _param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            for i, (det, is_active) in enumerate(zip(dets, active)):
                if not is_active:
                    continue
                x1, y1, x2, y2 = BoundingBoxVector.to_xyxy(det, image.shape)
                if x1 <= x <= x2 and y1 <= y <= y2:
                    active[i] = False
                    break

        while True:
            cv2.imshow(self.window_name, self._draw(image, dets, masks, active))
            cv2.setMouseCallback(self.window_name, on_mouse)
            key = cv2.waitKey(0) & 0xFF

            if key in (ord(" "), 27, ord("c"), ord("C")):
                result.detections = [d for d, a in zip(dets, active) if a]
                result.masks = [m for m, a in zip(masks, active) if a]
                cv2.destroyAllWindows()
                if not result.detections:
                    result.category = "rejected"
                    return False
                result.category = "confident" if key in (ord("c"), ord("C")) else "uncertain"
                return True

            if key in (ord("q"), ord("Q")):
                result.category = "rejected"
                cv2.destroyAllWindows()
                return False

    def _draw(self, image: np.ndarray, detections, masks, active) -> np.ndarray:
        display = image.copy()
        for det, mask, is_active in zip(detections, masks, active):
            if not is_active:
                continue

            color = tuple(self.config.class_colors.get(det.class_id, [0, 255, 0]))
            overlay = np.zeros_like(display)
            overlay[mask > 0] = color
            display = cv2.addWeighted(display, 1, overlay, 0.3, 0)

            x1, y1, x2, y2 = BoundingBoxVector.to_xyxy(det, image.shape)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)

            label = f"{self.config.classes[det.class_id]} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(display, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        info = "CLICK to remove | SPACE/ESC: uncertain | C: confident | Q: reject"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        counter = f"Active: {sum(active)}/{len(detections)}"
        cv2.putText(display, counter, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        return display
