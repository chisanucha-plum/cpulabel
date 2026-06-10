"""Human-in-the-Loop Viewer - Click to Remove Detections"""

import cv2
import numpy as np
import logging
from typing import List, Tuple
from schemas import Detection, ImageResult

logger = logging.getLogger(__name__)

class HITLViewer:
    """Interactive viewer - click to remove detections"""
    
    def __init__(self, config, confidence_threshold: float = 0.8):
        self.config = config
        self.window_name = "CPULabel - Click to Remove"
        self.confidence_threshold = confidence_threshold
    
    def should_review(self, detections: List[Detection]) -> bool:
        """Check if needs review"""
        return any(det.confidence < self.confidence_threshold for det in detections)
    
    def review(self, image: np.ndarray, result: ImageResult, masks: List) -> Tuple[bool, ImageResult, List, str]:
        """Review with click to remove
        
        Returns: (accepted, result, masks, category)
        """
        if len(result.detections) == 0:
            return False, result, masks, "rejected"
        
        # Auto-pass high confidence
        if not self.should_review(result.detections):
            logger.info(f"Auto-confident: {result.file_name}")
            return True, result, masks, "confident"
        
        logger.info(f"Reviewing: {result.file_name}")
        
        h, w = image.shape[:2]
        detections = list(result.detections)
        current_masks = list(masks)
        active = [True] * len(detections)
        
        def mouse_callback(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            if y > h:
                return
            
            # Check click on any detection
            for idx, (det, is_active) in enumerate(zip(detections, active)):
                if not is_active:
                    continue
                
                cx, cy, bw, bh = det.box
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                
                if x1 <= x <= x2 and y1 <= y <= y2:
                    active[idx] = False
                    break
        
        while True:
            display = self._draw(image, detections, current_masks, active, h, w)
            cv2.imshow(self.window_name, display)
            cv2.setMouseCallback(self.window_name, mouse_callback)
            
            key = cv2.waitKey(0) & 0xFF
            
            if key == ord(' ') or key == 27:  # SPACE or ESC
                result.detections = [d for d, a in zip(detections, active) if a]
                result.masks = [m for m, a in zip(current_masks, active) if a]
                cv2.destroyAllWindows()
                return True, result, result.masks, "uncertain"
            
            elif key == ord('c') or key == ord('C'):  # Confident
                result.detections = [d for d, a in zip(detections, active) if a]
                result.masks = [m for m, a in zip(current_masks, active) if a]
                cv2.destroyAllWindows()
                return True, result, result.masks, "confident"
            
            elif key == ord('q') or key == ord('Q'):  # Reject
                cv2.destroyAllWindows()
                return False, result, current_masks, "rejected"
    
    def _draw(self, image, detections, masks, active, h, w):
        """Draw image with detections"""
        display = image.copy()
        
        # Draw active detections
        for det, mask, is_active in zip(detections, masks, active):
            if not is_active:
                continue
            
            color = tuple(self.config.class_colors.get(det.class_id, [0, 255, 0]))
            
            # Draw mask
            overlay = np.zeros_like(display)
            overlay[mask > 0] = color
            display = cv2.addWeighted(display, 1, overlay, 0.3, 0)
            
            # Draw bbox
            cx, cy, bw, bh = det.box
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            # Class label
            class_name = self.config.classes[det.class_id]
            text = f"{class_name} {det.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
            cv2.putText(display, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Instructions
        info = "CLICK to remove | SPACE/ESC: uncertain | C: confident | Q: reject"
        cv2.putText(display, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        active_count = sum(active)
        counter = f"Active: {active_count}/{len(detections)}"
        cv2.putText(display, counter, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        return display
