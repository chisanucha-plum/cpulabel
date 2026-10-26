from typing import List, Tuple
import cv2
import numpy as np
from schemas import Detection


class ImageAugmenter:
    @staticmethod
    def rotate_with_coords(
        image: np.ndarray,
        angle: float,
        detections: List[Detection],
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        h, w = image.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))

        dets = []
        for det in detections:
            cx, cy, bw, bh = det.box
            new_cx, new_cy = matrix @ np.array([cx * w, cy * h, 1.0])
            dets.append(
                Detection(
                    class_id=det.class_id,
                    box=(new_cx / w, new_cy / h, bw, bh),
                    confidence=det.confidence,
                )
            )

        rotated_masks = [
            cv2.warpAffine(m.astype(np.uint8), matrix, (w, h)).astype(bool)
            for m in masks
        ]
        return rotated_image, dets, rotated_masks

    @staticmethod
    def add_noise_with_coords(
        image: np.ndarray,
        intensity: float,
        detections: List[Detection],
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, intensity, img_float.shape)
        noisy = (np.clip(img_float + noise, 0, 1) * 255).astype(np.uint8)
        return noisy, detections, masks

    @staticmethod
    def flip_h_with_coords(
        image: np.ndarray,
        detections: List[Detection],
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        dets = [
            Detection(
                class_id=d.class_id,
                box=(1.0 - d.box[0], d.box[1], d.box[2], d.box[3]),
                confidence=d.confidence,
            )
            for d in detections
        ]
        flipped_masks = [cv2.flip(m.astype(np.uint8), 1).astype(bool) for m in masks]
        return cv2.flip(image, 1), dets, flipped_masks

    @staticmethod
    def darken_with_coords(
        image: np.ndarray,
        factor: float,
        detections: List[Detection],
        masks: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        dark = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return dark, detections, masks
