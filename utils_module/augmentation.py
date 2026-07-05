import cv2
import numpy as np
from typing import Tuple, List, Optional
from schemas import Detection

class ImageAugmenter:
    """Image augmentation with coordinate transformation"""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def rotate_with_coords(image: np.ndarray, angle: float, 
                          detections: List[Detection], 
                          masks: List[np.ndarray]) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        """Rotate image and transform bounding boxes and masks"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (w, h))
        
        # Transform detections
        transformed_detections = []
        for det in detections:
            cx, cy, bw, bh = det.box
            # Convert normalized to pixel coordinates
            cx_px, cy_px = cx * w, cy * h
            bw_px, bh_px = bw * w, bh * h
            
            # Rotate center point
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            
            # Translate to origin, rotate, translate back
            cx_new = cos_a * (cx_px - center[0]) - sin_a * (cy_px - center[1]) + center[0]
            cy_new = sin_a * (cx_px - center[0]) + cos_a * (cy_px - center[1]) + center[1]
            
            # Convert back to normalized
            cx_norm = cx_new / w
            cy_norm = cy_new / h
            bw_norm = bw_px / w
            bh_norm = bh_px / h
            
            transformed_detections.append(Detection(
                class_id=det.class_id,
                box=(cx_norm, cy_norm, bw_norm, bh_norm),
                confidence=det.confidence
            ))
        
        # Transform masks
        transformed_masks = []
        for mask in masks:
            # Rotate mask
            rotated_mask = cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h))
            transformed_masks.append(rotated_mask.astype(bool))
        
        return rotated_image, transformed_detections, transformed_masks
    
    @staticmethod
    def add_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to image (no coordinate transformation needed)"""
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, intensity, img_float.shape)
        noisy = np.clip(img_float + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    @staticmethod
    def add_noise_with_coords(image: np.ndarray, intensity: float, 
                             detections: List[Detection], 
                             masks: List[np.ndarray]) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        """Add noise (coordinates unchanged)"""
        noisy_image = ImageAugmenter.add_noise(image, intensity)
        return noisy_image, detections, masks
    
    @staticmethod
    def flip_h(image: np.ndarray) -> np.ndarray:
        """Flip horizontally"""
        return cv2.flip(image, 1)
    
    @staticmethod
    def flip_h_with_coords(image: np.ndarray, 
                          detections: List[Detection], 
                          masks: List[np.ndarray]) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        """Flip horizontally and transform bounding boxes and masks"""
        flipped_image = cv2.flip(image, 1)
        
        # Transform detections: flip x-coordinate
        transformed_detections = []
        for det in detections:
            cx, cy, bw, bh = det.box
            new_cx = 1.0 - cx
            transformed_detections.append(Detection(
                class_id=det.class_id,
                box=(new_cx, cy, bw, bh),
                confidence=det.confidence
            ))
        
        # Transform masks
        transformed_masks = []
        for mask in masks:
            flipped_mask = cv2.flip(mask.astype(np.uint8), 1)
            transformed_masks.append(flipped_mask.astype(bool))
        
        return flipped_image, transformed_detections, transformed_masks
    
    @staticmethod
    def flip_v(image: np.ndarray) -> np.ndarray:
        """Flip vertically"""
        return cv2.flip(image, 0)
    
    @staticmethod
    def flip_v_with_coords(image: np.ndarray, 
                          detections: List[Detection], 
                          masks: List[np.ndarray]) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        """Flip vertically and transform bounding boxes and masks"""
        flipped_image = cv2.flip(image, 0)
        
        # Transform detections: flip y-coordinate
        transformed_detections = []
        for det in detections:
            cx, cy, bw, bh = det.box
            new_cy = 1.0 - cy
            transformed_detections.append(Detection(
                class_id=det.class_id,
                box=(cx, new_cy, bw, bh),
                confidence=det.confidence
            ))
        
        # Transform masks
        transformed_masks = []
        for mask in masks:
            flipped_mask = cv2.flip(mask.astype(np.uint8), 0)
            transformed_masks.append(flipped_mask.astype(bool))
        
        return flipped_image, transformed_detections, transformed_masks
    
    @staticmethod
    def darken(image: np.ndarray, factor: float = 0.5) -> np.ndarray:
        """Darken image by multiplying pixel values"""
        img_float = image.astype(np.float32)
        darkened = np.clip(img_float * factor, 0, 255)
        return darkened.astype(np.uint8)
    
    @staticmethod
    def darken_with_coords(image: np.ndarray, factor: float,
                           detections: List[Detection],
                           masks: List[np.ndarray]) -> Tuple[np.ndarray, List[Detection], List[np.ndarray]]:
        """Darken image (coordinates unchanged)"""
        darkened_image = ImageAugmenter.darken(image, factor)
        return darkened_image, detections, masks
