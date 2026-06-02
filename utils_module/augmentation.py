import cv2
import numpy as np
from typing import Tuple, List
from schemas import Detection

class ImageAugmenter:
    """Simple image augmentation"""
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by angle"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    @staticmethod
    def add_noise(image: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """Add Gaussian noise to image"""
        img_float = image.astype(np.float32) / 255.0
        noise = np.random.normal(0, intensity, img_float.shape)
        noisy = np.clip(img_float + noise, 0, 1)
        return (noisy * 255).astype(np.uint8)
    
    @staticmethod
    def flip_h(image: np.ndarray) -> np.ndarray:
        """Flip horizontally"""
        return cv2.flip(image, 1)
    
    @staticmethod
    def flip_v(image: np.ndarray) -> np.ndarray:
        """Flip vertically"""
        return cv2.flip(image, 0)
