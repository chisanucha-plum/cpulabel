from dataclasses import dataclass
from typing import Tuple

@dataclass
class Detection:
    """Single detection result"""
    class_id: int
    box: Tuple[float, float, float, float]  # cx, cy, bw, bh
    confidence: float = 0.0
