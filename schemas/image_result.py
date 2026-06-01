from dataclasses import dataclass, field
from typing import List
from .detection import Detection

@dataclass
class ImageResult:
    """Result for a single image"""
    image_id: int
    file_name: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    masks: List = field(default_factory=list)
