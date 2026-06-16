from dataclasses import dataclass, field
from typing import Dict


@dataclass
class ProcessingStats:
    """Statistics from processing"""

    total_images: int = 0
    processed_images: int = 0
    total_detections: int = 0
    detections_by_class: Dict[str, int] = field(default_factory=dict)
