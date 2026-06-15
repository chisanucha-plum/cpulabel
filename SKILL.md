# Coding Style Guide

สไตล์การเขียนโค้ดสำหรับโปรเจกต์นี้

## 🎯 หลักการหลัก

### 1. ความเรียบง่าย (Simplicity First)
- **เขียนโค้ดที่อ่านง่าย ไม่ซับซ้อนเกินไป**
- หลีกเลี่ยง abstraction ที่ไม่จำเป็น
- ใช้ชื่อตัวแปรและฟังก์ชันที่สื่อความหมายชัดเจน
- ไม่ใส่ feature ที่ไม่ได้ใช้งาน

```python
# ✅ Good - ชัดเจน ตรงไปตรงมา
def save_image(image: np.ndarray, path: str):
    img_pil = Image.fromarray(image)
    img_pil.save(path, quality=95)

# ❌ Bad - ซับซ้อนเกินไป
class ImageSaveStrategy:
    def execute(self, image, path, **kwargs):
        ...
```

### 2. โครงสร้างโมดูลที่ชัดเจน
จัดระเบียบโค้ดตามหน้าที่:

```
project/
├── schemas/          # Data structures (dataclasses)
├── utils_module/     # Utility functions
├── models_module/    # Model wrappers
├── processors/       # Business logic
└── main.py          # Entry point
```

### 3. ใช้ Dataclass สำหรับ Data Structures
```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple  # (cx, cy, w, h) normalized
    class_name: str = ""

@dataclass
class ImageResult:
    image_id: int
    file_name: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    masks: List = field(default_factory=list)
```

### 4. Type Hints ทุกที่
```python
def process_image(
    self, 
    image_path: Path, 
    image_id: int
) -> List[Tuple[ImageResult, np.ndarray]]:
    """Process single image with optional augmentation"""
    ...
```

### 5. Configuration แยกออกมา
- ใช้ YAML สำหรับ configuration
- ไม่ hardcode ค่าในโค้ด
- สร้าง ConfigLoader class เพื่อจัดการ config

```python
# configuration.yaml
models:
  grounding_dino: "GroundingDINO/weights/groundingdino_swint_ogc.pth"
  mobile_sam: "MobileSAM/weights/mobile_sam.pt"

detection:
  box_threshold: 0.35
  text_threshold: 0.25

classes:
  - "person"
  - "car"
  - "bicycle"
```

### 6. Error Handling และ Logging
```python
import logging

logger = logging.getLogger(__name__)

def process_image(self, image_path: Path, image_id: int):
    try:
        # Process logic
        ...
    except Exception as e:
        logger.error(f"Failed to process {image_path.name}: {e}")
        return None
    finally:
        # Cleanup temp files
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
```

### 7. ลด Print Statement ที่ไม่จำเป็น
```python
# ❌ Bad - print เยอะเกินไป
print("Loading model...")
print("Model loaded successfully!")
print(f"Processing image 1/10...")
print(f"Found {len(boxes)} boxes")
print("Done!")

# ✅ Good - สรุปสั้นๆ เฉพาะที่สำคัญ
logger.info(f"✓ Complete!")
logger.info(f"  Images: {stats.processed_images}/{total_images}")
logger.info(f"  Annotations: {stats.total_detections}")
```

### 8. ตั้งชื่อแบบมีความหมาย
```python
# ✅ Good
def detect_multiclass(self, image, class_names: List[str]) -> List[Detection]:
    """Detect multiple classes in single pass"""
    ...

# ❌ Bad
def process(self, img, cn):
    ...
```

### 9. Separate Concerns
แยกหน้าที่ออกจากกัน:

```python
# File operations
def get_image_files(folder: Path) -> List[Path]:
    ...

# Annotation formatting
def save_yolo_annotation(detections, image_shape, output_path):
    ...

# Model operations
class GroundingDINOModel:
    def detect(self, image, text):
        ...

# Business logic
class ImageProcessor:
    def process_image(self, image_path, image_id):
        ...
```

### 10. Resource Management
```python
def _process_single_version(self, img_source, image_path, image_id, suffix):
    temp_path = None
    try:
        # Create temp file
        temp_path = self._save_to_temp(img_source)
        # Process
        ...
    finally:
        # Always cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
```

### 11. Docstrings แบบกระชับ
```python
def process_image(self, image_path: Path, image_id: int) -> List[Tuple[ImageResult, np.ndarray]]:
    """Process single image with optional augmentation
    
    Args:
        image_path: Path to image file
        image_id: Unique image ID
        
    Returns:
        List of (result, image_source) tuples
    """
```

### 12. Private Methods (Underscore Convention)
```python
class ImageProcessor:
    # Public API
    def process_image(self, image_path, image_id):
        ...
    
    # Internal helpers
    def _load_image(self, image_path):
        ...
    
    def _segment_objects(self, result, image_source):
        ...
    
    def _create_visualization(self, image_source, detections, masks):
        ...
```

## 🚫 สิ่งที่ควรหลีกเลี่ยง

1. ❌ **Abstract factories และ complex patterns** (ถ้าไม่จำเป็น)
2. ❌ **Nested classes ที่ไม่จำเป็น**
3. ❌ **Magic numbers** - ใช้ config แทน
4. ❌ **Global variables** - ส่ง parameters แทน
5. ❌ **Print statements ในโค้ด production** - ใช้ logging
6. ❌ **Hardcoded paths** - ใช้ config
7. ❌ **ฟังก์ชันยาวเกิน 50 บรรทัด** - แยกย่อยออกมา

## ✅ Best Practices ที่ใช้

1. ✅ **Dataclasses** สำหรับ data structures
2. ✅ **Type hints** ทุกที่ที่เป็นไปได้
3. ✅ **YAML configuration** สำหรับ settings
4. ✅ **Logging module** แทน print
5. ✅ **Try-finally** สำหรับ resource cleanup
6. ✅ **List comprehension** เมื่อเหมาะสม
7. ✅ **Pathlib** แทน string paths
8. ✅ **ชื่อแบบ descriptive** แม้จะยาวหน่อย

## 📝 Example: Good Code Structure

```python
# schemas/detection.py
from dataclasses import dataclass

@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple
    class_name: str = ""

# utils_module/file_utils.py
from pathlib import Path
from typing import List

def get_image_files(folder: Path) -> List[Path]:
    """Get all image files from folder"""
    extensions = {'.jpg', '.jpeg', '.png'}
    return [f for f in folder.iterdir() if f.suffix.lower() in extensions]

# processors/image_processor.py
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process images for auto-labeling"""
    
    def __init__(self, config: Config, augment: bool = False):
        self.config = config
        self.dino = GroundingDINOModel(config)
        self.sam = MobileSAMModel(config)
        self.augment = augment
    
    def process_image(self, image_path: Path, image_id: int) -> List[Tuple[ImageResult, np.ndarray]]:
        """Process single image with optional augmentation"""
        try:
            results = self._do_processing(image_path, image_id)
            return results
        except Exception as e:
            logger.error(f"Failed to process {image_path.name}: {e}")
            return [None]
    
    def _do_processing(self, image_path, image_id):
        """Internal processing logic"""
        ...
```

---

**สรุป**: เน้นความเรียบง่าย อ่านง่าย บำรุงรักษาง่าย และทำงานได้จริง 🚀
