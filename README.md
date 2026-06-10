# CPULabel

Efficient image labeling using GroundingDINO + MobileSAM. Runs on CPU. No GPU required.

Generates YOLO and COCO format datasets with automatic segmentation masks.

## Features

- ✅ **CPU-friendly** - No GPU required
- ✅ **Multi-class detection** - Detect multiple object classes
- ✅ **Human-in-the-loop** - Review & edit annotations interactively
- ✅ **Data augmentation** - 5x dataset expansion (rotation, noise, flip)
- ✅ **YOLO + COCO format** - Export to both formats
- ✅ **Automatic segmentation** - Precise masks with MobileSAM
- ✅ **Batch processing** - Process multiple images
- ✅ **Bounding box vectors** - Convert to vector representations
- ✅ **Production ready** - Error handling, logging, clean code

## Quick Start

### 1. Setup (ครั้งแรก)
```bash
# Clone/download project
cd testsam_groud

# สร้าง virtual environment (ครั้งแรก)
python -m venv venv

# Activate environment
# Windows CMD:
venv\Scripts\activate.bat
# Windows PowerShell:
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
cd GroundingDINO
pip install -e .
cd ..
```

### 2. Prepare Images
```bash
mkdir images
# Copy ภาพของคุณ ไปใส่ในโฟลเดอร์ images/
```

### 3. Configure
Edit `configuration.json`:
```json
{
  "classes": {
    "0": "helmet",
    "1": "person",
    "2": "safety vest"
  },
  "detection": {
    "box_threshold": 0.35,
    "text_threshold": 0.25
  }
}
```

### 4. Run

#### **ง่ายสุด: ใช้ Script**
```bash
# Windows CMD
run.bat

# Windows PowerShell
.\run.ps1

# Linux/Mac
chmod +x run.sh
./run.sh
```

#### **ตัวเลือกการรัน**
```bash
# ปกติ
run.bat

# ตรวจสอบแต่ละภาพ
run.bat review

# เพิ่มภาพจาก augmentation
run.bat augment

# ตรวจสอบ + Augment
run.bat review augment
```

#### **Manual (เข้า env ก่อน)**
```bash
# เข้า environment
venv\Scripts\activate.bat

# รันตามต้องการ
python main.py                    # ปกติ
python main.py --review           # ตรวจสอบ
python main.py --augment          # Augment
python main.py --review --augment # ทั้งสอง
python main.py --verbose          # Debug mode
```

## Output Structure

```
dataset/
├── yolo/
│   ├── images/              # Images
│   ├── labels/              # YOLO format (.txt)
│   └── data.yaml            # YOLO config
├── coco/
│   ├── images/              # Images
│   └── annotations/
│       └── instances.json   # COCO format with masks
└── visualizations/          # Overlay visualizations
```

## Data Augmentation

When using `--augment`, each image generates 3 versions:

- ✅ Original
- ✅ Rotate +15°
- ✅ Gaussian noise

**Result:** 2 images → 6 labeled images

## Human-in-the-Loop Review

Use `--review` flag to interactively review and edit annotations based on **confidence scores**:

### Smart Auto-Sorting
- ✅ **Auto-confident** - Detections >= 0.80 confidence skip review
- ✅ **Manual review** - Detections < 0.80 confidence need approval
- ✅ **Organized output** - Files sorted by confidence

### Output Structure (with --review)
```
dataset/yolo/
├── confident/      # Auto-pass (high confidence)
│   ├── images/
│   └── labels/
├── uncertain/      # Manual reviewed
│   ├── images/
│   └── labels/
├── images/         # All images
└── labels/         # All labels
```

### Controls (Super Simple!)
- **CLICK** - Remove detection (disappears instantly)
- **SPACE** - Accept & move to uncertain
- **C** - Mark as confident
- **Q** - Reject image
- **ESC** - Accept & exit

### Visual Feedback
- Each class has its own color
- Click bbox to remove (gone!)
- Counter shows active detections
- Simple & fast!

### Workflow

#### Step 1: Auto-label everything
```bash
python main.py --review
# Or with augmentation
python main.py --review --augment
```

#### Step 2: Review uncertain folder later
Images flagged as uncertain are saved separately for review

#### Step 3: Train with confident data
```bash
# Use only high-confidence data
yolo detect train data=dataset/yolo/data.yaml model=yolov8n.pt
```

### Tips
- Reduce `text_threshold` in config to get more detections (need more review)
- Increase `box_threshold` to filter obvious detections
- Use C key to mark confident detections as you go

## Configuration

### Basic Settings
```json
{
  "paths": {
    "input_folder": "images",
    "output_folder": "dataset"
  },
  "detection": {
    "box_threshold": 0.35,
    "text_threshold": 0.25
  }
}
```

### Classes
```json
// Single class
{
  "classes": {
    "0": "helmet"
  }
}

// Multiple classes
{
  "classes": {
    "0": "helmet",
    "1": "person",
    "2": "safety vest",
    "3": "gloves"
  }
}
```

### Visualization Colors
```json
{
  "colors": {
    "0": [0, 255, 0],
    "1": [255, 0, 0],
    "2": [0, 255, 255]
  }
}
```

## Advanced Usage

### Bounding Box Vectors
```python
from utils_module import BoundingBoxVector
from schemas import Detection

# Convert to vector [cx, cy, bw, bh, class_id]
vectors = BoundingBoxVector.to_vectors(detections)

# Convert to pixel coordinates [x1, y1, x2, y2]
xyxy = BoundingBoxVector.to_xyxy(detection, image_shape)

# Calculate IoU between boxes
iou = BoundingBoxVector.iou(box1, box2)
```

### Custom Augmentation
```python
from utils_module import ImageAugmenter

augmenter = ImageAugmenter()
rotated = augmenter.rotate(image, 15)
noisy = augmenter.add_noise(image, 0.1)
flipped = augmenter.flip_h(image)
```

### YOLO Format
Each `.txt` file contains:
```
class_id x_center y_center width height
0 0.5 0.3 0.1 0.15
1 0.7 0.6 0.2 0.3
```

### COCO Format
JSON with segmentation masks:
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [[x1, y1, x2, y2, ...]],
      "area": 1200
    }
  ],
  "categories": [
    {"id": 1, "name": "helmet"}
  ]
}
```

## Troubleshooting

### No images found
- Verify `images/` folder exists
- Check image formats: `.jpg`, `.png`, `.jpeg`

### No detections
- Lower `box_threshold` (e.g., 0.25)
- Lower `text_threshold` (e.g., 0.15)
- Verify class names match objects

### Import errors
```bash
cd GroundingDINO
pip install -e .
cd ..
```

### Memory issues
- Process fewer images
- Disable augmentation
- Lower image resolution

## Project Structure

```
cpulabel/
├── main.py                 # Entry point
├── configuration.json      # Config file
├── requirements.txt
├── schemas/
│   ├── configuration.py    # Config + Loader
│   ├── detection.py        # Detection dataclass
│   ├── image_result.py     # Result dataclass
│   └── processing_stats.py # Stats dataclass
├── utils_module/
│   ├── file_utils.py       # File operations
│   ├── annotation_utils.py # YOLO/COCO format
│   ├── image_utils.py      # Image processing
│   ├── augmentation.py     # Data augmentation
│   └── box_vector.py       # Bounding box vectors
├── models_module/
│   ├── grounding_dino_model.py  # GroundingDINO
│   └── mobile_sam_model.py      # MobileSAM
└── processors/
    └── image_processor.py  # Main processing logic
```

## Requirements

- Python 3.8+
- ~5GB disk space (for models)
- 4GB+ RAM
- CPU (GPU optional but not required)

## License

MIT

## Credits

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
