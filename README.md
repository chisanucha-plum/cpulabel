# CPULabel

Efficient image labeling using GroundingDINO + MobileSAM. Runs on CPU. No GPU required.

Generates YOLO and COCO format datasets with automatic segmentation masks.

## Features

- ✅ **CPU-friendly** - No GPU required
- ✅ **Multi-class detection** - Detect multiple object classes
- ✅ **Data augmentation** - 5x dataset expansion (rotation, noise, flip)
- ✅ **YOLO + COCO format** - Export to both formats
- ✅ **Automatic segmentation** - Precise masks with MobileSAM
- ✅ **Batch processing** - Process multiple images
- ✅ **Bounding box vectors** - Convert to vector representations
- ✅ **Production ready** - Error handling, logging, clean code

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd GroundingDINO
pip install -e .
cd ..
```

### 2. Prepare Images
```bash
mkdir images
# Copy your images to images/ folder
```

### 3. Configure
Edit `configuration.yaml`:
```yaml
classes:
  0: "helmet"
  1: "person"
  2: "safety vest"

detection:
  box_threshold: 0.35
  text_threshold: 0.25
```

### 4. Run
```bash
# Basic labeling
python main.py

# With data augmentation (5x images)
python main.py --augment
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

When using `--augment`, each image generates 5 versions:

- ✅ Original
- ✅ Rotate +15°
- ✅ Rotate -15°
- ✅ Gaussian noise
- ✅ Horizontal flip

**Result:** 2 images → 10 labeled images

## Configuration

### Basic Settings
```yaml
paths:
  input_folder: "images"
  output_folder: "dataset"

detection:
  box_threshold: 0.35      # Detection confidence
  text_threshold: 0.25     # Text matching confidence
```

### Classes
```yaml
# Single class
classes:
  0: "helmet"

# Multiple classes
classes:
  0: "helmet"
  1: "person"
  2: "safety vest"
  3: "gloves"
```

### Visualization Colors
```yaml
colors:
  0: [0, 255, 0]      # Green
  1: [255, 0, 0]      # Blue
  2: [0, 255, 255]    # Yellow
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

## Output Formats

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
├── configuration.yaml      # Config file
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
