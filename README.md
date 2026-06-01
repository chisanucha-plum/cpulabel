# CPULabel

Efficient image labeling using GroundingDINO + MobileSAM. Runs on CPU. No GPU required.

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

### 3. Configure (Optional)
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
python main.py
```

## Output

```
dataset/
├── yolo/
│   ├── images/          # Original images
│   ├── labels/          # YOLO format (.txt)
│   └── data.yaml        # YOLO config
├── coco/
│   ├── images/          # Original images
│   └── annotations/
│       └── instances.json  # COCO format
└── visualizations/      # Masks overlay
```

## Configuration

Edit `configuration.yaml`:

```yaml
models:
  dino_config: "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
  dino_weights: "weights/groundingdino_swint_ogc.pth"
  sam_weights: "MobileSAM/weights/mobile_sam.pt"

paths:
  input_folder: "images"
  output_folder: "dataset"

detection:
  box_threshold: 0.35      # Lower = more detections
  text_threshold: 0.25     # Lower = more detections

classes:
  0: "helmet"
  1: "person"
  2: "safety vest"

colors:
  0: [0, 255, 0]      # Green
  1: [255, 0, 0]      # Blue
  2: [0, 255, 255]    # Yellow
```

## Usage Examples

### Single Class
```yaml
classes:
  0: "helmet"
```

### Multiple Classes
```yaml
classes:
  0: "helmet"
  1: "person"
  2: "safety vest"
  3: "gloves"
```

### Adjust Detection Sensitivity
```yaml
detection:
  box_threshold: 0.25      # More detections
  text_threshold: 0.15     # More detections
```

## Troubleshooting

**No images found:**
- Check `images/` folder exists and contains images

**No detections:**
- Lower `box_threshold` and `text_threshold`
- Verify class names match objects in images

**Import errors:**
- Reinstall GroundingDINO: `cd GroundingDINO && pip install -e .`

## Output Formats

### YOLO Format
```
class_id x_center y_center width height
0 0.5 0.3 0.1 0.15
1 0.7 0.6 0.2 0.3
```

### COCO Format
```json
{
  "images": [...],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 1,
      "bbox": [x, y, width, height],
      "segmentation": [...]
    }
  ],
  "categories": [
    {"id": 1, "name": "helmet"}
  ]
}
```

## Project Structure

```
cpulabel/
├── main.py                 # Entry point
├── configuration.yaml      # Config file
├── requirements.txt
├── schemas/
│   ├── configuration.py    # Config + Loader
│   ├── detection.py
│   ├── image_result.py
│   └── processing_stats.py
├── utils_module/
│   ├── file_utils.py
│   ├── annotation_utils.py
│   └── image_utils.py
├── models_module/
│   ├── grounding_dino_model.py
│   └── mobile_sam_model.py
└── processors/
    └── image_processor.py
```

## Features

- ✅ CPU-friendly 
- ✅ Multi-class detection
- ✅ YOLO + COCO format export
- ✅ Automatic segmentation masks
- ✅ Visualization with overlays
- ✅ Batch processing
- ✅ Easy configuration

## License

MIT
