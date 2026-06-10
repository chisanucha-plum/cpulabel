# CPULabel

**AI-powered auto-labeling** for creating YOLO and COCO datasets  
Runs on **standard CPU** – no GPU required

🚀 **Tech Stack**: GroundingDINO (detection) + MobileSAM (segmentation)

---

## ✨ Features

- ✅ **CPU-friendly** – No GPU needed
- ✅ **Multi-class detection** – Detect multiple object classes simultaneously
- ✅ **Human-in-the-loop** – Interactive review with click-to-remove detections
- ✅ **Per-class thresholds** – Fine-tune detection sensitivity per class
- ✅ **Data augmentation** – Generate 3 versions per image (original, rotate, noise)
- ✅ **YOLO + COCO export** – Export both formats with segmentation masks
- ✅ **Production ready** – Error handling, logging, and clean architecture

---

## 🎯 Quick Start

### 1️⃣ Setup (First Time)
See `SETUP.md` for detailed instructions

```bash
# Clone/download project
cd testsam_groud

# Create venv and install dependencies
python -m venv venv
venv\Scripts\activate.bat

pip install -r requirements.txt
cd GroundingDINO
pip install -e .
cd ..
```

### 2️⃣ Prepare Images
```bash
mkdir images
# Copy your images into images/ folder
```

### 3️⃣ Run
```bash
# Basic processing
python main.py

# ✨ Recommended: Interactive review
python main.py --review

# With 3x data augmentation per image
python main.py --review --augment

# Debug mode with verbose logging
python main.py --review --verbose
```

---

## 🎮 Human-in-the-Loop Review

Use `--review` flag to interactively review and refine detections

### 🔦 Smart Auto-Sorting
- **Confidence ≥ 0.80** → Auto-pass (saved to `confident/`)
- **Confidence < 0.80** → Manual review (saved to `uncertain/`)

### 🖱️ Keyboard Controls
| Key | Action |
|-----|--------|
| **CLICK** | Remove detection (disappears immediately!) |
| **SPACE / ESC** | Accept → mark as uncertain |
| **C** | Mark as confident |
| **Q** | Reject entire image |

### 📁 Output Structure (with --review)
```
dataset/yolo/
├── confident/          # Auto-pass ≥ 0.80 confidence
│   ├── images/
│   └── labels/
├── uncertain/          # Manual reviewed < 0.80 confidence
│   ├── images/
│   └── labels/
└── data.yaml
```

---

## ⚙️ Configuration

Edit `configuration.json` to customize detection behavior:

### Current Setup
```json
{
  "paths": {
    "input_folder": "images",
    "output_folder": "dataset"
  },
  "classes": {
    "0": {
      "name": "helmet",
      "box_threshold": 0.45,      // Conservative: high-quality only
      "text_threshold": 0.30
    },
    "1": {
      "name": "head",
      "box_threshold": 0.25,      // Aggressive: catch more instances
      "text_threshold": 0.20
    }
  },
  "colors": {
    "0": [0, 255, 0],      // helmet = green
    "1": [0, 0, 255]       // head = red
  }
}
```

### Add a New Class
```json
"2": {
  "name": "safety_vest",
  "box_threshold": 0.40,
  "text_threshold": 0.25
},
```

### Understanding Per-Class Thresholds
- **box_threshold** → Confidence threshold for bounding box detection (↑ = stricter, fewer false positives)
- **text_threshold** → Confidence threshold for text recognition (↑ = stricter, more selective)

💡 **helmet**: Higher thresholds (0.45/0.30) = focus on high-quality detections only  
💡 **head**: Lower thresholds (0.25/0.20) = catch more instances, reduce false negatives

---

## 📊 Output Structure

```
dataset/
├── yolo/
│   ├── confident/       # High-confidence detections
│   ├── uncertain/       # Manually reviewed detections
│   ├── images/          # All processed images
│   ├── labels/          # YOLO format annotations
│   └── data.yaml        # YOLO configuration
├── coco/
│   ├── images/
│   └── annotations/instances.json  # COCO format with masks
└── visualizations/      # Preview images with overlays
```

---

## 📈 Data Augmentation

When using `--augment`, each image generates 3 versions:

```
1 image → 3 versions:
  ✅ Original
  ✅ Rotated +15°
  ✅ Gaussian noise applied
```

**Result**: 5 images → 15 labeled images (faster processing, maintained quality)

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| No images found | Verify `images/` folder exists, check for `.jpg/.png` files |
| No detections | Lower `box_threshold` and `text_threshold` in config |
| ImportError | Run `cd GroundingDINO && pip install -e . && cd ..` |
| Slow processing | Reduce augmentation or disable it, lower image resolution |
| Memory errors | Process fewer images, disable augmentation, use smaller images |

---

## 📁 Project Structure

```
cpulabel/
├── main.py                       # Entry point
├── configuration.json            # Configuration
├── README.md / SETUP.md / SKILL.md
├── schemas/                      # Data structures
│   ├── configuration.py
│   ├── detection.py
│   ├── image_result.py
│   └── processing_stats.py
├── utils_module/                 # Utility functions
│   ├── file_utils.py             # File I/O operations
│   ├── annotation_utils.py       # YOLO/COCO format handlers
│   ├── augmentation.py           # Image augmentation
│   ├── hitl_viewer.py            # Interactive review interface
│   └── box_vector.py             # Box operations & IoU calculation
├── models_module/                # AI Models
│   ├── grounding_dino_model.py   # GroundingDINO wrapper
│   └── mobile_sam_model.py       # MobileSAM wrapper
└── processors/
    └── image_processor.py        # Main processing pipeline
```

---

## 📦 Requirements

- Python 3.8+
- RAM: 4GB minimum
- Disk space: ~5GB (for model weights)
- CPU: Any modern processor

**GPU**: Optional (CPU-only mode is fully supported)

---

## 🚀 Workflow Example

```bash
# 1. Initial setup
python -m venv venv
venv\Scripts\activate.bat
pip install -r requirements.txt && cd GroundingDINO && pip install -e . && cd ..

# 2. Place your images
# Copy images to images/ folder

# 3. Auto-label with interactive review (recommended)
python main.py --review --augment

# 4. Check results
# - High-confidence labels: dataset/yolo/confident/
# - Manual-reviewed labels: dataset/yolo/uncertain/

# 5. Train your YOLO model
# yolo detect train data=dataset/yolo/data.yaml model=yolov8n.pt epochs=50
```

---

## 📚 Documentation

- **SETUP.md** – Detailed setup guide with troubleshooting
- **SKILL.md** – Coding style guide for contributors
- **ARCHITECTURE.html** – System architecture and design

---

## 🏆 Credits

- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) – Open-vocabulary object detection
- [MobileSAM](https://github.com/ChaoningZhang/MobileSAM) – Lightweight segmentation model

---

## 📝 License

MIT
