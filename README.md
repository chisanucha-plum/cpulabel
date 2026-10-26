# CPULabel

CPULabel generates YOLO and COCO datasets from unlabeled images using GroundingDINO for object detection and MobileSAM for segmentation. It runs on CPU, though PyTorch will use CUDA if available.

## Setup

Requires Python 3.8+, Git, 4 GB RAM, and about 5 GB of disk space for dependencies and model weights.

```bash
python -m venv venv
venv\Scripts\activate.bat      # Windows CMD
# venv\Scripts\Activate.ps1    # Windows PowerShell
# source venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
pip install -e ./src/groundingdino
git clone https://github.com/ChaoningZhang/MobileSAM.git MobileSAM
```

Download the model weights into their expected paths:

- GroundingDINO: [groundingdino_swint_ogc.pth](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) into `weights/groundingdino_swint_ogc.pth`
- MobileSAM: [mobile_sam.pt](https://huggingface.co/ChaoningZhang/MobileSAM/resolve/main/mobile_sam.pt) into `MobileSAM/weights/mobile_sam.pt`

## Usage

Place `.jpg` or `.png` files in `images/` and run `main.py`:

```bash
# Automatic labeling
python main.py

# Interactive review before saving
python main.py --review

# Interactive review with data augmentation
python main.py --review --augment

# Enable debug logging
python main.py --verbose
```

### Interactive review (`--review`)

When `--review` is enabled, images where every detection has confidence `>= 0.80` are saved directly to `dataset/yolo/confident/`. Images with lower confidence open in an OpenCV window for manual inspection:

| Input | Action |
|---|---|
| Left click on box | Remove that detection |
| `Space` or `Esc` | Save remaining detections to `dataset/yolo/uncertain/` |
| `C` | Save remaining detections to `dataset/yolo/confident/` |
| `Q` | Reject the image and skip saving |

Images with no detections are rejected automatically.

### Data augmentation (`--augment`)

Passing `--augment` creates 4 additional variants for each accepted image (5 total):

- Original image
- Rotated +15 degrees (`_rot15`)
- Gaussian noise (`_noise`)
- Horizontal flip (`_fliph`)
- Darkened (`_dark`)

Bounding boxes and segmentation masks are transformed along with the rotated and flipped images.

## Configuration

Settings live in `configuration.json`:

```json
{
  "models": {
    "dino_config": "src/groundingdino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "dino_weights": "weights/groundingdino_swint_ogc.pth",
    "sam_weights": "MobileSAM/weights/mobile_sam.pt"
  },
  "paths": {
    "input_folder": "images",
    "output_folder": "dataset"
  },
  "detection": {
    "box_threshold": 0.45,
    "text_threshold": 0.30
  },
  "helmet": {
    "iou_threshold": 0.30
  },
  "classes": {
    "0": {
      "name": "helmet",
      "box_threshold": 0.45,
      "text_threshold": 0.30
    },
    "1": {
      "name": "head",
      "box_threshold": 0.15,
      "text_threshold": 0.10
    },
    "2": {
      "name": "no_helmet",
      "detect": false
    }
  },
  "colors": {
    "0": [0, 255, 0],
    "1": [255, 140, 0],
    "2": [0, 0, 255]
  }
}
```

Each entry in `classes` can be a plain string (`"0": "person"`) or an object with per-class `box_threshold`, `text_threshold`, and `detect` settings. Setting `"detect": false` excludes that class from GroundingDINO prompts while keeping it in the exported class list.

When `helmet`, `head`, and `no_helmet` are all defined in `classes`, CPULabel compares each detected `head` against detected `helmet` boxes using `helmet.iou_threshold`. Heads that overlap a helmet above the threshold become `helmet`; the rest become `no_helmet`.

## Output layout

```text
dataset/
├── yolo/
│   ├── images/          # Standard output (without --review)
│   ├── labels/
│   ├── confident/       # High-confidence or C-approved output (with --review)
│   │   ├── images/
│   │   └── labels/
│   ├── uncertain/       # Space/Esc-approved output (with --review)
│   │   ├── images/
│   │   └── labels/
│   └── data.yaml
├── coco/
│   ├── images/
│   └── annotations/
│       └── instances.json
└── visualizations/      # Bounding box and mask overlays
```

## Project structure

```text
cpulabel/
├── main.py                         # CLI entry point
├── configuration.json              # Paths, thresholds, and class definitions
├── schemas/
│   ├── __init__.py                 # Package exports
│   └── models.py                   # Config, ConfigLoader, Detection, ImageResult, ProcessingStats
├── models/
│   ├── dino.py                     # GroundingDINO detection wrapper
│   └── sam.py                      # MobileSAM segmentation wrapper
├── pipeline/
│   ├── labeler.py                  # Detection, review, augmentation, and export pipeline
│   └── helmet.py                   # Head-to-helmet IoU matching
├── utils/
│   ├── annotations.py              # YOLO and COCO writers
│   ├── augment.py                  # Image and bounding-box augmentations
│   ├── boxes.py                    # Box coordinate conversions and IoU
│   ├── files.py                    # Dataset directory setup and visualization output
│   └── viewer.py                   # OpenCV review window
└── tests/                          # Unit tests
```

## Tests

Run the unit test suite from the virtual environment:

```bash
.\venv\Scripts\python.exe -m unittest discover tests
```

## Credits and license

Built on [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) and [MobileSAM](https://github.com/ChaoningZhang/MobileSAM). Licensed under MIT.
