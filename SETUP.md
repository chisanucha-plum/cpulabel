# 🚀 CPULabel - Complete Setup Guide

Step-by-step guide to get CPULabel running from scratch.

---

## 📋 Prerequisites

- **Python 3.8+** (recommended 3.10 or 3.11)
- **Git** (for cloning)
- **~8GB disk space** (for models)
- **4GB+ RAM**
- **Windows / Linux / Mac**

---

## 🔧 Installation Steps

### Step 1: Clone Repository

```bash
git clone <your-repo-url> cpulabel
cd cpulabel
```

---

### Step 2: Create Virtual Environment

#### **Windows (CMD)**
```bash
python -m venv venv
venv\Scripts\activate.bat
```

#### **Windows (PowerShell)**
```bash
python -m venv venv
venv\Scripts\Activate.ps1
```

#### **Linux / Mac**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- numpy, opencv-python, Pillow
- torch, torchvision, transformers
- GroundingDINO (via git+https)
- And other ML dependencies

---

### Step 4: Setup GroundingDINO (Important!)

GroundingDINO needs to be installed from source:

```bash
cd GroundingDINO
pip install -e .
cd ..
```

**This step is CRITICAL** - without it, detection won't work.

---

### Step 5: Download Model Weights

Models need to be downloaded and placed in the correct folders.

#### **Option A: Automatic Download (Recommended)**

First run will auto-download models to these locations:
- GroundingDINO weights → `weights/groundingdino_swint_ogc.pth`
- MobileSAM weights → `MobileSAM/weights/mobile_sam.pt`

Just run:
```bash
python main.py
```

Models will download automatically (~1GB total) on first run.

#### **Option B: Manual Download**

If auto-download fails, download manually:

**1. GroundingDINO weights**
- URL: https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
- Save to: `weights/groundingdino_swint_ogc.pth`

```bash
mkdir weights
# Download file to weights/ folder
```

**2. MobileSAM weights**
- URL: https://huggingface.co/ChaoningZhang/MobileSAM/resolve/main/mobile_sam.pt
- Save to: `MobileSAM/weights/mobile_sam.pt`

```bash
mkdir -p MobileSAM/weights
# Download file to MobileSAM/weights/ folder
```

---

## 📁 Folder Structure After Setup

```
cpulabel/
├── venv/                          # Virtual environment (created)
├── weights/                       # Model weights (auto-created)
│   └── groundingdino_swint_ogc.pth (downloaded)
├── GroundingDINO/                 # GroundingDINO repository
│   ├── setup.py
│   ├── groundingdino/
│   └── weights/
│       └── mobile_sam.pt (downloaded)
├── MobileSAM/                     # MobileSAM repository
│   ├── setup.py
│   ├── mobile_sam/
│   └── weights/
│       └── mobile_sam.pt (downloaded)
├── images/                        # Input images (create yourself)
├── dataset/                       # Output (created on first run)
│   ├── yolo/
│   ├── coco/
│   └── visualizations/
├── schemas/                       # Data structures
├── utils_module/                  # Utilities
├── models_module/                 # Model wrappers
├── processors/                    # Processing logic
├── main.py                        # Entry point
├── configuration.json             # Settings
├── requirements.txt               # Dependencies
└── SETUP.md                       # This file
```

---

## 🎯 Quick Start After Installation

### 1. Prepare Input Images

```bash
mkdir images
# Copy your images (JPG/PNG) to images/ folder
```

### 2. Configure Classes

Edit `configuration.json`:

```json
{
  "classes": {
    "0": "helmet",
    "1": "black hair"
  },
  "colors": {
    "0": [0, 255, 0],
    "1": [0, 0, 255]
  },
  "detection": {
    "box_threshold": 0.45,
    "text_threshold": 0.30
  }
}
```

### 3. Run with Human Review (Recommended)

```bash
python main.py --review
```

**What happens:**
1. Models load automatically (first run is slower)
2. Images are detected and segmented
3. Interactive viewer opens for each image
4. Click boxes to remove, press C for confident, SPACE for uncertain
5. Results saved to `dataset/`

### 4. Check Results

```
dataset/
├── yolo/confident/      # High-confidence results
├── yolo/uncertain/      # Manual-reviewed results
├── coco/annotations/    # instances.json (COCO format)
└── visualizations/      # Preview images with masks
```

---

## 🐛 Troubleshooting

### Error: "Module 'GroundingDINO' not found"

**Solution:**
```bash
cd GroundingDINO
pip install -e .
cd ..
python main.py
```

### Error: "weights/groundingdino_swint_ogc.pth not found"

**Solution:** Download manually or run with internet connection:
```bash
python main.py
# Will auto-download on first run
```

### Error: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

### Slow on First Run

**Normal behavior:** Models are loading and downloading. Wait 2-5 minutes.
Subsequent runs are faster.

### "No images found"

**Check:**
- `images/` folder exists
- Images are JPG/PNG format
- Images are in `images/` folder (not subdirectories)

```bash
# Verify:
ls images/
```

---

## 🚀 Usage Examples

### Basic Auto-Labeling (No Review)
```bash
python main.py
```

### With Human Review (Recommended)
```bash
python main.py --review
```

### With Data Augmentation
```bash
python main.py --augment
```
Creates 3 versions per image (original, rotated, noise)

### Review + Augment
```bash
python main.py --review --augment
```
Best for maximum quality dataset

### Debug Mode
```bash
python main.py --verbose
```
Shows detailed processing info

---

## 📊 Expected Performance

- **First run:** 5-10 minutes (downloading/initializing models)
- **Subsequent runs:**
  - Without augmentation: 5-20 sec per image
  - With augmentation: 15-60 sec per image
  - With HITL review: +10-30 sec per image (depends on review time)

**Total time for 10 images (with review):** ~5-10 minutes

---

## 🔄 Using on Another Machine

### What to Copy

```bash
# These folders/files are needed:
cpulabel/
├── schemas/
├── utils_module/
├── models_module/
├── processors/
├── main.py
├── configuration.json
├── requirements.txt
├── SETUP.md
├── README.md
└── run.bat (or run.ps1 / run.sh)

# These are created locally, DON'T copy:
❌ venv/
❌ weights/
❌ GroundingDINO/weights/
❌ MobileSAM/weights/
❌ images/
❌ dataset/
```

### Setup on New Machine

```bash
# 1. Clone / copy code to new machine
cd cpulabel

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate.bat  # or source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install GroundingDINO
cd GroundingDINO
pip install -e .
cd ..

# 5. First run (auto-downloads models)
mkdir images
python main.py

# Models will auto-download on first run
```

---

## 💾 Disk Space Estimate

| Component | Size |
|-----------|------|
| GroundingDINO weights | 300MB |
| MobileSAM weights | 40MB |
| Dependencies (venv) | 2-3GB |
| Dataset (varies) | 100MB - 1GB+ |
| **Total** | **~3-5GB** |

Make sure you have **8GB free space** for safety.

---

## ✅ Verification Checklist

After setup, verify everything works:

```bash
# 1. Virtual environment activated?
echo %VIRTUAL_ENV%  # Should show venv path

# 2. Dependencies installed?
pip list | grep torch  # Should show torch

# 3. GroundingDINO installed?
python -c "from groundingdino.util.inference import load_model; print('OK')"

# 4. Models downloaded?
ls weights/
ls MobileSAM/weights/

# 5. Can run?
python main.py --help  # Should show help
```

---

## 🆘 Need Help?

1. Check `README.md` for features and usage
2. Check `ARCHITECTURE.html` for system design
3. Check `SKILL.md` for code style and structure
4. Run with `--verbose` flag for debugging:
   ```bash
   python main.py --verbose
   ```

---

**Last Updated:** June 2026
**Python Version:** 3.8+
**Status:** Production Ready ✅
