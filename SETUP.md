# Setup Guide - CPULabel

## Prerequisites

- Python 3.8+
- Git
- ~5GB disk space (for models)

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd cpulabel
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install GroundingDINO
```bash
cd GroundingDINO
pip install -e .
cd ..
```

### 5. Download Models

#### GroundingDINO Weights
```bash
mkdir -p weights
cd weights
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
cd ..
```

#### MobileSAM Weights
```bash
mkdir -p MobileSAM/weights
cd MobileSAM/weights
wget https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt
cd ../..
```

Or download manually:
- [GroundingDINO weights](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth)
- [MobileSAM weights](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt)

### 6. Verify Installation
```bash
python -c "from schemas import ConfigLoader; print('✓ Installation successful')"
```

## Quick Start

```bash
# 1. Prepare images
mkdir images
# Copy your images to images/ folder

# 2. Run
python main.py

# 3. Check results
ls dataset/
```

## Folder Structure After Setup

```
cpulabel/
├── GroundingDINO/          # Clone from GitHub
├── MobileSAM/              # Clone from GitHub
├── weights/                # Download models here
│   └── groundingdino_swint_ogc.pth
├── images/                 # Your input images
├── dataset/                # Output (auto-created)
├── main.py
├── configuration.yaml
└── ...
```

## Troubleshooting

### ModuleNotFoundError: No module named 'groundingdino'
```bash
cd GroundingDINO
pip install -e .
cd ..
```

### FileNotFoundError: weights/groundingdino_swint_ogc.pth
- Download weights manually from [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO)
- Place in `weights/` folder

### CUDA/GPU errors
- This project runs on CPU by default
- Remove any GPU-specific code if needed

### Memory issues
- Reduce image size
- Process fewer images at once
- Lower `box_threshold` in configuration.yaml

## Notes

- GroundingDINO and MobileSAM folders are large (~2GB each)
- They are in `.gitignore` - clone/download separately
- Models are downloaded on first run if not present
- CPU processing is slower than GPU but works fine for most use cases

## Next Steps

1. Read [README.md](README.md) for usage
2. Edit `configuration.yaml` for your classes
3. Run `python main.py`
