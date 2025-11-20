# Spheroid Segmentation Pipeline - Setup Guide

Complete installation and setup guide for the UNet-based spheroid segmentation pipeline.

---

## 📋 Table of Contents

1. [System Requirements](#system-requirements)
2. [Environment Setup](#environment-setup)
3. [Package Installation](#package-installation)
4. [Dataset Organization](#dataset-organization)
5. [Training Pipeline](#training-pipeline)
6. [Troubleshooting](#troubleshooting)

---

## 🖥️ System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (e.g., RTX 3050 with 4GB VRAM)
- **RAM**: At least 8GB system RAM
- **Storage**: ~5GB free space for dataset and models

### Software
- **OS**: Windows, Linux, or macOS
- **Python**: 3.9 or 3.10 (recommended)
- **CUDA**: 11.8 or 12.1 (for GPU acceleration)
- **Conda**: Anaconda or Miniconda

---

## 🔧 Environment Setup

### Step 1: Create Conda Environment

```bash
# Create new environment with Python 3.10
conda create -n spheroid_seg python=3.10

# Activate the environment
conda activate spheroid_seg
```

**Verify environment:**
```bash
python --version
# Should output: Python 3.10.x
```

---

## 📦 Package Installation

### Step 2: Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Check if you have CUDA 11 or 12:**
```bash
nvidia-smi
# Look at the top right corner for "CUDA Version: XX.X"
```

### Step 3: Install Core Dependencies

```bash
# Install FastAI and segmentation library
pip install fastai>=2.7.12
pip install semtorch>=0.2.0

# Install image processing libraries
pip install opencv-python>=4.8.0
pip install Pillow>=9.5.0

# Install scientific computing
pip install numpy>=1.24.0
pip install scipy>=1.10.0

# Install visualization
pip install matplotlib>=3.7.0

# Install metrics and utilities
pip install scikit-learn>=1.3.0
pip install scikit-image>=0.21.0
pip install shapely>=2.0.0
pip install tabulate>=0.9.0
pip install tqdm>=4.65.0
```

### Step 4: Install from requirements.txt (Alternative)

If you have the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Step 5: Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Expected output:**
```
PyTorch: 2.x.x
CUDA available: True
CUDA version: 11.8 (or 12.1)
```

```bash
# Check FastAI
python -c "import fastai; print(f'FastAI: {fastai.__version__}')"

# Check other key packages
python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
python -c "import semtorch; print('semtorch installed successfully')"
```

---

## 📂 Dataset Organization

### Step 6: Organize Your Dataset

Your dataset should be structured as follows:

```
project/
├── data/
│   ├── images/          # Raw images
│   ├── labels/          # Raw masks
│   └── organized/       # Organized dataset (created by script)
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── masks/
│           ├── train/
│           ├── val/
│           └── test/
├── models/              # Saved trained models (created automatically)
├── train_unet.py        # Training script
├── split_dataset.py     # Dataset splitting script
└── requirements.txt     # Dependencies
```

### Step 7: Run Dataset Splitting

```bash
# First, check what suffixes exist in your data
python check_suffixes.py

# Then organize the dataset
python split_dataset.py

# Verify test folder is clean
python verify_test_clean.py
```

**What this does:**
- Groups files by base name
- Assigns files with `_t` suffix to training
- Assigns files with `_v` suffix to validation
- Assigns standalone files to test
- Moves augmentations with their base images

---

## 🚀 Training Pipeline

### Step 8: Visualize Data

```bash
# Create visualization script to check your data
python -c "
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

img_path = Path('./data/organized/images/train')
mask_path = Path('./data/organized/masks/train')

img_files = list(img_path.glob('*.png'))
img = Image.open(img_files[0])
mask = Image.open(mask_path / img_files[0].name)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(img)
axes[0].set_title('Image')
axes[0].axis('off')
axes[1].imshow(mask, cmap='gray')
axes[1].set_title('Mask')
axes[1].axis('off')
plt.tight_layout()
plt.savefig('data_sample.png')
print('✅ Saved to data_sample.png')
"
```

### Step 9: Configure Training Parameters

Edit the `CONFIG` dictionary in `train_unet.py`:

```python
CONFIG = {
    'model_name': 'unet_resnet34_spheroid',
    'architecture': 'unet',
    'backbone': 'resnet34',
    'pretrained': True,
    'resize_factor': 0.5,      # Adjust based on your image size
    'batch_size': 4,           # 4 for RTX 3050 (4GB), 8 for larger GPUs
    'epochs': 20,
    'learning_rate': None,     # Auto-find optimal LR
    'loss_function': CombinedLossCE(alpha=1.0),
    'optimizer': Adam,
    'use_augmentation': True,
    'show_batch': True
}
```

**Configuration Guide:**

| Parameter | RTX 3050 (4GB) | RTX 3060 (6GB) | RTX 3080 (10GB) |
|-----------|----------------|----------------|-----------------|
| batch_size | 4 | 8 | 16 |
| resize_factor | 0.5 | 0.5-0.75 | 1.0 |
| Expected time | 1-1.5 hours | 40-60 min | 20-30 min |

### Step 10: Start Training

```bash
# Monitor GPU usage in another terminal
watch -n 1 nvidia-smi

# Start training
python train_unet.py
```

**What happens during training:**
1. ✅ Dataset loaded and split
2. ✅ Model initialized with pretrained weights
3. ✅ Learning rate automatically found
4. ✅ Training starts with progress bars
5. ✅ Model checkpoints saved automatically
6. ✅ Early stopping if validation doesn't improve
7. ✅ Final validation and metrics computed
8. ✅ Predictions visualized and saved

**Output files created:**
```
training_batch_sample.png       # Sample training batch
learning_rate_finder.png        # LR finder plot
models/unet_resnet34_spheroid.pth  # Best model weights
predictions_unet_resnet34_spheroid/  # Prediction overlays
    sample_000_overlay.png
    sample_000_pred_mask.png
    ...
```

---

## 🐛 Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory (OOM)

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# In train_unet.py, reduce batch_size
'batch_size': 2,  # or even 1

# Or reduce image size
'resize_factor': 0.4,  # or 0.3
```

#### 2. Module Not Found

**Error:**
```
ModuleNotFoundError: No module named 'semtorch'
```

**Solution:**
```bash
# Make sure environment is activated
conda activate spheroid_seg

# Reinstall the package
pip install semtorch
```

#### 3. CUDA Not Available

**Error:**
```
CUDA available: False
```

**Solution:**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 4. Slow Training

**Solutions:**
- Reduce `resize_factor` to make images smaller
- Reduce `epochs` to 10-15
- Disable `use_augmentation` (faster but less robust)
- Use smaller backbone: `'backbone': 'resnet18'`

#### 5. Poor Validation Metrics

**Solutions:**
- Increase `epochs` to 30-40
- Try different loss function: `CombinedLossFocal()`
- Increase `resize_factor` to 0.75 or 1.0
- Enable augmentation: `'use_augmentation': True`
- Use larger backbone: `'backbone': 'resnet50'`

#### 6. Dataset Path Errors

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: './data/organized/images/train'
```

**Solution:**
```bash
# Check if dataset organization was run
ls -la ./data/organized/

# If empty, run the split script
python split_dataset.py
```

---

## 📊 Monitoring Training

### Using TensorBoard (Optional)

```bash
# Install tensorboard
pip install tensorboard

# During training, logs are saved automatically
# View in browser
tensorboard --logdir=./logs
```

### GPU Monitoring

```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or
nvidia-smi dmon -s u
```

### Training Progress

The training script will show:
```
Epoch 1/20: [===================] 100% | Loss: 0.234 | Dice: 0.892 | Time: 2m 15s
Validation: Loss: 0.198 | Dice: 0.915 | Jaccard: 0.845
```

---

## 🎯 Expected Results

### Good Performance Indicators

| Metric | Good | Excellent |
|--------|------|-----------|
| Dice Score | > 0.85 | > 0.90 |
| IoU (Jaccard) | > 0.75 | > 0.85 |
| Accuracy | > 0.95 | > 0.98 |

### Training Time Estimates

| Dataset Size | RTX 3050 | RTX 3060 | RTX 3080 |
|--------------|----------|----------|----------|
| 200 images | ~1 hour | ~40 min | ~20 min |
| 500 images | ~2 hours | ~1 hour | ~30 min |
| 1000 images | ~4 hours | ~2 hours | ~1 hour |

---

## 📝 Quick Reference Commands

```bash
# Environment
conda create -n spheroid_seg python=3.10
conda activate spheroid_seg

# Installation
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Dataset
python split_dataset.py

# Training
python train_unet.py

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## 🆘 Getting Help

If you encounter issues:

1. Check the error message carefully
2. Verify environment is activated: `conda activate spheroid_seg`
3. Check GPU memory: `nvidia-smi`
4. Review the [Troubleshooting](#troubleshooting) section
5. Check package versions: `pip list`

---

## 📚 Additional Resources

- [FastAI Documentation](https://docs.fast.ai/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Semtorch GitHub](https://github.com/WaterKnight1998/SemTorch)
- [UNet Paper](https://arxiv.org/abs/1505.04597)

---

**Last Updated:** November 2025
**Version:** 1.0