#!/usr/bin/env python3
"""
YOLOv8-seg Training Script (boundary-friendly)

Key changes vs your older script:
1) Removes intensity normalization + RandomBrightnessContrast (these often hurt boundary learning).
2) Uses a conservative augmentation set (flip + mild blur only).
3) Sets imgsz default to 768 (change to 512 if VRAM is insufficient).
4) Optionally enables Ultralytics built-in multi-scale training (off by default).

Run:
  python src_new/train_yolo_seg_boundary_friendly.py

Outputs:
  yolo_training_output/<NAME>/weights/best.pt
"""

from __future__ import annotations

import multiprocessing
import os

import albumentations as A
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
DATA_YAML = "data/consolidated_yolo_augmented_v1/data.yaml"

# Base model checkpoint
MODEL = "yolov8n-seg.pt"

# Training
EPOCHS = 1000

# Try 768 first; if GPU OOM, change to 512.
IMG_SIZE = 768

PROJECT = "yolo_training_output"
NAME = "yolov8n_seg_spheroids_boundary_friendly_768_v1"  # change freely

# Device selection
if torch.cuda.is_available():
    device = 0
    batch_size = 8  # start lower for 768; raise if you have VRAM
    print("🔥 GPU detected — using CUDA.")
else:
    device = "cpu"
    batch_size = 4
    print("🧠 Training on CPU — multi-core enabled.")

NUM_WORKERS = min(4, multiprocessing.cpu_count())
print(f"🧵 Using up to {NUM_WORKERS} CPU workers for dataloading.")

# Optional Ultralytics multiscale training
# - This can help robustness but may slow training.
# - Keep False for now, enable later if stable.
USE_MULTISCALE = False

# ============================================================
# CONSERVATIVE AUGMENTATIONS (boundary-friendly)
# ============================================================
# Note: we are intentionally NOT using:
# - A.Normalize(...)
# - A.RandomBrightnessContrast(...)
# Because boundary precision is highly sensitive to inconsistent intensity transforms.

custom_augs = A.Compose(
    [
        A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.6), p=0.20),
    ]
)

# ------------------------------------------------------------
# Helper: apply Albumentations to a single image tensor/array
# ------------------------------------------------------------
def apply_albumentations_to_img(img):
    """
    Takes an image (torch.Tensor [C,H,W] or np.ndarray [H,W,C]),
    runs Albumentations, and returns the same type back.

    Important:
    - Ultralytics dataset images are typically float32 (0..255) tensors.
    - Albumentations expects uint8 HWC.
    """
    original_is_tensor = isinstance(img, torch.Tensor)

    if original_is_tensor:
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
    elif isinstance(img, np.ndarray):
        img_np = img
    else:
        return img

    # Ensure uint8 for Albumentations
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    augmented = custom_augs(image=img_np)
    aug_img = augmented["image"]

    if original_is_tensor:
        # Back to CHW float32
        return torch.from_numpy(aug_img).permute(2, 0, 1).float()
    return aug_img


# ============================================================
# CALLBACK TO WRAP ORIGINAL TRANSFORMS
# ============================================================
def add_custom_transforms(trainer):
    print("🔧 Injecting Albumentations wrapper around YOLO dataset transforms...")
    try:
        ds = None
        if hasattr(trainer, "train_loader"):
            ds = getattr(trainer.train_loader, "dataset", None)
            if hasattr(ds, "dataset"):
                ds = ds.dataset  # unwrap inner dataset

        if ds is None:
            print("⚠️ Could not locate underlying dataset to apply transforms.")
            return

        original_transforms = getattr(ds, "transforms", None)
        if original_transforms is None:
            print("⚠️ Dataset has no 'transforms' attribute, skipping injection.")
            return

        def wrapped_transforms(sample):
            sample = original_transforms(sample)
            if isinstance(sample, dict) and "img" in sample:
                sample["img"] = apply_albumentations_to_img(sample["img"])
            return sample

        ds.transforms = wrapped_transforms
        print("✅ Albumentations wrapper successfully attached on top of YOLO transforms.")

    except Exception as e:
        print(f"⚠️ Transform injection failed: {e}")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Starting YOLO training (boundary-friendly settings)...\n")
    print(f"DATA_YAML: {DATA_YAML}")
    print(f"MODEL:     {MODEL}")
    print(f"IMG_SIZE:  {IMG_SIZE}")
    print(f"EPOCHS:    {EPOCHS}")
    print(f"BATCH:     {batch_size}")
    print(f"DEVICE:    {device}")
    print(f"PROJECT:   {PROJECT}")
    print(f"NAME:      {NAME}")
    print(f"MULTISCALE:{USE_MULTISCALE}")
    print("")

    model = YOLO(MODEL)
    model.add_callback("on_train_start", add_custom_transforms)

    # Train
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        device=device,
        workers=NUM_WORKERS,
        project=PROJECT,
        name=NAME,
        batch=batch_size,
        verbose=True,
        augment=True,
        # Keep Ultralytics augmentations on, but do NOT stack aggressive
        # brightness jitter in Albumentations (we removed that above).
        multi_scale=USE_MULTISCALE,
    )

    # Log parameters
    os.makedirs(f"{PROJECT}/{NAME}", exist_ok=True)
    with open(f"{PROJECT}/{NAME}/params.txt", "w", encoding="utf-8") as f:
        f.write("YOLO Training Parameters (boundary-friendly)\n")
        f.write("==========================================\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"CPU Workers: {NUM_WORKERS}\n")
        f.write(f"Ultralytics multi_scale: {USE_MULTISCALE}\n")
        f.write("\n--- Albumentations (conservative) ---\n")
        f.write("GaussianBlur: blur_limit=(3,3), sigma=(0.1..0.6), p=0.20\n")
        f.write("HorizontalFlip: p=0.50\n")
        f.write("\nNotes:\n")
        f.write("- Removed Normalize + RandomBrightnessContrast to improve boundary consistency.\n")
        f.write("- If GPU OOM at 768, reduce imgsz to 512 and/or batch_size.\n")

    print("\n🎉 Training complete!")
    print(f"📁 Output saved to: {PROJECT}/{NAME}")
    print(f"➡️ Weights should be at: {PROJECT}/{NAME}/weights/best.pt")
