import multiprocessing
from ultralytics import YOLO
import torch
import albumentations as A
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================
DATA_YAML = "yolo_splits/data.yaml"
MODEL = "yolov8n-seg.pt"
EPOCHS = 100
IMG_SIZE = 1024
PROJECT = "yolo_training_output"
NAME = "cpu_run_final_tensordeepfix2"  # change name so it doesn't overwrite

# ============================================================
# DEVICE SELECTION
# ============================================================
if torch.cuda.is_available():
    device = 0  # GPU index 0
    batch_size = 16
    print("🔥 GPU detected — using CUDA.")
else:
    device = "cpu"
    batch_size = 8
    print("🧠 Training on CPU — multi-core enabled.")

NUM_WORKERS = min(4, multiprocessing.cpu_count())  # keep small for CPU
print(f"🧵 Using up to {NUM_WORKERS} CPU workers for dataloading.")

# ============================================================
# CUSTOM AUGMENTATIONS
# ============================================================
gaussian_sigma = 1.0

custom_augs = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=gaussian_sigma, p=0.4),
    A.Normalize(mean=(0.531, 0.531, 0.531), std=(0.207, 0.207, 0.207)),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
])

# ------------------------------------------------------------
# Helper: apply Albumentations to a single image tensor/array
# ------------------------------------------------------------
def apply_albumentations_to_img(img):
    """
    Takes an image (torch.Tensor [C,H,W] or np.ndarray [H,W,C]),
    runs Albumentations, and returns the same type back.
    """
    original_type = type(img)

    # Convert to numpy HWC uint8 for Albumentations
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.dtype != np.uint8:
            # Albumentations expects uint8 typically; adapt if needed
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    elif isinstance(img, np.ndarray):
        img_np = img
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        # Unknown type, just return
        return img

    augmented = custom_augs(image=img_np)
    aug_img = augmented["image"]

    # Convert back
    if original_type is torch.Tensor:
        aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1).float()
        # You can normalize/scale here if needed to match YOLO's expectations
        return aug_tensor
    else:
        return aug_img

# ============================================================
# CALLBACK TO WRAP ORIGINAL TRANSFORMS
# ============================================================
def add_custom_transforms(trainer):
    print("🔧 Injecting Albumentations wrapper around YOLO transforms (safe)...")
    try:
        ds = None
        if hasattr(trainer, "train_loader"):
            ds = getattr(trainer.train_loader, "dataset", None)
            # In YOLO this is usually a WrappedDataset/NestedDataset
            if hasattr(ds, "dataset"):
                ds = ds.dataset  # unwrap inner dataset

        if ds is None:
            print("⚠️ Could not locate underlying dataset to apply transforms.")
            return

        original_transforms = getattr(ds, "transforms", None)

        if original_transforms is None:
            print("⚠️ Dataset has no 'transforms' attribute, skipping injection.")
            return

        # ----------------------------
        # Define wrapped transform
        # ----------------------------
        def wrapped_transforms(sample):
            # First, let YOLO do all its usual stuff (including adding 'batch_idx')
            sample = original_transforms(sample)

            # Then apply Albumentations only on the image
            if isinstance(sample, dict) and "img" in sample:
                sample["img"] = apply_albumentations_to_img(sample["img"])

            return sample

        ds.transforms = wrapped_transforms
        print("✅ Albumentations wrapper successfully attached on top of YOLO transforms.")

    except Exception as e:
        print(f"⚠️ Transform injection failed: {e}")

# ============================================================
# BUILD MODEL
# ============================================================
model = YOLO(MODEL)
model.add_callback("on_train_start", add_custom_transforms)

# ============================================================
# TRAIN
# ============================================================
if __name__ == "__main__":
    print("\n🚀 Starting YOLO training...\n")

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
    )

    # ========================================================
    # LOG TRAINING PARAMETERS
    # ========================================================
    os.makedirs(f"{PROJECT}/{NAME}", exist_ok=True)
    with open(f"{PROJECT}/{NAME}/params.txt", "w") as f:
        f.write("YOLO Training Parameters\n")
        f.write("=========================\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"CPU Workers: {NUM_WORKERS}\n")
        f.write("\n--- Augmentations ---\n")
        f.write(f"GaussianBlur: σ={gaussian_sigma}, prob=0.4, kernel=3–5\n")
        f.write(f"Brightness normalization: mean=0.531, std=0.207\n")
        f.write(f"HorizontalFlip: p=0.5\n")
        f.write(f"RandomBrightnessContrast: p=0.3\n")

    print("\n🎉 Training complete!")
    print(f"📁 Output saved to: {PROJECT}/{NAME}")
