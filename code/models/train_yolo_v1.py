import multiprocessing
from ultralytics import YOLO
import torch
import albumentations as A
import numpy as np
import os
import cv2
from glob import glob
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================
DATA_YAML = "data/consolidated_yolo_augmented_v1/data.yaml"
MODEL = "yolov8n-seg.pt"
EPOCHS = 1000
IMG_SIZE = 512
PROJECT = "yolo_training_output"
NAME = "yolov8n_seg_spheroids_1024_v1"  # change name so it doesn't overwrite

# Folder with images & GT masks for boundary evaluation after training
# Adjust these to your validation set layout
EVAL_IMG_DIR = "data/consolidated_yolo_augmented_v1/val/images"
EVAL_MASK_DIR = "data/consolidated_yolo_augmented_v1/val/masks"  # you may need to adapt this

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
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    elif isinstance(img, np.ndarray):
        img_np = img
        if img_np.dtype != np.uint8:
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    else:
        return img

    augmented = custom_augs(image=img_np)
    aug_img = augmented["image"]

    # Convert back
    if original_type is torch.Tensor:
        aug_tensor = torch.from_numpy(aug_img).permute(2, 0, 1).float()
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
            sample = original_transforms(sample)
            if isinstance(sample, dict) and "img" in sample:
                sample["img"] = apply_albumentations_to_img(sample["img"])
            return sample

        ds.transforms = wrapped_transforms
        print("✅ Albumentations wrapper successfully attached on top of YOLO transforms.")

    except Exception as e:
        print(f"⚠️ Transform injection failed: {e}")

# ============================================================
# SOFT BOUNDARY LOSS (for METRIC, not training loss)
# ============================================================
def soft_boundary_loss(pred_mask, gt_mask):
    """
    pred_mask, gt_mask: [H, W] or [1, H, W] tensors (float 0-1)
    Returns a scalar boundary discrepancy (lower is better).
    Used here as a METRIC, not part of YOLO's training loss.
    """
    if pred_mask.dim() == 3:
        pred_mask = pred_mask[0]
    if gt_mask.dim() == 3:
        gt_mask = gt_mask[0]

    pred_mask = pred_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gt_mask = gt_mask.unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred_mask.device).unsqueeze(0).unsqueeze(0)

    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=pred_mask.device).unsqueeze(0).unsqueeze(0)

    pred_dx = F.conv2d(pred_mask, sobel_x, padding=1)
    pred_dy = F.conv2d(pred_mask, sobel_y, padding=1)
    gt_dx = F.conv2d(gt_mask, sobel_x, padding=1)
    gt_dy = F.conv2d(gt_mask, sobel_y, padding=1)

    pred_grad = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + 1e-6)
    gt_grad = torch.sqrt(gt_dx ** 2 + gt_dy ** 2 + 1e-6)

    return torch.mean(torch.abs(pred_grad - gt_grad)).item()


def boundary_iou(pred_mask, gt_mask, boundary_width=3):
    """
    Very simple boundary IoU-like metric: extracts boundary bands and computes IoU.
    pred_mask, gt_mask: [H,W] numpy arrays (0/1).
    """
    pred_mask = (pred_mask > 0.5).astype(np.uint8)
    gt_mask = (gt_mask > 0.5).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)

    pred_erode = cv2.erode(pred_mask, kernel, iterations=boundary_width)
    gt_erode = cv2.erode(gt_mask, kernel, iterations=boundary_width)

    pred_boundary = pred_mask - pred_erode
    gt_boundary = gt_mask - gt_erode

    inter = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()
    if union == 0:
        return 1.0  # both empty
    return inter / union


def hausdorff_like(pred_mask, gt_mask):
    """
    Approximate Hausdorff distance between boundaries using OpenCV contours.
    Returns distance in pixels (lower is better).
    """
    pred = (pred_mask > 0.5).astype(np.uint8)
    gt = (gt_mask > 0.5).astype(np.uint8)

    contours_p, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_g, _ = cv2.findContours(gt, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if not contours_p or not contours_g:
        return 0.0

    pts_p = np.concatenate(contours_p, axis=0).reshape(-1, 2)
    pts_g = np.concatenate(contours_g, axis=0).reshape(-1, 2)

    # directed distances
    def directed_hd(a, b):
        dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        return dists.min(axis=1).max()

    hd_pg = directed_hd(pts_p, pts_g)
    hd_gp = directed_hd(pts_g, pts_p)

    return max(hd_pg, hd_gp)


# ============================================================
# EVALUATION: RUN BOUNDARY METRICS ON A VAL SET
# ============================================================
def evaluate_boundaries(model, img_dir, mask_dir, imgsz=512, device="cpu", max_samples=None):
    """
    Runs inference on a folder of images + masks and computes:
      - mean Soft Boundary Loss
      - mean Boundary IoU
      - mean Hausdorff-like distance
    Assumes:
      - image files in img_dir
      - corresponding binary masks in mask_dir, with same stem + .png or .tif etc.
    """
    model.to(device)
    model.eval()

    image_paths = sorted(glob(os.path.join(img_dir, "*")))
    if max_samples is not None:
        image_paths = image_paths[:max_samples]

    sbl_list = []
    biou_list = []
    hd_list = []

    print(f"\n🔍 Evaluating boundary metrics on {len(image_paths)} samples...\n")

    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Try a few possible mask extensions; adapt if needed
        mask_path = None
        for ext in [".png", ".tif", ".tiff", ".jpg"]:
            candidate = os.path.join(mask_dir, basename + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break

        if mask_path is None:
            print(f"⚠️ No mask found for {basename}, skipping.")
            continue

        # Load image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠️ Failed to read {img_path}, skipping.")
            continue

        # Load GT mask as binary
        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if gt is None:
            print(f"⚠️ Failed to read mask {mask_path}, skipping.")
            continue
        gt = (gt > 0).astype(np.uint8)

        # Resize both to model size
        img_resized = cv2.resize(img, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST)

        # Run model
        results = model.predict(img_resized, imgsz=imgsz, device=device, verbose=False)
        r = results[0]

        if r.masks is None:
            print(f"⚠️ No masks predicted for {basename}, skipping.")
            continue

        # Convert instance masks -> single semantic mask by union
        pm = r.masks.data  # [N, H, W]
        pm_semantic = (pm.sigmoid().max(dim=0)[0] > 0.5).float()  # [H, W]

        gt_tensor = torch.from_numpy(gt_resized).float().to(pm_semantic.device)

        # Soft boundary loss (as metric)
        sbl = soft_boundary_loss(pm_semantic, gt_tensor)

        # Boundary IoU + Hausdorff-like using numpy
        pm_np = pm_semantic.detach().cpu().numpy()
        gt_np = gt_resized.astype(np.float32)

        biou = boundary_iou(pm_np, gt_np)
        hd = hausdorff_like(pm_np, gt_np)

        sbl_list.append(sbl)
        biou_list.append(biou)
        hd_list.append(hd)

    if not sbl_list:
        print("⚠️ No samples evaluated for boundary metrics.")
        return

    mean_sbl = float(np.mean(sbl_list))
    mean_biou = float(np.mean(biou_list))
    mean_hd = float(np.mean(hd_list))

    print("\n📏 Boundary metrics on validation set:")
    print(f"  ▸ Mean Soft Boundary Loss (lower is better): {mean_sbl:.6f}")
    print(f"  ▸ Mean Boundary IoU         (higher is better): {mean_biou:.4f}")
    print(f"  ▸ Mean Hausdorff-like Dist  (lower is better, px): {mean_hd:.2f}\n")

    return {
        "soft_boundary_loss": mean_sbl,
        "boundary_iou": mean_biou,
        "hausdorff_like": mean_hd,
    }

# ============================================================
# BUILD MODEL
# ============================================================
model = YOLO(MODEL)
model.add_callback("on_train_start", add_custom_transforms)

# ============================================================
# TRAIN + (OPTIONAL) EVAL
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

    # ========================================================
    # OPTIONAL: EVALUATE BOUNDARY QUALITY ON VAL SET
    # ========================================================
    # Comment this out if you don't want to run evaluation automatically
    if os.path.isdir(EVAL_IMG_DIR) and os.path.isdir(EVAL_MASK_DIR):
        evaluate_boundaries(model, EVAL_IMG_DIR, EVAL_MASK_DIR, imgsz=IMG_SIZE,
                            device="cuda" if torch.cuda.is_available() else "cpu",
                            max_samples=None)
    else:
        print("ℹ️ Skipping boundary evaluation: EVAL_IMG_DIR or EVAL_MASK_DIR not found.")



