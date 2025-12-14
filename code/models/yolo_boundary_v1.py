import multiprocessing
from types import SimpleNamespace
from ultralytics import YOLO
import torch
import torch.nn.functional as F
import albumentations as A
import numpy as np
import os

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask

# ============================================================
# CONFIGURATION
# ============================================================

# Use your new split dataset
DATA_YAML = "data_new/yolo_splits_v1/data.yaml"  # assumes running from project root

MODEL = "yolov8n-seg.pt"

# You can change this to 512, 768, 1024, etc.
IMG_SIZE = 512

EPOCHS = 1000
PROJECT = "yolo_training_output"
NAME = "yolov8n_seg_spheroids_softboundary_v2"

# Weight for boundary term in the segmentation loss
LAMBDA_BOUNDARY = 0.5  # tune in [0.1, 1.0] range

# ============================================================
# DEVICE SELECTION (CUDA if available)
# ============================================================

if torch.cuda.is_available():
    device = 0  # GPU index 0 for Ultralytics
    batch_size = 16
    print("🔥 GPU detected — using CUDA.")
else:
    device = "cpu"
    batch_size = 8
    print("🧠 Training on CPU — multi-core enabled.")

NUM_WORKERS = min(4, multiprocessing.cpu_count())
print(f"🧵 Using up to {NUM_WORKERS} CPU workers for dataloading.")

# ============================================================
# CUSTOM AUGMENTATIONS (Albumentations wrapper)
# ============================================================

gaussian_sigma = 1.0

# From your augment_echo_v3 stats (RGB)
MEAN = (0.46446225, 0.46446225, 0.46446225)
STD  = (0.15398581, 0.15398581, 0.15398581)

custom_augs = A.Compose([
    A.GaussianBlur(blur_limit=(3, 5), sigma_limit=gaussian_sigma, p=0.4),
    A.Normalize(mean=MEAN, std=STD),
])


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


def add_custom_transforms(trainer):
    """
    Callback: wraps YOLO's existing transforms with Albumentations on 'img'.
    """
    print("🔧 Injecting Albumentations wrapper around YOLO transforms...")
    try:
        ds = None
        if hasattr(trainer, "train_loader"):
            ds = getattr(trainer.train_loader, "dataset", None)
            if hasattr(ds, "dataset"):
                ds = ds.dataset  # unwrap inner dataset

        if ds is None:
            print("⚠ Could not locate underlying dataset to apply transforms.")
            return

        original_transforms = getattr(ds, "transforms", None)

        if original_transforms is None:
            print("⚠ Dataset has no 'transforms' attribute, skipping injection.")
            return

        def wrapped_transforms(sample):
            sample = original_transforms(sample)
            if isinstance(sample, dict) and "img" in sample:
                sample["img"] = apply_albumentations_to_img(sample["img"])
            return sample

        ds.transforms = wrapped_transforms
        print("✅ Albumentations wrapper successfully attached on top of YOLO transforms.")

    except Exception as e:
        print(f"⚠ Transform injection failed: {e}")


# ============================================================
# SOFT BOUNDARY SEGMENTATION LOSS
# ============================================================

class SoftBoundarySegmentationLoss(v8SegmentationLoss):
    """
    Custom segmentation loss for YOLOv8-seg:
      L_seg = L_BCE (Ultralytics style) + λ * L_boundary

    We override only `single_mask_loss`, so all detection-related parts
    (box, cls, dfl) stay exactly the same.
    """

    def __init__(self, model, lambda_boundary: float = 1.0):
        # ----- Patch model.args so v8SegmentationLoss doesn't explode -----
        args = getattr(model, "args", None)

        if isinstance(args, dict):
            args_ns = SimpleNamespace(**args)
        elif isinstance(args, SimpleNamespace):
            args_ns = args
        elif args is None:
            args_ns = SimpleNamespace()
        else:
            # some other type, just use as-is
            args_ns = args

        # Provide defaults expected by v8SegmentationLoss
        if not hasattr(args_ns, "overlap_mask"):
            args_ns.overlap_mask = False  # default used in Ultralytics for seg

        model.args = args_ns
        # ------------------------------------------------------------------

        super().__init__(model)
        self.lambda_boundary = lambda_boundary

        # Sobel kernels as plain tensors (not buffers, since this isn't nn.Module)
        self.sobel_x = torch.tensor(
            [[1, 0, -1],
             [2, 0, -2],
             [1, 0, -1]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

        self.sobel_y = torch.tensor(
            [[1, 2, 1],
             [0, 0, 0],
             [-1, -2, -1]],
            dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0)  # [1,1,3,3]

    def soft_boundary_term(self, pred_probs: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        pred_probs, gt_mask: [N, H, W] tensors in [0,1]
        Returns scalar boundary discrepancy.
        """
        if pred_probs.ndim == 2:
            pred_probs = pred_probs.unsqueeze(0)
        if gt_mask.ndim == 2:
            gt_mask = gt_mask.unsqueeze(0)

        # Make sure kernels are on the same device as the predictions
        sobel_x = self.sobel_x.to(pred_probs.device)
        sobel_y = self.sobel_y.to(pred_probs.device)

        # [N,1,H,W]
        pred = pred_probs.unsqueeze(1)
        gt = gt_mask.unsqueeze(1).float()

        pred_dx = F.conv2d(pred, sobel_x, padding=1)
        pred_dy = F.conv2d(pred, sobel_y, padding=1)
        gt_dx = F.conv2d(gt, sobel_x, padding=1)
        gt_dy = F.conv2d(gt, sobel_y, padding=1)

        pred_grad = torch.sqrt(pred_dx ** 2 + pred_dy ** 2 + 1e-6)
        gt_grad = torch.sqrt(gt_dx ** 2 + gt_dy ** 2 + 1e-6)

        # Mean absolute difference in gradient magnitude
        return torch.mean(torch.abs(pred_grad - gt_grad))

    @staticmethod
    def _bce_crop_area(pred_mask: torch.Tensor,
                       gt_mask: torch.Tensor,
                       xyxy: torch.Tensor,
                       area: torch.Tensor) -> torch.Tensor:
        """
        Re-implements Ultralytics' original BCE mask loss with cropping & area scaling.
        """
        loss_map = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        loss_cropped = crop_mask(loss_map, xyxy)              # [N, Hc, Wc]
        return (loss_cropped.mean(dim=(1, 2)) / area).sum()   # scalar

    def single_mask_loss(
        self,
        gt_mask: torch.Tensor,     # [N, H, W]
        pred: torch.Tensor,        # [N, 32] mask coeffs
        proto: torch.Tensor,       # [32, H, W]
        xyxy: torch.Tensor,        # [N, 4] normalized boxes
        area: torch.Tensor,        # [N]
    ) -> torch.Tensor:
        """
        Override original Ultralytics mask loss:
          1. Build logits mask via proto * coeffs
          2. Compute BCE-in-box as before
          3. Add soft boundary difference (prob gradients)
        """
        # 1) Proto-based mask reconstruction (logits)
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)  # [N, H, W]

        # 2) Standard BCE loss (inside bbox), same logic as Ultralytics
        bce_loss = self._bce_crop_area(pred_mask, gt_mask, xyxy, area)

        # 3) Soft boundary term — part of the gradient
        prob = torch.sigmoid(pred_mask)              # [N, H, W], 0–1
        prob_cropped = crop_mask(prob, xyxy)         # [N, Hc, Wc]
        gt_cropped = crop_mask(gt_mask, xyxy)        # [N, Hc, Wc]

        boundary_loss = self.soft_boundary_term(prob_cropped, gt_cropped)

        return bce_loss + self.lambda_boundary * boundary_loss


# ============================================================
# BUILD MODEL AND PATCH LOSS
# ============================================================

model = YOLO(MODEL)

# Replace the default segmentation loss with our custom version
model.model.loss = SoftBoundarySegmentationLoss(model.model, lambda_boundary=LAMBDA_BOUNDARY)

# Attach Albumentations wrapper
model.add_callback("on_train_start", add_custom_transforms)


# ============================================================
# TRAIN
# ============================================================

if __name__ == "__main__":
    print("\n🚀 Starting YOLO training with soft boundary segmentation loss...\n")

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,          # controls resize; 1024x1024 here
        device=device,           # 0 (GPU) or "cpu"
        workers=NUM_WORKERS,
        project=PROJECT,
        name=NAME,
        batch=batch_size,
        verbose=True,
        augment=True,            # YOLO's built-in aug, on top of your Albumentations
    )

    # Log training settings
    os.makedirs(f"{PROJECT}/{NAME}", exist_ok=True)
    with open(f"{PROJECT}/{NAME}/params.txt", "w") as f:
        f.write("YOLO Training Parameters (Soft Boundary Loss)\n")
        f.write("===========================================\n")
        f.write(f"Device: {device}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Image Size: {IMG_SIZE}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"CPU Workers: {NUM_WORKERS}\n")
        f.write("\n--- Albumentations ---\n")
        f.write(f"GaussianBlur: σ={gaussian_sigma}, prob=0.4, kernel=3–5\n")
        f.write(f"Brightness normalization: mean={MEAN}, std={STD}\n")
        f.write("\n--- Segmentation Loss ---\n")
        f.write("L_seg = BCE_in_box + lambda_boundary * SoftBoundaryLoss\n")
        f.write(f"lambda_boundary = {LAMBDA_BOUNDARY}\n")

    print("\n🎉 Training complete!")
    print(f"📁 Output saved to: {PROJECT}/{NAME}")
    print("ℹ Test-set evaluation (e.g. boundary metrics) should be done in a separate script.")
                                                                     
