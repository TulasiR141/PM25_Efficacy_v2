import os
import cv2
import numpy as np
from glob import glob
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ============================================================
# CONFIGURATION
# ============================================================
IMG_SIZE = 512

# Path to your trained YOLO segmentation model
# 👉 Set this to the best.pt from your training run
MODEL_WEIGHTS = "yolo_training_output/cpu_run_final_tensordeepfix24/weights/best.pt"

# YOLO-style data root (read-only)
DATA_ROOT = "data/yolo_splits"
SPLIT = "test"  # use "val" or "train" if you want

IMAGES_DIR = os.path.join(DATA_ROOT, SPLIT, "images")
LABELS_DIR = os.path.join(DATA_ROOT, SPLIT, "labels")

# Output directories (DO NOT TOUCH yolo_splits)
CONVERTED_MASK_DIR = os.path.join("data", "converted_masks", SPLIT)
EVAL_FIG_DIR = os.path.join("data", "boundary_eval_results", SPLIT)

os.makedirs(CONVERTED_MASK_DIR, exist_ok=True)
os.makedirs(EVAL_FIG_DIR, exist_ok=True)

# ============================================================
# DEVICE
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# ============================================================
# YOLO SEG LABEL (.txt, normalized) → BINARY MASK
# ============================================================
def yolo_seg_txt_to_mask(label_path, img_size):
    """
    Converts YOLO segmentation label to a binary mask of size (img_size, img_size).
    Assumes normalized polygon coords: class x1 y1 x2 y2 ...
    """
    mask = np.zeros((img_size, img_size), dtype=np.uint8)

    if not os.path.exists(label_path):
        return mask  # no objects

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        # class_id = int(parts[0])  # not needed, single class
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue

        xs = np.array(coords[0::2])  # normalized x
        ys = np.array(coords[1::2])  # normalized y

        px = (xs * img_size).astype(np.int32)
        py = (ys * img_size).astype(np.int32)

        poly = np.stack([px, py], axis=1)
        cv2.fillPoly(mask, [poly], 1)

    return mask

# ============================================================
# BOUNDARY METRIC FUNCTIONS
# ============================================================
def soft_boundary_loss(pred_mask_t, gt_mask_t):
    """
    pred_mask_t, gt_mask_t: torch tensors [H, W] with values in {0,1} or [0,1].
    Returns scalar (float), lower is better.
    """
    if pred_mask_t.dim() == 3:
        pred_mask_t = pred_mask_t[0]
    if gt_mask_t.dim() == 3:
        gt_mask_t = gt_mask_t[0]

    pred_mask_t = pred_mask_t.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    gt_mask_t = gt_mask_t.unsqueeze(0).unsqueeze(0)

    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred_mask_t.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=pred_mask_t.device).unsqueeze(0).unsqueeze(0)

    pred_dx = F.conv2d(pred_mask_t, sobel_x, padding=1)
    pred_dy = F.conv2d(pred_mask_t, sobel_y, padding=1)
    gt_dx = F.conv2d(gt_mask_t, sobel_x, padding=1)
    gt_dy = F.conv2d(gt_mask_t, sobel_y, padding=1)

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
        return 1.0  # both empty → perfect
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

    def directed_hd(a, b):
        dists = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=-1))
        return dists.min(axis=1).max()

    hd_pg = directed_hd(pts_p, pts_g)
    hd_gp = directed_hd(pts_g, pts_p)

    return float(max(hd_pg, hd_gp))

# ============================================================
# MAIN EVAL PIPELINE
# ============================================================
def main():
    # Load model
    print(f"Loading model from: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS).to(device)

    image_paths = sorted(glob(os.path.join(IMAGES_DIR, "*")))
    if not image_paths:
        print(f"No images found in {IMAGES_DIR}")
        return

    all_sbl = []
    all_biou = []
    all_hd = []

    print(f"\nEvaluating on {len(image_paths)} images from split: {SPLIT}\n")

    for img_path in image_paths:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        # --- 1) Build GT mask from YOLO seg labels ---
        gt_mask = yolo_seg_txt_to_mask(label_path, IMG_SIZE)  # [H,W], 0/1

        # Save GT mask (for inspection)
        gt_mask_out_path = os.path.join(CONVERTED_MASK_DIR, stem + ".png")
        cv2.imwrite(gt_mask_out_path, (gt_mask * 255).astype(np.uint8))

        # --- 2) Run YOLO prediction ---
        results = model.predict(img_path, imgsz=IMG_SIZE, device=device, verbose=False)
        r = results[0]

        if r.masks is None:
            print(f"⚠️ No masks predicted for {stem}, skipping metrics & fig.")
            continue

        pm = r.masks.data  # [N, H, W] torch tensor
        if pm.ndim != 3:
            print(f"⚠️ Unexpected mask shape for {stem}: {pm.shape}, skipping.")
            continue

        # Semantic union of all instance masks
        pm = pm.float()
        pm_semantic = (pm.sigmoid().max(dim=0)[0] > 0.5).float()  # [H, W]

        # Resize prediction to IMG_SIZE if needed
        pred_np = pm_semantic.detach().cpu().numpy().astype(np.float32)
        if pred_np.shape != (IMG_SIZE, IMG_SIZE):
            pred_np = cv2.resize(pred_np, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        # Ensure GT is also IMG_SIZE × IMG_SIZE (it already is)
        gt_np = gt_mask.astype(np.float32)

        # --- 3) Compute metrics ---
        gt_t = torch.from_numpy(gt_np).to(pm_semantic.device)
        pred_t = torch.from_numpy(pred_np).to(pm_semantic.device)

        sbl = soft_boundary_loss(pred_t, gt_t)
        biou = boundary_iou(pred_np, gt_np)
        hd = hausdorff_like(pred_np, gt_np)

        all_sbl.append(sbl)
        all_biou.append(biou)
        all_hd.append(hd)

        print(f"{stem}: SoftBoundary={sbl:.6f}, BIoU={biou:.4f}, HD≈{hd:.2f}px")

        # --- 4) Plot & save comparison figure ---
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(gt_np, cmap="gray")
        axes[0].set_title("GT Mask")
        axes[0].axis("off")

        axes[1].imshow(pred_np, cmap="gray")
        axes[1].set_title("Predicted Mask")
        axes[1].axis("off")

        fig.suptitle(
            f"{stem}\nSoftBoundary={sbl:.6f} | BIoU={biou:.4f} | HD≈{hd:.2f}px",
            fontsize=10
        )

        fig.tight_layout(rect=[0, 0, 1, 0.88])
        fig_out_path = os.path.join(EVAL_FIG_DIR, stem + "_eval.png")
        plt.savefig(fig_out_path, dpi=150)
        plt.close(fig)

    # --- 5) Print overall summary ---
    if all_sbl:
        mean_sbl = float(np.mean(all_sbl))
        mean_biou = float(np.mean(all_biou))
        mean_hd = float(np.mean(all_hd))

        print("\n================ Boundary Evaluation Summary ================")
        print(f"Split: {SPLIT}")
        print(f"  ▸ Mean Soft Boundary Loss  (lower better): {mean_sbl:.6f}")
        print(f"  ▸ Mean Boundary IoU        (higher better): {mean_biou:.4f}")
        print(f"  ▸ Mean Hausdorff-like Dist (lower better, px): {mean_hd:.2f}")
        print("=============================================================\n")
    else:
        print("No valid samples evaluated (no predictions or masks).")


if __name__ == "__main__":
    main()



