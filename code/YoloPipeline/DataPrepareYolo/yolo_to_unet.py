import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# ============================================================
# PATH CONFIG
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_BASE = PROJECT_ROOT / "data_new" / "yolo_splits_v1"
UNET_BASE = PROJECT_ROOT / "data_new" / "unet_splits_v1"

# ✅ process all three splits
SPLITS = ["train", "val", "test"]

# Global spotcheck directory (root-level)
GLOBAL_VIS_DIR = PROJECT_ROOT / "unet_mask_spotcheck"
GLOBAL_VIS_DIR.mkdir(parents=True, exist_ok=True)

# How many total comparisons to save across all splits
MAX_GLOBAL_VIS = 20


# ============================================================
# HELPERS
# ============================================================
def load_polygons_yolo_seg(label_path: Path, w: int, h: int):
    """
    Load polygons from a YOLOv8 segmentation label file.

    Line format:
      cls xc yc w h x1 y1 x2 y2 ...

    All coords are normalized [0,1]. We convert to pixel coords.
    Returns: list of polygons as np.ndarray [N,2] (int32).
    """
    polys = []
    if not label_path.exists():
        return polys

    text = label_path.read_text().strip()
    if not text:
        return polys

    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) <= 5:
            # Need at least one vertex pair
            continue

        coord_vals = list(map(float, parts[5:]))

        # Ensure even number of coords
        if len(coord_vals) % 2 != 0:
            coord_vals = coord_vals[:-1]

        pts = []
        for i in range(0, len(coord_vals), 2):
            x_norm = coord_vals[i]
            y_norm = coord_vals[i + 1]
            x_pix = int(round(x_norm * w))
            y_pix = int(round(y_norm * h))
            pts.append([x_pix, y_pix])

        if len(pts) >= 3:
            polys.append(np.array(pts, dtype=np.int32))

    return polys


def draw_polygons(img_bgr, polys, color=(0, 255, 0)):
    out = img_bgr.copy()
    for poly in polys:
        cv2.polylines(out, [poly], True, color, 2)
    return out


# ============================================================
# MAIN CONVERSION PER SPLIT
# ============================================================
def convert_split(split: str, global_vis_state: dict):
    """
    Convert one split (train/val/test) and update global_vis_state["count"].
    """
    print(f"\n===== Processing split: {split} =====")

    img_dir = YOLO_BASE / split / "images"
    lbl_dir = YOLO_BASE / split / "labels"

    if not img_dir.is_dir():
        print(f"⚠ Skipping {split}: image dir not found: {img_dir}")
        return
    if not lbl_dir.is_dir():
        print(f"⚠ Skipping {split}: label dir not found: {lbl_dir}")
        return

    out_img_dir = UNET_BASE / split / "images"
    out_msk_dir = UNET_BASE / split / "masks"

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_msk_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted(
        [p for p in img_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    print(f"🧩 Found {len(img_files)} images in {img_dir}")

    for img_path in tqdm(img_files, desc=f"{split} images"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"⚠ Could not read image: {img_path}")
            continue

        h, w = img_bgr.shape[:2]

        label_path = lbl_dir / f"{img_path.stem}.txt"
        polys = load_polygons_yolo_seg(label_path, w, h)

        # --------- build binary mask (0 background, 1 spheroid) ----------
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polys:
            cv2.fillPoly(mask, [poly], color=1)

        # --------- save UNet-style PNGs ----------
        # ✅ keep filename stem exactly the same, just change extension to .png
        out_img_path = out_img_dir / f"{img_path.stem}.png"
        out_msk_path = out_msk_dir / f"{img_path.stem}.png"

        cv2.imwrite(str(out_img_path), img_bgr)
        cv2.imwrite(str(out_msk_path), mask)

        # --------- global spot-checks (YOLO vs UNet mask) ----------
        if global_vis_state["count"] < MAX_GLOBAL_VIS:
            # Left: YOLO polygon overlay
            left = draw_polygons(img_bgr, polys, color=(0, 255, 0))
            left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

            # Right: mask overlay (red where mask==1)
            overlay = img_bgr.copy()
            red = np.zeros_like(img_bgr, dtype=np.uint8)
            red[:, :, 2] = 255  # red channel

            alpha = 0.5
            blended = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)
            overlay[mask == 1] = blended[mask == 1]
            right = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            fig = plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.title(f"{split} – YOLO Polygon")
            plt.imshow(left)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("UNet Binary Mask Overlay")
            plt.imshow(right)
            plt.axis("off")

            vis_path = GLOBAL_VIS_DIR / f"{split}_{img_path.stem}_comparison.png"
            fig.savefig(vis_path, dpi=140)
            plt.close(fig)

            global_vis_state["count"] += 1

    print(f"✅ Split '{split}' done.")
    print(f"   Images → {out_img_dir}")
    print(f"   Masks  → {out_msk_dir}")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print("Converting YOLO segmentation labels to UNet binary masks...\n")

    global_vis_state = {"count": 0}

    # ✅ Always run through train, val, test
    for split in SPLITS:
        convert_split(split, global_vis_state)

    print("\n🎉 Conversion complete!")
    print(f"UNet-style data is under: {UNET_BASE}")
    print(f"Spot-check figures (up to {MAX_GLOBAL_VIS}) in: {GLOBAL_VIS_DIR}")
                           
