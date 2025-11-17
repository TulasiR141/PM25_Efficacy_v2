import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================
# INPUT DIRECTORIES
# ============================================================
IMG_DIR = Path("pictures/augmented_final/images")
LBL_DIR = Path("pictures/augmented_final/labels")

# ============================================================
# OUTPUT DIRECTORIES (UNet format)
# ============================================================
OUT_IMG_DIR = Path("training_validation/images")
OUT_MSK_DIR = Path("training_validation/labels")
VIS_DIR = Path("comparison_output_unet_masks")

OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUT_MSK_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD YOLO POLYGONS
# ============================================================
def load_polygons(label_path, w, h):
    polys = []
    if not label_path.exists():
        return polys

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 3:
                continue

            coords = list(map(float, parts[1:]))

            pts = np.array([
                (coords[i] * w, coords[i + 1] * h)
                for i in range(0, len(coords), 2)
            ], dtype=np.int32)

            polys.append(pts)

    return polys


# ============================================================
# DRAW POLYGONS FOR VISUALIZATION
# ============================================================
def draw_polygons(img, polys, color=(0, 255, 0)):
    out = img.copy()
    for poly in polys:
        cv2.polylines(out, [poly], True, color, 2)
    return out


# ============================================================
# MAIN
# ============================================================
img_files = sorted([f for f in IMG_DIR.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]])

print(f"🧩 Found {len(img_files)} images. Converting to UNet format...\n")

vis_count = 0  # limit to 10 visualizations

for img_path in tqdm(img_files):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]

    label_path = LBL_DIR / (img_path.stem + ".txt")
    polygons = load_polygons(label_path, w, h)

    # ------------------------------
    # Create a binary mask (uint8)
    # ------------------------------
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly], color=1)

    # ------------------------------
    # Save UNet format PNGs
    # ------------------------------
    out_img_path = OUT_IMG_DIR / (img_path.stem + ".png")
    out_msk_path = OUT_MSK_DIR / (img_path.stem + ".png")

    cv2.imwrite(str(out_img_path), img)
    cv2.imwrite(str(out_msk_path), mask)

    # ------------------------------
    # Generate 10 comparison visualizations
    # ------------------------------
    if vis_count < 10:
        # Left visualization: YOLO boundary
        left = draw_polygons(img, polygons, color=(0, 255, 0))
        left = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)

        # Right visualization: mask overlay (binary, with red highlight)
        overlay = img.copy()

        red = np.zeros_like(img, dtype=np.uint8)
        red[:, :, 2] = 255  # pure red

        alpha = 0.5
        overlay_mask = (overlay * (1 - alpha) + red * alpha).astype(np.uint8)

        overlay[mask == 1] = overlay_mask[mask == 1]

        right = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        # Save figure
        fig = plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.title("YOLO Boundary")
        plt.imshow(left)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.title("UNet Mask Overlay")
        plt.imshow(right)
        plt.axis("off")

        fig.savefig(VIS_DIR / f"{img_path.stem}_comparison.png", dpi=140)
        plt.close()

        vis_count += 1


print("\n✅ Conversion complete!")
print(f"📁 Images saved to: {OUT_IMG_DIR}")
print(f"📁 Masks saved to:  {OUT_MSK_DIR}")
print(f"📁 10 verification figures saved to: {VIS_DIR}")
