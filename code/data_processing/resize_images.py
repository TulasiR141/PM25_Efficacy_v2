import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
SRC_DIR = "pictures/consolidated_yolo/test"
DEST_DIR = "pictures/resized_1024"
TARGET_SIZE = 1024
N_VISUALIZE = 10

# Create output folders
os.makedirs(os.path.join(DEST_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(DEST_DIR, "labels"), exist_ok=True)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_polygons(label_path, w, h):
    """Load YOLO-style polygon labels and scale to pixel coordinates."""
    polygons = []
    if not os.path.exists(label_path):
        return polygons

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) <= 3:
                continue
            cls = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            pts = np.array([(coords[i]*w, coords[i+1]*h) for i in range(0, len(coords), 2)], np.float32)
            polygons.append((cls, pts))
    return polygons


def resize_with_padding(img, target_size=1024):
    """Resize preserving aspect ratio and add padding to reach target_size."""
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w, pad_h = target_size - new_w, target_size - new_h
    pad_left, pad_top = pad_w // 2, pad_h // 2

    img_padded = cv2.copyMakeBorder(
        resized, pad_top, target_size - new_h - pad_top,
        pad_left, target_size - new_w - pad_left,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return img_padded, scale, pad_left, pad_top


def adjust_polygons(polygons, scale, pad_left, pad_top, target_size):
    """Apply resize and padding transformations to polygons, return normalized coordinates."""
    new_polys = []
    for cls, poly in polygons:
        scaled = poly * scale
        scaled[:, 0] += pad_left
        scaled[:, 1] += pad_top

        # Normalize again (YOLO format)
        norm = scaled.copy()
        norm[:, 0] /= target_size
        norm[:, 1] /= target_size
        new_polys.append((cls, norm))
    return new_polys


def save_yolo_label(label_path, polygons):
    """Save polygons back to YOLO segmentation format."""
    with open(label_path, "w") as f:
        for cls, pts in polygons:
            coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in pts])
            f.write(f"{cls} {coords}\n")


def visualize_before_after(orig_img, orig_polys, resized_img, resized_polys, target_size=1024):
    """Show side-by-side comparison of polygon alignment."""
    orig_vis = orig_img.copy()
    for _, poly in orig_polys:
        cv2.polylines(orig_vis, [poly.astype(np.int32)], True, (0, 255, 0), 2)

    resized_vis = resized_img.copy()
    for _, poly in resized_polys:
        scaled = (poly * target_size).astype(np.int32)
        cv2.polylines(resized_vis, [scaled], True, (0, 255, 0), 2)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(orig_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Original ({orig_img.shape[1]}×{orig_img.shape[0]})")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(resized_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Resized ({target_size}×{target_size})")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ============================================================
# MAIN PROCESSING LOOP
# ============================================================
img_dir = os.path.join(SRC_DIR, "images")
label_dir = os.path.join(SRC_DIR, "labels")

image_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
print(f"📸 Found {len(image_files)} images to process...")

# Randomly select samples for visualization
vis_samples = random.sample(image_files, min(N_VISUALIZE, len(image_files)))

for img_name in tqdm(image_files, desc="Resizing images"):
    img_path = os.path.join(img_dir, img_name)
    label_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + ".txt")

    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]

    # Load polygons
    polygons = load_polygons(label_path, w, h)

    # Resize + pad
    img_resized, scale, pad_left, pad_top = resize_with_padding(img, TARGET_SIZE)
    polygons_resized = adjust_polygons(polygons, scale, pad_left, pad_top, TARGET_SIZE)

    # Save results
    dest_img_path = os.path.join(DEST_DIR, "images", img_name)
    dest_label_path = os.path.join(DEST_DIR, "labels", os.path.splitext(img_name)[0] + ".txt")

    cv2.imwrite(dest_img_path, img_resized)
    save_yolo_label(dest_label_path, polygons_resized)

    # Visual check for selected samples
    if img_name in vis_samples:
        visualize_before_after(img, polygons, img_resized, polygons_resized, TARGET_SIZE)

print(f"\n✅ Done! Resized dataset saved to: {DEST_DIR}")
