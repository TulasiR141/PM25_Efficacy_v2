"""
consolidate_yolo_spheroid_only.py

Consolidates YOLO segmentation datasets:
 - Echo.v1i.yolov8  (3 classes)
 - EchoSpheroids.v1i.yolov8 (2 classes)

Keeps only the "spheroid" (or "Spheroid") class based on data.yaml mapping.
All other annotations are dropped.

Also visualizes 10 random side-by-side comparisons (original vs consolidated)
in Matplotlib figures that wait for you to close before continuing.
Includes automatic cleanup of old consolidated folder and sanity check on counts.
"""

import os
import glob
import shutil
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===========================================================
# CONFIG
# ===========================================================
BASE_DIR = "pictures"
OUTPUT_DIR = os.path.join(BASE_DIR, "consolidated_yolo", "test")

# 🧹 Clean previous consolidation to avoid duplicates
if os.path.exists(OUTPUT_DIR):
    print(f"🧹 Clearing old consolidated directory: {OUTPUT_DIR}")
    shutil.rmtree(OUTPUT_DIR)

os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "labels"), exist_ok=True)

DATASETS = [
    os.path.join(BASE_DIR, "Echo.v1i.yolov8"),
    os.path.join(BASE_DIR, "EchoSpheroids.v1i.yolov8")
]

# Which classes to keep per dataset (class names as defined in data.yaml)
KEEP_CLASS_NAMES = {
    "Echo.v1i.yolov8": ["spheroid"],
    "EchoSpheroids.v1i.yolov8": ["Spheroid"]
}


# ===========================================================
# HELPER FUNCTIONS
# ===========================================================
def parse_data_yaml(yaml_path):
    """Reads data.yaml and returns {class_index: class_name} mapping."""
    mapping = {}
    with open(yaml_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        if "names:" in line:
            names_str = line.split("names:")[1].strip()
            if names_str.startswith("["):
                names = [n.strip().strip("'\"") for n in names_str.strip("[]").split(",")]
                for i, n in enumerate(names):
                    mapping[i] = n
                return mapping
    # fallback for multi-line names
    names = []
    for line in lines:
        line = line.strip()
        if line.startswith("- "):
            names.append(line[2:].strip().strip("'\""))
    for i, n in enumerate(names):
        mapping[i] = n
    return mapping


def consolidate_dataset(dataset_dir):
    """Copies all train/val/test images & keeps only chosen spheroid polygons."""
    dataset_name = os.path.basename(dataset_dir)
    print(f"\n🧩 Consolidating {dataset_name} ...")

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        print(f"⚠️ Missing data.yaml in {dataset_dir}, skipping.")
        return 0

    idx_to_name = parse_data_yaml(yaml_path)
    keep_classes = KEEP_CLASS_NAMES[dataset_name]

    # gather all image/label pairs
    pairs = []
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        lbl_dir = os.path.join(dataset_dir, split, "labels")
        if not os.path.exists(img_dir):
            continue
        for img_path in glob.glob(os.path.join(img_dir, "*")):
            base = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(lbl_dir, base + ".txt")
            if os.path.exists(lbl_path):
                pairs.append((img_path, lbl_path))

    kept, dropped = 0, 0
    copied_images = 0
    for img_path, lbl_path in tqdm(pairs, desc=f"{dataset_name}"):
        new_lbl_lines = []
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    class_idx = int(float(parts[0]))
                except ValueError:
                    continue
                class_name = idx_to_name.get(class_idx, None)
                if class_name in keep_classes:
                    new_lbl_lines.append(" ".join(["1"] + parts[1:]))  # new class 1 = spheroid
                    kept += 1
                else:
                    dropped += 1

        if not new_lbl_lines:
            continue  # skip images that have no spheroid

        new_img_path = os.path.join(OUTPUT_DIR, "images", os.path.basename(img_path))
        new_lbl_path = os.path.join(OUTPUT_DIR, "labels", os.path.basename(lbl_path))
        shutil.copy(img_path, new_img_path)
        with open(new_lbl_path, "w") as f:
            f.write("\n".join(new_lbl_lines))
        copied_images += 1

    print(f"✅ Kept {kept} spheroid polygons, dropped {dropped} others.")
    print(f"📸 Copied {copied_images} images from {dataset_name}.")
    return copied_images


def draw_yolo_polygons(img, lbl_path, color):
    """Draw polygons from YOLO segmentation txt file onto an image."""
    h, w = img.shape[:2]
    overlay = img.copy()
    if not os.path.exists(lbl_path):
        return overlay
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            coords = list(map(float, parts[1:]))
            if len(coords) % 2 != 0:
                coords = coords[:-1]
            pts = np.array(
                [[int(x * w), int(y * h)] for x, y in zip(coords[0::2], coords[1::2])],
                np.int32
            )
            if len(pts) > 2:
                cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
    return overlay


def visualize_spot_checks(original_dirs, consolidated_dir, n=10):
    """Show side-by-side comparisons of original vs consolidated masks (Matplotlib)."""
    print("\n👁 Performing spot-check visualizations (close each window to continue)...")

    imgs = glob.glob(os.path.join(consolidated_dir, "images", "*"))
    sample_imgs = random.sample(imgs, min(n, len(imgs)))

    for img_path in sample_imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        new_lbl = os.path.join(consolidated_dir, "labels", base + ".txt")

        orig_img, orig_lbl = None, None
        for d in original_dirs:
            for split in ["train", "valid", "test"]:
                img_try = os.path.join(d, split, "images", base + ".jpg")
                lbl_try = os.path.join(d, split, "labels", base + ".txt")
                if os.path.exists(img_try) and os.path.exists(lbl_try):
                    orig_img, orig_lbl = img_try, lbl_try
                    break
            if orig_img:
                break

        if not orig_img:
            continue

        img_orig = cv2.cvtColor(cv2.imread(orig_img), cv2.COLOR_BGR2RGB)
        img_cons = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        drawn_orig = draw_yolo_polygons(img_orig, orig_lbl, (0, 255, 0))
        drawn_cons = draw_yolo_polygons(img_cons, new_lbl, (255, 0, 0))

        # Plot side-by-side with manageable size
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(drawn_orig)
        axes[0].set_title("Original (All Classes)")
        axes[0].axis("off")
        axes[1].imshow(drawn_cons)
        axes[1].set_title("Consolidated (Spheroid Only)")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()  # waits for you to close before showing next


# ===========================================================
# RUN
# ===========================================================
total_original = 0
total_consolidated = 0

for dataset_dir in DATASETS:
    # count total original images first
    img_count = 0
    for split in ["train", "valid", "test"]:
        img_dir = os.path.join(dataset_dir, split, "images")
        if os.path.exists(img_dir):
            img_count += len(glob.glob(os.path.join(img_dir, "*.jpg")))
            img_count += len(glob.glob(os.path.join(img_dir, "*.png")))
    total_original += img_count

    total_consolidated += consolidate_dataset(dataset_dir)

visualize_spot_checks(DATASETS, os.path.join(BASE_DIR, "consolidated_yolo", "test"), n=10)

print("\n📊 Summary:")
print(f" - Total original images: {total_original}")
print(f" - Total consolidated images: {total_consolidated}")

if total_consolidated > total_original:
    print("⚠️ Warning: Consolidated dataset has MORE images than original — check duplication!")
else:
    print("✅ Consolidation image count is within expected range.")

print("\n🎉 Consolidation complete! Check 'pictures/consolidated_yolo/test'.")
