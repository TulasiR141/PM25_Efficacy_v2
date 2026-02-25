"""
analyze_preprocessing_relevance_consolidated.py

Scientific preprocessing relevance analysis for the consolidated YOLO dataset.

Analyzes:
  - Resize necessity (aspect ratio std)
  - Normalization (dataset mean/std vs ImageNet)
  - Gaussian blur augmentation relevance
  - Random affine transform justification (translation, scaling)
  - Label integrity and class distribution

Outputs JSON report and visualizations in analysis_output/.
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
from collections import Counter

# ===========================================================
# CONFIGURATION
# ===========================================================

DATASET_DIR = "pictures/consolidated_yolo/test"  # consolidated dataset
VALID_EXTS = [".jpg", ".png", ".jpeg"]

os.makedirs("analysis_output", exist_ok=True)

# ===========================================================
# HELPER FUNCTIONS
# ===========================================================

def get_image_label_pairs(dataset_dir):
    """Return all image-label pairs under a YOLO directory (images/, labels/)."""
    image_dir = os.path.join(dataset_dir, "images")
    label_dir = os.path.join(dataset_dir, "labels")

    images = []
    for ext in VALID_EXTS:
        images.extend(glob.glob(os.path.join(image_dir, f"*{ext}")))

    pairs = []
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, base + ".txt")
        if os.path.exists(label_path):
            pairs.append((img_path, label_path))
    return pairs


def laplacian_var(img):
    """Compute variance of Laplacian (focus measure)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# ===========================================================
# MAIN ANALYSIS FUNCTION
# ===========================================================

def analyze_consolidated_dataset(dataset_dir):
    print(f"\n📊 Analyzing consolidated dataset: {dataset_dir}")
    pairs = get_image_label_pairs(dataset_dir)

    if not pairs:
        print("⚠️ No image-label pairs found, check paths.")
        return None

    # -------------------------------------------------------
    # 1. Aspect Ratio Variability
    # -------------------------------------------------------
    print("\n📏 Checking aspect ratio variability...")
    aspect_ratios = []
    for img_path, _ in tqdm(pairs, desc="Aspect Ratios", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        aspect_ratios.append(w / h)

    aspect_std = np.std(aspect_ratios) if aspect_ratios else 0
    resize_strategy = "pad" if aspect_std > 0.2 else "resize"
    print(f"Aspect ratio std: {aspect_std:.3f} → {resize_strategy.upper()} recommended.")

    # -------------------------------------------------------
    # 2. Normalization (ImageNet vs Dataset)
    # -------------------------------------------------------
    print("\n🎨 Computing color statistics for normalization check...")
    means, stds = [], []
    for img_path, _ in tqdm(pairs, desc="Color Stats", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = img / 255.0
        means.append(np.mean(img, axis=(0, 1)))
        stds.append(np.std(img, axis=(0, 1)))

    mean_rgb = np.mean(means, axis=0)
    std_rgb = np.mean(stds, axis=0)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])

    use_imagenet_norm = (
        np.all(np.abs(mean_rgb - imagenet_mean) < 0.05)
        and np.all(np.abs(std_rgb - imagenet_std) < 0.03)
    )

    print(f"Dataset mean: {mean_rgb}")
    print(f"Dataset std:  {std_rgb}")
    if use_imagenet_norm:
        print("✅ Dataset similar to ImageNet — ImageNet normalization appropriate.")
    else:
        print("⚠️ Dataset differs — compute dataset-specific normalization.")

    # -------------------------------------------------------
    # 3. Gaussian Blur Augmentation
    # -------------------------------------------------------
    print("\n🔍 Evaluating blur variability (Laplacian variance)...")
    blur_values = []
    for img_path, _ in tqdm(pairs, desc="Blur Var", leave=False):
        img = cv2.imread(img_path)
        if img is None:
            continue
        blur_values.append(laplacian_var(img))

    blur_mean, blur_std = np.mean(blur_values), np.std(blur_values)
    blur_augmentation = blur_std > 100
    print(f"Laplacian variance mean ± std: {blur_mean:.2f} ± {blur_std:.2f}")
    print("✅ Blur augmentation recommended.\n" if blur_augmentation else "⚠️ Blur augmentation unnecessary.\n")

    # -------------------------------------------------------
    # 4. Affine Transform (Translation & Scaling)
    # -------------------------------------------------------
    print("📐 Analyzing positional and scale variability...")

    centroids, areas = [], []
    for _, label_path in tqdm(pairs, desc="Affine Analysis", leave=False):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    continue
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                if len(coords) < 4:
                    continue

                xs = np.array(coords[0::2])
                ys = np.array(coords[1::2])

                x_min, x_max = xs.min(), xs.max()
                y_min, y_max = ys.min(), ys.max()
                x_c = (x_min + x_max) / 2
                y_c = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                area = width * height

                centroids.append((x_c, y_c))
                areas.append(area)

    if centroids:
        centroids = np.array(centroids)
        areas = np.array(areas)
        centroid_std = np.std(centroids, axis=0)
        area_std = np.std(areas)
    else:
        centroid_std = [0, 0]
        area_std = 0

    translation_needed = centroid_std[0] > 0.1 or centroid_std[1] > 0.1
    scaling_needed = area_std > 0.05
    print(f"Centroid std (x,y): {centroid_std}")
    print(f"Area std: {area_std:.3f}")
    if translation_needed:
        print("✅ Translation augmentation justified (≤25%).")
    else:
        print("⚠️ Objects centered — translation not critical.")
    if scaling_needed:
        print("✅ Scaling augmentation justified (75–125%).")
    else:
        print("⚠️ Object size consistent — scaling not critical.")
    print("Rotation/shear depend on camera angle variation.\n")

    # -------------------------------------------------------
    # 5. Label Integrity
    # -------------------------------------------------------
    print("🧾 Checking label integrity...")
    class_counts = Counter()
    for _, label_path in pairs:
        with open(label_path, "r") as f:
            for line in f:
                if line.strip():
                    cls = line.strip().split()[0]
                    class_counts[cls] += 1
    print(f"Class distribution: {dict(class_counts)}")

    # -------------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------------
    rec = {
        "dataset": os.path.basename(dataset_dir),
        "resize_strategy": resize_strategy,
        "use_imagenet_norm": bool(use_imagenet_norm),
        "blur_augmentation": bool(blur_augmentation),
        "translation_std": list(map(float, centroid_std)),
        "scaling_std": float(area_std),
        "aspect_ratio_std": float(aspect_std),
        "blur_var_mean": float(blur_mean),
        "blur_var_std": float(blur_std),
        "class_distribution": dict(class_counts)
    }

    fname = f"analysis_output/consolidated_recommendations.json"
    with open(fname, "w") as f:
        json.dump(rec, f, indent=2)
    print(f"✅ Saved recommendations to {fname}\n")

    # Visualizations
    if aspect_ratios:
        plt.figure(figsize=(8, 4))
        plt.hist(aspect_ratios, bins=20, color="skyblue", edgecolor="black")
        plt.title("Aspect Ratios - Consolidated Dataset")
        plt.xlabel("Width / Height")
        plt.ylabel("Count")
        plt.savefig("analysis_output/consolidated_aspect_ratios.png")

    if blur_values:
        plt.figure(figsize=(8, 4))
        plt.hist(blur_values, bins=30, color="salmon", edgecolor="black")
        plt.title("Blur Variance - Consolidated Dataset")
        plt.xlabel("Laplacian Variance")
        plt.ylabel("Count")
        plt.savefig("analysis_output/consolidated_blur_variation.png")

    return rec


# ===========================================================
# RUN ANALYSIS
# ===========================================================
results = analyze_consolidated_dataset(DATASET_DIR)

print("\n📋 Consolidated Dataset Summary")
print("=" * 80)
for k, v in results.items():
    print(f"{k}: {v}")
print("=" * 80)
print("🎉 Analysis complete. Results in analysis_output/")
