import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon
from sklearn.cluster import KMeans

# ============================================================
# CONFIGURATION
# ============================================================
IMG_DIR = Path("pictures/augmented_final/images")
LBL_DIR = Path("pictures/augmented_final/labels")

OUT_BASE = Path("yolo_splits")
SPLITS = ["train", "val", "test"]

# Desired split ratios
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

# Create directory structure
for split in SPLITS:
    (OUT_BASE / split / "images").mkdir(parents=True, exist_ok=True)
    (OUT_BASE / split / "labels").mkdir(parents=True, exist_ok=True)

print("\n==============================================")
print(" YOLO DATASET SPLITTING (Balanced + Safe)")
print("==============================================\n")


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
            if len(parts) < 4:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array([
                (coords[i] * w, coords[i+1] * h)
                for i in range(0, len(coords), 2)
            ], np.float32)
            polys.append(pts)
    return polys


# ============================================================
# COMPUTE CHARACTERISTICS
# ============================================================
def compute_characteristics(img, polygons):
    # Mean brightness
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = gray.mean()

    # Polygon-based geometry
    if len(polygons) > 0:
        poly = Polygon(polygons[0])
        size = poly.area
        circularity = 4 * np.pi * poly.area / (poly.length**2 + 1e-6)
    else:
        size = 0
        circularity = 0

    return brightness, size, circularity


# ============================================================
# GROUP AUGMENTATIONS TOGETHER (no leakage!)
# ============================================================
print("🔎 Grouping images by base name to prevent data leakage.")
print("   ➤ Ensures augmented variants NEVER go to different splits.\n")

groups = {}  # base → [augmented files]

for img_file in IMG_DIR.iterdir():
    if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    stem = img_file.stem

    # Remove augmentation suffix (e.g., _rot90, _vflip)
    base = stem.split("_rot")[0].split("_hflip")[0].split("_vflip")[0]

    groups.setdefault(base, []).append(img_file.name)

print(f"📦 Found {len(groups)} unique original image groups.\n")

# ============================================================
# COMPUTE FEATURES FOR STRATIFICATION
# ============================================================
print("📊 Computing brightness, size, and circularity for each base...")
print("   ➤ Ensures balanced distributions across train/val/test.\n")

features = []
base_names = []

for base, file_list in tqdm(groups.items()):
    fname = file_list[0]
    img_path = IMG_DIR / fname
    lbl_path = LBL_DIR / (Path(fname).stem + ".txt")

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    h, w = img.shape[:2]
    polygons = load_polygons(lbl_path, w, h)

    feat = compute_characteristics(img, polygons)
    features.append(feat)
    base_names.append(base)

features = np.array(features)


# ============================================================
# K-MEANS CLUSTERING FOR BALANCED SPLITS
# ============================================================
print("🔬 Clustering samples to avoid biased splits...")
print("   ➤ Prevents test set being accidentally too easy/hard.\n")

kmeans = KMeans(n_clusters=4, random_state=42)
cluster_ids = kmeans.fit_predict(features)


# ============================================================
# SPLIT INTO TRAIN / VAL / TEST
# ============================================================
print("✂ Performing stratified 70/15/15 split...\n")

train_bases, temp_bases, train_clust, temp_clust = train_test_split(
    base_names, cluster_ids, test_size=1-TRAIN_RATIO, shuffle=True,
    random_state=42, stratify=cluster_ids
)

val_bases, test_bases = train_test_split(
    temp_bases, test_size=TEST_RATIO/(TEST_RATIO+VAL_RATIO),
    shuffle=True, random_state=42, stratify=temp_clust
)

print(f"📁 TRAIN: {len(train_bases)} base images")
print(f"📁 VAL:   {len(val_bases)} base images")
print(f"📁 TEST:  {len(test_bases)} base images\n")


# ============================================================
# COPY FILES TO DESTINATION SPLIT FOLDERS
# ============================================================
def copy_group(base_list, split):
    print(f"📤 Copying files for {split}...")

    for base in tqdm(base_list):
        for fname in groups[base]:
            img_path = IMG_DIR / fname
            lbl_path = LBL_DIR / (Path(fname).stem + ".txt")

            shutil.copy(str(img_path), str(OUT_BASE / split / "images" / img_path.name))

            if lbl_path.exists():
                shutil.copy(str(lbl_path), str(OUT_BASE / split / "labels" / lbl_path.name))


copy_group(train_bases, "train")
copy_group(val_bases, "val")
copy_group(test_bases, "test")


# ============================================================
# CREATE data.yaml FOR YOLO
# ============================================================
print("\n📝 Creating YOLO data.yaml file...")

yaml_path = OUT_BASE / "data.yaml"

content = f"""
# YOLO Dataset Configuration
train: {str((OUT_BASE/'train/images')).replace("\\\\", "/")}
val: {str((OUT_BASE/'val/images')).replace("\\\\", "/")}
test: {str((OUT_BASE/'test/images')).replace("\\\\", "/")}

nc: 1
names: ["spheroid"]
"""

with open(yaml_path, "w") as f:
    f.write(content.strip() + "\n")

print(f"✅ data.yaml created at: {yaml_path}\n")


# ============================================================
# DONE
# ============================================================
print("==============================================")
print("🎉 YOLO dataset split completed successfully!")
print("📁 Dataset directory: yolo_splits/")
print("📄 YAML config:       yolo_splits/data.yaml")
print("==============================================\n")
