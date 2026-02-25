import os
import shutil
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from shapely.geometry import Polygon
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# ============================================================
# INPUT DIRECTORIES  (we modify files *inside* here)
# ============================================================
BASE_DIR = Path("training_validation")
IMG_DIR  = BASE_DIR / "images"
MSK_DIR  = BASE_DIR / "labels"

print("\n==============================================")
print(" UNET DATASET SPLITTING (Grouped + Suffix Only)")
print("==============================================\n")

# ============================================================
# GROUP FILENAMES BY BASE (same as YOLO)
# ============================================================
print("🔎 Grouping augmented images by base name...")

groups = {}

for img_file in IMG_DIR.iterdir():
    if img_file.suffix.lower() != ".png":
        continue

    stem = img_file.stem

    # remove augment suffixes
    base = stem.split("_rot")[0].split("_hflip")[0].split("_vflip")[0].split("_t")[0].split("_v")[0]

    groups.setdefault(base, []).append(img_file.name)

print(f"📦 Found {len(groups)} groups.\n")

# ============================================================
# FEATURE EXTRACTION (brightness + shape)
# ============================================================
def compute_characteristics(mask):
    brightness = mask.mean()

    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return brightness, 0, 0

    cnt = cnts[0].squeeze()
    if len(cnt) < 3:
        return brightness, 0, 0

    poly = Polygon(cnt)
    area = poly.area
    circ = 4*np.pi*area / (poly.length**2 + 1e-6)
    return brightness, area, circ


print("📊 Extracting mask-based features...")

features = []
base_names = []

for base, files in tqdm(groups.items()):
    sample_file = files[0]
    mask_path = MSK_DIR / sample_file
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    feat = compute_characteristics(mask)
    features.append(feat)
    base_names.append(base)

features = np.array(features)

# ============================================================
# KMEANS SPLITTING
# ============================================================
print("\n🔬 Running KMeans for balanced splits...\n")

kmeans = KMeans(n_clusters=6, random_state=42)
cluster_ids = kmeans.fit_predict(features)

# ============================================================
# GROUPED SPLITS
# ============================================================
print("✂ Performing grouped 70 / 15 / 15 split...\n")

train_bases, temp_bases, train_c, temp_c = train_test_split(
    base_names, cluster_ids, test_size=0.30, random_state=42, shuffle=True,
    stratify=cluster_ids
)

val_bases, test_bases = train_test_split(
    temp_bases, test_size=0.50, random_state=42, shuffle=True,
    stratify=temp_c
)

print(f"📁 TRAIN: {len(train_bases)} groups")
print(f"📁 VAL:   {len(val_bases)} groups")
print(f"📁 TEST:  {len(test_bases)} groups\n")

# ============================================================
# APPLY SUFFIXES _v and _t (train = unchanged)
# ============================================================
print("📝 Renaming files to apply suffix-based splits...")

def rename_group(bases, suffix):
    for base in bases:
        for filename in groups[base]:
            old_img = IMG_DIR / filename
            old_msk = MSK_DIR / filename

            new_name = filename.replace(".png", f"{suffix}.png")

            new_img = IMG_DIR / new_name
            new_msk = MSK_DIR / new_name

            os.rename(old_img, new_img)
            os.rename(old_msk, new_msk)


# IMPORTANT: rename TEST first so `.png` doesn't collide
rename_group(test_bases, "_t")
rename_group(val_bases, "_v")

print("\n==============================================")
print("🎉 UNet suffix-only split completed!")
print("📁 Updated dataset: training_validation/")
print("==============================================\n")
