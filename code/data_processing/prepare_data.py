import os
import shutil
from pathlib import Path
import random


# ==== CONFIG ====
BASE_PATH = Path("./data/Echo.v1i_UNet/")  # current directory
TRAIN_DIR = BASE_PATH / "train"
VALID_DIR = BASE_PATH / "valid"
TEST_DIR  = BASE_PATH / "test"


OUTPUT_DIR = BASE_PATH / "training_validation"
IMAGES_DIR = OUTPUT_DIR / "images"
LABELS_DIR = OUTPUT_DIR / "labels"


IMAGES_DIR.mkdir(parents=True, exist_ok=True)
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def find_matching_mask(mask_dir, img_name_stem):
    """
    Looks in mask_dir for a mask with the same stem as the image, regardless of extension.
    Returns the matching Path or None if not found.
    """
    for ext in [".png", ".jpg", ".jpeg", ".tif"]:
        candidate = mask_dir / f"{img_name_stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def copy_pairs(src_img_dir, src_mask_dir, tag=None):
    img_files = sorted(os.listdir(src_img_dir))
    for img_name in img_files:
        img_path = Path(src_img_dir) / img_name
        img_stem = Path(img_name).stem


        # Find matching mask by stem
        mask_path = find_matching_mask(Path(src_mask_dir), img_stem)
        if mask_path is None:
            print(f"⚠ Missing mask for {img_name}, skipping.")
            continue


        # Rename output files (optional suffix _t)
        name, ext = os.path.splitext(img_name)
        if tag:
            new_name = f"{name}{tag}{ext}"
        else:
            new_name = img_name


        shutil.copy(img_path, IMAGES_DIR / new_name)


        # Always save masks as .png
        new_mask_name = f"{name}{tag if tag else ''}.png"
        shutil.copy(mask_path, LABELS_DIR / new_mask_name)


# ==== Copy files ====
print("Copying training files...")
copy_pairs(TRAIN_DIR / "images", TRAIN_DIR / "masks")


print("Copying validation files (tagged as _t)...")
copy_pairs(VALID_DIR / "images", VALID_DIR / "masks", tag="_t")


print("Copying test files (tagged as _t)...")
copy_pairs(TEST_DIR / "images", TEST_DIR / "masks", tag="_t")


print("✅ Dataset successfully reformatted!")
print(f"Images are now in: {IMAGES_DIR}")
print(f"Masks  are now in: {LABELS_DIR}")





