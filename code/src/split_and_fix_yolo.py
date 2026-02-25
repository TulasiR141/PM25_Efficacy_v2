"""
split_and_fix_yolo.py

1. Takes augmented YOLO dataset from:
     data_new/echo_v3_consolidated_augmented/{images,labels}

2. Groups images by base key so that all augmentations of the same
   base sample (_hflip, _vflip, _rot90, _rot270) stay together.

3. Splits groups into train / val / test and writes to:
     yolo_splits/train/{images,labels}
     yolo_splits/val/{images,labels}
     yolo_splits/test/{images,labels}

4. Prints per-split stats about group counts and augmentation types.

5. Fixes YOLO label classes so that all class IDs are set to 0
   (spheroid only), making .bak backups of label files.
"""

from pathlib import Path
import shutil
import random
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================
# Project root = one level up from this script (assuming in src_new/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Source: augmented consolidated dataset
SRC_BASE = PROJECT_ROOT / "data_new" / "echo_v3_consolidated_augmented"
SRC_IMG_DIR = SRC_BASE / "images"
SRC_LBL_DIR = SRC_BASE / "labels"

# Output: YOLO train/val/test splits
OUT_BASE = PROJECT_ROOT / "yolo_splits_v1"
SPLITS = ["train", "val", "test"]

# Train/Val/Test ratios over base groups
SPLIT_RATIOS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# Augmentation suffixes that define variants of the same base sample
AUG_SUFFIXES = ("_hflip", "_vflip", "_rot90", "_rot270")

# Random seed for reproducible splits
random.seed(42)


# ============================================================
# HELPERS: GROUPING & SPLITTING
# ============================================================
def base_key_from_stem(stem: str) -> str:
    """
    Given a filename stem, strip known augmentation suffixes to get the base key.
    Example:
      A_0005_jpg.rf.abcd_hflip  -> A_0005_jpg.rf.abcd
      A_0005_jpg.rf.abcd       -> A_0005_jpg.rf.abcd
    """
    for suf in AUG_SUFFIXES:
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def build_groups():
    """
    Build groups of (image, label) pairs by base key.
    Returns:
      groups: dict[base_key] = list[(img_path, lbl_path)]
    """
    if not SRC_IMG_DIR.is_dir():
        raise RuntimeError(f"Image directory not found: {SRC_IMG_DIR}")
    if not SRC_LBL_DIR.is_dir():
        raise RuntimeError(f"Label directory not found: {SRC_LBL_DIR}")

    groups = {}
    missing_labels = 0

    for img_path in sorted(SRC_IMG_DIR.iterdir()):
        if not img_path.is_file():
            continue
        stem = img_path.stem
        lbl_path = SRC_LBL_DIR / f"{stem}.txt"
        if not lbl_path.is_file():
            print(f"⚠ No label for image {img_path.name}, skipping this image.")
            missing_labels += 1
            continue

        key = base_key_from_stem(stem)
        groups.setdefault(key, []).append((img_path, lbl_path))

    print(f"📦 Found {len(groups)} base groups.")
    print(f"⚠ Images skipped due to missing labels: {missing_labels}\n")
    return groups


def split_groups(groups: dict):
    """
    Split base groups into train/val/test by SPLIT_RATIOS.
    Returns:
      split_assignment: dict[split] = list[base_key]
    """
    base_keys = list(groups.keys())
    random.shuffle(base_keys)

    n = len(base_keys)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    n_test = n - n_train - n_val  # remainder

    train_keys = base_keys[:n_train]
    val_keys = base_keys[n_train : n_train + n_val]
    test_keys = base_keys[n_train + n_val :]

    split_assignment = {
        "train": train_keys,
        "val": val_keys,
        "test": test_keys,
    }

    print("🔀 Group split sizes (by base_key):")
    for split, keys in split_assignment.items():
        print(f"  {split}: {len(keys)} groups")
    print()

    return split_assignment


def ensure_split_dirs():
    """
    Create output directories:
      yolo_splits/{train,val,test}/{images,labels}
    """
    for split in SPLITS:
        img_dir = OUT_BASE / split / "images"
        lbl_dir = OUT_BASE / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)


def copy_to_splits(groups: dict, split_assignment: dict):
    """
    Copy images and labels into train/val/test directories according to split_assignment.
    Also collects augmentation statistics per split.
    """
    ensure_split_dirs()

    # Stats: per split, counts of images and augmentation types
    split_counts = {
        split: {
            "total_images": 0,
            "suffix_counts": Counter(),  # counts of hflip/vflip/rot90/rot270/base
        }
        for split in SPLITS
    }

    for split, keys in split_assignment.items():
        img_out_dir = OUT_BASE / split / "images"
        lbl_out_dir = OUT_BASE / split / "labels"

        for key in keys:
            pairs = groups[key]
            for img_path, lbl_path in pairs:
                stem = img_path.stem
                ext = img_path.suffix

                # Determine augmentation type for stats
                aug_type = "base"
                for suf in AUG_SUFFIXES:
                    if stem.endswith(suf):
                        aug_type = suf.lstrip("_")
                        break

                split_counts[split]["total_images"] += 1
                split_counts[split]["suffix_counts"][aug_type] += 1

                # Copy image and label
                dst_img = img_out_dir / img_path.name
                dst_lbl = lbl_out_dir / lbl_path.name

                shutil.copy2(img_path, dst_img)
                shutil.copy2(lbl_path, dst_lbl)

    # Print stats
    print("📊 Split statistics (per split):")
    for split in SPLITS:
        stats = split_counts[split]
        print(f"\nSplit: {split}")
        print(f"  Total images: {stats['total_images']}")
        for aug_type, cnt in stats["suffix_counts"].items():
            print(f"  {aug_type:>8}: {cnt}")
    print("\n✅ Copying to splits complete.\n")


# ============================================================
# LABEL FIXING (like fix_yolo_label_classes.py)
# ============================================================
def collect_class_stats(label_dir: Path) -> Counter:
    """
    Collect class ID counts from all .txt YOLO label files in label_dir.
    Returns a Counter {class_id: count}.
    """
    class_counts = Counter()
    for label_file in label_dir.rglob("*.txt"):
        text = label_file.read_text().strip()
        if not text:
            continue
        for line in text.splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            class_counts[cls_id] += 1
    return class_counts


def fix_labels_in_dir(label_dir: Path) -> int:
    """
    Converts all class IDs in YOLO label files under label_dir to 0.
    Creates a .bak backup for each file that is changed.
    Returns number of files changed.
    """
    fixed_files = 0
    for label_file in label_dir.rglob("*.txt"):
        text = label_file.read_text().strip()
        if not text:
            continue
        lines = text.splitlines()
        new_lines = []
        changed = False
        for line in lines:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                old_class = int(float(parts[0]))
            except ValueError:
                # keep line as is if class malformed
                new_lines.append(line)
                continue
            if old_class != 0:
                parts[0] = "0"
                changed = True
            new_lines.append(" ".join(parts))
        if changed:
            backup = label_file.with_suffix(".bak")
            shutil.copy(label_file, backup)
            label_file.write_text("\n".join(new_lines) + "\n")
            fixed_files += 1
    return fixed_files


def fix_all_splits_and_report():
    """
    For each split (train/val/test), print class stats before fixing,
    then apply fix_labels_in_dir to set all class IDs to 0,
    then print how many files were fixed.
    """
    total_fixed = 0
    print("====== Checking and fixing label classes ======\n")

    for split in SPLITS:
        label_dir = OUT_BASE / split / "labels"
        if not label_dir.exists():
            print(f"⚠ Skipping {split}: no labels directory found.")
            continue

        print(f"🔍 Split: {split}")
        before_stats = collect_class_stats(label_dir)
        if before_stats:
            print("   Class distribution BEFORE fix:")
            for cls_id, cnt in sorted(before_stats.items()):
                print(f"     class {cls_id}: {cnt} boxes")
        else:
            print("   No labels found in this split.")

        fixed = fix_labels_in_dir(label_dir)
        total_fixed += fixed
        print(f"✅ {fixed} label files modified in '{split}'.\n")

    print("🎯 Summary of label fixing:")
    print(f"   Total label files corrected: {total_fixed}")
    print("   (Backups saved as .bak next to each changed label file)")
    print("   All class IDs are now set to 0 (spheroid only).\n")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("====== STEP 1: Build groups from augmented dataset ======")
    groups = build_groups()

    print("====== STEP 2: Split groups into train/val/test ======")
    split_assignment = split_groups(groups)

    print("====== STEP 3: Copy images & labels into yolo_splits/ ======")
    copy_to_splits(groups, split_assignment)

    print("====== STEP 4: Fix label classes to 0 & report ======")
    fix_all_splits_and_report()

    print(f"📂 YOLO splits ready under: {OUT_BASE}")
                                                       
