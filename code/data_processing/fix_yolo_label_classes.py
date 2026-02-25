"""
fix_yolo_label_classes.py

Fixes YOLO label class IDs after consolidation.
 - Converts all class IDs to 0 (spheroid only)
 - Works for train, val, and test splits
 - Creates a backup of each label file before overwriting
"""

from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================
BASE_DIR = Path("yolo_splits")  # path where train/val/test exist
SPLITS = ["train", "val", "test"]

# ============================================================
# FUNCTION
# ============================================================
def fix_labels_in_dir(label_dir):
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

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    total_fixed = 0
    for split in SPLITS:
        label_dir = BASE_DIR / split / "labels"
        if not label_dir.exists():
            print(f"⚠️ Skipping {split}: no labels directory found.")
            continue
        print(f"🔍 Fixing labels in: {label_dir}")
        fixed = fix_labels_in_dir(label_dir)
        total_fixed += fixed
        print(f"✅ {fixed} files fixed in '{split}'.")

    print("\n🎯 Summary:")
    print(f"   Total label files corrected: {total_fixed}")
    print("   (Backups saved as .bak next to each label file)")
    print("\n💡 All class IDs are now set to 0 (spheroid only).")
