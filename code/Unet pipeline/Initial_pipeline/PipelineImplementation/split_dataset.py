from pathlib import Path
import shutil
import re
from collections import defaultdict

# Source paths
image_dir = Path("./data/images")
label_dir = Path("./data/labels")

# Destination paths
output_dir = Path("./data/organized")
train_img_dir = output_dir / "images" / "train"
train_mask_dir = output_dir / "masks" / "train"
val_img_dir = output_dir / "images" / "val"
val_mask_dir = output_dir / "masks" / "val"
test_img_dir = output_dir / "images" / "test"
test_mask_dir = output_dir / "masks" / "test"

# Create directories
for dir_path in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, test_img_dir, test_mask_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("🗂️  Smart dataset organization...\n")
print("="*70)

# Get all files
image_files = {f.name: f for f in image_dir.glob("*.png")}
label_files = {f.name: f for f in label_dir.glob("*.png")}

print(f"📊 Total images: {len(image_files)}")
print(f"📊 Total labels: {len(label_files)}")

def extract_base_name(filename):
    """Extract base name by removing augmentation suffixes"""
    stem = Path(filename).stem
    # Remove known suffixes: _v, _t, _hflip, _vflip, _rot90, _rot270
    base = re.sub(r'_(v|t|hflip|vflip|rot90|rot270)$', '', stem)
    return base

# Step 1: Group all files by their base name
print("\n" + "="*70)
print("STEP 1: Grouping files by base name...\n")

base_name_groups = defaultdict(list)

for filename in image_files.keys():
    base = extract_base_name(filename)
    base_name_groups[base].append(filename)

print(f"📦 Found {len(base_name_groups)} unique base name groups")

# Step 2: Classify each base group as train/val/test/extra
print("\n" + "="*70)
print("STEP 2: Classifying each base group...\n")

train_groups = []
val_groups = []
test_candidates = []
extra_files = []

for base_name, file_list in base_name_groups.items():
    stems = [Path(f).stem for f in file_list]
    
    # Check if this group has _v or _t anchor
    has_v = any(s.endswith('_v') for s in stems)
    has_t = any(s.endswith('_t') for s in stems)
    has_root = any(s == base_name for s in stems)  # Has file without any suffix
    
    if has_t:
        # Group goes to training (including ALL augmentations)
        train_groups.extend(file_list)
        print(f"  🟢 TRAIN: {base_name} ({len(file_list)} files) - {file_list}")
    elif has_v:
        # Group goes to validation (including ALL augmentations)
        val_groups.extend(file_list)
        print(f"  🔵 VAL:   {base_name} ({len(file_list)} files) - {file_list}")
    elif has_root and len(file_list) == 1:
        # Single root image with no augmentations → test candidate
        test_candidates.append(file_list[0])
        print(f"  🟡 TEST:  {base_name} (standalone root) - {file_list[0]}")
    else:
        # Orphan augmentations without root and no _v/_t → extra pool
        extra_files.extend(file_list)
        print(f"  🟠 EXTRA: {base_name} ({len(file_list)} orphan augmentations) - {file_list}")

print(f"\n✅ Training groups: {len(train_groups)} files")
print(f"✅ Validation groups: {len(val_groups)} files")
print(f"✅ Test candidates: {len(test_candidates)} files")
print(f"✅ Extra files: {len(extra_files)} files")

# Step 3: Split extra files into train/val/test
print("\n" + "="*70)
print("STEP 3: Distributing extra files...\n")

extra_files.sort()
n_extra = len(extra_files)
n_train_extra = int(n_extra * 0.4)
n_val_extra = int(n_extra * 0.3)

extra_train = extra_files[:n_train_extra]
extra_val = extra_files[n_train_extra:n_train_extra + n_val_extra]
extra_test = extra_files[n_train_extra + n_val_extra:]

print(f"  Training: +{len(extra_train)} extra files")
print(f"  Validation: +{len(extra_val)} extra files")
print(f"  Test: +{len(extra_test)} extra files")

# Combine final splits
final_train = train_groups + extra_train
final_val = val_groups + extra_val
final_test = test_candidates + extra_test

print("\n" + "="*70)
print("📊 FINAL DISTRIBUTION:")
print("-"*70)
print(f"Training:   {len(final_train)} files")
print(f"Validation: {len(final_val)} files")
print(f"Test:       {len(final_test)} files")
print(f"Total:      {len(final_train) + len(final_val) + len(final_test)} files")

# Step 4: Copy files
print("\n" + "="*70)
print("STEP 4: Copying files to organized structure...\n")

def copy_files(file_list, img_dest, mask_dest, split_name):
    """Copy images and masks to destination"""
    copied = 0
    skipped = 0
    for filename in file_list:
        img_path = image_files.get(filename)
        mask_path = label_files.get(filename)
        
        if img_path and mask_path:
            shutil.copy2(img_path, img_dest / filename)
            shutil.copy2(mask_path, mask_dest / filename)
            copied += 1
        else:
            print(f"  ⚠️  Missing pair for: {filename}")
            skipped += 1
    
    print(f"  ✅ {split_name}: Copied {copied} files" + (f" (skipped {skipped})" if skipped > 0 else ""))

print("📁 Copying training files...")
copy_files(final_train, train_img_dir, train_mask_dir, "Training")

print("📁 Copying validation files...")
copy_files(final_val, val_img_dir, val_mask_dir, "Validation")

print("📁 Copying test files...")
copy_files(final_test, test_img_dir, test_mask_dir, "Test")

print("\n" + "="*70)
print("✨ Dataset organization complete!")
print(f"📂 Output directory: {output_dir.absolute()}")
print("\nStructure created:")
print("  data/organized/")
print("  ├── images/")
print("  │   ├── train/")
print("  │   ├── val/")
print("  │   └── test/")
print("  └── masks/")
print("      ├── train/")
print("      ├── val/")
print("      └── test/")
print("="*70)