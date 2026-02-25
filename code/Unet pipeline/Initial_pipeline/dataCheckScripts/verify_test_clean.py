from pathlib import Path
import shutil
import re

# Paths
organized_dir = Path("./data/organized")
train_img_dir = organized_dir / "images" / "train"
train_mask_dir = organized_dir / "masks" / "train"
val_img_dir = organized_dir / "images" / "val"
val_mask_dir = organized_dir / "masks" / "val"
test_img_dir = organized_dir / "images" / "test"
test_mask_dir = organized_dir / "masks" / "test"

print("🔍 Verifying test folder integrity...\n")
print("="*70)

def extract_base_name(filename):
    """Extract base name by removing augmentation suffixes"""
    stem = Path(filename).stem
    # Remove known suffixes: _v, _t, _hflip, _vflip, _rot90, _rot270
    base = re.sub(r'_(v|t|hflip|vflip|rot90|rot270)$', '', stem)
    return base

def has_suffix(filename):
    """Check if filename has augmentation suffix"""
    stem = Path(filename).stem
    return bool(re.search(r'_(v|t|hflip|vflip|rot90|rot270)$', stem))

# Get all files
train_files = [f.name for f in train_img_dir.glob("*.png")]
val_files = [f.name for f in val_img_dir.glob("*.png")]
test_files = [f.name for f in test_img_dir.glob("*.png")]

print(f"📊 Current distribution:")
print(f"  Training:   {len(train_files)} files")
print(f"  Validation: {len(val_files)} files")
print(f"  Test:       {len(test_files)} files")

# Extract base names from train and val
train_base_names = set(extract_base_name(f) for f in train_files)
val_base_names = set(extract_base_name(f) for f in val_files)

print(f"\n📦 Unique base names:")
print(f"  Training:   {len(train_base_names)} unique bases")
print(f"  Validation: {len(val_base_names)} unique bases")

# Check test files
print("\n" + "="*70)
print("STEP 1: Checking for files with suffixes in test...\n")

files_with_suffix = []
for test_file in test_files:
    if has_suffix(test_file):
        files_with_suffix.append(test_file)
        print(f"  ⚠️  {test_file} has suffix")

print(f"\n📊 Found {len(files_with_suffix)} test files with suffixes")

# Check for base name conflicts
print("\n" + "="*70)
print("STEP 2: Checking for base name conflicts (both directions)...\n")

conflicts = []
for test_file in test_files:
    test_base = extract_base_name(test_file)
    
    # Check if test file's base name appears in train or val
    if test_base in train_base_names:
        conflicts.append((test_file, "train", test_base))
        print(f"  ❌ CONFLICT: {test_file} (base: {test_base}) shares base with train")
    elif test_base in val_base_names:
        conflicts.append((test_file, "val", test_base))
        print(f"  ❌ CONFLICT: {test_file} (base: {test_base}) shares base with val")
    
    # Also check reverse: if baseName1.png in test, check if baseName1_hflip etc. in train/val
    # This catches cases where test has root but train/val has augmentations
    for train_file in train_files:
        train_base = extract_base_name(train_file)
        if test_base == train_base and test_file != train_file:
            conflicts.append((test_file, "train", test_base))
            print(f"  ❌ CONFLICT: {test_file} (base: {test_base}) - train has {train_file}")
            break
    
    for val_file in val_files:
        val_base = extract_base_name(val_file)
        if test_base == val_base and test_file != val_file:
            conflicts.append((test_file, "val", test_base))
            print(f"  ❌ CONFLICT: {test_file} (base: {test_base}) - val has {val_file}")
            break

print(f"\n📊 Found {len(conflicts)} conflicts")

# Create removal list (files with suffix OR conflicts)
files_to_remove = set(files_with_suffix + [c[0] for c in conflicts])

if len(files_to_remove) == 0:
    print("\n" + "="*70)
    print("✅ Test folder is clean! No issues found.")
    print("="*70)
else:
    print("\n" + "="*70)
    print(f"STEP 3: Removing {len(files_to_remove)} problematic files from test...\n")
    
    removed_dir = organized_dir / "removed_from_test"
    removed_img_dir = removed_dir / "images"
    removed_mask_dir = removed_dir / "masks"
    removed_img_dir.mkdir(parents=True, exist_ok=True)
    removed_mask_dir.mkdir(parents=True, exist_ok=True)
    
    for filename in files_to_remove:
        img_path = test_img_dir / filename
        mask_path = test_mask_dir / filename
        
        # Move to removed folder (backup)
        if img_path.exists():
            shutil.move(str(img_path), str(removed_img_dir / filename))
        if mask_path.exists():
            shutil.move(str(mask_path), str(removed_mask_dir / filename))
        
        print(f"  ✅ Removed: {filename}")
    
    # Recount
    test_files_after = [f.name for f in test_img_dir.glob("*.png")]
    
    print("\n" + "="*70)
    print("📊 FINAL DISTRIBUTION:")
    print("-"*70)
    print(f"Training:   {len(train_files)} files (unchanged)")
    print(f"Validation: {len(val_files)} files (unchanged)")
    print(f"Test:       {len(test_files_after)} files (was {len(test_files)})")
    print(f"Removed:    {len(files_to_remove)} files → {removed_dir}")
    print("\n" + "="*70)
    print("✨ Test folder cleaned!")
    print(f"📂 Removed files backed up to: {removed_dir.absolute()}")
    print("="*70)

# Final verification
print("\n" + "="*70)
print("FINAL VERIFICATION:")
print("-"*70)

test_files_final = [f.name for f in test_img_dir.glob("*.png")]
issues = 0

for test_file in test_files_final:
    if has_suffix(test_file):
        print(f"  ⚠️  Still has suffix: {test_file}")
        issues += 1
    
    test_base = extract_base_name(test_file)
    if test_base in train_base_names or test_base in val_base_names:
        print(f"  ⚠️  Still has conflict: {test_file}")
        issues += 1

if issues == 0:
    print("  ✅ All test files are clean (no suffixes, no conflicts)")
else:
    print(f"  ❌ Found {issues} remaining issues")

print("="*70)