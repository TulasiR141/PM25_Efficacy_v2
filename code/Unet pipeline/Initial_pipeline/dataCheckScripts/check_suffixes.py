from pathlib import Path
from collections import Counter
import re

# Paths
image_dir = Path("./data/images")
label_dir = Path("./data/labels")

print("🔍 Analyzing file naming patterns...\n")
print("="*70)

# Get all image files
image_files = sorted(image_dir.glob("*.png"))
label_files = sorted(label_dir.glob("*.png"))

print(f"📊 Total images: {len(image_files)}")
print(f"📊 Total labels: {len(label_files)}")
print("\n" + "="*70)

def extract_suffix(filepath):
    """Extract the suffix pattern from filename"""
    name = filepath.stem
    pattern = r'_([a-z0-9]+)$'
    match = re.search(pattern, name)
    if match:
        return match.group(1)
    return "no_suffix"

# Collect all suffixes
image_suffixes = [extract_suffix(f) for f in image_files]
label_suffixes = [extract_suffix(f) for f in label_files]

# Count occurrences
image_suffix_counts = Counter(image_suffixes)
label_suffix_counts = Counter(label_suffixes)

print("\n📁 IMAGE SUFFIXES FOUND:")
print("-"*70)
for suffix, count in sorted(image_suffix_counts.items(), key=lambda x: -x[1]):
    print(f"  _{suffix:15} : {count:4} files")

print("\n📁 LABEL SUFFIXES FOUND:")
print("-"*70)
for suffix, count in sorted(label_suffix_counts.items(), key=lambda x: -x[1]):
    print(f"  _{suffix:15} : {count:4} files")

print("\n" + "="*70)
print("\n💡 SUFFIX CATEGORIES DETECTED:")
print("-"*70)

# Categorize suffixes
validation_suffixes = [s for s in image_suffix_counts.keys() if s.startswith('v')]
training_suffixes = [s for s in image_suffix_counts.keys() if s.startswith('t')]
augmentation_suffixes = [s for s in image_suffix_counts.keys() if 'flip' in s or 'rot' in s]

print(f"\n✅ Validation suffixes (start with 'v'):")
for s in validation_suffixes:
    print(f"   _{s}")

print(f"\n✅ Training suffixes (start with 't'):")
for s in training_suffixes:
    print(f"   _{s}")

print(f"\n✅ Augmentation suffixes (flip/rotation):")
for s in augmentation_suffixes:
    print(f"   _{s}")

print(f"\n✅ Other suffixes:")
other_suffixes = [s for s in image_suffix_counts.keys() 
                  if s not in validation_suffixes 
                  and s not in training_suffixes 
                  and s not in augmentation_suffixes]
for s in other_suffixes:
    print(f"   _{s}")

print("\n" + "="*70)
print("\n📋 SAMPLE FILENAMES:")
print("-"*70)
print("\nFirst 10 image files:")
for f in image_files[:10]:
    suffix = extract_suffix(f)
    print(f"  {f.name} → suffix: _{suffix}")

print("\n" + "="*70)
print("\n✨ Analysis complete!")