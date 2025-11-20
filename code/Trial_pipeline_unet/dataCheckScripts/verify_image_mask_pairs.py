from pathlib import Path
from fastai.vision.core import get_image_files

PATH_TO_DATASET = Path('./data/organized')

print("=" * 70)
print("VERIFYING IMAGE-MASK PAIRS")
print("=" * 70)

def check_split(split_name):
    print(f"\n{split_name.upper()} Split:")
    print("-" * 70)
    
    img_dir = PATH_TO_DATASET / 'images' / split_name
    mask_dir = PATH_TO_DATASET / 'masks' / split_name
    
    # Get all images
    images = get_image_files(img_dir)
    print(f"  Images found: {len(images)}")
    
    if len(images) == 0:
        print(f"  ❌ ERROR: No images in {img_dir}")
        return False
    
    # Check each image has a mask
    missing_masks = []
    for img in images:
        mask_path = mask_dir / img.name
        if not mask_path.exists():
            missing_masks.append(img.name)
    
    if missing_masks:
        print(f"  ❌ {len(missing_masks)} images missing masks:")
        for name in missing_masks[:5]:
            print(f"     - {name}")
        if len(missing_masks) > 5:
            print(f"     ... and {len(missing_masks) - 5} more")
        return False
    else:
        print(f"  ✅ All {len(images)} images have matching masks")
        print(f"  Sample pairs:")
        for img in images[:3]:
            print(f"     {img.name} → masks/{split_name}/{img.name}")
        return True

# Check all splits
all_good = True
for split in ['train', 'val', 'test']:
    if not check_split(split):
        all_good = False

print("\n" + "=" * 70)
if all_good:
    print("✅ ALL CHECKS PASSED - Ready to train!")
else:
    print("❌ ISSUES FOUND - Fix missing masks before training")
print("=" * 70)
