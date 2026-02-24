from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Paths
mask_dir = Path("./data/labels")
image_dir = Path("./data/images")

# Get only 2 masks to visualize
mask_files = sorted(mask_dir.glob("*.png"))[:2]

print("🎨 Creating mask overlay visualization...\n")

# Create figure with 2 rows and 2 columns
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

for idx, mask_path in enumerate(mask_files):
    # Find corresponding image
    image_path = image_dir / mask_path.name
    
    if not image_path.exists():
        print(f"⚠️  Image not found: {image_path.name}")
        continue
    
    # Load image and mask
    img = np.array(Image.open(image_path).convert('RGB'))
    mask = np.array(Image.open(mask_path))
    
    # Normalize mask to 0-1 if needed
    if mask.max() > 1:
        mask_norm = mask / 255.0
    else:
        mask_norm = mask
    
    # Column 1: Original Image
    axes[idx, 0].imshow(img)
    axes[idx, 0].set_title(f"Original\n{mask_path.name[:30]}", fontsize=11, pad=8)
    axes[idx, 0].axis('off')
    
    # Column 2: Overlay
    axes[idx, 1].imshow(img)
    axes[idx, 1].imshow(mask_norm, cmap='Blues', alpha=0.2, vmin=0, vmax=1)
    axes[idx, 1].set_title(f"Overlay (Blue = Mask)", fontsize=11, pad=8)
    axes[idx, 1].axis('off')
    
    print(f"✅ Processed: {mask_path.name}")

# Reduce spacing between subplots
plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.02, wspace=0.05, hspace=0.1)

plt.savefig("mask_overlay_visualization.png", dpi=200, bbox_inches='tight', pad_inches=0.1)
print("\n✅ Saved to 'mask_overlay_visualization.png'")
plt.show()

print("\n✨ Done!")