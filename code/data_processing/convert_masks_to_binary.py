from pathlib import Path
import numpy as np
from PIL import Image


# Path to your labeled masks
mask_dir = Path("data/Echo.v1i_UNet/training_validation/labels")


# ✅ Correct class ID for the inner spheroid
SPHEROID_CLASS = 2


for mask_path in mask_dir.glob("*.png"):
    mask = np.array(Image.open(mask_path))
    
    # Keep only spheroid class (2), everything else → background (0)
    mask_binary = (mask == SPHEROID_CLASS).astype(np.uint8)
    
    # Save as binary image (255 for spheroid, 0 for background)
    Image.fromarray(mask_binary * 255).save(mask_path)


print("✅ All masks converted to show ONLY the spheroid (class 2 inner region, no outer border).")





