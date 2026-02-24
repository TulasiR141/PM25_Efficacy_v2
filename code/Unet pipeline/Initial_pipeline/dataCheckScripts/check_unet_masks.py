from pathlib import Path
import numpy as np
from PIL import Image


mask_dir = Path("./data/labels")


print("🔍 Checking unique label values in masks...\n")


for mask_path in sorted(mask_dir.glob("*.png")):
    mask = np.array(Image.open(mask_path))
    unique_values, counts = np.unique(mask, return_counts=True)


    print(f"{mask_path.name}:")
    for val, count in zip(unique_values, counts):
        print(f"   label {val}: {count} pixels")


    print("-" * 40)

