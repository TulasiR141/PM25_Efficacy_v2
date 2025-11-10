from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


mask_path = Path("data/Echo.v1i_UNet/training_validation/labels/A_0005_jpg.rf.6a9779180a39f9359028a262fcee7f4e.png")
img_path = Path("data/Echo.v1i_UNet/training_validation/images/A_0005_jpg.rf.6a9779180a39f9359028a262fcee7f4e.jpg")


mask = np.array(Image.open(mask_path))
img = np.array(Image.open(img_path).convert("RGB"))


# Try both class IDs
mask_class1 = (mask == 1)
mask_class2 = (mask == 2)
mask_class3 = (mask == 3)


fig, ax = plt.subplots(1, 4, figsize=(16, 6))
ax[0].imshow(img); ax[0].set_title("Original")
ax[1].imshow(img); ax[1].imshow(mask_class1, alpha=0.5, cmap="Reds"); ax[1].set_title("Class 1 overlay")
ax[2].imshow(img); ax[2].imshow(mask_class2, alpha=0.5, cmap="Reds"); ax[2].set_title("Class 2 overlay")
ax[3].imshow(img); ax[3].imshow(mask_class3, alpha=0.5, cmap="Reds"); ax[3].set_title("Class 3 overlay")
for a in ax: a.axis("off")
plt.tight_layout()
plt.show()





