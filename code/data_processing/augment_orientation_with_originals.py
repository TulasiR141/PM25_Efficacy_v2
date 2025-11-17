import os
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# ============================================================
# CONFIGURATION
# ============================================================
SRC_IMG_DIR = "pictures/resized_1024/images"
SRC_LABEL_DIR = "pictures/resized_1024/labels"

DEST_IMG_DIR = "pictures/augmented_final/images"
DEST_LABEL_DIR = "pictures/augmented_final/labels"

TARGET_SIZE = 1024

os.makedirs(DEST_IMG_DIR, exist_ok=True)
os.makedirs(DEST_LABEL_DIR, exist_ok=True)

# ============================================================
# AUGMENTATIONS WITH YOUR PROBABILITIES
# ============================================================
AUGMENTATIONS = {
    "hflip": A.HorizontalFlip(p=0.35),
    "vflip": A.VerticalFlip(p=0.35),
    "rot90": A.Rotate(limit=(90, 90), p=0.18, border_mode=cv2.BORDER_CONSTANT),
    "rot270": A.Rotate(limit=(270, 270), p=0.18, border_mode=cv2.BORDER_CONSTANT),
}

# ============================================================
# LOAD / SAVE POLYGONS
# ============================================================
def load_polygons(label_path, w, h):
    polygons = []
    if not os.path.exists(label_path):
        return polygons
    
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 3:
                cls = int(parts[0])
                coords = list(map(float, parts[1:]))
                pts = np.array([(coords[i] * w, coords[i+1] * h)
                                for i in range(0, len(coords), 2)], np.float32)
                polygons.append((cls, pts))
    return polygons


def save_polygons(label_path, polygons, w, h):
    with open(label_path, "w") as f:
        for cls, pts in polygons:
            pts_norm = pts.copy()
            pts_norm[:, 0] /= w
            pts_norm[:, 1] /= h
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts_norm)
            f.write(f"{cls} {coords}\n")


# ============================================================
# MAIN AUGMENTATION LOOP
# ============================================================
images = [f for f in os.listdir(SRC_IMG_DIR)
          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"🔄 Found {len(images)} images. Creating final dataset...")

for img_name in tqdm(images, desc="Processing"):
    img_path = os.path.join(SRC_IMG_DIR, img_name)
    lbl_path = os.path.join(SRC_LABEL_DIR, img_name.replace(".jpg", ".txt"))

    img = cv2.imread(img_path)
    if img is None or not os.path.exists(lbl_path):
        continue

    h, w = img.shape[:2]
    polygons = load_polygons(lbl_path, w, h)
    keypoints = [tuple(pt) for _, poly in polygons for pt in poly]

    # ============================================================
    # SAVE ORIGINAL IMAGE + LABEL
    # ============================================================
    cv2.imwrite(os.path.join(DEST_IMG_DIR, img_name), img)
    dest_label_path = os.path.join(DEST_LABEL_DIR, img_name.replace(".jpg", ".txt"))
    with open(lbl_path, "r") as src, open(dest_label_path, "w") as dst:
        dst.write(src.read())

    # ============================================================
    # APPLY AUGMENTATIONS (ONLY IF THEY HAPPEN)
    # ============================================================
    for aug_name, aug in AUGMENTATIONS.items():
        pipeline = A.Compose(
            [aug],
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False)
        )

        result = pipeline(image=img, keypoints=keypoints)
        aug_img = result["image"]

        # augmentation didn't happen → skip
        if np.array_equal(aug_img, img):
            continue  

        aug_kpts = result["keypoints"]

        # rebuild polygons
        new_polygons = []
        k = 0
        for cls, poly in polygons:
            pts = np.array(aug_kpts[k:k+len(poly)], np.float32)
            new_polygons.append((cls, pts))
            k += len(poly)

        # save augmented files
        new_name = img_name.replace(".jpg", f"_{aug_name}.jpg")
        cv2.imwrite(os.path.join(DEST_IMG_DIR, new_name), aug_img)

        save_polygons(
            os.path.join(DEST_LABEL_DIR, new_name.replace(".jpg", ".txt")),
            new_polygons, TARGET_SIZE, TARGET_SIZE
        )

print("\n✅ Final dataset creation complete!")
print(f"📁 All images saved to: {DEST_IMG_DIR}")
print(f"📁 All labels saved to: {DEST_LABEL_DIR}")
