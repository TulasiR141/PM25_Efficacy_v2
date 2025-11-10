import random
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# === PATH CONFIGURATION ===
data_root = Path("data")
yolo_root = data_root / "EchoRoboflow.v1i.yolov8"
unet_mask_dir = data_root / "Echo.v1i_UNet" / "training_validation" / "labels"


# === FUNCTION: Load YOLO segmentation polygons ===
def load_yolo_polygons(label_path, img_shape):
    """Reads YOLO segmentation polygons (class + normalized xy pairs)."""
    h, w = img_shape[:2]
    polygons = []
    if not label_path.exists():
        return polygons


    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            coords = np.array(list(map(float, parts[1:])))
            coords = coords.reshape(-1, 2)
            coords[:, 0] *= w
            coords[:, 1] *= h
            coords = coords.astype(np.int32)
            polygons.append((cls, coords))
    return polygons




# === COLLECT ALL YOLO IMAGES ===
yolo_images = []
for subset in ["train", "valid", "test"]:
    subset_path = yolo_root / subset / "images"
    if subset_path.exists():
        yolo_images.extend(list(subset_path.glob("*.jpg")))


random.shuffle(yolo_images)
sample_images = yolo_images[:5]


print(f"🎯 Spot-checking {len(sample_images)} random samples across YOLO splits...")


# === MAIN COMPARISON LOOP ===
for img_path in sample_images:
    base_name = img_path.stem
    subset = img_path.parent.parent.name
    label_path = yolo_root / subset / "labels" / f"{base_name}.txt"


    # Load YOLO image
    yolo_img = cv2.imread(str(img_path))
    if yolo_img is None:
        print(f"⚠ Cannot read {img_path}")
        continue
    yolo_img = cv2.cvtColor(yolo_img, cv2.COLOR_BGR2RGB)
    img_h, img_w = yolo_img.shape[:2]


    # --- YOLO overlay ---
    yolo_overlay = yolo_img.copy()
    polygons = load_yolo_polygons(label_path, (img_h, img_w))
    for cls, pts in polygons:
        cv2.polylines(yolo_overlay, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(
            yolo_overlay,
            f"class {cls}",
            (pts[0][0], pts[0][1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


    # --- U-Net mask overlay ---
    unet_mask_path = unet_mask_dir / f"{base_name}.png"
    if not unet_mask_path.exists():
        print(f"⚠ No U-Net mask found for {base_name}")
        continue
    mask = np.array(Image.open(unet_mask_path).resize((img_w, img_h)))


    # Normalize and create colored overlay
    mask_norm = (mask > 128).astype(np.uint8)  # binary mask
    unet_overlay = yolo_img.copy()
    red_mask = np.zeros_like(unet_overlay)
    red_mask[:, :, 0] = mask_norm * 255  # red channel only


    alpha = 0.5  # transparency strength
    unet_overlay = cv2.addWeighted(unet_overlay, 1.0, red_mask, alpha, 0)


    # === DISPLAY SIDE BY SIDE ===
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(yolo_overlay)
    ax[0].set_title("YOLO Image + Polygon Overlay", fontsize=11)
    ax[1].imshow(unet_overlay)
    ax[1].set_title("U-Net Mask Overlay (Red Transparent)", fontsize=11)


    for a in ax:
        a.axis("off")


    plt.tight_layout()
    plt.show()





