import os
import random
import cv2
import numpy as np
from pathlib import Path

# === Resolve paths relative to the project root ===
project_root = Path(__file__).resolve().parents[1]  # one level up from metric_tools/
dataset_path = project_root / "pictures" / "Echo.v1i.yolov8" / "train"
images_dir = dataset_path / "images"
labels_dir = dataset_path / "labels"
output_dir = project_root / "pictures" / "visualized_samples"

# === Class names and colors (from your Echo dataset) ===
class_names = ['background', 'spheroid', 'spheroidOuterBorder']
colors = {
    0: (255, 0, 0),     # background → blue
    1: (0, 255, 0),     # spheroid → green
    2: (0, 0, 255)      # spheroidOuterBorder → red
}

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Collect images ===
image_files = [f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
if len(image_files) < 3:
    raise ValueError(f"Not enough images found in {images_dir} (found {len(image_files)})")

# Pick 3 random images
sample_images = random.sample(image_files, 3)

def draw_polygon(img, points, color):
    """Draw a filled polygon with an outline."""
    points = np.array(points, np.int32).reshape((-1, 1, 2))
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    cv2.polylines(img, [points], True, color, 2)
    return img

for img_file in sample_images:
    img_path = images_dir / img_file
    label_path = labels_dir / (Path(img_file).stem + ".txt")

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"⚠️ Could not read {img_file}")
        continue

    h, w = img.shape[:2]

    if not label_path.exists():
        print(f"⚠️ No label found for {img_file}")
        continue

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        cls_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        xy = [(int(coords[i] * w), int(coords[i + 1] * h)) for i in range(0, len(coords), 2)]
        img = draw_polygon(img, xy, colors.get(cls_id, (255, 255, 255)))

    # === Add legend ===
    y0 = 30
    for cid, cname in enumerate(class_names):
        cv2.rectangle(img, (10, y0 - 20), (30, y0), colors[cid], -1)
        cv2.putText(img, cname, (40, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y0 += 30

    out_path = output_dir / img_file
    cv2.imwrite(str(out_path), img)
    print(f"✅ Saved visualization: {out_path}")

print("\n✨ Visualization complete! Check the 'pictures/visualized_samples/' folder.")
