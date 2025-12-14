
import random
from pathlib import Path
import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

IMG_DIR = BASE_DIR / "data_new" / "echo_v3_consolidated_augmented" / "images"
LBL_DIR = BASE_DIR / "data_new" / "echo_v3_consolidated_augmented" / "labels"

OUT_DIR = BASE_DIR / "polygon_spotcheck"
OUT_DIR.mkdir(exist_ok=True)


# ============================================================
# PARSE YOLO SEG LABEL
# ============================================================
def load_yolo_seg_label(path):
    objs = []
    text = path.read_text().strip()
    if not text:
        return objs

    for line in text.splitlines():
        parts = line.split()
        cls = int(float(parts[0]))
        xc = float(parts[1])
        yc = float(parts[2])
        w  = float(parts[3])
        h  = float(parts[4])
        poly = [float(x) for x in parts[5:]] if len(parts) > 5 else []
        objs.append((cls, xc, yc, w, h, poly))
    return objs


# ============================================================
# DRAW POLYGON ON IMAGE
# ============================================================
def draw_polygon_on_image(img, poly_norm):
    """
    img = RGB image loaded by cv2
    poly_norm = [x1, y1, x2, y2, ...] normalized coords in YOLO format
    """
    h, w, _ = img.shape

    pts = []
    for i in range(0, len(poly_norm), 2):
        x = int(poly_norm[i] * w)
        y = int(poly_norm[i+1] * h)
        pts.append([x, y])

    pts = np.array(pts, dtype=np.int32)

    if len(pts) >= 3:
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return img


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    all_imgs = sorted(list(IMG_DIR.glob("*.jpg")) + list(IMG_DIR.glob("*.png")))

    if len(all_imgs) < 25:
        print("⚠ Less than 25 images found — sampling all.")
        sample = all_imgs
    else:
        sample = random.sample(all_imgs, 25)

    print(f"🔍 Spot-checking {len(sample)} augmented images...")

    for img_path in sample:
        stem = img_path.stem
        lbl_path = LBL_DIR / f"{stem}.txt"

        if not lbl_path.is_file():
            print(f"⚠ Missing label for {img_path.name}, skipping.")
            continue

        # Load & convert image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        objs = load_yolo_seg_label(lbl_path)

        # Draw all polygons in this image
        for cls, xc, yc, w, h, poly in objs:
            if poly:
                img = draw_polygon_on_image(img, poly)

        # Save output visualization
        out_path = OUT_DIR / f"{stem}_viz.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"   ✔ Saved: {out_path.name}")

    print(f"\n📂 Visualization output saved in: {OUT_DIR}")
    print("👀 Open the PNG files to verify polygon alignment.")
                                                                  
