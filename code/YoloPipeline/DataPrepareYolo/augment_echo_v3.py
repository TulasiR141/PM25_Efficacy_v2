import os
from pathlib import Path
import random
import shutil

import cv2
import numpy as np

# ============================================================
# CONFIG
# ============================================================
BASE_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = BASE_DIR / "data_new" / "echo_v3_consolidated"
SRC_IMG_DIR = SRC_DIR / "images"
SRC_LBL_DIR = SRC_DIR / "labels"

OUT_DIR = BASE_DIR / "data_new" / "echo_v3_consolidated_augmented"
OUT_IMG_DIR = OUT_DIR / "images"
OUT_LBL_DIR = OUT_DIR / "labels"

P_HFLIP = 0.40
P_VFLIP = 0.40
P_ROT90 = 0.20   # 90° clockwise
P_ROT270 = 0.20  # 270° clockwise

random.seed(42)


# ============================================================
# YOLO SEG LABEL I/O
# ============================================================
def parse_yolo_seg_label_file(path: Path):
    """
    YOLOv8 segmentation line:
      cls xc yc w h x1 y1 x2 y2 ...
    """
    objs = []
    if not path.is_file():
        return objs

    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            poly = []
            if len(parts) > 5:
                coords = [float(v) for v in parts[5:]]
                if len(coords) % 2 != 0:
                    coords = coords[:-1]
                poly = coords

            objs.append([cls, xc, yc, w, h, poly])
    return objs


def write_yolo_seg_label_file(path: Path, objs):
    with path.open("w") as f:
        for cls, xc, yc, w, h, poly in objs:
            parts = [
                str(int(cls)),
                f"{xc:.6f}",
                f"{yc:.6f}",
                f"{w:.6f}",
                f"{h:.6f}",
            ]
            if poly:
                parts.extend(f"{v:.6f}" for v in poly)
            f.write(" ".join(parts) + "\n")


# ============================================================
# POLYGON TRANSFORMS
# ============================================================
def _hflip_coords(poly):
    """Horizontal flip: (x, y) -> (1 - x, y)."""
    if not poly:
        return poly
    new = poly.copy()
    for i in range(0, len(new), 2):
        new[i] = 1.0 - new[i]        # x
    return new


def _vflip_coords(poly):
    """Vertical flip: (x, y) -> (x, 1 - y)."""
    if not poly:
        return poly
    new = poly.copy()
    for i in range(0, len(new), 2):
        new[i + 1] = 1.0 - new[i + 1]   # y
    return new


def _rot90_coords(poly):
    """
    90° clockwise rotation (cv2.ROTATE_90_CLOCKWISE).

    For normalized coords:
      (x, y) -> (1 - y, x)
    """
    if not poly:
        return poly
    new = poly.copy()
    for i in range(0, len(new), 2):
        x = new[i]
        y = new[i + 1]
        new[i]     = 1.0 - y   # x'
        new[i + 1] = x         # y'
    return new


def _rot270_coords(poly):
    """
    270° clockwise / 90° CCW (cv2.ROTATE_90_COUNTERCLOCKWISE).

    For normalized coords:
      (x, y) -> (y, 1 - x)
    """
    if not poly:
        return poly
    new = poly.copy()
    for i in range(0, len(new), 2):
        x = new[i]
        y = new[i + 1]
        new[i]     = y         # x'
        new[i + 1] = 1.0 - x   # y'
    return new


# ============================================================
# OBJECT-LEVEL TRANSFORMS (bbox + polygon)
# ============================================================
def seg_hflip(objs):
    """Horizontal flip: x' = 1 - x, polygons included."""
    flipped = []
    for cls, xc, yc, w, h, poly in objs:
        new_xc = 1.0 - xc
        new_poly = _hflip_coords(poly)
        flipped.append([cls, new_xc, yc, w, h, new_poly])
    return flipped


def seg_vflip(objs):
    """Vertical flip: y' = 1 - y, polygons included."""
    flipped = []
    for cls, xc, yc, w, h, poly in objs:
        new_yc = 1.0 - yc
        new_poly = _vflip_coords(poly)
        flipped.append([cls, xc, new_yc, w, h, new_poly])
    return flipped


def seg_rot90(objs):
    """
    90° clockwise rotation (cv2.ROTATE_90_CLOCKWISE).

    For bbox:
      (xc, yc) -> (1 - yc, xc)
      (w, h)   -> (h, w)
    For polygon:
      (x, y)   -> (1 - y, x)
    """
    rotated = []
    for cls, xc, yc, w, h, poly in objs:
        new_xc = 1.0 - yc
        new_yc = xc
        new_w  = h
        new_h  = w
        new_poly = _rot90_coords(poly)
        rotated.append([cls, new_xc, new_yc, new_w, new_h, new_poly])
    return rotated


def seg_rot270(objs):
    """
    270° clockwise / 90° CCW (cv2.ROTATE_90_COUNTERCLOCKWISE).

    For bbox:
      (xc, yc) -> (yc, 1 - xc)
      (w, h)   -> (h, w)
    For polygon:
      (x, y)   -> (y, 1 - x)
    """
    rotated = []
    for cls, xc, yc, w, h, poly in objs:
        new_xc = yc
        new_yc = 1.0 - xc
        new_w  = h
        new_h  = w
        new_poly = _rot270_coords(poly)
        rotated.append([cls, new_xc, new_yc, new_w, new_h, new_poly])
    return rotated


# ============================================================
# MEAN/STD
# ============================================================
def compute_mean_std():
    img_files = sorted([p for p in SRC_IMG_DIR.iterdir() if p.is_file()])
    if not img_files:
        raise RuntimeError(f"No images found in {SRC_IMG_DIR}")

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    print(f"🔍 Computing mean/std over {len(img_files)} base images...")

    for img_path in img_files:
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠ Could not read image {img_path}, skipping.")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        h, w, c = img.shape
        pixel_count += h * w

        reshaped = img.reshape(-1, 3)
        channel_sum += reshaped.sum(axis=0)
        channel_sq_sum += (reshaped ** 2).sum(axis=0)

    mean = channel_sum / pixel_count
    sq_mean = channel_sq_sum / pixel_count
    std = np.sqrt(sq_mean - mean ** 2)

    print("✅ Mean (RGB):", mean)
    print("✅ Std  (RGB):", std)
    print()

    return mean, std


# ============================================================
# AUGMENTATION LOOP
# ============================================================
def augment_dataset():
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_LBL_DIR.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in SRC_IMG_DIR.iterdir() if p.is_file()])

    count_base = count_hflip = count_vflip = 0
    count_rot90 = count_rot270 = 0

    print(f"🚀 Starting augmentation from {SRC_IMG_DIR} -> {OUT_IMG_DIR}")
    print(f"Total base images found: {len(img_files)}\n")

    for img_path in img_files:
        stem = img_path.stem
        ext = img_path.suffix
        lbl_path = SRC_LBL_DIR / f"{stem}.txt"

        if not lbl_path.is_file():
            print(f"⚠ No label for image {img_path.name}, skipping.")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"⚠ Could not read image {img_path}, skipping.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        objs = parse_yolo_seg_label_file(lbl_path)

        # Base copy
        out_img_base = OUT_IMG_DIR / f"{stem}{ext}"
        out_lbl_base = OUT_LBL_DIR / f"{stem}.txt"
        cv2.imwrite(str(out_img_base), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        write_yolo_seg_label_file(out_lbl_base, objs)
        count_base += 1

        # Horizontal flip
        if random.random() < P_HFLIP:
            img_h = cv2.flip(img_rgb, 1)
            objs_h = seg_hflip(objs)
            out_img = OUT_IMG_DIR / f"{stem}_hflip{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_hflip.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(img_h, cv2.COLOR_RGB2BGR))
            write_yolo_seg_label_file(out_lbl, objs_h)
            count_hflip += 1

        # Vertical flip
        if random.random() < P_VFLIP:
            img_v = cv2.flip(img_rgb, 0)
            objs_v = seg_vflip(objs)
            out_img = OUT_IMG_DIR / f"{stem}_vflip{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_vflip.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(img_v, cv2.COLOR_RGB2BGR))
            write_yolo_seg_label_file(out_lbl, objs_v)
            count_vflip += 1

        # 90° clockwise
        if random.random() < P_ROT90:
            img_r90 = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            objs_r90 = seg_rot90(objs)
            out_img = OUT_IMG_DIR / f"{stem}_rot90{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_rot90.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(img_r90, cv2.COLOR_RGB2BGR))
            write_yolo_seg_label_file(out_lbl, objs_r90)
            count_rot90 += 1

        # 270° clockwise / 90° CCW
        if random.random() < P_ROT270:
            img_r270 = cv2.rotate(img_rgb, cv2.ROTATE_90_COUNTERCLOCKWISE)
            objs_r270 = seg_rot270(objs)
            out_img = OUT_IMG_DIR / f"{stem}_rot270{ext}"
            out_lbl = OUT_LBL_DIR / f"{stem}_rot270.txt"
            cv2.imwrite(str(out_img), cv2.cvtColor(img_r270, cv2.COLOR_RGB2BGR))
            write_yolo_seg_label_file(out_lbl, objs_r270)
            count_rot270 += 1

    total_out_images = len([p for p in OUT_IMG_DIR.iterdir() if p.is_file()])

    print("\n🎯 Augmentation summary:")
    print(f"  Base images copied:          {count_base}")
    print(f"  Horizontal flips created:    {count_hflip}")
    print(f"  Vertical flips created:      {count_vflip}")
    print(f"  90° rotations created:       {count_rot90}")
    print(f"  270° rotations created:      {count_rot270}")
    total_augmented = count_hflip + count_vflip + count_rot90 + count_rot270
    print(f"  Total augmented images:      {total_augmented}")
    print(f"  Expected total in out dir:   {count_base + total_augmented}")
    print(f"  Actual total in out dir:     {total_out_images}")

    if total_out_images == count_base + total_augmented:
        print("✅ Sanity check passed.")
    else:
        print("❗ Sanity check FAILED: counts don't match, please investigate.")


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("====== STEP 1: Compute mean/std for normalization ======")
    compute_mean_std()

    print("\n====== STEP 2: Augment dataset ======")
    augment_dataset()

    print(f"\n📂 Augmented dataset is in: {OUT_DIR}")
                
