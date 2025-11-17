import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# CONFIG
# =========================================================
RAW_IMG_DIR = "pictures/consolidated_yolo/test/images"
RAW_LABEL_DIR = "pictures/consolidated_yolo/test/labels"

AUG_IMG_DIR = "pictures/augmented_final/images"
AUG_LABEL_DIR = "pictures/augmented_final/labels"

OUTPUT_DIR = "comparison_output"
NUM_SAMPLES = 20

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# LOAD POLYGONS
# =========================================================
def load_polygons(label_path, w, h):
    polys = []
    if not os.path.exists(label_path):
        return polys

    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) > 3:
                coords = list(map(float, parts[1:]))
                pts = np.array([(coords[i] * w, coords[i+1] * h)
                                for i in range(0, len(coords), 2)], dtype=np.int32)
                polys.append(pts)
    return polys


# =========================================================
# DRAW POLYGONS
# =========================================================
def draw_polygons(img, polys, color=(0, 255, 0)):
    out = img.copy()
    for pts in polys:
        cv2.polylines(out, [pts], True, color, 2)
    return out


# =========================================================
# FIND VARIANTS (robust matching)
# =========================================================
def find_variants(base):
    variants = []
    for fname in os.listdir(AUG_IMG_DIR):
        if fname.lower().startswith(base.lower()):
            variants.append(fname)
    return variants


# =========================================================
# MAIN
# =========================================================
def main():
    raw_files = [f for f in os.listdir(RAW_IMG_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    if len(raw_files) == 0:
        print("❌ No raw images found!")
        return

    sampled = random.sample(raw_files, min(NUM_SAMPLES, len(raw_files)))
    print(f"🖼 Selected {len(sampled)} images for comparison.\n")

    for img_name in sampled:
        base = os.path.splitext(img_name)[0]

        # Load RAW
        raw_img_path = os.path.join(RAW_IMG_DIR, img_name)
        raw_label_path = os.path.join(RAW_LABEL_DIR, base + ".txt")

        raw_img = cv2.imread(raw_img_path)
        if raw_img is None:
            print(f"⚠️ Failed to load RAW image: {img_name}")
            continue

        h, w = raw_img.shape[:2]
        raw_polys = load_polygons(raw_label_path, w, h)
        raw_vis = cv2.cvtColor(draw_polygons(raw_img, raw_polys), cv2.COLOR_BGR2RGB)

        # Find AUG variants
        variants = find_variants(base)
        if len(variants) == 0:
            print(f"⚠️ No augmented versions found for: {base}")
            continue

        print(f"🔍 Found {len(variants)} variants for {img_name}")

        for aug in variants:
            aug_path = os.path.join(AUG_IMG_DIR, aug)
            aug_label_path = os.path.join(AUG_LABEL_DIR, aug.replace(".jpg", ".txt"))

            aug_img = cv2.imread(aug_path)
            if aug_img is None:
                continue

            ah, aw = aug_img.shape[:2]
            aug_polys = load_polygons(aug_label_path, aw, ah)
            aug_vis = cv2.cvtColor(draw_polygons(aug_img, aug_polys, (255, 0, 0)), cv2.COLOR_BGR2RGB)

            # Save figure
            plt.figure(figsize=(14, 6))
            plt.suptitle(f"{img_name}  vs  {aug}", fontsize=14)

            plt.subplot(1, 2, 1)
            plt.title("RAW")
            plt.imshow(raw_vis)
            plt.axis("off")

            plt.subplot(1, 2, 2)
            plt.title("AUGMENTED")
            plt.imshow(aug_vis)
            plt.axis("off")

            out_path = os.path.join(OUTPUT_DIR, f"{base}__{aug}.png")
            plt.savefig(out_path, dpi=140)
            plt.close()

    print(f"\n✅ Comparison complete! Files saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
