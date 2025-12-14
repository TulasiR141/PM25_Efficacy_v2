import os
import shutil

# Base paths relative to project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data_new", "echo_v3")

CONSOLIDATED_DIR = os.path.join(BASE_DIR, "data_new", "echo_v3_consolidated")
IMG_OUT = os.path.join(CONSOLIDATED_DIR, "images")
LBL_OUT = os.path.join(CONSOLIDATED_DIR, "labels")

# Create output directories
os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

splits = ["train", "valid", "test"]

count_images = 0
count_labels = 0

print(f"🔍 Starting consolidation from: {DATA_DIR}")

for split in splits:
    img_dir = os.path.join(DATA_DIR, split, "images")
    lbl_dir = os.path.join(DATA_DIR, split, "labels")

    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"⚠ Missing directory in '{split}', skipping.")
        continue

    print(f"📁 Processing split: {split}")

    for filename in os.listdir(img_dir):
        img_path = os.path.join(img_dir, filename)

        if not os.path.isfile(img_path):
            continue

        stem = os.path.splitext(filename)[0]
        label_path = os.path.join(lbl_dir, stem + ".txt")

        # Only copy if label exists
        if os.path.isfile(label_path):
            shutil.copy2(img_path, IMG_OUT)
            shutil.copy2(label_path, LBL_OUT)

            count_images += 1
            count_labels += 1
        else:
            print(f"⚠ No label for: {filename} (skipping)")

print("\n✅ Consolidation complete!")
print(f"📸 Total images moved: {count_images}")
print(f"📝 Total labels moved: {count_labels}")
print(f"📂 Output located at: {CONSOLIDATED_DIR}")
                                                         
