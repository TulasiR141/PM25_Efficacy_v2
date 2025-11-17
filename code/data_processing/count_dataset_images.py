"""
count_dataset_images.py

Counts the number of images in train/valid/test for each dataset
and compares it to the number of consolidated images.
"""

import os
import glob
from tabulate import tabulate  # nice formatted table, pip install tabulate if needed

# ===========================================================
# CONFIG
# ===========================================================
BASE_DIR = "pictures"
DATASETS = [
    os.path.join(BASE_DIR, "Echo.v1i.yolov8"),
    os.path.join(BASE_DIR, "EchoSpheroids.v1i.yolov8"),
]
CONSOLIDATED_DIR = os.path.join(BASE_DIR, "consolidated_yolo", "test", "images")


# ===========================================================
# HELPER FUNCTIONS
# ===========================================================
def count_images_in_split(dataset_path, split):
    """Count .jpg or .png images in split/images directory."""
    img_dir = os.path.join(dataset_path, split, "images")
    if not os.path.exists(img_dir):
        return 0
    count = len(glob.glob(os.path.join(img_dir, "*.jpg"))) + len(
        glob.glob(os.path.join(img_dir, "*.png"))
    )
    return count


def count_dataset_images(dataset_path):
    """Return dict of counts for train/valid/test."""
    counts = {}
    total = 0
    for split in ["train", "valid", "test"]:
        n = count_images_in_split(dataset_path, split)
        counts[split] = n
        total += n
    counts["total"] = total
    return counts


# ===========================================================
# MAIN LOGIC
# ===========================================================
summary = []

for dataset in DATASETS:
    name = os.path.basename(dataset)
    counts = count_dataset_images(dataset)
    summary.append(
        [
            name,
            counts["train"],
            counts["valid"],
            counts["test"],
            counts["total"],
        ]
    )

# count consolidated images
consolidated_count = len(glob.glob(os.path.join(CONSOLIDATED_DIR, "*.jpg"))) + len(
    glob.glob(os.path.join(CONSOLIDATED_DIR, "*.png"))
)

# print results
headers = ["Dataset", "Train", "Valid", "Test", "Total"]
print("\n📊 Image Count per Dataset:\n")
print(tabulate(summary, headers=headers, tablefmt="fancy_grid"))

print("\n📦 Consolidated Directory:")
print(f" -> {CONSOLIDATED_DIR}")
print(f" -> Total consolidated images: {consolidated_count}")

# calculate total from originals
total_original = sum(row[4] for row in summary)
if total_original > 0:
    ratio = (consolidated_count / total_original) * 100
    print(f"\n✅ Consolidated dataset contains {ratio:.2f}% of the original total.")
else:
    print("\n⚠️ No original images found — please check dataset paths.")

print("\nDone ✅")
