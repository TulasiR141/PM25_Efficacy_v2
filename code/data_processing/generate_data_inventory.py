import os
import cv2
import csv
import hashlib
from pathlib import Path
from datetime import datetime
import yaml

# === Paths ===
project_root = Path(__file__).resolve().parents[1]
pictures_dir = project_root / "pictures"
datasets = {
    "Echo": pictures_dir / "Echo.v1i.yolov8" / "train" / "images",
    "EchoSpheroids": pictures_dir / "EchoSpheroids.v1i.yolov8" / "train" / "images",
}

# === Output folder ===
output_dir = project_root / "pictures" / "data_inventory"
output_dir.mkdir(exist_ok=True)

# === Helper functions ===
def file_hash(filepath):
    """Compute MD5 hash of an image for duplicate detection."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_class_info(dataset_dir):
    """Get class info from data.yaml in dataset folder."""
    yaml_path = dataset_dir.parent.parent / "data.yaml"
    if not yaml_path.exists():
        return None, None
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("nc"), data.get("names")

def inspect_image(img_path):
    """Check image quality and extract info."""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return "Unreadable", None, None
        h, w = img.shape[:2]
        return "OK", w, h
    except Exception as e:
        return f"Error: {e}", None, None

# === Collect hashes to detect duplicates ===
print("🔍 Calculating image hashes...")
all_hashes = {}
for dataset_name, img_dir in datasets.items():
    for img_file in os.listdir(img_dir):
        if img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
            path = img_dir / img_file
            h = file_hash(path)
            if h in all_hashes:
                all_hashes[h].append((dataset_name, img_file))
            else:
                all_hashes[h] = [(dataset_name, img_file)]

# === Function to process one dataset ===
def process_dataset(dataset_name, img_dir):
    print(f"📁 Processing dataset: {dataset_name}")
    rows = []
    dataset_dir = img_dir.parent.parent  # one level up (train/)
    nc, names = load_class_info(dataset_dir)

    for img_file in os.listdir(img_dir):
        if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".tiff", ".bmp")):
            continue
        path = img_dir / img_file
        fmt = path.suffix.lower().replace(".", "").upper()

        quality, width, height = inspect_image(path)
        img_hash = file_hash(path)

        # Check uniqueness
        duplicates = [d for d in all_hashes[img_hash] if d[0] != dataset_name]
        unique_flag = "Unique" if not duplicates else f"Duplicate of {duplicates[0][0]}:{duplicates[0][1]}"

        rows.append({
            "Dataset": dataset_name,
            "Image Filename": img_file,
            "Format": fmt,
            "Resolution": f"{width}x{height}" if width and height else "Unknown",
            "Quality Notes": quality,
            "Treatment Condition (if known)": "",
            "Date Taken (if known)": "",
            "Unique": unique_flag,
            "Num Classes": nc if nc is not None else "Unknown",
            "Class Names": ", ".join(names) if names else "Unknown",
            "Timestamp Analyzed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        })

    # Write CSV
    csv_path = output_dir / f"Data_Inventory_{dataset_name}.csv"
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Saved inventory: {csv_path} ({len(rows)} images)")

# === Run for both datasets ===
for name, path in datasets.items():
    process_dataset(name, path)

print("\n✨ Data inventory generation complete!")
print(f"Check your CSVs in: {output_dir}")
