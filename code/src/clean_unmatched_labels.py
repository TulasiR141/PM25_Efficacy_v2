import os

# ------------------------------------------------------------
# Paths (relative to project root)
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONSOLIDATED_DIR = os.path.join(BASE_DIR, "data_new", "echo_v3_consolidated")
IMG_DIR = os.path.join(CONSOLIDATED_DIR, "images")
LBL_DIR = os.path.join(CONSOLIDATED_DIR, "labels")

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def main():
    if not os.path.isdir(IMG_DIR):
        print(f"❌ Image directory not found: {IMG_DIR}")
        return
    if not os.path.isdir(LBL_DIR):
        print(f"❌ Label directory not found: {LBL_DIR}")
        return

    print(f"🖼  Images dir: {IMG_DIR}")
    print(f"📝 Labels dir: {LBL_DIR}\n")

    # --------------------------------------------------------
    # 1) Collect all image stems (filenames without extension)
    # --------------------------------------------------------
    image_stems = set()

    for fname in os.listdir(IMG_DIR):
        img_path = os.path.join(IMG_DIR, fname)
        if not os.path.isfile(img_path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in ALLOWED_IMG_EXTS:
            # skip weird files in images dir
            continue

        image_stems.add(stem)

    print(f"📸 Found {len(image_stems)} image stems.\n")

    # --------------------------------------------------------
    # 2) Iterate labels and delete those with no matching image
    # --------------------------------------------------------
    removed = 0
    kept = 0

    for fname in os.listdir(LBL_DIR):
        lbl_path = os.path.join(LBL_DIR, fname)
        if not os.path.isfile(lbl_path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            # skip non-.txt files in labels dir
            print(f"⚠ Non-txt file in labels dir, skipping: {fname}")
            continue

        if stem not in image_stems:
            print(f"🗑  Removing label without image: {fname}")
            os.remove(lbl_path)
            removed += 1
        else:
            kept += 1

    print("\n✅ Cleanup complete.")
    print(f"   Labels kept:    {kept}")
    print(f"   Labels removed: {removed}")
    print(f"   Final label count should match images (or be very close).")


if __name__ == "__main__":
    main()

     
