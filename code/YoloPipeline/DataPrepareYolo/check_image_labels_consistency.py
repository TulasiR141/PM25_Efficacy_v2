import os

# ------------------------------------------------------------
# Paths (relative to project root)
# ------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

CONSOLIDATED_DIR = os.path.join(BASE_DIR, "data_new", "echo_v3_consolidated")
IMG_DIR = os.path.join(CONSOLIDATED_DIR, "images")
LBL_DIR = os.path.join(CONSOLIDATED_DIR, "labels")

ALLOWED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def collect_image_stems():
    stems = set()
    bad_files = []

    for fname in os.listdir(IMG_DIR):
        path = os.path.join(IMG_DIR, fname)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() not in ALLOWED_IMG_EXTS:
            bad_files.append(fname)
            continue

        stems.add(stem)

    return stems, bad_files


def collect_label_stems():
    stems = set()
    bad_files = []

    for fname in os.listdir(LBL_DIR):
        path = os.path.join(LBL_DIR, fname)
        if not os.path.isfile(path):
            continue

        stem, ext = os.path.splitext(fname)
        if ext.lower() != ".txt":
            bad_files.append(fname)
            continue

        stems.add(stem)

    return stems, bad_files


def main():
    if not os.path.isdir(IMG_DIR):
        print(f"❌ Image directory not found: {IMG_DIR}")
        return
    if not os.path.isdir(LBL_DIR):
        print(f"❌ Label directory not found: {LBL_DIR}")
        return

    print(f"🖼  Images dir: {IMG_DIR}")
    print(f"📝 Labels dir: {LBL_DIR}\n")

    image_stems, bad_img_files = collect_image_stems()
    label_stems, bad_lbl_files = collect_label_stems()

    print(f"📸 # valid image files: {len(image_stems)}")
    print(f"📝 # valid label files: {len(label_stems)}\n")

    if bad_img_files:
        print("⚠ Non-image or weird files in images/ (ignored):")
        for f in bad_img_files:
            print(f"   - {f}")
        print()

    if bad_lbl_files:
        print("⚠ Non-.txt or weird files in labels/ (ignored):")
        for f in bad_lbl_files:
            print(f"   - {f}")
        print()

    # --------------------------------------------------------
    # 1) Images missing labels
    # --------------------------------------------------------
    imgs_missing_labels = sorted(image_stems - label_stems)
    # 2) Labels missing images
    labels_missing_images = sorted(label_stems - image_stems)

    if not imgs_missing_labels and not labels_missing_images:
        print("✅ Perfect match: every image has a label and every label has an image.")
    else:
        print("❗ Inconsistencies detected:\n")

        if imgs_missing_labels:
            print(f"🖼 Images without labels: {len(imgs_missing_labels)}")
            for stem in imgs_missing_labels[:50]:  # show up to 50
                print(f"   - {stem}")
            if len(imgs_missing_labels) > 50:
                print(f"   ... and {len(imgs_missing_labels) - 50} more")
            print()

        if labels_missing_images:
            print(f"📝 Labels without images: {len(labels_missing_images)}")
            for stem in labels_missing_images[:50]:  # show up to 50
                print(f"   - {stem}")
            if len(labels_missing_images) > 50:
                print(f"   ... and {len(labels_missing_images) - 50} more")
            print()

    print("\n📊 Summary:")
    print(f"   Images (valid): {len(image_stems)}")
    print(f"   Labels (valid): {len(label_stems)}")
    print(f"   Images without labels: {len(imgs_missing_labels)}")
    print(f"   Labels without images: {len(labels_missing_images)}")


if __name__ == "__main__":
    main()
                
