"""
Detect the spheroid boundary and draw it on each image in ../pictures/.
Outputs are saved to ./output as JPG overlays.
"""

import os
import cv2
import numpy as np

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(os.path.dirname(HERE), "pictures")
OUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUT_DIR, exist_ok=True)

def detect_spheroid_contour(gray: np.ndarray):
    """Return the largest external contour after Otsu-threshold segmentation."""
    # 1) Noise suppression (keeps edges but smooths small specks)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2) Otsu threshold (auto picks the cutoff)
    _, bin_img = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If foreground came out white but is actually background, invert
    # (choose the polarity that makes the darker region = foreground)
    if np.mean(gray[bin_img == 255]) > np.mean(gray[bin_img == 0]):
        bin_img = cv2.bitwise_not(bin_img)

    # 3) Morphology: fill small holes, remove tiny noise, smooth edge
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    # 4) Keep the largest connected component (assumed spheroid)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, bin_img
    contour = max(contours, key=cv2.contourArea)

    return contour, bin_img

def process_image(path):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"⚠️  Could not read: {path}")
        return

    contour, mask = detect_spheroid_contour(gray)
    if contour is None:
        print(f"⚠️  No contour found in: {os.path.basename(path)}")
        return

    # Draw contour on the original grayscale (convert to BGR for color overlay)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    # Save
    base = os.path.splitext(os.path.basename(path))[0]
    out_overlay = os.path.join(OUT_DIR, f"{base}_contour.jpg")
    cv2.imwrite(out_overlay, overlay)

    print(f"✅ Saved: {out_overlay}")

def main():
    files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(".jpg")]
    if not files:
        print(f"⚠️  No .jpg images found in {IMG_DIR}")
        return
    for f in sorted(files):
        process_image(os.path.join(IMG_DIR, f))
    print("🎉 Done.")

if __name__ == "__main__":
    main()
