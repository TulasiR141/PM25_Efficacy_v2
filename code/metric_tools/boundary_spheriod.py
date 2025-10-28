"""
Detect the spheroid boundary and draw it on each image in ../pictures/.
Also computes area, perimeter, and circularity of each detected spheroid.
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

# --- Optional: calibration (set your real-world scale here) ---
# Example: 1 pixel = 0.5 micrometers → scale = 0.5
# Leave as 1.0 if you just want results in pixels
SCALE = 1.0


def detect_spheroid_contour(gray: np.ndarray):
    """Return the largest external contour after Otsu-threshold segmentation."""
    # 1) Noise suppression (keeps edges but smooths small specks)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # 2) Otsu threshold (auto picks the cutoff)
    _, bin_img = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # If foreground came out white but is actually background, invert
    if np.mean(gray[bin_img == 255]) > np.mean(gray[bin_img == 0]):
        bin_img = cv2.bitwise_not(bin_img)

    # 3) Morphology: fill small holes, remove tiny noise, smooth edge
    bin_img = cv2.morphologyEx(
        bin_img, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    bin_img = cv2.morphologyEx(
        bin_img, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    )

    # 4) Keep the largest connected component (assumed spheroid)
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, bin_img
    contour = max(contours, key=cv2.contourArea)

    return contour, bin_img


def check_and_load_image(path):
    """Check if the image is grayscale or RGB, and convert only if needed."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"⚠️  Could not read: {path}")
        return None, None

    # Determine the number of channels
    if len(img.shape) == 2:
        # Single channel (already grayscale)
        print(f"🟢 {os.path.basename(path)} is grayscale.")
        gray = img

    elif len(img.shape) == 3:
        h, w, c = img.shape
        if c == 3:
            b, g, r = cv2.split(img)
            # Check if all channels are identical
            if (b == g).all() and (g == r).all():
                print(f"🟢 {os.path.basename(path)} is RGB but identical across channels (effectively grayscale).")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                print(f"🎨 {os.path.basename(path)} is true RGB (color image) — converting to grayscale for analysis.")
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif c == 4:
            print(f"🎨 {os.path.basename(path)} has 4 channels (RGBA) — converting to grayscale.")
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            print(f"⚠️  {os.path.basename(path)} has unexpected channel count: {c}")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        print(f"⚠️  Unknown image format: {os.path.basename(path)}")
        gray = None

    return gray, img


def process_image(path):
    gray, original = check_and_load_image(path)
    if gray is None:
        return

    contour, mask = detect_spheroid_contour(gray)
    if contour is None:
        print(f"⚠️  No contour found in: {os.path.basename(path)}")
        return

    # --- Compute area & perimeter ---
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)

    # Convert to real-world units (if scale != 1.0)
    area_scaled = area * (SCALE ** 2)
    perimeter_scaled = perimeter * SCALE

    # --- Compute circularity ---
    circularity = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0

    print(
        f"   • Area: {area_scaled:.2f} {'µm²' if SCALE != 1.0 else 'px²'}\n"
        f"   • Perimeter: {perimeter_scaled:.2f} {'µm' if SCALE != 1.0 else 'px'}\n"
        f"   • Circularity: {circularity:.3f}\n"
    )

    # --- Draw contour overlay ---
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    # --- Save result ---
    base = os.path.splitext(os.path.basename(path))[0]
    out_overlay = os.path.join(OUT_DIR, f"{base}_contour.jpg")
    cv2.imwrite(out_overlay, overlay)
    print(f"✅ Saved: {out_overlay}\n")


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
