#!/usr/bin/env python3
"""
Compute Local Binary Pattern (LBP) texture features
inside spheroid regions detected by spheroid_boundary.py
"""

import os
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "output")  # outputs from spheroid_boundary.py
OUT_DIR = os.path.join(HERE, "texture_output")
os.makedirs(OUT_DIR, exist_ok=True)


def extract_spheroid_mask(gray):
    """Generate a binary mask for the spheroid region."""
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def compute_lbp_features(gray, mask, P=8, R=3):
    """Compute LBP histogram inside the spheroid mask."""
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    spheroid_pixels = lbp[mask == 255]
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(spheroid_pixels, bins=n_bins, range=(0, n_bins), density=True)
    return hist, lbp


def process_image(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"⚠️  Cannot read {img_path}")
        return

    mask = extract_spheroid_mask(gray)
    if mask is None:
        print(f"⚠️  No spheroid found in {img_path}")
        return

    hist, lbp_img = compute_lbp_features(gray, mask)

    # --- Visualization ---
    lbp_norm = (lbp_img / lbp_img.max() * 255).astype(np.uint8)
    lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_JET)
    lbp_color[mask == 0] = 0
    base = os.path.splitext(os.path.basename(img_path))[0]
    out_img = os.path.join(OUT_DIR, f"{base}_lbp.jpg")
    cv2.imwrite(out_img, lbp_color)

    # Plot histogram
    plt.figure(figsize=(6, 3))
    plt.bar(range(len(hist)), hist, color='gray')
    plt.title(f"LBP Histogram – {base}")
    plt.xlabel("LBP Code")
    plt.ylabel("Normalized Frequency")
    plt.tight_layout()
    plt.show()

    print(f"✅ Processed {base}")
    print(f"   Saved LBP map → {out_img}")


def main():
    imgs = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".jpg")]
    if not imgs:
        print(f"⚠️  No images found in {IN_DIR}")
        return
    for img in imgs:
        process_image(os.path.join(IN_DIR, img))
    print("🎉 Finished computing LBP features for all spheroids.")


if __name__ == "__main__":
    main()
