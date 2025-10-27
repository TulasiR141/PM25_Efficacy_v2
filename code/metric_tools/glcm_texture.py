#!/usr/bin/env python3
"""
Compute Gray-Level Co-occurrence Matrix (GLCM) texture features
inside spheroid regions detected by spheroid_boundary.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# --- Paths ---
HERE = os.path.dirname(os.path.abspath(__file__))
IN_DIR = os.path.join(HERE, "output")       # same input folder as LBP script
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


def compute_glcm_features(gray, mask):
    """Compute GLCM texture features inside spheroid mask."""
    # Crop to masked region to save compute time
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    roi = gray[y:y+h, x:x+w]
    roi_mask = mask[y:y+h, x:x+w]

    # Apply mask
    roi_masked = cv2.bitwise_and(roi, roi, mask=roi_mask)

    # Reduce gray levels for smaller GLCM (recommended)
    roi_scaled = (roi_masked / 8).astype(np.uint8)  # 0–31 levels

    # Compute GLCM at several angles & distance = 1
    glcm = graycomatrix(roi_scaled, distances=[1],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=32, symmetric=True, normed=True)

    # Extract statistical features averaged across angles
    features = {
        'Contrast': graycoprops(glcm, 'contrast').mean(),
        'Dissimilarity': graycoprops(glcm, 'dissimilarity').mean(),
        'Homogeneity': graycoprops(glcm, 'homogeneity').mean(),
        'Energy': graycoprops(glcm, 'energy').mean(),
        'Correlation': graycoprops(glcm, 'correlation').mean(),
        'ASM': graycoprops(glcm, 'ASM').mean()
    }

    # Optional: compute entropy manually
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    features['Entropy'] = entropy

    return features, glcm


def process_image(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"⚠️  Cannot read {img_path}")
        return None

    mask = extract_spheroid_mask(gray)
    if mask is None:
        print(f"⚠️  No spheroid found in {img_path}")
        return None

    features, glcm = compute_glcm_features(gray, mask)
    base = os.path.splitext(os.path.basename(img_path))[0]

    # --- Visualization (optional) ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gray, cmap='gray')
    plt.imshow(mask, cmap='jet', alpha=0.3)
    plt.title(f"Spheroid Mask – {base}")

    plt.subplot(1, 2, 2)
    plt.imshow(glcm[:, :, 0, 0], cmap='inferno')
    plt.title("GLCM (0° direction)")
    plt.colorbar(label='Co-occurrence probability')
    plt.tight_layout()
    plt.show()

    print(f"✅ Processed {base}")
    return {'Image': base, **features}


def main():
    imgs = [f for f in os.listdir(IN_DIR) if f.lower().endswith(".jpg")]
    if not imgs:
        print(f"⚠️  No images found in {IN_DIR}")
        return

    all_results = []
    for img in imgs:
        result = process_image(os.path.join(IN_DIR, img))
        if result:
            all_results.append(result)

    if all_results:
        df = pd.DataFrame(all_results)
        csv_path = os.path.join(OUT_DIR, "glcm_features.csv")
        df.to_csv(csv_path, index=False)
        print(f"📊 Saved GLCM features → {csv_path}")

    print("🎉 Finished GLCM texture analysis for all spheroids.")


if __name__ == "__main__":
    main()
