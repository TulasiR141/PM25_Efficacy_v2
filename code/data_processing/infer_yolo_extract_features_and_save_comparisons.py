#!/usr/bin/env python3
"""
infer_yolo_extract_features_and_save_comparisons.py

Paired Control vs Treated pipeline:
  1) Brightness-normalize images BEFORE YOLO inference (robust; auto mode)
  2) Run YOLOv8 segmentation
  3) Optional boundary-friendly refinement (morph + DT)
  4) Compute shape metrics (area, perimeter, circularity)
  5) Compute scalar GLCM texture metrics inside predicted mask
  6) Save CSV (one row per image per condition)
  7) Save side-by-side PNGs with boundary overlays + metrics

Adds guard rails:
  - If prediction "explodes" (mask too large / too border-touching), retry with safer preprocessing
  - If still bad, mark mask invalid for downstream comparisons
"""

from __future__ import annotations

from pathlib import Path
import math
import csv
from typing import Dict, Tuple, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage.feature import graycomatrix, graycoprops


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = Path("~/thws/semester2/project/data/b_a_cropped_gray_sorted").expanduser()
CONTROL_DIR = DATA_ROOT / "Control"
TREATED_DIR = DATA_ROOT / "Treated"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

OUT_CSV = Path("~/thws/semester2/project/data/spheroid_features_control_treated.csv").expanduser()
COMPARISON_DIR = Path("~/thws/semester2/project/data/spheroid_comparisons").expanduser()

# ✅ UPDATE TO YOUR LATEST MODEL
YOLO_WEIGHTS = Path(
    "~/thws/semester2/project/yolo_training_output/"
    "yolov8n_seg_spheroids_boundary_friendly_768_v1/weights/best.pt"
).expanduser()

# Inference settings
IMG_SIZE = 768          # match your new training; set to 512 if GPU can’t handle
CONF = 0.10             # you found 0.1 works well
IOU_NMS = 0.70
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Mask postprocess
KEEP_LARGEST_COMPONENT = True
MIN_MASK_AREA_PX = 300
ERODE_INNER_PX_FOR_GLCM = 3

# Optional refinement (recommended if you used it in eval)
USE_REFINEMENT = True
REFINE_CLOSE_K = 5          # close kernel size
REFINE_DT_THRESH = 1.5      # distance-transform threshold in pixels

# Guard rails / retries to prevent “exploding” masks
MAX_MASK_FRAC = 0.60        # if mask covers > 60% of image, it’s suspicious → retry
MAX_BORDER_TOUCH_FRAC = 0.20  # if >20% of boundary pixels touch image border → retry
RETRY_CONF = 0.20           # stricter conf for retry
RETRY_NORM_MODE = "clahe"   # safer fallback normalization

# Normalization / preprocessing
# Modes: "auto", "percentile", "clahe", "none"
NORM_MODE = "auto"

# Percentile stretch (milder than 2–98; reduces “blowouts”)
NORM_P_LO = 5
NORM_P_HI = 95
NORM_MIN_RANGE = 15.0       # if (hi-lo) too small, skip stretching

# CLAHE params
CLAHE_CLIP = 2.0
CLAHE_TILE = 8

# Optional gamma correction (set None to disable; or use "auto")
GAMMA = "auto"              # None | float | "auto"
AUTO_GAMMA_DARK = 0.85      # brighten dark images slightly
AUTO_GAMMA_BRIGHT = 1.10    # tame very bright images slightly

# Light denoise before YOLO (helps with harsh backgrounds)
PRE_BLUR_SIGMA = 0.0        # 0.0 disables; try 0.5–1.0 if needed

# GLCM
GLCM_LEVELS = 32
GLCM_DISTANCES = [1, 2, 4]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Visualization
BOUNDARY_COLOR_RGB = (0, 255, 0)
BOUNDARY_THICKNESS = 2


# ============================================================
# HELPERS
# ============================================================

def list_images(d: Path) -> dict[str, Path]:
    return {
        p.name: p
        for p in d.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    }


def _apply_gamma(gray_u8: np.ndarray, gamma: float) -> np.ndarray:
    # gamma < 1 brightens; gamma > 1 darkens
    x = gray_u8.astype(np.float32) / 255.0
    y = np.power(x, gamma)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)


def normalize_percentile(gray_u8: np.ndarray,
                         p_lo: float,
                         p_hi: float,
                         min_range: float = 10.0) -> np.ndarray:
    lo, hi = np.percentile(gray_u8, (p_lo, p_hi))
    if (hi - lo) < min_range:
        return gray_u8.copy()
    out = (gray_u8.astype(np.float32) - float(lo)) * 255.0 / float(hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize_clahe(gray_u8: np.ndarray,
                    clip: float,
                    tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    return clahe.apply(gray_u8)


def auto_normalize(gray_u8: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Auto chooses a safer normalization:
      - If very low contrast → CLAHE
      - Else mild percentile stretch
    """
    mean = float(gray_u8.mean())
    std = float(gray_u8.std())

    # Low contrast / washed background → CLAHE helps local contrast without blowing out globally
    if std < 25.0:
        return normalize_clahe(gray_u8, CLAHE_CLIP, CLAHE_TILE), "clahe"

    # Otherwise percentile is fine (but mild)
    return normalize_percentile(gray_u8, NORM_P_LO, NORM_P_HI, NORM_MIN_RANGE), "percentile"


def preprocess_for_yolo(gray_u8: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """
    Returns (preprocessed_gray, mode_used).
    """
    if mode == "none":
        g = gray_u8.copy()
        mode_used = "none"
    elif mode == "percentile":
        g = normalize_percentile(gray_u8, NORM_P_LO, NORM_P_HI, NORM_MIN_RANGE)
        mode_used = "percentile"
    elif mode == "clahe":
        g = normalize_clahe(gray_u8, CLAHE_CLIP, CLAHE_TILE)
        mode_used = "clahe"
    elif mode == "auto":
        g, mode_used = auto_normalize(gray_u8)
    else:
        raise ValueError(f"Unknown NORM_MODE: {mode}")

    # Optional gamma (helps match training intensity distribution)
    if GAMMA is None:
        return g, mode_used

    if GAMMA == "auto":
        m = float(g.mean())
        # Heuristic: if still dark → brighten; if very bright → slightly compress
        if m < 90:
            g = _apply_gamma(g, AUTO_GAMMA_DARK)
            mode_used += "+gamma(dark)"
        elif m > 170:
            g = _apply_gamma(g, AUTO_GAMMA_BRIGHT)
            mode_used += "+gamma(bright)"
    else:
        g = _apply_gamma(g, float(GAMMA))
        mode_used += f"+gamma({GAMMA})"

    # Optional light blur to reduce background artifacts
    if PRE_BLUR_SIGMA and PRE_BLUR_SIGMA > 0:
        k = int(max(3, round(PRE_BLUR_SIGMA * 6) | 1))  # odd kernel
        g = cv2.GaussianBlur(g, (k, k), PRE_BLUR_SIGMA)
        mode_used += f"+blur({PRE_BLUR_SIGMA})"

    return g, mode_used


def yolo_predict_semantic_mask(model: YOLO, img_bgr: np.ndarray,
                               imgsz: int, conf: float, iou_nms: float) -> np.ndarray:
    results = model.predict(
        img_bgr, imgsz=imgsz, conf=conf, iou=iou_nms, device=DEVICE, verbose=False
    )
    r = results[0]
    h, w = img_bgr.shape[:2]

    if r.masks is None or r.masks.data is None:
        return np.zeros((h, w), dtype=np.uint8)

    pm = r.masks.data  # [N,H,W] (torch)
    sem = (pm.sigmoid().max(dim=0)[0] > 0.5).to(torch.uint8)
    sem_np = sem.detach().cpu().numpy().astype(np.uint8)

    if sem_np.shape != (h, w):
        sem_np = cv2.resize(sem_np, (w, h), interpolation=cv2.INTER_NEAREST)
    return sem_np


def keep_largest_component(mask01: np.ndarray) -> np.ndarray:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask01, 8)
    if num <= 1:
        return mask01
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == idx).astype(np.uint8)


def refine_mask(mask01: np.ndarray) -> np.ndarray:
    """
    Boundary-friendly refinement similar to your eval idea:
      - Close small gaps
      - Fill holes (via close + floodfill)
      - DT threshold to smooth boundary slightly
      - Keep largest component (optional)
    """
    if not USE_REFINEMENT or mask01.sum() == 0:
        return mask01

    m = mask01.astype(np.uint8)

    # Close
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (REFINE_CLOSE_K, REFINE_CLOSE_K))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)

    # Fill holes (flood fill background, invert)
    h, w = m.shape
    ff = m.copy()
    mask_ff = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, mask_ff, seedPoint=(0, 0), newVal=1)  # fill bg with 1
    holes = (ff == 0).astype(np.uint8)  # holes were 0 after bg fill
    m = np.clip(m + holes, 0, 1).astype(np.uint8)

    # Distance transform smoothing (shrinks away hairline artifacts)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    m = (dist > float(REFINE_DT_THRESH)).astype(np.uint8)

    if KEEP_LARGEST_COMPONENT:
        m = keep_largest_component(m)

    return m


def mask_border_touch_fraction(mask01: np.ndarray) -> float:
    """
    Fraction of boundary pixels that lie on the image border.
    Exploding masks often "stick" to borders.
    """
    if mask01.sum() == 0:
        return 0.0
    m = (mask01.astype(np.uint8) * 255)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    b = (grad > 0).astype(np.uint8)

    if b.sum() == 0:
        return 0.0

    border = np.zeros_like(b)
    border[0, :] = 1
    border[-1, :] = 1
    border[:, 0] = 1
    border[:, -1] = 1

    touch = (b & border).sum()
    return float(touch) / float(b.sum())


def is_mask_suspicious(mask01: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    h, w = mask01.shape
    frac = float(mask01.sum()) / float(h * w)
    bt = mask_border_touch_fraction(mask01)
    suspicious = (frac > MAX_MASK_FRAC) or (bt > MAX_BORDER_TOUCH_FRAC)
    return suspicious, {"mask_frac": frac, "border_touch_frac": bt}


def compute_shape_metrics(mask01: np.ndarray) -> dict:
    area = int(mask01.sum())
    if area == 0:
        return {"area_px": 0, "perimeter_px": 0.0, "circularity": 0.0}

    cnts, _ = cv2.findContours(mask01, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return {"area_px": area, "perimeter_px": 0.0, "circularity": 0.0}

    c = max(cnts, key=cv2.contourArea)
    p = float(cv2.arcLength(c, True))
    circ = (4 * math.pi * area) / (p * p) if p > 1e-6 else 0.0
    return {"area_px": area, "perimeter_px": p, "circularity": circ}


def quantize_to_levels(gray: np.ndarray, levels: int) -> np.ndarray:
    q = (gray.astype(np.uint16) * levels) // 256
    return np.clip(q, 0, levels - 1).astype(np.uint8)


def glcm_entropy(glcm: np.ndarray) -> float:
    eps = 1e-12
    P = glcm.astype(np.float64)
    return float((-(P * np.log2(P + eps))).sum(axis=(0, 1)).mean())


def compute_glcm_metrics(gray: np.ndarray, mask: np.ndarray) -> dict:
    if mask.sum() < MIN_MASK_AREA_PX:
        return {k: 0.0 for k in [
            "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
            "glcm_energy", "glcm_ASM", "glcm_correlation", "glcm_entropy"
        ]}

    m = mask.copy()
    if ERODE_INNER_PX_FOR_GLCM > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ERODE_INNER_PX_FOR_GLCM + 1, 2 * ERODE_INNER_PX_FOR_GLCM + 1)
        )
        m = cv2.erode(m, k, 1)

    if m.sum() < MIN_MASK_AREA_PX:
        return {k: 0.0 for k in [
            "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
            "glcm_energy", "glcm_ASM", "glcm_correlation", "glcm_entropy"
        ]}

    ys, xs = np.where(m > 0)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()

    roi = gray[y0:y1 + 1, x0:x1 + 1].copy()
    roi[m[y0:y1 + 1, x0:x1 + 1] == 0] = 0

    q = quantize_to_levels(roi, GLCM_LEVELS)

    glcm = graycomatrix(
        q, GLCM_DISTANCES, GLCM_ANGLES,
        levels=GLCM_LEVELS, symmetric=True, normed=True
    )

    return {
        "glcm_contrast": float(graycoprops(glcm, "contrast").mean()),
        "glcm_dissimilarity": float(graycoprops(glcm, "dissimilarity").mean()),
        "glcm_homogeneity": float(graycoprops(glcm, "homogeneity").mean()),
        "glcm_energy": float(graycoprops(glcm, "energy").mean()),
        "glcm_ASM": float(graycoprops(glcm, "ASM").mean()),
        "glcm_correlation": float(graycoprops(glcm, "correlation").mean()),
        "glcm_entropy": glcm_entropy(glcm),
    }


def overlay_boundary(gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        cv2.drawContours(
            rgb,
            [max(cnts, key=cv2.contourArea)],
            -1,
            BOUNDARY_COLOR_RGB,
            BOUNDARY_THICKNESS
        )
    return rgb


def save_side_by_side(out_path: Path, title: str,
                      img_c, img_t, feats_c, feats_t,
                      meta_c: str, meta_t: str):
    fig = plt.figure(figsize=(12, 6))

    for i, (img, feats, label, meta) in enumerate(
        [(img_c, feats_c, "Control", meta_c), (img_t, feats_t, "Treated", meta_t)]
    ):
        ax = fig.add_subplot(1, 2, i + 1)
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(label)

        ax.text(
            0.01, -0.18,
            f"Area(px):        {feats['area_px']}\n"
            f"Perimeter(px):   {feats['perimeter_px']:.1f}\n"
            f"Circularity:     {feats['circularity']:.3f}\n"
            f"GLCM Energy:     {feats['glcm_energy']:.3f}\n"
            f"GLCM Entropy:    {feats['glcm_entropy']:.3f}\n"
            f"GLCM Contrast:   {feats['glcm_contrast']:.3f}\n"
            f"MaskValid:       {feats['mask_nonempty']}\n"
            f"{meta}",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace"
        )

    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.08, 1, 0.92])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def process_one(model: YOLO, img_path: Path) -> Tuple[dict, np.ndarray, str]:
    """
    Returns: (features, vis_rgb, meta_string)
    """
    g0 = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if g0 is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")

    # Pass 1
    g1, mode_used = preprocess_for_yolo(g0, NORM_MODE)
    bgr1 = cv2.cvtColor(g1, cv2.COLOR_GRAY2BGR)
    m1 = yolo_predict_semantic_mask(model, bgr1, IMG_SIZE, CONF, IOU_NMS)

    if KEEP_LARGEST_COMPONENT:
        m1 = keep_largest_component(m1)

    m1 = refine_mask(m1)

    suspicious, stats = is_mask_suspicious(m1)

    # Retry if suspicious
    retry_used = False
    if suspicious:
        retry_used = True
        g2, mode2 = preprocess_for_yolo(g0, RETRY_NORM_MODE)
        bgr2 = cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)
        m2 = yolo_predict_semantic_mask(model, bgr2, IMG_SIZE, RETRY_CONF, IOU_NMS)

        if KEEP_LARGEST_COMPONENT:
            m2 = keep_largest_component(m2)

        m2 = refine_mask(m2)

        suspicious2, stats2 = is_mask_suspicious(m2)
        # Keep whichever is less suspicious (or smaller mask frac)
        if (not suspicious2) or (stats2["mask_frac"] < stats["mask_frac"]):
            g1, mode_used = g2, mode2 + f"+conf({RETRY_CONF})"
            m1, stats = m2, stats2
            suspicious = suspicious2

    # Final validity
    valid = int(m1.sum() >= MIN_MASK_AREA_PX and (not suspicious))

    feats = compute_shape_metrics(m1) | compute_glcm_metrics(g1, m1)
    feats["mask_area_px"] = int(m1.sum())
    feats["mask_nonempty"] = valid

    vis = overlay_boundary(g1, m1)

    meta = (
        f"Preproc: {mode_used}\n"
        f"MaskFrac: {stats['mask_frac']:.3f}\n"
        f"BorderTouch: {stats['border_touch_frac']:.3f}\n"
        f"Retried: {int(retry_used)}"
    )

    return feats, vis, meta


# ============================================================
# MAIN
# ============================================================

def main():
    control = list_images(CONTROL_DIR)
    treated = list_images(TREATED_DIR)
    common = sorted(set(control) & set(treated))

    if not common:
        raise RuntimeError("No matching filenames between Control/ and Treated/.")

    if not YOLO_WEIGHTS.exists():
        raise FileNotFoundError(f"YOLO weights not found: {YOLO_WEIGHTS}")

    model = YOLO(str(YOLO_WEIGHTS))

    fields = [
        "filename", "condition",
        "area_px", "perimeter_px", "circularity",
        "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity",
        "glcm_energy", "glcm_ASM", "glcm_correlation", "glcm_entropy",
        "mask_area_px", "mask_nonempty",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name in common:
            stem = Path(name).stem

            fc, vc, meta_c = process_one(model, control[name])
            ft, vt, meta_t = process_one(model, treated[name])

            writer.writerow({"filename": name, "condition": "Control", **fc})
            writer.writerow({"filename": name, "condition": "Treated", **ft})

            save_side_by_side(
                COMPARISON_DIR / f"{stem}.png",
                stem,
                vc, vt,
                fc, ft,
                meta_c, meta_t
            )

            print(f"[OK] {stem}")

    print("DONE")


if __name__ == "__main__":
    main()
