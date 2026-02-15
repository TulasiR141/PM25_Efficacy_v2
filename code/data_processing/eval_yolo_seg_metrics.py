#!/usr/bin/env python3
"""
Evaluate YOLOv8 segmentation model on a YOLO-seg dataset split.

Run from project root, script stored in src_new/.

Outputs:
- eval_outputs/yolo_seg_metrics/run_YYYYMMDD_HHMMSS/metrics_per_image.csv
- eval_outputs/yolo_seg_metrics/run_YYYYMMDD_HHMMSS/metrics_summary.json
- eval_outputs/yolo_seg_metrics/run_YYYYMMDD_HHMMSS/overlays/pred/*.png
- eval_outputs/yolo_seg_metrics/run_YYYYMMDD_HHMMSS/overlays/pred_gt/*.png
- eval_outputs/yolo_seg_metrics/run_YYYYMMDD_HHMMSS/debug/zero_pred.txt

Metrics (per image + summary):
- IoU
- Boundary IoU (dilated boundary overlap; tolerance in pixels)
- Hausdorff distance (pixel units; NaN if one mask is empty)
- Area ratio = GT_area / Pred_area

Important fix:
- Pred mask is constructed from result.masks.xy (polygons in ORIGINAL image pixel coords),
  not from result.masks.data resized (which can be misaligned due to letterboxing/padding).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2

from ultralytics import YOLO
from scipy.spatial.distance import directed_hausdorff


# -----------------------------
# IO helpers
# -----------------------------

def read_image_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img


def maybe_apply_clahe_bgr(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """
    Optional brightness normalization that tends to help when one domain is much darker.
    Applies CLAHE to the L channel in LAB space.
    """
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    l2 = clahe.apply(l)
    lab2 = cv2.merge((l2, a, b))
    out = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    return out


def yolo_seg_txt_to_mask(label_path: Path, h: int, w: int) -> np.ndarray:
    """
    YOLOv8 segmentation label format (per line):
      class x1 y1 x2 y2 ... (normalized polygon coords in [0,1])

    Returns union of all polygons as boolean mask (H,W).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if not label_path.exists():
        return mask.astype(bool)

    with label_path.open("r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines() if ln.strip()]

    for ln in lines:
        parts = ln.split()
        if len(parts) < 3:
            continue

        coords = parts[1:]
        if len(coords) < 6 or (len(coords) % 2 != 0):
            continue

        xy = np.array(coords, dtype=np.float32).reshape(-1, 2)
        xy[:, 0] = np.clip(xy[:, 0] * w, 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1] * h, 0, h - 1)
        pts = xy.astype(np.int32)

        cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


def pred_masks_xy_to_union_mask(result, h: int, w: int) -> np.ndarray:
    """
    Build a single union prediction mask in ORIGINAL image coordinates (H,W)
    using result.masks.xy polygons (already in pixel coordinates).
    This avoids letterbox/padding alignment issues.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    if result.masks is None:
        return mask.astype(bool)

    xy_list = getattr(result.masks, "xy", None)
    if xy_list is None or len(xy_list) == 0:
        return mask.astype(bool)

    for poly in xy_list:
        if poly is None:
            continue
        poly = np.asarray(poly, dtype=np.float32)
        if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2:
            continue

        poly[:, 0] = np.clip(poly[:, 0], 0, w - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, h - 1)
        pts = poly.astype(np.int32)

        cv2.fillPoly(mask, [pts], 1)

    return mask.astype(bool)


# -----------------------------
# Metrics
# -----------------------------

def iou(gt: np.ndarray, pr: np.ndarray) -> float:
    gt = gt.astype(bool)
    pr = pr.astype(bool)
    inter = np.logical_and(gt, pr).sum()
    union = np.logical_or(gt, pr).sum()
    if union == 0:
        return 1.0  # both empty
    return float(inter / union)


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """
    Boundary pixels using morphological gradient.
    """
    m = mask.astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    grad = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0)


def boundary_iou(gt: np.ndarray, pr: np.ndarray, tolerance_px: int = 2) -> float:
    b_gt = mask_boundary(gt)
    b_pr = mask_boundary(pr)

    if not b_gt.any() and not b_pr.any():
        return 1.0

    if tolerance_px <= 0:
        inter = np.logical_and(b_gt, b_pr).sum()
        union = np.logical_or(b_gt, b_pr).sum()
        return float(inter / union) if union > 0 else 1.0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tolerance_px + 1, 2 * tolerance_px + 1))
    d_gt = cv2.dilate(b_gt.astype(np.uint8), k, iterations=1).astype(bool)
    d_pr = cv2.dilate(b_pr.astype(np.uint8), k, iterations=1).astype(bool)

    inter = np.logical_and(d_gt, d_pr).sum()
    union = np.logical_or(d_gt, d_pr).sum()
    if union == 0:
        return 1.0
    return float(inter / union)


def hausdorff_distance_px(gt: np.ndarray, pr: np.ndarray) -> float:
    """
    Symmetric Hausdorff distance (px) between boundaries.
    Returns NaN if either mask has no boundary points.
    """
    b_gt = mask_boundary(gt)
    b_pr = mask_boundary(pr)

    pts_gt = np.column_stack(np.where(b_gt))
    pts_pr = np.column_stack(np.where(b_pr))

    if pts_gt.shape[0] == 0 or pts_pr.shape[0] == 0:
        return float("nan")

    d1 = directed_hausdorff(pts_gt, pts_pr)[0]
    d2 = directed_hausdorff(pts_pr, pts_gt)[0]
    return float(max(d1, d2))


def area_ratio_gt_to_pred(gt: np.ndarray, pr: np.ndarray, eps: float = 1e-9) -> float:
    gt_area = float(gt.astype(bool).sum())
    pr_area = float(pr.astype(bool).sum())
    return float(gt_area / (pr_area + eps))


# -----------------------------
# Visualization
# -----------------------------

def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], alpha: float = 0.35) -> np.ndarray:
    out = image_bgr.copy()
    m = mask.astype(bool)
    if not m.any():
        return out
    overlay = out.copy()
    overlay[m] = color_bgr
    return cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)


def draw_contour(image_bgr: np.ndarray, mask: np.ndarray, color_bgr: Tuple[int, int, int], thickness: int = 2) -> np.ndarray:
    out = image_bgr.copy()
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(out, contours, -1, color_bgr, thickness)
    return out


# -----------------------------
# Summary stats
# -----------------------------

@dataclass
class MetricsRow:
    image: str
    iou: float
    boundary_iou: float
    hausdorff_px: float
    gt_area: int
    pred_area: int
    area_ratio_gt_to_pred: float
    pred_instances: int


def safe_mean(values: List[float]) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else float("nan")


def safe_median(values: List[float]) -> float:
    v = [x for x in values if not (isinstance(x, float) and math.isnan(x))]
    return float(np.median(v)) if v else float("nan")


# -----------------------------
# Weight discovery
# -----------------------------

def find_weights(model_dir: Path) -> Path:
    """
    Find Ultralytics weights under model_dir.

    Priority:
      1) model_dir/weights/best.pt
      2) model_dir/weights/last.pt
      3) recursive best.pt
      4) recursive last.pt
    """
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir does not exist: {model_dir}")

    direct_best = model_dir / "weights" / "best.pt"
    if direct_best.exists():
        return direct_best

    direct_last = model_dir / "weights" / "last.pt"
    if direct_last.exists():
        return direct_last

    best = sorted(model_dir.rglob("best.pt"))
    if best:
        return best[0]

    last = sorted(model_dir.rglob("last.pt"))
    if last:
        return last[0]

    raise FileNotFoundError(
        f"Could not find best.pt/last.pt under {model_dir}. "
        f"Tip: check that {model_dir}/weights exists and contains best.pt."
    )


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model_dir",
        type=str,
        default="yolo_training_output/yolov8n_seg_spheroids_boundary_friendly_768_v1",
        help="Path to the YOLO training run directory (expects weights/best.pt).",
    )
    ap.add_argument(
        "--test_dir",
        type=str,
        default="data/consolidated_yolo_augmented_v1/test",
        help="Path to YOLO split directory with images/ and labels/.",
    )
    ap.add_argument("--imgsz", type=int, default=512, help="Inference image size (recommend: match training imgsz).")
    ap.add_argument("--conf", type=float, default=0.10, help="YOLO confidence threshold (lower for debugging).")
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU threshold.")
    ap.add_argument("--device", type=str, default="", help="Device for ultralytics (e.g. '0' or 'cpu').")
    ap.add_argument("--boundary_tol", type=int, default=2, help="Boundary IoU tolerance (pixels).")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE brightness normalization before inference.")
    ap.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clipLimit.")
    ap.add_argument("--clahe_tile", type=int, default=8, help="CLAHE tileGridSize (one int => square).")
    ap.add_argument(
        "--out_dir",
        type=str,
        default="eval_outputs/yolo_seg_metrics",
        help="Output directory root.",
    )
    ap.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = no limit).")
    args = ap.parse_args()

    root = Path.cwd()
    model_dir = root / args.model_dir
    test_dir = root / args.test_dir
    img_dir = test_dir / "images"
    lbl_dir = test_dir / "labels"

    if not img_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {img_dir}")
    if not lbl_dir.exists():
        raise FileNotFoundError(f"Missing labels dir: {lbl_dir}")

    weights = find_weights(model_dir)
    print(f"[INFO] Using weights: {weights}")

    model = YOLO(str(weights))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = root / args.out_dir / f"run_{ts}"
    pred_overlay_dir = out_root / "overlays" / "pred"
    predgt_overlay_dir = out_root / "overlays" / "pred_gt"
    debug_dir = out_root / "debug"
    out_root.mkdir(parents=True, exist_ok=True)
    pred_overlay_dir.mkdir(parents=True, exist_ok=True)
    predgt_overlay_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted([p for p in img_dir.iterdir()
                          if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]])
    if args.limit and args.limit > 0:
        image_paths = image_paths[: args.limit]

    if not image_paths:
        raise RuntimeError(f"No images found in: {img_dir}")

    rows: List[MetricsRow] = []
    zero_pred: List[str] = []

    for idx, img_path in enumerate(image_paths, start=1):
        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"

        img = read_image_bgr(img_path)
        h, w = img.shape[:2]

        gt = yolo_seg_txt_to_mask(lbl_path, h, w)

        # Optional brightness normalization for inference
        infer_img = img
        if args.clahe:
            infer_img = maybe_apply_clahe_bgr(img, clip_limit=args.clahe_clip, tile_grid=args.clahe_tile)

        # IMPORTANT: pass the array (infer_img), not the path, so we control preprocessing.
        results = model.predict(
            source=infer_img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device if args.device != "" else None,
            verbose=False,
        )
        result = results[0]

        pred_instances = 0 if result.masks is None else len(getattr(result.masks, "xy", []) or [])
        if pred_instances == 0:
            zero_pred.append(img_path.name)

        # Build pred mask in original coordinates from polygons
        pr = pred_masks_xy_to_union_mask(result, h, w)

        m_iou = iou(gt, pr)
        m_biou = boundary_iou(gt, pr, tolerance_px=args.boundary_tol)
        m_hd = hausdorff_distance_px(gt, pr)
        gt_area = int(gt.sum())
        pr_area = int(pr.sum())
        m_ar = area_ratio_gt_to_pred(gt, pr)

        rows.append(
            MetricsRow(
                image=img_path.name,
                iou=m_iou,
                boundary_iou=m_biou,
                hausdorff_px=m_hd,
                gt_area=gt_area,
                pred_area=pr_area,
                area_ratio_gt_to_pred=m_ar,
                pred_instances=int(pred_instances),
            )
        )

        # Overlays (always save)
        pred_vis = overlay_mask(img, pr, color_bgr=(0, 0, 255), alpha=0.35)
        pred_vis = draw_contour(pred_vis, pr, color_bgr=(0, 0, 255), thickness=2)
        cv2.imwrite(str(pred_overlay_dir / f"{stem}.png"), pred_vis)

        both = img.copy()
        both = overlay_mask(both, gt, color_bgr=(0, 255, 0), alpha=0.30)  # GT green
        both = overlay_mask(both, pr, color_bgr=(0, 0, 255), alpha=0.30)  # Pred red
        both = draw_contour(both, gt, color_bgr=(0, 255, 0), thickness=2)
        both = draw_contour(both, pr, color_bgr=(0, 0, 255), thickness=2)
        cv2.imwrite(str(predgt_overlay_dir / f"{stem}.png"), both)

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"[INFO] Processed {idx}/{len(image_paths)} images")

    # Debug: list files with zero predictions
    if zero_pred:
        zp = debug_dir / "zero_pred.txt"
        zp.write_text("\n".join(zero_pred) + "\n", encoding="utf-8")
        print(f"[WARN] {len(zero_pred)} images had 0 predicted instances. See: {zp}")

    # Write CSV
    csv_path = out_root / "metrics_per_image.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))

    # Summary
    ious = [r.iou for r in rows]
    bious = [r.boundary_iou for r in rows]
    hds = [r.hausdorff_px for r in rows]
    ars = [r.area_ratio_gt_to_pred for r in rows]
    preds = [r.pred_instances for r in rows]

    summary = {
        "count": len(rows),
        "model_dir": str(model_dir),
        "weights": str(weights),
        "test_dir": str(test_dir),
        "params": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "device": args.device,
            "boundary_tol_px": args.boundary_tol,
            "clahe": bool(args.clahe),
            "clahe_clip": args.clahe_clip,
            "clahe_tile": args.clahe_tile,
        },
        "predictions": {
            "zero_pred_count": int(sum(1 for p in preds if p == 0)),
            "mean_instances": float(np.mean(preds)),
            "median_instances": float(np.median(preds)),
        },
        "mean": {
            "iou": safe_mean(ious),
            "boundary_iou": safe_mean(bious),
            "hausdorff_px": safe_mean(hds),
            "area_ratio_gt_to_pred": safe_mean(ars),
        },
        "median": {
            "iou": safe_median(ious),
            "boundary_iou": safe_median(bious),
            "hausdorff_px": safe_median(hds),
            "area_ratio_gt_to_pred": safe_median(ars),
        },
    }

    summary_path = out_root / "metrics_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[DONE]")
    print(f"CSV:      {csv_path}")
    print(f"Summary:  {summary_path}")
    print(f"Overlays (pred):    {pred_overlay_dir}")
    print(f"Overlays (pred+gt): {predgt_overlay_dir}")


if __name__ == "__main__":
    main()
