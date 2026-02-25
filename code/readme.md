# PM2.5 Spheroid Segmentation Code Guide

This README describes the current code pipeline from data setup to model training and post-treatment comparison outputs.

## 1) Final Pipeline (Data -> Training -> Before/After Comparison)

### A) Prepare YOLO training data

1. Put your source YOLO data in split format (images + labels).
2. Consolidate/clean/augment/split using the `code/src` pipeline:
   - `python code/src/consolidate_echo_v3.py`
   - `python code/src/check_image_labels_consistency.py`
   - `python code/src/clean_unmatched_labels.py`
   - `python code/src/augment_echo_v3.py`
   - `python code/src/split_and_fix_yolo.py`
3. Confirm output split dataset exists (with `data.yaml`) for training.

### B) Train YOLO segmentation model
=======

4. Train with:
   - `python code/models/yolo_train.py`
5. Training output:
   - `yolo_training_output/<run_name>/weights/best.pt`

### C) Evaluate segmentation quality on labeled split

6. Run segmentation metric evaluation:
   - `python code/data_processing/eval_yolo_seg_metrics.py --help`
7. Outputs include per-image metrics, summary JSON, and overlays under:
   - `eval_outputs/yolo_seg_metrics/...`

### D) Run before vs after treatment comparison (core analysis step)

8. Run paired Control/Treated inference + feature extraction:
   - `python code/data_processing/infer_yolo_extract_features_and_save_comparisons.py`
9. This script performs:
   - YOLO inference on Control and Treated images
   - mask cleanup/refinement
   - shape metrics (area, perimeter, circularity)
   - texture metrics (GLCM)
   - side-by-side comparison image generation
10. Main outputs:
   - feature CSV (one row per image per condition)
   - side-by-side comparison PNGs for each matched filename pair

## 2) Data Setup Requirements

Most scripts use hard-coded `CONFIG` paths. Update each script before running.

### Required YOLO label format

For most scripts:
- `class x1 y1 x2 y2 ...` (normalized polygon coordinates)

Some `code/src` scripts can parse extended variants, but keep one consistent format per dataset.

### Expected paired comparison structure

For `infer_yolo_extract_features_and_save_comparisons.py`, your filenames must match between:

```text
<DATA_ROOT>/Control/
<DATA_ROOT>/Treated/
```

If `Control/img001.png` exists, `Treated/img001.png` must also exist.

## 3) Environment Setup

From repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install ultralytics torch torchvision albumentations opencv-python numpy matplotlib tqdm scikit-learn scikit-image shapely scipy pillow pyyaml pandas tabulate
```

## 4) Current File-by-File Purpose Map (Matches Existing Directory)

### `code/data_processing/`

- `code/data_processing/analyze_preprocessing_relevance_multi_yolo.py`: Dataset-level preprocessing relevance analysis (resize/normalization/blur/class stats).
- `code/data_processing/augment_orientation_with_originals.py`: Orientation augmentation while updating YOLO polygons.
- `code/data_processing/check_unet_masks.py`: Prints unique class values found in mask files.
- `code/data_processing/compare_flipped_resized_to_raw_data.py`: Visual comparison of raw vs augmented images with polygon overlays.
- `code/data_processing/compare_yolo_v_unet_masks.py`: Side-by-side comparison of YOLO polygons and U-Net masks.
- `code/data_processing/consolidate_yolo_spheroids.py`: Consolidates datasets and retains spheroid class only.
- `code/data_processing/convert_masks_to_binary.py`: Converts multi-class masks to binary masks.
- `code/data_processing/convert_yolo_to_unet.py`: Converts YOLO polygon annotations into U-Net-style mask dataset.
- `code/data_processing/count_dataset_images.py`: Counts images by split and compares with consolidated totals.
- `code/data_processing/eval_yolo_seg_metrics.py`: Evaluates YOLO segmentation using IoU, boundary IoU, Hausdorff, area ratio; saves CSV/JSON/overlays.
- `code/data_processing/fix_yolo_label_classes.py`: Forces YOLO class IDs to a single class.
- `code/data_processing/generate_data_inventory.py`: Builds dataset inventory and quality metadata.
- `code/data_processing/infer_yolo_extract_features_and_save_comparisons.py`: Paired Control vs Treated inference, feature extraction, and side-by-side comparison export.
- `code/data_processing/prepare_data.py`: Merges train/valid/test image-mask pairs into unified training/validation structure.
- `code/data_processing/resize_images.py`: Resizes images with padding and updates polygon coordinates.
- `code/data_processing/split_unet_dataset.py`: Train/val/test split creation for U-Net-style data.
- `code/data_processing/split_yolo_dataset.py`: Train/val/test split creation for YOLO-style data.
- `code/data_processing/visualize_yolo_segmentation.py`: Random sample visualization of YOLO segmentation labels.
- `code/data_processing/visually_confirm_unet_masks.py`: Manual visualization check for selected U-Net mask/image pairs.

### `code/src/`

- `code/src/augment_echo_v3.py`: Augments consolidated YOLO data while preserving polygon labels.
- `code/src/bring_augmented_base_images.py`: Ensures base/original images are present in augmented dataset output.
- `code/src/check_image_labels_consistency.py`: Verifies image-label filename consistency.
- `code/src/clean_unmatched_labels.py`: Removes labels with no matching images.
- `code/src/consolidate_echo_v3.py`: Consolidates split data into one pool.
- `code/src/eval_yolo_seg_metrics.py`: Alternate YOLO segmentation evaluation script with overlay/report outputs.
- `code/src/split_and_fix_yolo.py`: Group-aware split generation and class-id fixing.
- `code/src/visualize_polygons.py`: Polygon spot-check visualization tool.
- `code/src/yolo_to_unet.py`: Converts YOLO splits into U-Net image-mask splits.

### `code/models/`

- `code/models/eval_yolo.py`: Evaluates trained YOLO predictions against YOLO-label-derived masks with boundary-focused metrics.
- `code/models/yolo_train.py`: Main YOLOv8 segmentation training script currently used in this repository.

### `code/metric_tools/`

- `code/metric_tools/boundary_spheriod.py`: Detects spheroid boundary and computes geometric descriptors.
- `code/metric_tools/glcm_texture.py`: GLCM texture feature extraction from spheroid regions.
- `code/metric_tools/lbp_texture.py`: LBP texture feature extraction from spheroid regions.

## 5) Notes

- Run from repo root: `/home/riki/PM25_Efficacy`.
- Many scripts are still path-configured internally; edit their config blocks before running.
- For the final drug-treatment comparison stage, keep filename pairing between Control and Treated folders exact.


#### Project title:Spheroid Metrics

#### Team members:

Tulasi Rajgopal
Richard Van Winkle
Sara Prasla


## Description
Since this project involves experimenting with multiple pipelines, each following its own structure, dependencies, and processing steps, we have provided a dedicated README file within each pipeline’s folder. These README files contain detailed installation instructions, environment setup steps, and usage guidelines tailored specifically to that pipeline.


