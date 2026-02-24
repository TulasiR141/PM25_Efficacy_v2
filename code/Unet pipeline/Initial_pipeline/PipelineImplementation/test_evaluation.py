from pathlib import Path
import numpy as np
import torch
from fastai.vision.core import PILImage, PILMask
from fastai.data.block import DataBlock
from fastai.data.transforms import FuncSplitter, Normalize, get_image_files
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.augment import Resize, ResizeMethod
from fastai.metrics import Dice, JaccardCoeff
from fastai.optimizer import Adam
from semtorch import get_segmentation_learner
import cv2
import time

# Configuration
PATH_TO_DATASET = Path('./data/organized')
CODES = np.array(['Background', 'Spheroid'])
MODEL_NAME = 'unet_resnet34_spheroid'
RESIZE_FACTOR = 0.5
BATCH_SIZE = 4

print("="*70)
print("🧪 TEST SET EVALUATION")
print("="*70)


class CombinedLossCE:
    """Dice and CrossEntropyFlat combined"""
    def __init__(self, axis=1, smooth=1., alpha=1.):
        from fastai.vision.core import store_attr, CrossEntropyLossFlat, DiceLoss, F
        store_attr()
        self.ce_loss = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis)
        self.axis = axis

    def __call__(self, pred, targ):
        return self.ce_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)
    
    def activation(self, x):
        import torch.nn.functional as F
        return F.softmax(x, dim=self.axis)


def get_mask(img_path):
    """Get corresponding mask path"""
    img_path = Path(img_path)
    mask_path = PATH_TO_DATASET / 'masks' / img_path.parent.name / img_path.name
    return mask_path


def train_val_splitter(path):
    """Split based on folder structure"""
    return 'val' not in Path(path).parts


def accuracy_spheroid(input, target):
    """Calculate pixel-wise accuracy"""
    target = target.squeeze(1)
    return (input.argmax(dim=1) == target).float().mean()


def calculate_metrics(pred, target):
    """Calculate Dice, IoU, and Accuracy for a single image"""
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    
    # Dice Score
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
    
    # IoU (Jaccard)
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    
    # Accuracy
    accuracy = (pred == target).sum() / target.size
    
    return dice, iou, accuracy


def evaluate_test_set():
    """
    SIMPLE EXPLANATION:
    1. Load the trained model
    2. Load all test images one by one
    3. Predict masks for each test image
    4. Compare predictions with ground truth masks
    5. Calculate metrics (Dice, IoU, Accuracy)
    6. Save visualizations
    """
    
    print(f"\n📊 Step 1: Loading trained model '{MODEL_NAME}'...")
    
    # Get image size
    first_train_img = list((PATH_TO_DATASET / 'images' / 'train').glob('*.png'))[0]
    img_size_original = PILImage.create(first_train_img).shape
    img_size = (int(img_size_original[0] * RESIZE_FACTOR), 
                int(img_size_original[1] * RESIZE_FACTOR))
    print(f"  Image size: {img_size}")
    
    # Create dataloader (needed to load model structure)
    datablock = DataBlock(
        blocks=(ImageBlock, MaskBlock(CODES)),
        get_items=get_image_files,
        get_y=get_mask,
        splitter=FuncSplitter(train_val_splitter),
        item_tfms=Resize(img_size, ResizeMethod.Squish),
        batch_tfms=[Normalize.from_stats(*imagenet_stats)]
    )
    
    dls = datablock.dataloaders(PATH_TO_DATASET / 'images', bs=BATCH_SIZE)
    
    # Initialize model
    learn = get_segmentation_learner(
        dls=dls,
        number_classes=2,
        segmentation_type="Semantic Segmentation",
        architecture_name='unet',
        backbone_name='resnet34',
        loss_func=CombinedLossCE(alpha=1.0),
        opt_func=Adam,
        metrics=[Dice(), JaccardCoeff(), accuracy_spheroid],
        pretrained=False
    ).to_fp16()
    
    # Load trained weights
    learn.load(MODEL_NAME)
    print(f"  ✅ Model loaded successfully")
    
    # Get test images
    test_img_dir = PATH_TO_DATASET / 'images' / 'test'
    test_mask_dir = PATH_TO_DATASET / 'masks' / 'test'
    test_images = sorted(test_img_dir.glob('*.png'))
    
    print(f"\n📂 Step 2: Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("❌ No test images found!")
        return None
    
    # Run inference
    print(f"\n🚀 Step 3: Running inference...")
    start_time = time.time()
    
    all_dice = []
    all_iou = []
    all_accuracy = []
    predictions_data = []
    
    learn.model.eval()
    
    with torch.no_grad():
        for idx, img_path in enumerate(test_images):
            # Load image and mask
            img_pil = PILImage.create(img_path)
            mask_path = test_mask_dir / img_path.name
            
            if not mask_path.exists():
                print(f"  ⚠️  Warning: Mask not found for {img_path.name}")
                continue
            
            mask_pil = PILMask.create(mask_path)
            
            # Get original size
            original_size = img_pil.size  # (width, height)
            
            # Predict
            pred_class, pred_idx, pred_probs = learn.predict(img_pil)
            
            # Convert to numpy
            pred = np.array(pred_class)
            target = np.array(mask_pil)
            image = np.array(img_pil)
            
            # Resize prediction to match original mask size if needed
            if pred.shape != target.shape:
                pred = cv2.resize(pred.astype(np.uint8), 
                                (target.shape[1], target.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)
            
            # Calculate metrics
            dice, iou, accuracy = calculate_metrics(pred, target)
            
            all_dice.append(dice)
            all_iou.append(iou)
            all_accuracy.append(accuracy)
            
            # Store for visualization
            predictions_data.append({
                'image': image,
                'pred': pred,
                'target': target,
                'dice': dice,
                'iou': iou,
                'accuracy': accuracy,
                'filename': img_path.name
            })
            
            if (idx + 1) % 10 == 0 or (idx + 1) == len(test_images):
                print(f"  Processed {idx + 1}/{len(test_images)} images...")
    
    inference_time = time.time() - start_time
    
    print(f"\n⏱️  Total inference time: {inference_time:.2f} seconds")
    print(f"⏱️  Average per image: {inference_time/len(test_images):.3f} seconds")
    
    # Calculate statistics
    print(f"\n📊 Step 4: Computing statistics...")
    
    results = {
        'num_samples': len(test_images),
        'inference_time': inference_time,
        'avg_time_per_image': inference_time / len(test_images),
        'dice_mean': np.mean(all_dice),
        'dice_std': np.std(all_dice),
        'dice_min': np.min(all_dice),
        'dice_max': np.max(all_dice),
        'iou_mean': np.mean(all_iou),
        'iou_std': np.std(all_iou),
        'iou_min': np.min(all_iou),
        'iou_max': np.max(all_iou),
        'accuracy_mean': np.mean(all_accuracy),
        'accuracy_std': np.std(all_accuracy),
        'accuracy_min': np.min(all_accuracy),
        'accuracy_max': np.max(all_accuracy),
    }
    
    # Print results
    print("\n" + "="*70)
    print("📋 TEST SET RESULTS")
    print("="*70)
    print(f"\n📊 Overall Metrics:")
    print(f"  Number of samples:     {results['num_samples']}")
    print(f"  Total inference time:  {results['inference_time']:.2f}s")
    print(f"  Avg time per image:    {results['avg_time_per_image']:.3f}s")
    print(f"\n🎯 Dice Score:")
    print(f"  Mean:  {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"  Range: [{results['dice_min']:.4f}, {results['dice_max']:.4f}]")
    print(f"\n🎯 IoU (Jaccard):")
    print(f"  Mean:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print(f"  Range: [{results['iou_min']:.4f}, {results['iou_max']:.4f}]")
    print(f"\n🎯 Pixel Accuracy:")
    print(f"  Mean:  {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
    print(f"  Range: [{results['accuracy_min']:.4f}, {results['accuracy_max']:.4f}]")
    print("="*70)
    
    # Save visualizations
    print(f"\n💾 Step 5: Saving visualizations...")
    save_visualizations(predictions_data)
    
    # Save results to file
    results_file = Path(f'test_results_{MODEL_NAME}.txt')
    with open(results_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TEST SET EVALUATION RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {MODEL_NAME}\n")
        f.write(f"Test images: {results['num_samples']}\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"  ✅ Results saved to: {results_file}")
    
    return results


def save_visualizations(predictions_data):
    """Save prediction visualizations with beautiful comparisons"""
    output_dir = Path('test_predictions')
    output_dir.mkdir(exist_ok=True)
    
    for idx, data in enumerate(predictions_data):
        image = data['image']
        pred = data['pred']
        target = data['target']
        dice = data['dice']
        iou = data['iou']
        
        # Ensure RGB for visualization
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Ensure image matches mask size
        if image.shape[:2] != target.shape:
            image = cv2.resize(image, (target.shape[1], target.shape[0]))
        
        # Create visualizations
        h, w = target.shape
        
        # 1. Original image
        img_original = image.copy()
        
        # 2. Ground truth mask (green overlay)
        img_gt = image.copy()
        gt_overlay = np.zeros_like(img_gt)
        gt_overlay[:, :, 1] = target * 255  # Green channel
        img_gt = cv2.addWeighted(img_gt, 0.7, gt_overlay, 0.3, 0)
        
        # 3. Prediction mask (blue overlay)
        img_pred = image.copy()
        pred_overlay = np.zeros_like(img_pred)
        pred_overlay[:, :, 0] = pred * 255  # Blue channel
        img_pred = cv2.addWeighted(img_pred, 0.7, pred_overlay, 0.3, 0)
        
        # 4. Both overlays (green=GT, blue=Pred, purple=overlap)
        img_both = image.copy()
        both_overlay = np.zeros_like(img_both)
        both_overlay[:, :, 0] = pred * 255      # Blue for prediction
        both_overlay[:, :, 1] = target * 255    # Green for ground truth
        img_both = cv2.addWeighted(img_both, 0.6, both_overlay, 0.4, 0)
        
        # 5. Contour overlay
        img_contours = image.copy()
        target_contours, _ = cv2.findContours(target.astype(np.uint8), 
                                             cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred.astype(np.uint8), 
                                           cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img_contours, target_contours, -1, (0, 255, 0), 2)  # Green = GT
        cv2.drawContours(img_contours, pred_contours, -1, (255, 0, 0), 2)    # Blue = Pred
        
        # Add metrics text to contour image
        text = f"Dice: {dice:.3f} | IoU: {iou:.3f}"
        cv2.putText(img_contours, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Add labels
        cv2.putText(img_gt, "Ground Truth", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_pred, "Prediction", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_both, "Overlay (Green=GT, Blue=Pred)", (10, h-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Create comparison grid (2x2)
        top_row = np.hstack([img_original, img_contours])
        bottom_row = np.hstack([img_gt, img_pred])
        grid = np.vstack([top_row, bottom_row])
        
        # Save individual files
        cv2.imwrite(str(output_dir / f'test_{idx:03d}_grid.png'), grid)
        cv2.imwrite(str(output_dir / f'test_{idx:03d}_overlay.png'), img_both)
        cv2.imwrite(str(output_dir / f'test_{idx:03d}_contours.png'), img_contours)
        cv2.imwrite(str(output_dir / f'test_{idx:03d}_pred_mask.png'), pred * 255)
        cv2.imwrite(str(output_dir / f'test_{idx:03d}_gt_mask.png'), target * 255)
    
    print(f"  ✅ Saved {len(predictions_data)} visualizations to '{output_dir}/'")
    print(f"\n  📁 Files saved per test image:")
    print(f"     - *_grid.png: 2x2 comparison (Original, Contours, GT, Prediction)")
    print(f"     - *_overlay.png: Combined overlay (Green=GT, Blue=Pred, Purple=Overlap)")
    print(f"     - *_contours.png: Contour visualization with metrics")
    print(f"     - *_pred_mask.png: Binary prediction mask")
    print(f"     - *_gt_mask.png: Binary ground truth mask")


if __name__ == "__main__":
    print("\n🎯 What this script does:")
    print("  1. Loads your trained model")
    print("  2. Runs it on test images (unseen data)")
    print("  3. Compares predictions with ground truth")
    print("  4. Calculates performance metrics")
    print("  5. Saves visual results\n")
    
    if not (PATH_TO_DATASET / 'images' / 'test').exists():
        print(f"❌ Error: Test set not found at {PATH_TO_DATASET / 'images' / 'test'}")
    else:
        results = evaluate_test_set()
        
        if results:
            print("\n✨ Test evaluation completed!")
            print(f"\n💡 Summary: Your model achieved {results['dice_mean']:.2%} Dice score on test data")