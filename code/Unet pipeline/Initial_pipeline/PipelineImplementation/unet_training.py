from pathlib import Path
import random
import numpy as np
import torch
from torch import nn, backends
from fastai.data.block import DataBlock
from fastai.data.transforms import FuncSplitter, Normalize, get_image_files
from fastai.metrics import Dice, JaccardCoeff
from fastai.callback.tracker import EarlyStoppingCallback, SaveModelCallback
from fastai.torch_core import set_seed
from fastai.vision.core import PILImage, PILMask, DiceLoss, CrossEntropyLossFlat, FocalLossFlat, store_attr, F
from fastai.vision.data import ImageBlock, MaskBlock, imagenet_stats
from fastai.vision.augment import Resize, aug_transforms, ResizeMethod
from fastai.optimizer import ranger, Adam
from matplotlib import pyplot as plt
from semtorch import get_segmentation_learner
import cv2
from tabulate import tabulate
import time

# Set random seeds for reproducibility
SEED = 2021
set_seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
backends.cudnn.deterministic = True
backends.cudnn.benchmark = False

# Paths to organized dataset
PATH_TO_DATASET = Path('./data/organized')

# Class codes for segmentation
CODES = np.array(['Background', 'Spheroid'])
background_code = np.where(CODES == 'Background')[0][0]

print("="*70)
print("🔬 UNet Training Pipeline for Spheroid Segmentation")
print("="*70)
print(f"Dataset path: {PATH_TO_DATASET}")
print(f"Classes: {CODES}")
print(f"Random seed: {SEED}")
print("="*70)


def get_mask(img_path):
    """Get corresponding mask path for an image"""
    # Convert from images/split to masks/split
    img_path = Path(img_path)
    mask_path = PATH_TO_DATASET / 'masks' / img_path.parent.name / img_path.name
    return mask_path


def train_val_splitter(path):
    """Split data based on folder structure (train/val)"""
    return 'val' not in Path(path).parts


class CombinedLossCE:
    """Dice and CrossEntropyFlat combined"""
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.ce_loss = CrossEntropyLossFlat(axis=axis)
        self.dice_loss = DiceLoss(axis)

    def __call__(self, pred, targ):
        return self.ce_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)
    
    def activation(self, x):
        return F.softmax(x, dim=self.axis)


class CombinedLossFocal:
    """Dice and FocalLossFlat combined"""
    def __init__(self, axis=1, smooth=1., alpha=1.):
        store_attr()
        self.fo_loss = FocalLossFlat(axis=axis, gamma=2.0)
        self.dice_loss = DiceLoss(axis)

    def __call__(self, pred, targ):
        return self.fo_loss(pred, targ) + self.alpha * self.dice_loss(pred, targ)

    def decodes(self, x):
        return x.argmax(dim=self.axis)
    
    def activation(self, x):
        return F.softmax(x, dim=self.axis)


def accuracy_spheroid(input, target):
    """Calculate pixel-wise accuracy"""
    target = target.squeeze(1)
    return (input.argmax(dim=1) == target).float().mean()


def custom_jaccard(inputs, targets):
    """Calculate Jaccard index (IoU)"""
    jccs = 0
    count = 0
    for input, target in zip(inputs, targets):
        pred = input.argmax(dim=0)
        intersection = (pred * target).sum()
        union = (pred + target).sum() - intersection
        jccs += intersection / union
        count += 1
    return jccs / count


def get_image_size(resize_factor=1.0):
    """Get image dimensions from dataset"""
    path_to_images = PATH_TO_DATASET / 'images' / 'train'
    image_path = get_image_files(path_to_images)[0]
    image = PILImage.create(image_path)
    size = image.shape
    
    if resize_factor == 1.0:
        return size
    
    return (int(size[0] * resize_factor), int(size[1] * resize_factor))


def get_stopping_callbacks(model_name, monitor='valid_loss', patience=5):
    """Create early stopping and model saving callbacks"""
    esc = EarlyStoppingCallback(monitor=monitor, patience=patience)
    smc = SaveModelCallback(monitor=monitor, fname=model_name)
    return [esc, smc]


def show_sample_batch(dls):
    """Visualize a batch from dataloader"""
    dls.show_batch(vmin=0, vmax=1, figsize=(14, 6))
    plt.tight_layout()
    plt.savefig('training_batch_sample.png', dpi=150, bbox_inches='tight')
    print("✅ Saved sample batch to 'training_batch_sample.png'")


def get_dataloader(size, bs=8, show_batch=False, use_augmentation=True):
    """Create FastAI dataloader for training"""
    print(f"\n📊 Creating dataloader...")
    print(f"  Image size: {size}")
    print(f"  Batch size: {bs}")
    print(f"  Augmentation: {use_augmentation}")
    
    # Define augmentation transforms
    batch_tfms = [Normalize.from_stats(*imagenet_stats)]
    if use_augmentation:
        batch_tfms = aug_transforms(size=size, min_scale=0.75) + batch_tfms
    
    spheroids = DataBlock(
        blocks=(ImageBlock, MaskBlock(CODES)),
        get_items=get_image_files,
        get_y=get_mask,
        splitter=FuncSplitter(train_val_splitter),
        item_tfms=Resize(size, ResizeMethod.Squish),
        batch_tfms=batch_tfms
    )
    
    dls = spheroids.dataloaders(PATH_TO_DATASET / 'images', bs=bs)
    
    print(f"  ✅ Train batches: {len(dls.train)}")
    print(f"  ✅ Valid batches: {len(dls.valid)}")
    
    if show_batch:
        spheroids.summary(PATH_TO_DATASET / 'images', bs=bs)
        show_sample_batch(dls)
    
    return dls


def find_learning_rate(learn):
    """Find optimal learning rate"""
    print("\n🔍 Finding optimal learning rate...")
    sugg_lr = learn.lr_find()
    print(f"  Suggested LR: {sugg_lr}")
    plt.tight_layout()
    plt.savefig('learning_rate_finder.png', dpi=150, bbox_inches='tight')
    print("  ✅ Saved LR plot to 'learning_rate_finder.png'")
    return sugg_lr[0]


def init_learner(dls, arch_name='unet', backbone='resnet34', pretrained=True, 
                 loss_func=None, opt_func=Adam):
    """Initialize segmentation learner"""
    print(f"\n🏗️  Initializing model...")
    print(f"  Architecture: {arch_name}")
    print(f"  Backbone: {backbone}")
    print(f"  Pretrained: {pretrained}")
    print(f"  Loss function: {loss_func.__class__.__name__ if loss_func else 'Default'}")
    
    learn = get_segmentation_learner(
        dls=dls,
        number_classes=2,
        segmentation_type="Semantic Segmentation",
        architecture_name=arch_name,
        backbone_name=backbone,
        loss_func=loss_func,
        opt_func=opt_func,
        metrics=[Dice(), JaccardCoeff(), accuracy_spheroid],
        pretrained=pretrained
    ).to_fp16()
    
    print("  ✅ Model initialized")
    return learn


def train_model(learn, epochs, model_name, lr=None, monitor='valid_loss'):
    """Train the model"""
    print(f"\n🚀 Starting training...")
    print(f"  Model name: {model_name}")
    print(f"  Epochs: {epochs}")
    
    callbacks = get_stopping_callbacks(model_name, monitor=monitor, patience=5)
    
    if lr is None:
        lr = find_learning_rate(learn)
    
    print(f"  Learning rate: {lr}")
    print("\n" + "="*70)
    
    learn.fit_one_cycle(epochs, lr, cbs=callbacks)
    
    print("\n✅ Training completed!")
    print(f"📁 Model saved as: models/{model_name}.pth")


def validate_model(learn, model_name, save_visualizations=False):
    """Validate trained model and compute metrics"""
    print(f"\n📊 Validating model: {model_name}")
    print("="*70)
    
    # Load trained weights
    learn.load(model_name)
    
    # Get validation dataset
    valid_dataset = learn.dls.valid_ds
    
    # Perform inference
    start_time = time.time()
    val_preds, val_targets, val_preds_decoded = learn.get_preds(with_decoded=True)
    inference_time = time.time() - start_time
    
    print(f"⏱️  Inference time: {inference_time:.2f} seconds")
    print(f"📈 Number of validation samples: {len(valid_dataset)}")
    
    # Calculate metrics
    metric_names = ['valid_loss'] + [item.name for item in learn.metrics.items]
    metric_values = learn.validate()
    
    # Create results table
    results = dict(zip(metric_names, metric_values))
    results['inference_time'] = inference_time
    results['samples'] = len(valid_dataset)
    
    print("\n📋 Validation Results:")
    print("-"*70)
    for name, value in results.items():
        if isinstance(value, (int, float)):
            print(f"  {name:20s}: {value:.4f}")
        else:
            print(f"  {name:20s}: {value}")
    print("="*70)
    
    # Save visualizations
    if save_visualizations:
        save_predictions(valid_dataset, val_preds_decoded, model_name)
    
    return results


def save_predictions(valid_dataset, decoded_preds, model_name, max_samples=10):
    """Save prediction visualizations"""
    print(f"\n💾 Saving prediction visualizations...")
    
    output_dir = Path(f'predictions_{model_name}')
    output_dir.mkdir(exist_ok=True)
    
    for idx, (sample, pred) in enumerate(zip(valid_dataset[:max_samples], decoded_preds[:max_samples])):
        # Resize prediction to match original image size
        pred = cv2.resize(np.array(pred, np.uint8), sample[0].size, 
                         interpolation=cv2.INTER_NEAREST)
        
        # Get image and target
        image = np.array(sample[0])
        target = np.array(sample[1])
        
        # Find contours
        target_contours, _ = cv2.findContours(target, cv2.RETR_EXTERNAL, 
                                             cv2.CHAIN_APPROX_SIMPLE)
        pred_contours, _ = cv2.findContours(pred, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours: green for ground truth, blue for prediction
        vis_image = image.copy()
        cv2.drawContours(vis_image, target_contours, -1, (0, 255, 0), 2)  # Green
        cv2.drawContours(vis_image, pred_contours, -1, (255, 0, 0), 2)    # Blue
        
        # Save
        cv2.imwrite(str(output_dir / f'sample_{idx:03d}_overlay.png'), vis_image)
        cv2.imwrite(str(output_dir / f'sample_{idx:03d}_pred_mask.png'), pred * 255)
    
    print(f"  ✅ Saved {min(max_samples, len(valid_dataset))} predictions to '{output_dir}/'")


def main_training_pipeline():
    """Main training function"""
    print("\n" + "="*70)
    print("🎯 MAIN TRAINING PIPELINE")
    print("="*70)
    
    # Configuration (Optimized for RTX 3050 4GB)
    CONFIG = {
        'model_name': 'unet_resnet34_spheroid',
        'architecture': 'unet',
        'backbone': 'resnet34',
        'pretrained': True,
        'resize_factor': 0.5,  # Resize images to 50% for faster training
        'batch_size': 4,  # Reduced for RTX 3050 4GB VRAM (use 8 if you have headroom)
        'epochs': 20,
        'learning_rate': None,  # Auto-find if None
        'loss_function': CombinedLossCE(alpha=1.0),  # Can also try: CombinedLossFocal()
        'optimizer': Adam,
        'use_augmentation': True,
        'show_batch': True
    }
    
    print("\n📋 Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Get image size
    img_size = get_image_size(CONFIG['resize_factor'])
    
    # Create dataloader
    dls = get_dataloader(
        size=img_size,
        bs=CONFIG['batch_size'],
        show_batch=CONFIG['show_batch'],
        use_augmentation=CONFIG['use_augmentation']
    )
    
    # Initialize model
    learn = init_learner(
        dls=dls,
        arch_name=CONFIG['architecture'],
        backbone=CONFIG['backbone'],
        pretrained=CONFIG['pretrained'],
        loss_func=CONFIG['loss_function'],
        opt_func=CONFIG['optimizer']
    )
    
    # Train model
    train_model(
        learn=learn,
        epochs=CONFIG['epochs'],
        model_name=CONFIG['model_name'],
        lr=CONFIG['learning_rate'],
        monitor='dice'  # Monitor Dice score for early stopping
    )
    
    # Validate model
    results = validate_model(
        learn=learn,
        model_name=CONFIG['model_name'],
        save_visualizations=True
    )
    
    print("\n" + "="*70)
    print("✨ Training pipeline completed successfully!")
    print("="*70)
    
    return learn, results


if __name__ == "__main__":
    # Check if dataset exists
    if not PATH_TO_DATASET.exists():
        print(f"❌ Error: Dataset not found at {PATH_TO_DATASET}")
        print("Please run the data organization script first!")
    else:
        learn, results = main_training_pipeline()