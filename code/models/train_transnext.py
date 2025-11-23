import os
import cv2
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------------------------------------
# IMPORT THE MODEL
# --------------------------------------------------------
from transnext_upernet import (
    TransNeXtUperNet_Tiny,
    TransNeXtUperNet_Small,
    TransNeXtUperNet_Base
)

# ============================================================
# CONFIG
# ============================================================
DATA_ROOT = "data/consolidated_unet_augmented_v1"

IMG_SIZE = 512
BATCH_SIZE = 2
LR = 1e-4
EPOCHS = 1000      # max epochs
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_VARIANT = "tiny"   # tiny | small | base
SAVE_DIR = "transnext_training_output"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Training Device: {DEVICE}")

# ============================================================
# DATASET
# ============================================================
class SpheroidDataset(Dataset):
    def __init__(self, images_dir, masks_dir, augment=False):
        self.images = sorted(glob(images_dir + "/*"))
        self.masks = sorted(glob(masks_dir + "/*"))
        self.augment = augment

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.GaussianBlur(p=0.2),
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

        self.val_transform = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(mean=(0.5, 0.5, 0.5),
                        std=(0.5, 0.5, 0.5)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        mask_path = self.masks[idx]

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)

        if self.augment:
            transformed = self.transform(image=img, mask=mask)
        else:
            transformed = self.val_transform(image=img, mask=mask)

        img = transformed["image"]
        mask = transformed["mask"].unsqueeze(0)  # [1, H, W]

        return img, mask


# ============================================================
# LOSS FUNCTIONS
# ============================================================
bce = nn.BCEWithLogitsLoss()

def dice_loss(pred, target):
    pred = torch.sigmoid(pred)
    smooth = 1e-6
    intersection = (pred * target).sum()
    return 1 - ((2 * intersection + smooth) /
                (pred.sum() + target.sum() + smooth))


# ============================================================
# METRICS
# ============================================================
def compute_iou(pred, mask):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * mask).sum()
    union = (pred + mask).clamp(0, 1).sum()
    return (inter / (union + 1e-6)).item()

def compute_dice(pred, mask):
    pred = (torch.sigmoid(pred) > 0.5).float()
    inter = (pred * mask).sum()
    return (2 * inter / (pred.sum() + mask.sum() + 1e-6)).item()


# ============================================================
# EARLY STOPPING
# ============================================================
class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        # Not improved enough
        if score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            # Improved
            self.best_score = score
            self.counter = 0

        return False


# ============================================================
# LOAD DATA
# ============================================================
train_ds = SpheroidDataset(
    f"{DATA_ROOT}/train/images",
    f"{DATA_ROOT}/train/masks",
    augment=True
)

val_ds = SpheroidDataset(
    f"{DATA_ROOT}/val/images",
    f"{DATA_ROOT}/val/masks",
    augment=False
)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)

print(f"Training samples:   {len(train_ds)}")
print(f"Validation samples: {len(val_ds)}\n")


# ============================================================
# MODEL INIT
# ============================================================
if MODEL_VARIANT == "tiny":
    model = TransNeXtUperNet_Tiny(in_size=IMG_SIZE, in_channels=3, out_channels=1)
elif MODEL_VARIANT == "small":
    model = TransNeXtUperNet_Small(in_size=IMG_SIZE, in_channels=3, out_channels=1)
else:
    model = TransNeXtUperNet_Base(in_size=IMG_SIZE, in_channels=3, out_channels=1)

model.to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

early_stopper = EarlyStopping(patience=50, min_delta=1e-4)
best_val_dice = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

    for imgs, masks in pbar:
        imgs = imgs.to(DEVICE)
        masks = masks.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
            outputs = model(imgs)
            bce_loss = bce(outputs, masks)
            d_loss = dice_loss(outputs, masks)
            loss = bce_loss + d_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        # GPU VRAM in GB
        gpu_mem_gb = torch.cuda.memory_allocated() / (1024**3) if DEVICE == "cuda" else 0.0

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "gpu": f"{gpu_mem_gb:.2f}GB"
        })

    # ----------------------
    # VALIDATION
    # ----------------------
    model.eval()
    val_iou = 0.0
    val_dice = 0.0
    count = 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(DEVICE)
            masks = masks.to(DEVICE)

            outputs = model(imgs)

            val_iou += compute_iou(outputs, masks)
            val_dice += compute_dice(outputs, masks)
            count += 1

    val_iou /= count
    val_dice /= count

    print(f"\nEpoch {epoch}:")
    print(f"Train Loss: {train_loss / len(train_loader):.4f}")
    print(f"Val IoU:    {val_iou:.4f}")
    print(f"Val Dice:   {val_dice:.4f}")

    # ----------------------
    # SAVE BEST MODEL
    # ----------------------
    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")
        print("🔥 Saved new BEST model!\n")

    # ----------------------
    # EARLY STOPPING
    # ----------------------
    if early_stopper.step(val_dice):
        print("🛑 Early stopping triggered — no improvement.")
        torch.save(model.state_dict(), f"{SAVE_DIR}/earlystop_best.pth")
        break

print("🎉 Training complete!")
torch.save(model.state_dict(), f"{SAVE_DIR}/last_model.pth")
