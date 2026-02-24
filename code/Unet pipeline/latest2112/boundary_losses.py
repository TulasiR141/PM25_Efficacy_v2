import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class BoundaryIoULoss(nn.Module):
    """Boundary IoU Loss - focuses on edge alignment"""
    def __init__(self, theta0=3, theta=5):
        super().__init__()
        self.theta0 = theta0
        self.theta = theta
        
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - model output logits
        target: (B, H, W) - ground truth labels
        """
        # Get prediction mask
        pred_mask = pred.argmax(dim=1)  # (B, H, W)
        
        # Convert to float
        pred_mask = pred_mask.float()
        target = target.float()
        
        batch_loss = 0.0
        for i in range(pred.shape[0]):
            pred_i = pred_mask[i].cpu().numpy().astype(np.uint8)
            target_i = target[i].cpu().numpy().astype(np.uint8)
            
            # Get boundaries
            pred_boundary = self._get_boundary(pred_i)
            target_boundary = self._get_boundary(target_i)
            
            # Calculate boundary IoU
            pred_boundary_tensor = torch.from_numpy(pred_boundary).float().to(pred.device)
            target_boundary_tensor = torch.from_numpy(target_boundary).float().to(pred.device)
            
            intersection = (pred_boundary_tensor * target_boundary_tensor).sum()
            union = pred_boundary_tensor.sum() + target_boundary_tensor.sum() - intersection
            
            boundary_iou = (intersection + 1e-6) / (union + 1e-6)
            batch_loss += (1.0 - boundary_iou)
        
        return batch_loss / pred.shape[0]
    
    def _get_boundary(self, mask):
        """Extract boundary from mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.theta0, self.theta0))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - mask_eroded
        
        # Dilate boundary
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (self.theta, self.theta))
        boundary = cv2.dilate(boundary, kernel_dilate, iterations=1)
        
        return boundary


class AreaMatchingLoss(nn.Module):
    """Loss based on area difference between prediction and target"""
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - model output logits
        target: (B, H, W) - ground truth labels
        """
        pred_mask = pred.argmax(dim=1).float()  # (B, H, W)
        target = target.float()
        
        # Calculate areas
        pred_area = pred_mask.sum(dim=(1, 2))  # (B,)
        target_area = target.sum(dim=(1, 2))    # (B,)
        
        # Area difference loss
        area_diff = torch.abs(pred_area - target_area) / (target_area + 1e-6)
        
        return area_diff.mean()


class CombinedBoundaryLoss(nn.Module):
    """
    Combined loss: BCE + IoU + Boundary IoU + Area Matching
    """
    def __init__(self, alpha_bce=1.0, alpha_iou=1.0, alpha_boundary=2.0, alpha_area=0.5):
        super().__init__()
        self.alpha_bce = alpha_bce
        self.alpha_iou = alpha_iou
        self.alpha_boundary = alpha_boundary
        self.alpha_area = alpha_area
        
        self.bce_loss = nn.CrossEntropyLoss()
        self.boundary_iou_loss = BoundaryIoULoss(theta0=3, theta=5)
        self.area_loss = AreaMatchingLoss()
        
    def forward(self, pred, target):
        """
        pred: (B, C, H, W) - model output logits
        target: (B, H, W) - ground truth labels
        """
        # BCE Loss
        bce = self.bce_loss(pred, target)
        
        # IoU Loss (Dice-based)
        pred_soft = F.softmax(pred, dim=1)[:, 1]  # Get foreground probability
        target_float = target.float()
        
        intersection = (pred_soft * target_float).sum(dim=(1, 2))
        union = pred_soft.sum(dim=(1, 2)) + target_float.sum(dim=(1, 2))
        iou = (2.0 * intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1.0 - iou.mean()
        
        # Boundary IoU Loss
        boundary_loss = self.boundary_iou_loss(pred, target)
        
        # Area Matching Loss
        area_loss = self.area_loss(pred, target)
        
        # Combined loss
        total_loss = (self.alpha_bce * bce + 
                     self.alpha_iou * iou_loss + 
                     self.alpha_boundary * boundary_loss +
                     self.alpha_area * area_loss)
        
        return total_loss
    
    def decodes(self, x):
        return x.argmax(dim=1)
    
    def activation(self, x):
        return F.softmax(x, dim=1)
