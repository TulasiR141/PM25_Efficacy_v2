import torch
import cv2
import numpy as np
from fastai.metrics import Metric


class BoundaryIoU(Metric):
    """Boundary IoU metric - measures edge alignment"""
    def __init__(self, theta0=3, theta=5):
        self.theta0 = theta0
        self.theta = theta
        self.reset()
        
    def reset(self):
        self.total = 0.0
        self.count = 0
        
    def accumulate(self, pred, target):
        """
        pred: (B, C, H, W) or (B, H, W)
        target: (B, H, W)
        """
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
            
        pred = pred.cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        
        for i in range(pred.shape[0]):
            pred_boundary = self._get_boundary(pred[i])
            target_boundary = self._get_boundary(target[i])
            
            intersection = (pred_boundary * target_boundary).sum()
            union = pred_boundary.sum() + target_boundary.sum() - intersection
            
            boundary_iou = intersection / (union + 1e-8)
            self.total += boundary_iou
            self.count += 1
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def name(self):
        return "boundary_iou"
    
    def _get_boundary(self, mask):
        """Extract boundary from mask"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.theta0, self.theta0))
        mask_eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = mask - mask_eroded
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (self.theta, self.theta))
        boundary = cv2.dilate(boundary, kernel_dilate, iterations=1)
        
        return boundary


class BoundaryF1(Metric):
    """Boundary F1 Score"""
    def __init__(self, theta=5):
        self.theta = theta
        self.reset()
        
    def reset(self):
        self.total = 0.0
        self.count = 0
        
    def accumulate(self, pred, target):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
            
        pred = pred.cpu().numpy().astype(np.uint8)
        target = target.cpu().numpy().astype(np.uint8)
        
        for i in range(pred.shape[0]):
            # Get boundaries
            pred_contours, _ = cv2.findContours(pred[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            target_contours, _ = cv2.findContours(target[i], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(pred_contours) == 0 or len(target_contours) == 0:
                continue
                
            # Get boundary points
            pred_pts = np.vstack(pred_contours) if len(pred_contours) > 0 else np.array([])
            target_pts = np.vstack(target_contours) if len(target_contours) > 0 else np.array([])
            
            if len(pred_pts) == 0 or len(target_pts) == 0:
                continue
            
            # Calculate boundary precision and recall
            pred_pts = pred_pts.squeeze()
            target_pts = target_pts.squeeze()
            
            # Count matched points
            matched_pred = 0
            for pt in pred_pts:
                distances = np.sqrt(((target_pts - pt) ** 2).sum(axis=1))
                if distances.min() <= self.theta:
                    matched_pred += 1
            
            matched_target = 0
            for pt in target_pts:
                distances = np.sqrt(((pred_pts - pt) ** 2).sum(axis=1))
                if distances.min() <= self.theta:
                    matched_target += 1
            
            precision = matched_pred / (len(pred_pts) + 1e-8)
            recall = matched_target / (len(target_pts) + 1e-8)
            
            f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
            self.total += f1
            self.count += 1
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def name(self):
        return "boundary_f1"


class AreaAccuracy(Metric):
    """Measures area prediction accuracy"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.total = 0.0
        self.count = 0
        
    def accumulate(self, pred, target):
        if pred.dim() == 4:
            pred = pred.argmax(dim=1)
            
        pred = pred.float()
        target = target.float()
        
        pred_area = pred.sum(dim=(1, 2))
        target_area = target.sum(dim=(1, 2))
        
        area_accuracy = 1.0 - torch.abs(pred_area - target_area) / (target_area + 1e-6)
        
        self.total += area_accuracy.sum().item()
        self.count += pred.shape[0]
    
    @property
    def value(self):
        return self.total / self.count if self.count > 0 else 0.0
    
    @property
    def name(self):
        return "area_accuracy"


def accuracy_spheroid(input, target):
    """Calculate pixel-wise accuracy"""
    target = target.squeeze(1) if target.dim() == 4 else target
    return (input.argmax(dim=1) == target).float().mean()


def calculate_boundary_metrics(pred, target):
    """Calculate all boundary metrics for a single image"""
    pred = pred.astype(np.uint8)
    target = target.astype(np.uint8)
    
    # Standard metrics
    intersection = (pred * target).sum()
    dice = (2.0 * intersection) / (pred.sum() + target.sum() + 1e-8)
    union = pred.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-8)
    accuracy = (pred == target).sum() / target.size
    
    # Boundary IoU
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    pred_eroded = cv2.erode(pred, kernel, iterations=1)
    target_eroded = cv2.erode(target, kernel, iterations=1)
    pred_boundary = pred - pred_eroded
    target_boundary = target - target_eroded
    
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    pred_boundary = cv2.dilate(pred_boundary, kernel_dilate, iterations=1)
    target_boundary = cv2.dilate(target_boundary, kernel_dilate, iterations=1)
    
    boundary_intersection = (pred_boundary * target_boundary).sum()
    boundary_union = pred_boundary.sum() + target_boundary.sum() - boundary_intersection
    boundary_iou = boundary_intersection / (boundary_union + 1e-8)
    
    # Area accuracy
    pred_area = pred.sum()
    target_area = target.sum()
    area_accuracy = 1.0 - abs(pred_area - target_area) / (target_area + 1e-8)
    
    return {
        'dice': dice,
        'iou': iou,
        'accuracy': accuracy,
        'boundary_iou': boundary_iou,
        'area_accuracy': area_accuracy
    }
