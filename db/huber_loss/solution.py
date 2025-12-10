"""
Custom Loss Functions - Solution

Complete implementation of custom loss functions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def huber_loss(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """Huber loss (Smooth L1 Loss)."""
    # Compute absolute error
    error = pred - target
    abs_error = torch.abs(error)
    
    # Apply Huber loss formula
    quadratic = 0.5 * error ** 2
    linear = delta * abs_error - 0.5 * delta ** 2
    
    loss = torch.where(abs_error < delta, quadratic, linear)
    
    # Apply reduction
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


class HuberLoss(nn.Module):
    """Huber Loss as a module."""
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return huber_loss(pred, target, self.delta, self.reduction)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits (batch, num_classes)
            target: Class labels (batch,)
        """
        # Convert logits to probabilities
        probs = F.softmax(pred, dim=-1)
        
        # Get probability of true class
        batch_size = pred.size(0)
        p_t = probs[torch.arange(batch_size), target]
        
        # Compute focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Compute cross entropy: -log(p_t)
        ce = -torch.log(p_t + 1e-8)
        
        # Combine: -alpha * (1 - p_t)^gamma * log(p_t)
        loss = self.alpha * focal_weight * ce
        
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()


class LabelSmoothingLoss(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Create one-hot encoding
        one_hot = torch.zeros_like(pred)
        one_hot.scatter_(1, target.unsqueeze(1), 1)
        
        # Create smoothed labels
        smooth_labels = (1 - self.smoothing) * one_hot + self.smoothing / self.num_classes
        
        # Compute log softmax
        log_probs = F.log_softmax(pred, dim=-1)
        
        # Compute cross entropy with smoothed labels
        loss = -(smooth_labels * log_probs).sum(dim=-1)
        
        return loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks."""
    
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test Huber loss
    pred = torch.randn(10, 1)
    target = torch.randn(10, 1)
    
    loss = huber_loss(pred, target, delta=1.0)
    print(f"Huber loss: {loss.item():.4f}")
    
    # Compare with PyTorch
    expected = F.huber_loss(pred, target, delta=1.0)
    print(f"PyTorch Huber: {expected.item():.4f}")
    print(f"Matches: {torch.allclose(loss, expected)}")
    
    # Test Focal loss
    print("\nTesting Focal Loss...")
    focal = FocalLoss(gamma=2.0)
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    loss = focal(logits, labels)
    print(f"Focal loss: {loss.item():.4f}")
    
    # Test Label Smoothing
    print("\nTesting Label Smoothing Loss...")
    label_smooth = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
    loss = label_smooth(logits, labels)
    print(f"Label smoothing loss: {loss.item():.4f}")
