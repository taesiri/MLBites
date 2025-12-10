"""
Data Augmentation with Transforms - Solution

Complete implementation of data augmentation pipelines.
"""

import torch
import numpy as np
import random
from torchvision import transforms
from PIL import Image


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """Get augmentation pipeline for training."""
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224):
    """Get transforms for validation (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # Slightly larger
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_light_augmentation(img_size: int = 224):
    """Light augmentation for fine-tuning."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class CutoutAugmentation:
    """Cutout augmentation: randomly mask out square regions."""
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        self.n_holes = n_holes
        self.length = length
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """Apply cutout to image."""
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones_like(img)
        
        for _ in range(self.n_holes):
            y = random.randint(0, h - 1)
            x = random.randint(0, w - 1)
            
            y1 = max(0, y - self.length // 2)
            y2 = min(h, y + self.length // 2)
            x1 = max(0, x - self.length // 2)
            x2 = min(w, x + self.length // 2)
            
            mask[:, y1:y2, x1:x2] = 0
        
        return img * mask


class RandomErasingAugmentation:
    """Random erasing augmentation (similar to cutout but with random fill)."""
    
    def __init__(self, p: float = 0.5, scale: tuple = (0.02, 0.33), ratio: tuple = (0.3, 3.3)):
        self.transform = transforms.RandomErasing(p=p, scale=scale, ratio=ratio)
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.transform(img)


class MixupAugmentation:
    """Mixup augmentation: blend two images and their labels."""
    
    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply mixup to a batch."""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        perm = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[perm]
        mixed_labels = lam * labels + (1 - lam) * labels[perm]
        
        return mixed_images, mixed_labels


class CutMixAugmentation:
    """CutMix augmentation: paste a patch from one image onto another."""
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """Apply cutmix to a batch."""
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size, _, h, w = images.shape
        perm = torch.randperm(batch_size)
        
        # Generate random box
        cut_ratio = np.sqrt(1 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        
        bbx1 = max(0, cx - cut_w // 2)
        bbx2 = min(w, cx + cut_w // 2)
        bby1 = max(0, cy - cut_h // 2)
        bby2 = min(h, cy + cut_h // 2)
        
        # Apply cutmix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[perm, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust labels based on area
        lam_adjusted = 1 - (bbx2 - bbx1) * (bby2 - bby1) / (h * w)
        mixed_labels = lam_adjusted * labels + (1 - lam_adjusted) * labels[perm]
        
        return mixed_images, mixed_labels


if __name__ == "__main__":
    # Create sample image
    img = Image.new('RGB', (256, 256), color='red')
    
    # Test training transforms
    train_transform = get_train_transforms(224)
    print("Training transforms:")
    print(train_transform)
    
    augmented = train_transform(img)
    print(f"Augmented shape: {augmented.shape}")
    
    # Test validation transforms
    val_transform = get_val_transforms(224)
    print("\nValidation transforms:")
    print(val_transform)
    
    # Test custom cutout
    print("\nTesting Cutout:")
    cutout = CutoutAugmentation(n_holes=1, length=32)
    tensor_img = transforms.ToTensor()(img)
    cutout_img = cutout(tensor_img)
    print(f"Cutout applied, shape: {cutout_img.shape}")
    print(f"Number of zeros: {(cutout_img == 0).sum().item()}")
    
    # Test Mixup
    print("\nTesting Mixup:")
    mixup = MixupAugmentation(alpha=0.2)
    batch_imgs = torch.randn(4, 3, 32, 32)
    batch_labels = torch.eye(10)[torch.randint(0, 10, (4,))]  # One-hot
    
    mixed_imgs, mixed_labels = mixup(batch_imgs, batch_labels)
    print(f"Mixed images shape: {mixed_imgs.shape}")
    print(f"Mixed labels shape: {mixed_labels.shape}")
