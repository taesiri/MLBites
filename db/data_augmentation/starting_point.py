"""
Data Augmentation with Transforms - Starting Point

Learn to use torchvision.transforms for data augmentation.
Fill in the TODO sections to complete the implementation.
"""

import torch
import random
from torchvision import transforms
from PIL import Image


# ImageNet normalization values
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = 224):
    """
    Get augmentation pipeline for training.
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transforms for training
    """
    # TODO: Create training augmentation pipeline
    # Include:
    # - RandomResizedCrop to img_size
    # - RandomHorizontalFlip
    # - ColorJitter
    # - ToTensor
    # - Normalize
    
    pass


def get_val_transforms(img_size: int = 224):
    """
    Get transforms for validation (no augmentation).
    
    Args:
        img_size: Target image size
        
    Returns:
        Composed transforms for validation
    """
    # TODO: Create validation pipeline
    # Include:
    # - Resize to slightly larger than img_size
    # - CenterCrop to img_size
    # - ToTensor
    # - Normalize
    
    pass


class CutoutAugmentation:
    """
    Cutout augmentation: randomly mask out square regions.
    
    This helps the model learn to use context.
    """
    
    def __init__(self, n_holes: int = 1, length: int = 16):
        """
        Args:
            n_holes: Number of regions to cut out
            length: Side length of each cutout square
        """
        # TODO: Store parameters
        pass
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply cutout to image.
        
        Args:
            img: Tensor image (C, H, W)
            
        Returns:
            Augmented image
        """
        # TODO: Randomly select regions and set them to 0
        pass


class MixupAugmentation:
    """
    Mixup augmentation: blend two images and their labels.
    
    Apply at the batch level, not per-image.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: Beta distribution parameter
        """
        # TODO: Store alpha
        pass
    
    def __call__(self, images: torch.Tensor, labels: torch.Tensor):
        """
        Apply mixup to a batch.
        
        Args:
            images: Batch of images (N, C, H, W)
            labels: Batch of one-hot labels (N, num_classes)
            
        Returns:
            Mixed images and labels
        """
        # TODO: Sample lambda from Beta distribution
        # TODO: Create random permutation
        # TODO: Mix images: lambda * img + (1 - lambda) * img[perm]
        # TODO: Mix labels similarly
        pass


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
