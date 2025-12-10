# Data Augmentation with Transforms

## Problem Statement

Learn to use **torchvision.transforms** for data augmentation. Data augmentation artificially increases training data variety, helping models generalize better.

Your task is to:

1. Create augmentation pipelines for training
2. Create normalization-only pipelines for validation
3. Implement custom transforms
4. Handle both image and tensor inputs

## Requirements

- Use `torchvision.transforms` for standard augmentations
- Chain transforms using `Compose`
- Implement custom augmentation as a callable class
- Apply different augmentations for train vs. test

## Function Signature

```python
def get_train_transforms(img_size: int = 224) -> transforms.Compose:
    """Get augmentation pipeline for training."""
    pass

def get_val_transforms(img_size: int = 224) -> transforms.Compose:
    """Get transforms for validation (no augmentation)."""
    pass

class CustomAugmentation:
    """Custom augmentation as a callable class."""
    def __call__(self, img):
        pass
```

## Example

```python
from torchvision import transforms

# Training transforms with augmentation
train_transform = get_train_transforms(img_size=224)

# Validation transforms (no augmentation)
val_transform = get_val_transforms(img_size=224)

# Apply to dataset
train_dataset = ImageFolder(train_path, transform=train_transform)
val_dataset = ImageFolder(val_path, transform=val_transform)
```

## Common Augmentations

| Transform | Description |
|-----------|-------------|
| `RandomHorizontalFlip` | Flip horizontally with probability |
| `RandomRotation` | Rotate by random angle |
| `RandomResizedCrop` | Crop and resize randomly |
| `ColorJitter` | Randomly change brightness, contrast, saturation |
| `RandomAffine` | Random affine transformation |
| `Normalize` | Normalize with mean and std |

## Hints

- Always normalize with ImageNet stats for pretrained models
- Use different transforms for train and validation
- Consider using `transforms.v2` for newer API
- Random augmentations help prevent overfitting
