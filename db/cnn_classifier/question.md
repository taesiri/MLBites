# CNN Image Classifier

## Problem Statement

Implement a **Convolutional Neural Network (CNN)** for image classification using PyTorch's built-in modules. CNNs are fundamental architectures for computer vision tasks that use convolutional layers to extract spatial features from images.

Your task is to:

1. Build a CNN with convolutional layers, pooling layers, and fully connected layers
2. Use appropriate activation functions (ReLU)
3. Implement the forward pass
4. Train the model on MNIST-like data

## Requirements

- Use `nn.Conv2d` for convolutional layers
- Use `nn.MaxPool2d` for pooling
- Use `nn.ReLU` for activation functions
- Include at least 2 convolutional layers followed by fully connected layers
- Handle the flattening between conv and fc layers

## Function Signature

```python
class CNNClassifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        """Initialize CNN classifier.
        
        Args:
            num_classes: Number of output classes
        """
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input images of shape (batch, channels, height, width)
            
        Returns:
            Class logits of shape (batch, num_classes)
        """
        pass
```

## Example

```python
import torch

# Create CNN classifier
model = CNNClassifier(num_classes=10)

# Generate random MNIST-like input (batch of 4, 1 channel, 28x28 images)
x = torch.randn(4, 1, 28, 28)

# Forward pass
logits = model(x)

print(f"Input shape: {x.shape}")       # (4, 1, 28, 28)
print(f"Output shape: {logits.shape}") # (4, 10)
```

## Hints

- A typical architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten -> FC -> ReLU -> FC
- Calculate the size after convolutions carefully for the first FC layer
- Use `x.view(x.size(0), -1)` or `torch.flatten(x, 1)` to flatten
- Consider using `nn.Sequential` to organize layers cleanly
