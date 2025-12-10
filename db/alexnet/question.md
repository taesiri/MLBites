# AlexNet from Scratch

## Problem Statement

Implement **AlexNet** from scratch. AlexNet won the 2012 ImageNet competition and popularized deep learning for computer vision.

Your task is to:

1. Implement the convolutional layers with ReLU
2. Use overlapping max pooling
3. Implement Local Response Normalization (LRN)
4. Add fully connected layers with dropout

## AlexNet Architecture

```
Conv1: 96 filters, 11x11, stride 4 → ReLU → LRN → MaxPool 3x3 stride 2
Conv2: 256 filters, 5x5, pad 2 → ReLU → LRN → MaxPool 3x3 stride 2
Conv3: 384 filters, 3x3, pad 1 → ReLU
Conv4: 384 filters, 3x3, pad 1 → ReLU
Conv5: 256 filters, 3x3, pad 1 → ReLU → MaxPool 3x3 stride 2
Flatten → FC 4096 → ReLU → Dropout
FC 4096 → ReLU → Dropout
FC 1000 (num_classes)
```

## Function Signature

```python
class LocalResponseNorm(nn.Module):
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 2.0):
        pass

class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 1000):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
```

## Example

```python
model = AlexNet(num_classes=1000)

x = torch.randn(1, 3, 224, 224)
logits = model(x)  # (1, 1000)
```

## Key Innovations (2012)

| Innovation | Description |
|------------|-------------|
| ReLU | Faster training than tanh/sigmoid |
| Dropout | Regularization for FC layers |
| LRN | Local response normalization (less used today) |
| GPU Training | Split model across 2 GPUs |
| Data Augmentation | Crops, flips, color jitter |

## Hints

- Input size is 224x224 (or 227x227 in some implementations)
- LRN is rarely used now; BatchNorm replaced it
- Dropout rate is 0.5 in FC layers
- Original used 2 GPUs; we implement single-GPU version
