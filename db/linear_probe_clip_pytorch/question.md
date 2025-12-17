# Linear Probe on CLIP Features

## Problem

Linear probing is a standard evaluation technique for pre-trained representations. Given frozen features from a model like CLIP, we train only a single linear layer to perform classification. This tests how well the features separate classes without fine-tuning the backbone.

Your task is to implement a `LinearProbe` class that takes pre-extracted features (e.g., from CLIP's image encoder) and trains a linear classifier using gradient descent.

## Task

Implement a `LinearProbe` class that:
- Initializes a linear classification head (`nn.Linear`)
- Trains the linear layer using cross-entropy loss and SGD
- Predicts class labels for new feature inputs

## Function Signature

```python
class LinearProbe:
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        lr: float = 0.1,
    ) -> None: ...

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        epochs: int = 100,
    ) -> list[float]: ...

    def predict(self, features: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs

- **`__init__`**:
  - `feature_dim`: int, dimensionality of input features (e.g., 512 for CLIP ViT-B/32)
  - `num_classes`: int, number of target classes
  - `lr`: float, learning rate for SGD

- **`fit`**:
  - `features`: `torch.Tensor` of shape `(N, feature_dim)`, pre-extracted features
  - `labels`: `torch.Tensor` of shape `(N,)`, integer class labels in `[0, num_classes)`
  - `epochs`: int, number of training epochs
  - **Returns**: `list[float]` of length `epochs`, containing the loss at each epoch

- **`predict`**:
  - `features`: `torch.Tensor` of shape `(M, feature_dim)`
  - **Returns**: `torch.Tensor` of shape `(M,)`, predicted class labels (integers)

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: no need to load actual CLIP models or images.
- Assume inputs satisfy the documented contract (features are already extracted and normalized).
- Allowed libs: PyTorch (`torch`, `torch.nn`) and Python standard library.
- Use vanilla SGD (no momentum) for optimization.
- Process the full batch each epoch (no mini-batching required).

## Examples

### Example 1 (simple 2D features, 2 classes)

```python
import torch

# 4 samples, 2D features, 2 classes
features = torch.tensor([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
])
labels = torch.tensor([0, 0, 1, 1])

probe = LinearProbe(feature_dim=2, num_classes=2, lr=0.5)
losses = probe.fit(features, labels, epochs=50)

# After training, should correctly classify
preds = probe.predict(features)
# Expected: preds == tensor([0, 0, 1, 1])
```

### Example 2 (higher dimensional)

```python
torch.manual_seed(42)
features = torch.randn(100, 512)  # CLIP-like 512-dim features
labels = torch.randint(0, 10, (100,))  # 10 classes

probe = LinearProbe(feature_dim=512, num_classes=10, lr=0.1)
losses = probe.fit(features, labels, epochs=100)

# losses should be decreasing
assert losses[-1] < losses[0]
```

