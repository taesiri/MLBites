# Linear Probe on CLIP Features

## Problem

Implement a `LinearProbe` class that trains a classifier on pre-extracted features and predicts class labels.

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
  - `feature_dim`: int, dimensionality of input features
  - `num_classes`: int, number of target classes
  - `lr`: float, learning rate

- **`fit`**:
  - `features`: `torch.Tensor` of shape `(N, feature_dim)`
  - `labels`: `torch.Tensor` of shape `(N,)`, integer class labels in `[0, num_classes)`
  - `epochs`: int, number of training epochs
  - **Returns**: `list[float]` of length `epochs`, containing the loss at each epoch

- **`predict`**:
  - `features`: `torch.Tensor` of shape `(M, feature_dim)`
  - **Returns**: `torch.Tensor` of shape `(M,)`, predicted class labels (integers)

## Constraints

- Must be solvable in 20–30 minutes.
- Allowed libs: PyTorch (`torch`, `torch.nn`) and Python standard library.

## Examples

### Example 1

```python
import torch

features = torch.tensor([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
])
labels = torch.tensor([0, 0, 1, 1])

probe = LinearProbe(feature_dim=2, num_classes=2, lr=0.5)
losses = probe.fit(features, labels, epochs=50)

preds = probe.predict(features)
# Expected: preds == tensor([0, 0, 1, 1])
```

### Example 2

```python
torch.manual_seed(42)
features = torch.randn(100, 512)
labels = torch.randint(0, 10, (100,))

probe = LinearProbe(feature_dim=512, num_classes=10, lr=0.1)
losses = probe.fit(features, labels, epochs=100)

assert losses[-1] < losses[0]
```




