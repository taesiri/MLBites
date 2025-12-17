# Implement a Simple CNN

## Problem
Convolutional Neural Networks (CNNs) are the foundation of modern computer vision. A basic CNN consists of convolutional layers that extract spatial features, pooling layers that reduce dimensionality, and fully connected layers that perform classification. Understanding how to build a CNN from scratch is essential for any deep learning practitioner.

## Task
Implement a simple CNN architecture as a PyTorch `nn.Module`. The network should accept 28×28 grayscale images (like MNIST) and output class logits for 10 classes.

The architecture follows this structure:
1. **Conv1**: 32 filters of size 3×3 with padding=1, followed by ReLU and 2×2 max pooling → 14×14×32
2. **Conv2**: 64 filters of size 3×3 with padding=1, followed by ReLU and 2×2 max pooling → 7×7×64
3. **Flatten**: 7×7×64 = 3136 features
4. **FC1**: Fully connected layer with 128 output features, followed by ReLU
5. **FC2**: Fully connected layer with 10 output features (logits, no activation)

## Function Signature

```python
class SimpleCNN(nn.Module):
    def __init__(self) -> None: ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: a `torch.Tensor` of shape `(batch_size, 1, 28, 28)` — batch of grayscale images
- **outputs**:
  - `torch.Tensor` of shape `(batch_size, 10)` — raw logits (no softmax)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: use standard PyTorch layers (`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.ReLU`).
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`, `torch.nn`) and Python standard library.

## Examples

### Example 1 (single image)
```python
model = SimpleCNN()
x = torch.randn(1, 1, 28, 28)
out = model(x)
# out.shape == torch.Size([1, 10])
```

### Example 2 (batch of images)
```python
model = SimpleCNN()
x = torch.randn(16, 1, 28, 28)
out = model(x)
# out.shape == torch.Size([16, 10])
```

