# Implement LeNet-5 (PyTorch)

## Problem
LeNet-5 is one of the earliest convolutional neural networks, designed by Yann LeCun in 1998 for handwritten digit recognition. It introduced the core concepts of CNNs: convolution, pooling, and hierarchical feature extraction. Understanding LeNet-5 is foundational for working with modern deep learning architectures.

## Task
Implement the LeNet-5 architecture as a PyTorch `nn.Module`. The network should accept 32×32 grayscale images and output class logits for 10 classes.

The architecture follows this structure:
1. **Conv1**: 6 filters of size 5×5, followed by ReLU and 2×2 max pooling
2. **Conv2**: 16 filters of size 5×5, followed by ReLU and 2×2 max pooling
3. **FC1**: Fully connected layer with 120 output features, followed by ReLU
4. **FC2**: Fully connected layer with 84 output features, followed by ReLU
5. **FC3**: Fully connected layer with 10 output features (logits, no activation)

## Function Signature

```python
class LeNet5(nn.Module):
    def __init__(self) -> None: ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

## Inputs and Outputs
- **inputs**:
  - `x`: a `torch.Tensor` of shape `(batch_size, 1, 32, 32)` — batch of grayscale images
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
model = LeNet5()
x = torch.randn(1, 1, 32, 32)
out = model(x)
# out.shape == torch.Size([1, 10])
```

### Example 2 (batch of images)
```python
model = LeNet5()
x = torch.randn(8, 1, 32, 32)
out = model(x)
# out.shape == torch.Size([8, 10])
```


