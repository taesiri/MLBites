# Conv2D Layer

## Problem

The 2D convolution layer is the fundamental building block of Convolutional Neural Networks (CNNs). It applies learnable filters across spatial dimensions of the input to detect local patterns like edges, textures, and shapes. Understanding how convolution works at a low level is essential for deep learning practitioners.

## Task

Implement a `Conv2d` class in PyTorch from scratch. The class should apply 2D convolution with configurable kernel size, stride, and padding. Your implementation should support learnable weight and bias parameters.

Your implementation should match the behavior of `torch.nn.Conv2d` for the supported parameters.

## Function Signature

```python
import torch
import torch.nn as nn

class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

## Inputs and Outputs

### `__init__`
- **Inputs**:
  - `in_channels`: Number of channels in the input image (integer)
  - `out_channels`: Number of channels produced by the convolution (integer)
  - `kernel_size`: Size of the square convolving kernel (integer)
  - `stride`: Stride of the convolution (default 1)
  - `padding`: Zero-padding added to both sides of the input (default 0)
- The weight should be initialized with shape `(out_channels, in_channels, kernel_size, kernel_size)`
- The bias should be initialized with shape `(out_channels,)`
- Use Kaiming uniform initialization for weight and uniform initialization for bias (matching PyTorch defaults)

### `forward`
- **Inputs**:
  - `x`: Input tensor of shape `(batch_size, in_channels, height, width)`
- **Outputs**:
  - Output tensor of shape `(batch_size, out_channels, out_height, out_width)`
  - Where `out_height = (height + 2*padding - kernel_size) // stride + 1`
  - And `out_width = (width + 2*padding - kernel_size) // stride + 1`

## Constraints

- Must be solvable in 20–30 minutes.
- Interview-friendly: focus on correctness, not edge cases.
- Assume inputs satisfy the documented contract; avoid extra validation.
- Allowed libs: PyTorch (`torch`) and Python standard library.
- Do not call `torch.nn.Conv2d`, `F.conv2d`, or similar — implement the convolution yourself.
- Only square kernels are required (kernel_size is a single int, not a tuple).

## Examples

### Example 1 (basic convolution)
```python
import torch

conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
x = torch.randn(1, 1, 5, 5)  # batch=1, channels=1, height=5, width=5

out = conv(x)
# out has shape (1, 1, 3, 3) since (5 - 3) // 1 + 1 = 3
```

### Example 2 (with padding and stride)
```python
import torch

conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)
x = torch.randn(2, 3, 32, 32)  # batch=2, RGB image 32x32

out = conv(x)
# out has shape (2, 16, 16, 16) since (32 + 2*1 - 3) // 2 + 1 = 16
```

### Example 3 (multiple filters)
```python
import torch

conv = Conv2d(in_channels=1, out_channels=8, kernel_size=5, padding=2)
x = torch.randn(4, 1, 28, 28)  # batch of 4 grayscale 28x28 images

out = conv(x)
# out has shape (4, 8, 28, 28) — same spatial size due to padding=2 with kernel=5
```

