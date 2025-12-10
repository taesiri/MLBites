# 2D Convolution from Scratch

## Problem Statement

Implement **2D Convolution** from scratch using only NumPy/PyTorch tensor operations (no nn.Conv2d). This is a fundamental operation in CNNs that slides a kernel over an input to detect features.

Your task is to:

1. Implement the forward pass of 2D convolution with support for multiple channels and filters
2. Handle padding and stride parameters
3. Implement the backward pass to compute gradients

## Requirements

- Do **NOT** use `nn.Conv2d`, `F.conv2d`, or similar built-in convolution functions
- Support multi-channel inputs and multiple output filters
- Handle `padding` and `stride` parameters
- Implement gradient computation for backpropagation

## Function Signature

```python
def conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """Compute 2D convolution forward pass.
    
    Args:
        x: Input tensor of shape (batch, in_channels, H, W)
        weight: Kernel weights of shape (out_channels, in_channels, kH, kW)
        bias: Optional bias of shape (out_channels,)
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        
    Returns:
        Output tensor of shape (batch, out_channels, H_out, W_out)
    """
    pass

def conv2d_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute gradients for 2D convolution.
    
    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    pass
```

## Example

```python
import torch

# Input: batch=1, channels=3, size=5x5
x = torch.randn(1, 3, 5, 5)

# Filter: 2 output channels, 3 input channels, 3x3 kernel
weight = torch.randn(2, 3, 3, 3)
bias = torch.randn(2)

# Forward pass
output = conv2d_forward(x, weight, bias, stride=1, padding=0)

print(f"Input shape: {x.shape}")      # (1, 3, 5, 5)
print(f"Output shape: {output.shape}") # (1, 2, 3, 3)
```

## Hints

- Output size: `H_out = (H_in + 2*padding - kH) // stride + 1`
- Use nested loops or sliding window views to extract patches
- Each output position is the dot product of the kernel with the input patch
- For efficient implementation, consider im2col transformation
- Compare your output with `F.conv2d` to verify correctness
