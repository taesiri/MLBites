"""
2D Convolution from Scratch - Starting Point

Implement 2D convolution without using built-in conv functions.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn.functional as F


def conv2d_forward(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """
    Compute 2D convolution forward pass.
    
    Args:
        x: Input tensor of shape (batch, in_channels, H, W)
        weight: Kernel weights of shape (out_channels, in_channels, kH, kW)
        bias: Optional bias of shape (out_channels,)
        stride: Stride of the convolution
        padding: Zero-padding added to both sides
        
    Returns:
        Output tensor of shape (batch, out_channels, H_out, W_out)
    """
    batch_size, in_channels, H, W = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # TODO: Apply padding to input if needed
    # Hint: Use F.pad(x, (padding, padding, padding, padding))
    
    # TODO: Calculate output dimensions
    # H_out = (H + 2*padding - kH) // stride + 1
    # W_out = (W + 2*padding - kW) // stride + 1
    
    # TODO: Initialize output tensor
    
    # TODO: Perform convolution using nested loops
    # For each output position (i, j):
    #   1. Extract the input patch of size (in_channels, kH, kW)
    #   2. Multiply element-wise with weight
    #   3. Sum all values to get the output value
    
    # TODO: Add bias if provided
    
    pass


def conv2d_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gradients for 2D convolution.
    
    Args:
        grad_output: Gradient of loss w.r.t. output, shape (batch, out_channels, H_out, W_out)
        x: Original input tensor
        weight: Original kernel weights
        stride: Stride used in forward pass
        padding: Padding used in forward pass
        
    Returns:
        Tuple of (grad_input, grad_weight, grad_bias)
    """
    batch_size, in_channels, H, W = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # TODO: Compute grad_bias
    # grad_bias = sum over (batch, H_out, W_out) of grad_output
    
    # TODO: Apply padding to input
    
    # TODO: Compute grad_weight
    # For each weight position, correlate input with grad_output
    
    # TODO: Compute grad_input
    # This is the "full" convolution of grad_output with flipped weights
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create test input and weights
    x = torch.randn(1, 3, 5, 5, requires_grad=True)
    weight = torch.randn(2, 3, 3, 3, requires_grad=True)
    bias = torch.randn(2, requires_grad=True)
    
    # Test forward pass
    output = conv2d_forward(x, weight, bias, stride=1, padding=0)
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare with PyTorch's implementation
    expected = F.conv2d(x, weight, bias, stride=1, padding=0)
    print(f"Matches PyTorch: {torch.allclose(output, expected, atol=1e-5)}")
