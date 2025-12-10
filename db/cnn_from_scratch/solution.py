"""
2D Convolution from Scratch - Solution

Complete implementation of 2D convolution without built-in conv functions.
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
    
    # Apply padding to input if needed
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    
    # Get padded dimensions
    _, _, H_padded, W_padded = x.shape
    
    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1
    
    # Initialize output tensor
    output = torch.zeros(batch_size, out_channels, H_out, W_out, device=x.device, dtype=x.dtype)
    
    # Perform convolution using nested loops
    for i in range(H_out):
        for j in range(W_out):
            # Calculate input region
            h_start = i * stride
            h_end = h_start + kH
            w_start = j * stride
            w_end = w_start + kW
            
            # Extract input patch: (batch, in_channels, kH, kW)
            patch = x[:, :, h_start:h_end, w_start:w_end]
            
            # Compute output for all filters at once
            # patch: (batch, in_channels, kH, kW)
            # weight: (out_channels, in_channels, kH, kW)
            # Result: (batch, out_channels)
            for oc in range(out_channels):
                output[:, oc, i, j] = (patch * weight[oc]).sum(dim=(1, 2, 3))
    
    # Add bias if provided
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    
    return output


def conv2d_forward_im2col(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0
) -> torch.Tensor:
    """
    Efficient 2D convolution using im2col transformation.
    
    This converts the convolution into a matrix multiplication for efficiency.
    """
    batch_size, in_channels, H, W = x.shape
    out_channels, _, kH, kW = weight.shape
    
    # Apply padding
    if padding > 0:
        x = F.pad(x, (padding, padding, padding, padding))
    
    _, _, H_padded, W_padded = x.shape
    
    # Calculate output dimensions
    H_out = (H_padded - kH) // stride + 1
    W_out = (W_padded - kW) // stride + 1
    
    # im2col: extract all patches and reshape
    # Result shape: (batch, in_channels * kH * kW, H_out * W_out)
    cols = torch.zeros(batch_size, in_channels * kH * kW, H_out * W_out, device=x.device, dtype=x.dtype)
    
    idx = 0
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            patch = x[:, :, h_start:h_start+kH, w_start:w_start+kW]
            cols[:, :, idx] = patch.reshape(batch_size, -1)
            idx += 1
    
    # Reshape weight: (out_channels, in_channels * kH * kW)
    weight_reshaped = weight.view(out_channels, -1)
    
    # Matrix multiplication: (out_channels, in_channels * kH * kW) @ (batch, in_channels * kH * kW, H_out * W_out)
    # Transpose for batch matmul
    output = torch.matmul(weight_reshaped, cols)  # (batch, out_channels, H_out * W_out)
    
    # Reshape to spatial dimensions
    output = output.view(batch_size, out_channels, H_out, W_out)
    
    # Add bias
    if bias is not None:
        output = output + bias.view(1, -1, 1, 1)
    
    return output


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
    _, _, H_out, W_out = grad_output.shape
    
    # Compute grad_bias: sum over batch and spatial dimensions
    grad_bias = grad_output.sum(dim=(0, 2, 3))
    
    # Apply padding to input
    if padding > 0:
        x_padded = F.pad(x, (padding, padding, padding, padding))
    else:
        x_padded = x
    
    # Compute grad_weight
    grad_weight = torch.zeros_like(weight)
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            # Extract input patch
            patch = x_padded[:, :, h_start:h_start+kH, w_start:w_start+kW]
            
            # grad_output at this position: (batch, out_channels)
            grad_out_ij = grad_output[:, :, i, j]
            
            # Update grad_weight
            # grad_weight += outer product of grad_out and patch
            for b in range(batch_size):
                for oc in range(out_channels):
                    grad_weight[oc] += grad_out_ij[b, oc] * patch[b]
    
    # Compute grad_input (simplified, assumes stride=1)
    grad_input = torch.zeros_like(x)
    if padding > 0:
        grad_input_padded = F.pad(grad_input, (padding, padding, padding, padding))
    else:
        grad_input_padded = grad_input
    
    for i in range(H_out):
        for j in range(W_out):
            h_start = i * stride
            w_start = j * stride
            
            grad_out_ij = grad_output[:, :, i, j]  # (batch, out_channels)
            
            for b in range(batch_size):
                for oc in range(out_channels):
                    grad_input_padded[b, :, h_start:h_start+kH, w_start:w_start+kW] += \
                        grad_out_ij[b, oc] * weight[oc]
    
    # Remove padding from grad_input
    if padding > 0:
        grad_input = grad_input_padded[:, :, padding:-padding, padding:-padding]
    else:
        grad_input = grad_input_padded
    
    return grad_input, grad_weight, grad_bias


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create test input and weights
    x = torch.randn(1, 3, 5, 5)
    weight = torch.randn(2, 3, 3, 3)
    bias = torch.randn(2)
    
    # Test forward pass
    output = conv2d_forward(x, weight, bias, stride=1, padding=0)
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {weight.shape}")
    print(f"Output shape: {output.shape}")
    
    # Compare with PyTorch's implementation
    expected = F.conv2d(x, weight, bias, stride=1, padding=0)
    print(f"Matches PyTorch: {torch.allclose(output, expected, atol=1e-5)}")
    
    # Test im2col version
    output_im2col = conv2d_forward_im2col(x, weight, bias, stride=1, padding=0)
    print(f"im2col matches PyTorch: {torch.allclose(output_im2col, expected, atol=1e-5)}")
