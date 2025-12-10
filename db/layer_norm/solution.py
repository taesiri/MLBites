"""
Layer Normalization from Scratch - Solution

Complete implementation of Layer Normalization from scratch.
"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """Layer Normalization implemented from scratch."""
    
    def __init__(self, normalized_shape: int | tuple, eps: float = 1e-5):
        """
        Initialize Layer Normalization.
        
        Args:
            normalized_shape: Shape of the features to normalize
            eps: Small constant for numerical stability
        """
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable scale parameter (gamma) - initialized to 1
        self.gamma = nn.Parameter(torch.ones(normalized_shape))
        
        # Learnable shift parameter (beta) - initialized to 0
        self.beta = nn.Parameter(torch.zeros(normalized_shape))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., *normalized_shape)
            
        Returns:
            Normalized tensor (same shape as input)
        """
        # Calculate the dimensions to normalize over
        # For normalized_shape of (D,), normalize over the last dimension
        # For normalized_shape of (H, W), normalize over the last two dimensions
        dims = tuple(range(-len(self.normalized_shape), 0))
        
        # Compute mean across normalized dimensions
        mean = x.mean(dim=dims, keepdim=True)
        
        # Compute variance across normalized dimensions
        var = x.var(dim=dims, keepdim=True, unbiased=False)
        
        # Normalize: (x - mean) / sqrt(var + eps)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply scale and shift: gamma * x_norm + beta
        output = self.gamma * x_norm + self.beta
        
        return output


class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (used in LLaMA).
    
    Simpler variant that only uses RMS for scaling, no mean subtraction.
    """
    
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize RMSNorm.
        
        Args:
            dim: Dimension of features
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        
        # Learnable scale parameter
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        output = (x / rms) * self.weight
        
        return output


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test Layer Norm
    print("Testing LayerNorm...")
    layer_norm = LayerNorm(512)
    x = torch.randn(4, 10, 512)
    output = layer_norm(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mean per position (should be ~0): {output[0, 0].mean():.6f}")
    print(f"Std per position (should be ~1): {output[0, 0].std():.6f}")
    
    # Compare with PyTorch
    pytorch_ln = nn.LayerNorm(512)
    pytorch_ln.weight.data = layer_norm.gamma.data.clone()
    pytorch_ln.bias.data = layer_norm.beta.data.clone()
    expected = pytorch_ln(x)
    print(f"Matches PyTorch: {torch.allclose(output, expected, atol=1e-5)}")
    
    # Test RMSNorm
    print("\nTesting RMSNorm...")
    rms_norm = RMSNorm(512)
    rms_output = rms_norm(x)
    print(f"RMSNorm output shape: {rms_output.shape}")
    print(f"RMSNorm output range: {rms_output.min():.2f} to {rms_output.max():.2f}")
