"""
Layer Normalization from Scratch - Starting Point

Implement Layer Normalization without using built-in functions.
Fill in the TODO sections to complete the implementation.
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
        
        # TODO: Create learnable scale parameter (gamma)
        # Initialized to 1
        
        # TODO: Create learnable shift parameter (beta)
        # Initialized to 0
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., *normalized_shape)
            
        Returns:
            Normalized tensor (same shape as input)
        """
        # TODO: Calculate the dimensions to normalize over
        # For normalized_shape of (D,), normalize over the last dimension
        # For normalized_shape of (H, W), normalize over the last two dimensions
        
        # TODO: Compute mean across normalized dimensions
        # Hint: Use keepdim=True
        
        # TODO: Compute variance across normalized dimensions
        # Hint: Use unbiased=False
        
        # TODO: Normalize: (x - mean) / sqrt(var + eps)
        
        # TODO: Apply scale and shift: gamma * x_norm + beta
        
        pass


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
        
        # TODO: Create learnable scale parameter
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization.
        
        Args:
            x: Input tensor
            
        Returns:
            Normalized tensor
        """
        # TODO: Compute RMS: sqrt(mean(x^2))
        
        # TODO: Normalize and scale
        
        pass


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
    expected = pytorch_ln(x)
    print(f"Output range matches: {output.min():.2f} to {output.max():.2f}")
