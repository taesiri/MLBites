"""
Batch Normalization from Scratch - Starting Point

Implement Batch Normalization without using built-in functions.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn


class BatchNorm1d(nn.Module):
    """Batch Normalization for 1D inputs (fully connected layers)."""
    
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1,
        affine: bool = True
    ):
        """
        Initialize Batch Normalization.
        
        Args:
            num_features: Number of features/channels
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
            affine: Whether to learn scale and shift parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # TODO: Create learnable parameters if affine is True
        # gamma (scale): initialized to 1
        # beta (shift): initialized to 0
        
        # TODO: Register running statistics as buffers (not parameters)
        # running_mean: initialized to 0
        # running_var: initialized to 1
        # num_batches_tracked: counter for batches seen
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch normalization.
        
        Args:
            x: Input tensor of shape (batch, features)
            
        Returns:
            Normalized tensor
        """
        if self.training:
            # TODO: Compute batch statistics
            # mean over batch dimension, shape: (features,)
            # var over batch dimension, shape: (features,)
            
            # TODO: Update running statistics
            # running_mean = (1 - momentum) * running_mean + momentum * batch_mean
            # running_var = (1 - momentum) * running_var + momentum * batch_var
            
            # TODO: Normalize using batch statistics
            
            pass
        else:
            # TODO: Use running statistics for normalization
            
            pass
        
        # TODO: Apply affine transformation if enabled
        # output = gamma * x_norm + beta
        
        pass


class BatchNorm2d(nn.Module):
    """Batch Normalization for 2D inputs (convolutional layers)."""
    
    def __init__(
        self, 
        num_features: int, 
        eps: float = 1e-5, 
        momentum: float = 0.1,
        affine: bool = True
    ):
        """
        Initialize Batch Normalization for 2D inputs.
        
        Args:
            num_features: Number of channels (C in NCHW)
            eps: Small constant for numerical stability
            momentum: Momentum for running statistics update
            affine: Whether to learn scale and shift parameters
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        
        # TODO: Same as BatchNorm1d - create parameters and buffers
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch normalization to 2D input.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Normalized tensor
        """
        # TODO: Compute statistics over (N, H, W), keeping C independent
        # Hint: Use dimensions (0, 2, 3) for mean/var
        
        # TODO: Same logic as BatchNorm1d but reshape gamma/beta for broadcasting
        # gamma, beta need shape (1, C, 1, 1) for proper broadcasting
        
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test BatchNorm1d
    print("Testing BatchNorm1d...")
    bn1d = BatchNorm1d(64)
    bn1d.train()
    
    x = torch.randn(32, 64)
    output = bn1d(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Mean per feature (should be ~0): {output.mean(dim=0).mean():.6f}")
    print(f"Std per feature (should be ~1): {output.std(dim=0).mean():.6f}")
    
    # Test BatchNorm2d
    print("\nTesting BatchNorm2d...")
    bn2d = BatchNorm2d(64)
    bn2d.train()
    
    x_conv = torch.randn(8, 64, 32, 32)  # (N, C, H, W)
    output_conv = bn2d(x_conv)
    
    print(f"Input shape: {x_conv.shape}")
    print(f"Output shape: {output_conv.shape}")
