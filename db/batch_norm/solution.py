"""
Batch Normalization from Scratch - Solution

Complete implementation of Batch Normalization from scratch.
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
        
        # Learnable parameters
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running statistics (not parameters, but saved in state_dict)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch normalization.
        
        Args:
            x: Input tensor of shape (batch, features)
            
        Returns:
            Normalized tensor
        """
        if self.training:
            # Compute batch statistics
            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)
            
            # Update running statistics
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Normalize using batch statistics
            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        else:
            # Use running statistics for normalization
            x_norm = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            output = self.gamma * x_norm + self.beta
        else:
            output = x_norm
        
        return output


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
        
        # Learnable parameters
        if affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply batch normalization to 2D input.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            
        Returns:
            Normalized tensor
        """
        # Compute statistics over (N, H, W), keeping C independent
        if self.training:
            # Mean and var over dimensions (0, 2, 3), result shape: (C,)
            batch_mean = x.mean(dim=(0, 2, 3))
            batch_var = x.var(dim=(0, 2, 3), unbiased=False)
            
            # Update running statistics
            self.num_batches_tracked += 1
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var
            
            # Reshape for broadcasting: (C,) -> (1, C, 1, 1)
            mean = batch_mean.view(1, -1, 1, 1)
            var = batch_var.view(1, -1, 1, 1)
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
        
        # Normalize
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation
        if self.affine:
            gamma = self.gamma.view(1, -1, 1, 1)
            beta = self.beta.view(1, -1, 1, 1)
            output = gamma * x_norm + beta
        else:
            output = x_norm
        
        return output


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
    
    # Compare with PyTorch
    pytorch_bn = nn.BatchNorm1d(64)
    pytorch_bn.weight.data = bn1d.gamma.data.clone()
    pytorch_bn.bias.data = bn1d.beta.data.clone()
    pytorch_bn.running_mean.data = bn1d.running_mean.data.clone()
    pytorch_bn.running_var.data = bn1d.running_var.data.clone()
    
    # Test BatchNorm2d
    print("\nTesting BatchNorm2d...")
    bn2d = BatchNorm2d(64)
    bn2d.train()
    
    x_conv = torch.randn(8, 64, 32, 32)  # (N, C, H, W)
    output_conv = bn2d(x_conv)
    
    print(f"Input shape: {x_conv.shape}")
    print(f"Output shape: {output_conv.shape}")
    
    # Test evaluation mode
    print("\nTesting evaluation mode...")
    bn1d.eval()
    x_eval = torch.randn(32, 64)
    output_eval = bn1d(x_eval)
    print(f"Eval output shape: {output_eval.shape}")
