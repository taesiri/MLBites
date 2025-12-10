"""
AlexNet from Scratch - Starting Point

Implement AlexNet from scratch.
"""

import torch
import torch.nn as nn


class LocalResponseNorm(nn.Module):
    """Local Response Normalization (LRN) layer."""
    
    def __init__(self, size: int = 5, alpha: float = 1e-4, beta: float = 0.75, k: float = 2.0):
        """
        Args:
            size: Number of channels to normalize across
            alpha, beta, k: LRN hyperparameters
        """
        super().__init__()
        # TODO: Store hyperparameters
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply LRN: x_i / (k + α * Σ x_j²)^β
        
        Sum is over neighboring channels.
        """
        # TODO: Implement LRN
        pass


class AlexNet(nn.Module):
    """AlexNet architecture."""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # TODO: Convolutional layers (features)
        # Conv1: 96 filters, 11x11, stride 4
        # Conv2: 256 filters, 5x5, padding 2
        # Conv3: 384 filters, 3x3, padding 1
        # Conv4: 384 filters, 3x3, padding 1
        # Conv5: 256 filters, 3x3, padding 1
        
        # TODO: Fully connected layers (classifier)
        # FC1: 4096
        # FC2: 4096
        # FC3: num_classes
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input image (batch, 3, 224, 224)
        Returns:
            Logits (batch, num_classes)
        """
        # TODO: Forward through conv layers
        # TODO: Flatten
        # TODO: Forward through FC layers
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create AlexNet
    model = AlexNet(num_classes=1000)
    
    # Test forward
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {logits.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
