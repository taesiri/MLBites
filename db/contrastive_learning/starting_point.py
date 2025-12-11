"""
Contrastive Learning (SimCLR) - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ProjectionHead(nn.Module):
    """MLP projection head for SimCLR."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        # TODO: 2-layer MLP
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class SimCLR(nn.Module):
    """SimCLR contrastive learning framework."""
    
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        super().__init__()
        # TODO: Store encoder
        # TODO: Get encoder output dim
        # TODO: Create projection head
        pass
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor):
        """
        Args:
            x_i, x_j: Two augmented views (batch, C, H, W)
        Returns:
            z_i, z_j: Projected representations
        """
        # TODO: Encode both views
        # TODO: Project to contrastive space
        pass


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    
    Args:
        z_i, z_j: Projections of two views (batch, dim)
        temperature: Scaling temperature
    
    Returns:
        Contrastive loss
    """
    batch_size = z_i.shape[0]
    
    # TODO: Normalize embeddings
    
    # TODO: Concatenate z_i and z_j
    
    # TODO: Compute similarity matrix
    
    # TODO: Create labels (positive pairs)
    
    # TODO: Mask out self-similarity
    
    # TODO: Compute cross-entropy loss
    
    pass


def get_simclr_augmentations(size: int = 224):
    """Get SimCLR-style augmentations."""
    # TODO: Random crop, color jitter, grayscale, gaussian blur, horizontal flip
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Simple encoder
    encoder = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    
    simclr = SimCLR(encoder, projection_dim=128)
    
    # Dummy data
    x_i = torch.randn(32, 3, 32, 32)
    x_j = torch.randn(32, 3, 32, 32)
    
    z_i, z_j = simclr(x_i, x_j)
    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    
    print(f"Projection shape: {z_i.shape}")
    print(f"Loss: {loss.item():.4f}")
