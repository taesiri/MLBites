"""
Contrastive Learning (SimCLR) - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 128, encoder_dim: int = None):
        super().__init__()
        self.encoder = encoder
        
        if encoder_dim is None:
            encoder_dim = 2048  # Default for ResNet
        
        self.projector = ProjectionHead(encoder_dim, encoder_dim, projection_dim)
    
    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        
        return z_i, z_j


def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent Loss."""
    batch_size = z_i.shape[0]
    device = z_i.device
    
    # Normalize
    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    
    # Concatenate
    z = torch.cat([z_i, z_j], dim=0)  # (2N, dim)
    
    # Similarity matrix
    sim = torch.mm(z, z.T) / temperature  # (2N, 2N)
    
    # Mask self-similarity
    mask = torch.eye(2 * batch_size, device=device).bool()
    sim.masked_fill_(mask, float('-inf'))
    
    # Labels: positive pairs are at positions (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device)
    ])
    
    loss = F.cross_entropy(sim, labels)
    return loss


def info_nce_loss(query: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor, 
                  temperature: float = 0.1) -> torch.Tensor:
    """InfoNCE loss with explicit negatives."""
    query = F.normalize(query, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)
    
    pos_sim = (query * positive).sum(dim=-1, keepdim=True) / temperature
    neg_sim = torch.mm(query, negatives.T) / temperature
    
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(query.size(0), dtype=torch.long, device=query.device)
    
    return F.cross_entropy(logits, labels)


def get_simclr_augmentations(size: int = 224):
    """SimCLR augmentation pipeline."""
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=int(0.1 * size) // 2 * 2 + 1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class ContrastiveDataset:
    """Dataset wrapper that returns two augmented views."""
    
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), self.transform(img), label


if __name__ == "__main__":
    torch.manual_seed(42)
    
    encoder = nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten()
    )
    
    simclr = SimCLR(encoder, projection_dim=128, encoder_dim=64)
    
    x_i = torch.randn(32, 3, 32, 32)
    x_j = torch.randn(32, 3, 32, 32)
    
    z_i, z_j = simclr(x_i, x_j)
    loss = nt_xent_loss(z_i, z_j, temperature=0.5)
    
    print(f"Projection shape: {z_i.shape}")
    print(f"Loss: {loss.item():.4f}")
