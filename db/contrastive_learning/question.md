# Contrastive Learning (SimCLR)

## Problem Statement

Implement **SimCLR** (Simple Contrastive Learning of Visual Representations) from scratch. SimCLR learns visual representations by maximizing agreement between augmented views of the same image.

Your task is to:

1. Create augmented views of images
2. Implement the projection head
3. Compute NT-Xent (InfoNCE) loss
4. Train an encoder without labels

## SimCLR Pipeline

```
Image x
    ↓ Random augmentation (twice)
x_i, x_j (two views of same image)
    ↓ Encoder (ResNet)
h_i, h_j (representations)
    ↓ Projection head (MLP)
z_i, z_j (projections)
    ↓ NT-Xent Loss
Maximize similarity(z_i, z_j)
```

## Function Signature

```python
class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, projection_dim: int = 128):
        pass
    
    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> tuple:
        """Return (z_i, z_j) projections."""
        pass

def nt_xent_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Normalized Temperature-scaled Cross Entropy Loss."""
    pass
```

## NT-Xent Loss (InfoNCE)

```
sim(u, v) = u·v / (||u|| ||v||)

l(i, j) = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```

For each positive pair, treat other 2(N-1) samples as negatives.

## Example

```python
encoder = resnet18(pretrained=False)
encoder.fc = nn.Identity()  # Remove classifier

simclr = SimCLR(encoder, projection_dim=128)

# Two augmented views
x_i = augment(images)
x_j = augment(images)

z_i, z_j = simclr(x_i, x_j)
loss = nt_xent_loss(z_i, z_j, temperature=0.5)
```

## Hints

- Use strong augmentations: random crop, color jitter, Gaussian blur
- Projection head: 2-layer MLP with ReLU
- Temperature τ typically 0.1 - 0.5
- Large batch sizes (256-4096) help by providing more negatives
