# Diffusion Transformer (DiT)

## Problem Statement

Implement a **Diffusion Transformer (DiT)** for image generation. DiT replaces the U-Net backbone in diffusion models with a Transformer architecture.

Your task is to:

1. Implement patch embedding for images
2. Add adaptive layer norm (AdaLN) conditioning
3. Build DiT blocks with conditioning
4. Implement the diffusion training objective

## Requirements

- Patchify images and add position embeddings
- Condition on timestep and optional class labels
- Use AdaLN-Zero for conditioning
- Predict noise (or velocity) for denoising

## DiT Architecture

```
Image → Patch Embed → + Position Embeddings
                    ↓
DiT Block × N (conditioned on timestep t and class c):
    AdaLN → Self-Attention → Residual
    AdaLN → MLP → Residual
                    ↓
Final Layer → Unpatchify → Predicted Noise
```

## Function Signature

```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        pass
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x: patch tokens, c: conditioning (timestep + class embedding)."""
        pass

class DiT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int,
                 hidden_size: int, num_heads: int, num_layers: int, num_classes: int):
        pass
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Predict noise. x: noisy image, t: timestep, y: class label."""
        pass
```

## AdaLN-Zero

```python
# Instead of LayerNorm(x), use:
scale, shift = MLP(c).chunk(2, dim=-1)
out = LayerNorm(x) * (1 + scale) + shift

# AdaLN-Zero initializes the output to zero initially
```

## Example

```python
dit = DiT(img_size=32, patch_size=4, in_channels=3,
          hidden_size=384, num_heads=6, num_layers=12, num_classes=10)

x = torch.randn(4, 3, 32, 32)  # Noisy images
t = torch.randint(0, 1000, (4,))  # Timesteps
y = torch.randint(0, 10, (4,))  # Class labels

noise_pred = dit(x, t, y)  # (4, 3, 32, 32)
```

## Hints

- Timestep embedding: sinusoidal or learned
- Class embedding: learned embedding table
- Final layer predicts (patch_size² × in_channels) per patch
- Unpatchify reconstructs the image grid
