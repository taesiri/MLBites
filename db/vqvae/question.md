# VQ-VAE with Transformer Prior

## Problem Statement

Implement a **Vector Quantized Variational Autoencoder (VQ-VAE)** with a Transformer prior for generation. VQ-VAE learns a discrete latent representation using a codebook.

Your task is to:

1. Implement vector quantization with codebook lookup
2. Handle the straight-through gradient estimator
3. Train encoder/decoder with commitment loss
4. Train a transformer prior to model the latent sequence

## Requirements

- Implement codebook with nearest neighbor lookup
- Use straight-through estimator for gradients
- Include commitment loss and codebook loss
- Transformer prior autoregressively models codebook indices

## VQ-VAE Architecture

```
Encoder: x → z_e (continuous)
Quantize: z_e → z_q (discrete, from codebook)
Decoder: z_q → x_recon

Codebook: K embeddings of dimension D
z_q = e_k where k = argmin ||z_e - e_k||
```

## Function Signature

```python
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        pass
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (quantized, loss, indices)."""
        pass

class VQVAE(nn.Module):
    def __init__(self, in_channels: int, num_embeddings: int, embedding_dim: int):
        pass

class TransformerPrior(nn.Module):
    """Autoregressive prior over codebook indices."""
    def __init__(self, num_embeddings: int, seq_len: int, embed_dim: int, num_heads: int):
        pass
```

## Loss Function

```
L = Reconstruction + β * Commitment + Codebook

Reconstruction: ||x - decoder(z_q)||²
Commitment: ||z_e - sg[z_q]||²  (encoder commits to codebook)
Codebook: ||sg[z_e] - z_q||²   (codebook moves to encoder output)
```
(sg = stop gradient)

## Example

```python
vqvae = VQVAE(in_channels=3, num_embeddings=512, embedding_dim=64)

x = torch.randn(4, 3, 32, 32)
x_recon, vq_loss, indices = vqvae(x)

# Train transformer prior on indices
prior = TransformerPrior(num_embeddings=512, seq_len=64, embed_dim=256, num_heads=8)
```

## Hints

- Use `torch.argmin` for nearest neighbor lookup
- Straight-through: `z_q = z_e + (z_q - z_e).detach()`
- Indices shape for prior: (batch, height//patch * width//patch)
- EMA (exponential moving average) update for codebook is optional but helps
