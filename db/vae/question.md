# Variational Autoencoder (VAE)

## Problem Statement

Implement a **Variational Autoencoder (VAE)** from scratch. VAEs learn a probabilistic latent representation of data, enabling generation of new samples.

Your task is to:

1. Implement encoder that outputs μ and σ (not deterministic)
2. Implement reparameterization trick for backprop
3. Implement decoder
4. Compute ELBO loss (reconstruction + KL divergence)

## Requirements

- Encoder outputs mean and log-variance
- Use reparameterization: z = μ + σ * ε
- Combine reconstruction loss and KL divergence
- Generate new samples from latent space

## Function Signature

```python
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        pass
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mu, log_var)."""
        pass
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Sample z using reparameterization trick."""
        pass
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct from latent."""
        pass
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Return (reconstruction, mu, log_var)."""
        pass

def vae_loss(recon_x, x, mu, log_var) -> torch.Tensor:
    """ELBO loss = Reconstruction + KL Divergence."""
    pass
```

## VAE Loss (ELBO)

```
L = Reconstruction Loss + β * KL Divergence

Reconstruction: BCE or MSE between input and output
KL Divergence: -0.5 * Σ(1 + log(σ²) - μ² - σ²)
```

## Example

```python
vae = VAE(input_dim=784, latent_dim=20)

x = torch.randn(32, 784)
recon, mu, log_var = vae(x)

loss = vae_loss(recon, x, mu, log_var)
```

## Hints

- log_var is more stable than σ directly
- Reparameterization: z = mu + exp(0.5 * log_var) * epsilon
- KL divergence has closed form for Gaussian prior
- Use β-VAE (β > 1) for more disentangled latent space
