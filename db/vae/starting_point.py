"""
Variational Autoencoder (VAE) - Starting Point

Implement VAE from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, latent_dim: int = 20):
        super().__init__()
        self.latent_dim = latent_dim
        
        # TODO: Encoder layers
        # Input -> hidden -> (mu, log_var)
        
        # TODO: Decoder layers
        # latent -> hidden -> output
        
        pass
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution parameters.
        
        Returns:
            (mu, log_var) each of shape (batch, latent_dim)
        """
        # TODO: Pass through encoder
        # TODO: Output mu and log_var
        pass
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = mu + std * epsilon.
        
        This allows gradients to flow through the sampling.
        """
        # TODO: std = exp(0.5 * log_var)
        # TODO: epsilon ~ N(0, 1)
        # TODO: z = mu + std * epsilon
        pass
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        # TODO: Pass through decoder
        pass
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Full forward pass.
        
        Returns:
            (reconstruction, mu, log_var)
        """
        # TODO: Encode, reparameterize, decode
        pass
    
    def sample(self, num_samples: int, device: str = 'cpu') -> torch.Tensor:
        """Generate new samples from prior."""
        # TODO: Sample z from N(0, 1) and decode
        pass


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, log_var: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    VAE loss = Reconstruction + β * KL Divergence.
    
    Args:
        recon_x: Reconstructed input
        x: Original input
        mu: Latent mean
        log_var: Latent log variance
        beta: KL weight (β-VAE)
    
    Returns:
        Total loss
    """
    # TODO: Reconstruction loss (BCE or MSE)
    
    # TODO: KL divergence: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create VAE
    vae = VAE(input_dim=784, hidden_dim=256, latent_dim=20)
    
    # Test forward pass
    x = torch.randn(32, 784)
    recon, mu, log_var = vae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"Log var shape: {log_var.shape}")
    
    # Compute loss
    loss = vae_loss(recon, x, mu, log_var)
    print(f"Loss: {loss.item():.4f}")
    
    # Generate samples
    samples = vae.sample(10)
    print(f"Generated samples shape: {samples.shape}")
