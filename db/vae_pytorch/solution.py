from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """A simple Variational Autoencoder with MLP encoder and decoder.

    Architecture:
        Encoder: input_dim -> hidden_dim (ReLU) -> mu, logvar (each latent_dim)
        Decoder: latent_dim -> hidden_dim (ReLU) -> input_dim (Sigmoid)

    Args:
        input_dim: Dimension of input data.
        hidden_dim: Dimension of hidden layer.
        latent_dim: Dimension of latent space.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction."""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor:
    """Compute VAE loss: reconstruction loss + KL divergence.

    Args:
        recon_x: Reconstructed input (batch_size, input_dim).
        x: Original input (batch_size, input_dim).
        mu: Mean of latent distribution (batch_size, latent_dim).
        logvar: Log variance of latent distribution (batch_size, latent_dim).

    Returns:
        Scalar tensor containing total loss (summed over batch).
    """
    # Reconstruction loss (MSE with sum reduction)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum")

    # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss




