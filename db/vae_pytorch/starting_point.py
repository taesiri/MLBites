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
        # TODO: Define encoder layers (fc1, fc_mu, fc_logvar)
        # TODO: Define decoder layers (fc3, fc4)
        raise NotImplementedError

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (mu, logvar), each of shape (batch_size, latent_dim).
        """
        # TODO: Pass x through encoder, return mu and logvar
        raise NotImplementedError

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample from latent distribution using reparameterization trick.

        z = mu + std * epsilon, where std = exp(0.5 * logvar) and epsilon ~ N(0, 1)

        Args:
            mu: Mean of latent distribution (batch_size, latent_dim).
            logvar: Log variance of latent distribution (batch_size, latent_dim).

        Returns:
            Sampled latent vector z of shape (batch_size, latent_dim).
        """
        # TODO: Implement reparameterization trick
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Args:
            z: Latent vector of shape (batch_size, latent_dim).

        Returns:
            Reconstructed tensor of shape (batch_size, input_dim).
        """
        # TODO: Pass z through decoder, return reconstruction
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Tuple of (recon_x, mu, logvar).
        """
        # TODO: Encode, reparameterize, decode
        raise NotImplementedError


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

    Notes:
        - Reconstruction loss: MSE with sum reduction
        - KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    """
    # TODO: Compute reconstruction loss (MSE, sum reduction)
    # TODO: Compute KL divergence
    # TODO: Return total loss
    raise NotImplementedError




