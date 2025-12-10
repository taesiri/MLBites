"""
VQ-VAE with Transformer Prior - Starting Point

Implement VQ-VAE from scratch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with codebook."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        """
        Args:
            num_embeddings: Number of codebook entries (K)
            embedding_dim: Dimension of each embedding (D)
            commitment_cost: Weight for commitment loss (β)
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # TODO: Create embedding table (codebook)
        # self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        
        pass
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize encoder output using codebook.
        
        Args:
            z: Encoder output (B, D, H, W)
        
        Returns:
            quantized: Quantized tensor (B, D, H, W)
            loss: VQ loss (commitment + codebook)
            indices: Codebook indices (B, H, W)
        """
        # TODO: Reshape z for distance computation
        # (B, D, H, W) -> (B*H*W, D)
        
        # TODO: Compute distances to all codebook entries
        # distances = ||z - e_k||^2
        
        # TODO: Find nearest codebook entry
        # indices = argmin(distances)
        
        # TODO: Quantize: lookup embeddings
        # z_q = embedding(indices)
        
        # TODO: Compute losses
        # codebook_loss = ||sg[z_e] - z_q||^2
        # commitment_loss = ||z_e - sg[z_q]||^2
        
        # TODO: Straight-through estimator
        # z_q = z_e + (z_q - z_e).detach()
        
        pass


class Encoder(nn.Module):
    """VQ-VAE Encoder."""
    
    def __init__(self, in_channels: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        # TODO: Build encoder (downsample 4x)
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Decoder(nn.Module):
    """VQ-VAE Decoder."""
    
    def __init__(self, embedding_dim: int, hidden_dim: int, out_channels: int):
        super().__init__()
        # TODO: Build decoder (upsample 4x)
        pass
    
    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        pass


class VQVAE(nn.Module):
    """Complete VQ-VAE model."""
    
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, 
                 num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        
        # TODO: Create encoder, vector quantizer, decoder
        pass
    
    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_recon: Reconstructed input
            vq_loss: Vector quantization loss
            indices: Codebook indices
        """
        pass
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode to codebook indices."""
        pass
    
    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode from codebook indices."""
        pass


class TransformerPrior(nn.Module):
    """Autoregressive transformer prior over codebook indices."""
    
    def __init__(self, num_embeddings: int, seq_len: int, embed_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        # TODO: Token embedding for codebook indices
        # TODO: Position embedding
        # TODO: Transformer decoder blocks
        # TODO: Output projection to num_embeddings
        pass
    
    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Predict next token probabilities.
        
        Args:
            indices: (B, S) codebook indices
        
        Returns:
            logits: (B, S, num_embeddings)
        """
        pass
    
    @torch.no_grad()
    def sample(self, batch_size: int, temperature: float = 1.0) -> torch.Tensor:
        """Autoregressively sample codebook indices."""
        pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Test VQ-VAE
    vqvae = VQVAE(in_channels=3, num_embeddings=512, embedding_dim=64)
    
    x = torch.randn(4, 3, 32, 32)
    x_recon, vq_loss, indices = vqvae(x)
    
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Indices: {indices.shape}")
    
    # Test Transformer Prior
    seq_len = indices.shape[1] * indices.shape[2]  # H * W
    prior = TransformerPrior(num_embeddings=512, seq_len=seq_len)
    
    flat_indices = indices.flatten(1)  # (B, H*W)
    logits = prior(flat_indices)
    print(f"Prior logits: {logits.shape}")
