"""
VQ-VAE with Transformer Prior - Solution

Complete VQ-VAE implementation with transformer prior.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization with codebook."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float = 0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z: (B, D, H, W)
        B, D, H, W = z.shape
        
        # Reshape: (B, D, H, W) -> (B*H*W, D)
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        
        # Compute distances to codebook
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        d = (z_flat.pow(2).sum(1, keepdim=True) 
             + self.embedding.weight.pow(2).sum(1)
             - 2 * z_flat @ self.embedding.weight.t())
        
        # Find nearest
        indices = d.argmin(dim=1)
        
        # Quantize
        z_q = self.embedding(indices).view(B, H, W, D).permute(0, 3, 1, 2)
        
        # Losses
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z, z_q.detach())
        loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator
        z_q = z + (z_q - z).detach()
        
        indices = indices.view(B, H, W)
        
        return z_q, loss, indices


class VectorQuantizerEMA(nn.Module):
    """VQ with Exponential Moving Average codebook updates."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 commitment_cost: float = 0.25, decay: float = 0.99):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        
        self.register_buffer('embedding', torch.randn(num_embeddings, embedding_dim))
        self.register_buffer('cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('ema_w', torch.randn(num_embeddings, embedding_dim))
    
    def forward(self, z: torch.Tensor):
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)
        
        d = (z_flat.pow(2).sum(1, keepdim=True) 
             + self.embedding.pow(2).sum(1)
             - 2 * z_flat @ self.embedding.t())
        
        indices = d.argmin(dim=1)
        encodings = F.one_hot(indices, self.num_embeddings).float()
        
        z_q = self.embedding[indices].view(B, H, W, D).permute(0, 3, 1, 2)
        
        if self.training:
            # EMA update
            self.cluster_size.data.mul_(self.decay).add_(encodings.sum(0), alpha=1-self.decay)
            self.ema_w.data.mul_(self.decay).add_(encodings.t() @ z_flat, alpha=1-self.decay)
            
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + 1e-5) / (n + self.num_embeddings * 1e-5) * n
            self.embedding.data.copy_(self.ema_w / cluster_size.unsqueeze(1))
        
        loss = self.commitment_cost * F.mse_loss(z, z_q.detach())
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, indices.view(B, H, W)


class Encoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, padding=1),
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, out_channels, 4, stride=2, padding=1),
        )
    
    def forward(self, z_q):
        return self.net(z_q)


class VQVAE(nn.Module):
    def __init__(self, in_channels: int = 3, hidden_dim: int = 128,
                 num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, embedding_dim)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_dim, in_channels)
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss, indices
    
    def encode(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices
    
    def decode_from_indices(self, indices):
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2)
        return self.decoder(z_q)


class TransformerPrior(nn.Module):
    """Autoregressive prior over codebook indices."""
    
    def __init__(self, num_embeddings: int, seq_len: int, embed_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 6):
        super().__init__()
        self.seq_len = seq_len
        self.num_embeddings = num_embeddings
        
        self.tok_emb = nn.Embedding(num_embeddings, embed_dim)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, embed_dim))
        
        decoder_layer = nn.TransformerDecoderLayer(embed_dim, num_heads, 4*embed_dim, batch_first=True)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.head = nn.Linear(embed_dim, num_embeddings)
        
        # Causal mask
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool())
    
    def forward(self, indices):
        x = self.tok_emb(indices) + self.pos_emb[:, :indices.size(1)]
        x = self.transformer(x, x, tgt_mask=self.mask[:indices.size(1), :indices.size(1)])
        return self.head(x)
    
    @torch.no_grad()
    def sample(self, batch_size: int, temperature: float = 1.0):
        indices = torch.zeros(batch_size, 1, dtype=torch.long, device=self.pos_emb.device)
        
        for i in range(self.seq_len - 1):
            logits = self.forward(indices)
            logits = logits[:, -1] / temperature
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1)
            indices = torch.cat([indices, next_idx], dim=1)
        
        return indices


if __name__ == "__main__":
    torch.manual_seed(42)
    
    vqvae = VQVAE(in_channels=3, num_embeddings=512, embedding_dim=64)
    
    x = torch.randn(4, 3, 32, 32)
    x_recon, vq_loss, indices = vqvae(x)
    
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"VQ Loss: {vq_loss.item():.4f}")
    print(f"Indices: {indices.shape}")
    
    seq_len = indices.shape[1] * indices.shape[2]
    prior = TransformerPrior(num_embeddings=512, seq_len=seq_len, embed_dim=128, num_layers=2)
    
    flat_indices = indices.flatten(1)
    logits = prior(flat_indices)
    print(f"Prior logits: {logits.shape}")
