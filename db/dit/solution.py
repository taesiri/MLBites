"""
Diffusion Transformer (DiT) - Solution

Complete DiT implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=timesteps.device) / half)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """DiT block with AdaLN-Zero."""
    
    def __init__(self, hidden_size: int, num_heads: int, cond_dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.attn = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        
        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, hidden_size),
        )
        
        # AdaLN modulation: (shift, scale, gate) for each of attn and mlp
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * hidden_size),
        )
        
        # Zero initialization for gates
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # Get modulation parameters
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN(c).chunk(6, dim=-1)
        
        # Attention with AdaLN
        h = self.norm1(x) * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        x = x + gate_msa.unsqueeze(1) * self.attn(h)
        
        # MLP with AdaLN
        h = self.norm2(x) * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(h)
        
        return x


class FinalLayer(nn.Module):
    """Final layer for DiT."""
    
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 2 * hidden_size),
        )
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN(c).chunk(2, dim=-1)
        x = self.norm(x) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer."""
    
    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_size: int = 384,
        num_heads: int = 6,
        num_layers: int = 12,
        num_classes: int = 10,
        cond_dim: int = None
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.img_size = img_size
        cond_dim = cond_dim or hidden_size
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_size, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )
        
        # Class embedding
        self.class_embed = nn.Embedding(num_classes, cond_dim)
        
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, cond_dim)
            for _ in range(num_layers)
        ])
        
        self.final = FinalLayer(hidden_size, patch_size, in_channels, cond_dim)
        
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Patch embed
        x = self.patch_embed(x) + self.pos_embed
        
        # Conditioning
        t_emb = get_timestep_embedding(t, self.pos_embed.shape[-1])
        c = self.time_embed(t_emb) + self.class_embed(y)
        
        # DiT blocks
        for block in self.blocks:
            x = block(x, c)
        
        # Final layer and unpatchify
        x = self.final(x, c)
        x = self.unpatchify(x)
        
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        p = self.patch_size
        c = self.in_channels
        
        x = x.reshape(B, h, w, p, p, c)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, c, h * p, w * p)
        return x


if __name__ == "__main__":
    torch.manual_seed(42)
    
    dit = DiT(
        img_size=32, patch_size=4, in_channels=3,
        hidden_size=384, num_heads=6, num_layers=6,
        num_classes=10
    )
    
    x = torch.randn(4, 3, 32, 32)
    t = torch.randint(0, 1000, (4,))
    y = torch.randint(0, 10, (4,))
    
    noise_pred = dit(x, t, y)
    
    print(f"Input: {x.shape}")
    print(f"Noise prediction: {noise_pred.shape}")
    print(f"Parameters: {sum(p.numel() for p in dit.parameters()):,}")
