"""
CLIP from Scratch - Solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )
    
    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ImageEncoder(nn.Module):
    def __init__(self, image_size=224, patch_size=16, in_channels=3,
                 embed_dim=768, num_heads=12, num_layers=12):
        super().__init__()
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x[:, 0]


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=49408, max_len=77, embed_dim=512, num_heads=8, num_layers=12):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        
        # Causal mask
        self.register_buffer('mask', torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())
    
    def forward(self, text):
        B, N = text.shape
        x = self.token_embedding(text) + self.pos_embedding[:, :N]
        
        mask = self.mask[:N, :N]
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Take features from EOS token (last non-padding token)
        # For simplicity, take last position
        return x[torch.arange(B), text.argmax(dim=-1)]


class CLIP(nn.Module):
    def __init__(self, embed_dim=512, image_encoder=None, text_encoder=None,
                 image_embed_dim=None, text_embed_dim=None):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        image_embed_dim = image_embed_dim or embed_dim
        text_embed_dim = text_embed_dim or embed_dim
        
        self.image_projection = nn.Linear(image_embed_dim, embed_dim, bias=False)
        self.text_projection = nn.Linear(text_embed_dim, embed_dim, bias=False)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))
    
    def encode_image(self, images):
        x = self.image_encoder(images)
        x = self.image_projection(x)
        return F.normalize(x, dim=-1)
    
    def encode_text(self, text):
        x = self.text_encoder(text)
        x = self.text_projection(x)
        return F.normalize(x, dim=-1)
    
    def forward(self, images, text):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text)
        return image_features, text_features, self.logit_scale.exp()


def clip_loss(image_features, text_features, logit_scale):
    """Symmetric contrastive loss."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logits_per_image.T
    
    batch_size = image_features.shape[0]
    labels = torch.arange(batch_size, device=image_features.device)
    
    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    
    return (loss_i + loss_t) / 2


def zero_shot_classify(clip_model, images, text_features):
    """Zero-shot classification."""
    image_features = clip_model.encode_image(images)
    similarities = image_features @ text_features.T
    return similarities.argmax(dim=-1)


if __name__ == "__main__":
    torch.manual_seed(42)
    
    image_enc = ImageEncoder(image_size=32, patch_size=4, embed_dim=256, num_heads=4, num_layers=2)
    text_enc = TextEncoder(vocab_size=1000, max_len=16, embed_dim=256, num_heads=4, num_layers=2)
    
    clip = CLIP(embed_dim=256, image_encoder=image_enc, text_encoder=text_enc,
                image_embed_dim=256, text_embed_dim=256)
    
    images = torch.randn(8, 3, 32, 32)
    text = torch.randint(0, 1000, (8, 16))
    
    image_features, text_features, logit_scale = clip(images, text)
    loss = clip_loss(image_features, text_features, logit_scale)
    
    print(f"Image features: {image_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Logit scale: {logit_scale.item():.2f}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters: {sum(p.numel() for p in clip.parameters()):,}")
