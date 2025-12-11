"""
CLIP from Scratch - Starting Point
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Vision encoder (simplified ViT)."""
    
    def __init__(self, image_size: int = 224, patch_size: int = 16, 
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12):
        super().__init__()
        # TODO: Patch embedding
        # TODO: CLS token
        # TODO: Position embeddings
        # TODO: Transformer blocks
        # TODO: Final LayerNorm
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return [CLS] token embedding."""
        pass


class TextEncoder(nn.Module):
    """Text encoder (Transformer)."""
    
    def __init__(self, vocab_size: int = 49408, max_len: int = 77,
                 embed_dim: int = 512, num_heads: int = 8, num_layers: int = 12):
        super().__init__()
        # TODO: Token embedding
        # TODO: Position embedding
        # TODO: Transformer blocks with causal masking
        # TODO: Final LayerNorm
        pass
    
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Return [EOS] token embedding."""
        pass


class CLIP(nn.Module):
    """CLIP model."""
    
    def __init__(self, embed_dim: int = 512, image_encoder: nn.Module = None, 
                 text_encoder: nn.Module = None):
        super().__init__()
        # TODO: Store encoders
        # TODO: Image projection
        # TODO: Text projection
        # TODO: Learnable temperature (logit_scale)
        pass
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode and normalize images."""
        # TODO: Encode
        # TODO: Project
        # TODO: Normalize
        pass
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode and normalize text."""
        # TODO: Encode
        # TODO: Project
        # TODO: Normalize
        pass
    
    def forward(self, images: torch.Tensor, text: torch.Tensor):
        """Return image features, text features, logit_scale."""
        pass


def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor,
              logit_scale: torch.Tensor) -> torch.Tensor:
    """
    Symmetric contrastive loss.
    
    Each image should match its corresponding text and vice versa.
    """
    # TODO: Compute logits
    # TODO: Create labels (diagonal)
    # TODO: Compute symmetric cross-entropy
    pass


def zero_shot_classify(clip_model: CLIP, images: torch.Tensor, 
                       text_prompts: list[str], tokenizer) -> torch.Tensor:
    """Zero-shot classification using text prompts."""
    # TODO: Encode text prompts
    # TODO: Encode images
    # TODO: Compute similarities
    # TODO: Return predictions
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create simple encoders
    image_enc = ImageEncoder(image_size=32, patch_size=4, embed_dim=256, num_heads=4, num_layers=2)
    text_enc = TextEncoder(vocab_size=1000, max_len=16, embed_dim=256, num_heads=4, num_layers=2)
    
    clip = CLIP(embed_dim=256, image_encoder=image_enc, text_encoder=text_enc)
    
    # Dummy data
    images = torch.randn(8, 3, 32, 32)
    text = torch.randint(0, 1000, (8, 16))
    
    image_features, text_features, logit_scale = clip(images, text)
    loss = clip_loss(image_features, text_features, logit_scale)
    
    print(f"Image features: {image_features.shape}")
    print(f"Text features: {text_features.shape}")
    print(f"Logit scale: {logit_scale.exp().item():.2f}")
    print(f"Loss: {loss.item():.4f}")
