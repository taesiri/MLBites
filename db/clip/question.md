# CLIP from Scratch

## Problem Statement

Implement **CLIP (Contrastive Language-Image Pre-training)** from scratch. CLIP learns to align images and text in a shared embedding space using contrastive learning.

Your task is to:

1. Implement image encoder (ViT or ResNet)
2. Implement text encoder (Transformer)
3. Compute contrastive loss between image-text pairs
4. Enable zero-shot classification using text prompts

## CLIP Architecture

```
Image → Image Encoder → Image Embedding → Project → Normalize
Text  → Text Encoder  → Text Embedding  → Project → Normalize

Contrastive Loss: maximize similarity of matching pairs,
                  minimize similarity of non-matching pairs
```

## Function Signature

```python
class CLIP(nn.Module):
    def __init__(self, embed_dim: int, image_encoder: nn.Module, text_encoder: nn.Module):
        pass
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to normalized embeddings."""
        pass
    
    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        """Encode text to normalized embeddings."""
        pass
    
    def forward(self, images: torch.Tensor, text: torch.Tensor) -> tuple:
        """Return (image_features, text_features, logit_scale)."""
        pass

def clip_loss(image_features: torch.Tensor, text_features: torch.Tensor, 
              logit_scale: torch.Tensor) -> torch.Tensor:
    """Symmetric contrastive loss."""
    pass
```

## CLIP Loss

```python
# Compute similarity matrix
logits = logit_scale * image_features @ text_features.T

# Symmetric cross-entropy
labels = torch.arange(batch_size)
loss_i = F.cross_entropy(logits, labels)      # image -> text
loss_t = F.cross_entropy(logits.T, labels)    # text -> image
loss = (loss_i + loss_t) / 2
```

## Example

```python
clip = CLIP(embed_dim=512, image_encoder=vit, text_encoder=transformer)

images = torch.randn(32, 3, 224, 224)
text_tokens = torch.randint(0, 49408, (32, 77))

image_features, text_features, logit_scale = clip(images, text_tokens)
loss = clip_loss(image_features, text_features, logit_scale)

# Zero-shot classification
prompts = ["a photo of a cat", "a photo of a dog"]
text_features = clip.encode_text(tokenize(prompts))
similarities = image_features @ text_features.T
predictions = similarities.argmax(dim=-1)
```

## Hints

- `logit_scale` is learned (initialized to log(1/0.07))
- Text encoder uses causal masking
- Take [EOS] token embedding from text encoder
- Use LayerNorm after projection
