# CLIP Linear Probe

## Problem Statement

Implement a **linear probe** on CLIP features for image classification. Linear probing evaluates the quality of learned representations by training only a linear classifier on frozen features.

Your task is to:

1. Load a pre-trained CLIP model
2. Extract image features (frozen)  
3. Train a linear classifier on top
4. Evaluate on a downstream classification task

## Requirements

- Freeze CLIP's image encoder
- Only train the linear classification head
- Use extracted features for training efficiency
- Compare with zero-shot CLIP

## Function Signature

```python
def extract_clip_features(model, dataloader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract features for entire dataset."""
    pass

class LinearProbe(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int):
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass

def train_linear_probe(features, labels, num_classes, epochs=100):
    """Train linear classifier on extracted features."""
    pass
```

## Example

```python
import clip

# Load CLIP
model, preprocess = clip.load("ViT-B/32", device="cuda")

# Extract features
train_features, train_labels = extract_clip_features(model, train_loader, device)

# Train linear probe
probe = LinearProbe(512, num_classes=10)
train_linear_probe(probe, train_features, train_labels, epochs=100)

# Evaluate
test_features, test_labels = extract_clip_features(model, test_loader, device)
accuracy = evaluate(probe, test_features, test_labels)
```

## Zero-Shot vs Linear Probe

| Method | Description |
|--------|-------------|
| Zero-Shot | Use text prompts to classify (no training) |
| Linear Probe | Train linear classifier on frozen features |
| Fine-Tuning | Update all model weights (expensive) |

## Hints

- Features from CLIP's image encoder are already normalized
- Use `model.encode_image()` to get image features
- L-BFGS optimizer can converge faster than SGD for linear probe
- Try different learning rates (0.1 to 0.0001)
