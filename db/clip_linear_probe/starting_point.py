"""
CLIP Linear Probe - Starting Point

Implement linear probing on CLIP features.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """Linear classifier for probing."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        # TODO: Create linear layer
        pass
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pass


@torch.no_grad()
def extract_clip_features(model, dataloader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Extract CLIP features for entire dataset.
    
    Args:
        model: CLIP model
        dataloader: DataLoader with images
        device: Device to run on
        
    Returns:
        (features, labels) tensors
    """
    # TODO: Set model to eval mode
    # TODO: Iterate through dataloader
    # TODO: Extract image features using model.encode_image()
    # TODO: Collect all features and labels
    pass


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 256
) -> LinearProbe:
    """
    Train linear classifier on extracted features.
    """
    # TODO: Create LinearProbe
    
    # TODO: Create DataLoader
    
    # TODO: Training loop
    
    pass


def evaluate_probe(probe: LinearProbe, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Evaluate linear probe accuracy."""
    # TODO: Forward pass and compute accuracy
    pass


def zero_shot_clip(model, images, text_prompts, device):
    """Zero-shot classification with CLIP."""
    # TODO: Encode images and text
    # TODO: Compute similarity and predict
    pass


if __name__ == "__main__":
    # Note: This requires the 'clip' package
    # pip install git+https://github.com/openai/CLIP.git
    
    try:
        import clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"CLIP loaded on {device}")
        print(f"Image encoder output dim: 512")
        
        # Demo with random data
        dummy_images = torch.randn(100, 3, 224, 224)
        dummy_labels = torch.randint(0, 10, (100,))
        
        print("\nFor real usage:")
        print("1. Load your dataset with preprocess transform")
        print("2. Extract features with extract_clip_features()")
        print("3. Train linear probe on features")
        print("4. Evaluate on test set")
        
    except ImportError:
        print("CLIP not installed. Install with:")
        print("pip install git+https://github.com/openai/CLIP.git")
        
        # Demo without CLIP
        print("\nDemo with random features:")
        features = torch.randn(100, 512)
        labels = torch.randint(0, 10, (100,))
        
        probe = LinearProbe(512, 10)
        probe = train_linear_probe(features, labels, num_classes=10, epochs=10)
        
        acc = evaluate_probe(probe, features, labels)
        print(f"Training accuracy: {acc:.2%}")
