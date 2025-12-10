"""
CLIP Linear Probe - Solution

Complete implementation of linear probing on CLIP features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class LinearProbe(nn.Module):
    """Linear classifier for probing."""
    
    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(feature_dim, num_classes)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)


@torch.no_grad()
def extract_clip_features(model, dataloader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract CLIP features for entire dataset."""
    model.eval()
    
    all_features = []
    all_labels = []
    
    for images, labels in dataloader:
        images = images.to(device)
        
        # Extract image features
        features = model.encode_image(images)
        features = features.float()  # CLIP uses float16
        features = F.normalize(features, dim=-1)  # Normalize
        
        all_features.append(features.cpu())
        all_labels.append(labels)
    
    return torch.cat(all_features), torch.cat(all_labels)


def train_linear_probe(
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    epochs: int = 100,
    lr: float = 0.1,
    batch_size: int = 256,
    weight_decay: float = 0.0
) -> LinearProbe:
    """Train linear classifier on extracted features."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    probe = LinearProbe(features.shape[1], num_classes).to(device)
    
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            logits = probe(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += batch_labels.size(0)
            correct += predicted.eq(batch_labels).sum().item()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            acc = 100 * correct / total
            print(f"Epoch {epoch+1}: Loss={total_loss/len(loader):.4f}, Acc={acc:.2f}%")
    
    return probe


@torch.no_grad()
def evaluate_probe(probe: LinearProbe, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Evaluate linear probe accuracy."""
    device = next(probe.parameters()).device
    features = features.to(device)
    labels = labels.to(device)
    
    probe.eval()
    logits = probe(features)
    _, predicted = logits.max(1)
    
    accuracy = predicted.eq(labels).float().mean().item()
    return accuracy


@torch.no_grad()
def zero_shot_clip(model, images, text_prompts, device):
    """Zero-shot classification with CLIP."""
    import clip
    
    # Encode images
    image_features = model.encode_image(images.to(device))
    image_features = F.normalize(image_features, dim=-1)
    
    # Encode text prompts
    text_tokens = clip.tokenize(text_prompts).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = F.normalize(text_features, dim=-1)
    
    # Compute similarity
    similarity = image_features @ text_features.T
    predictions = similarity.argmax(dim=-1)
    
    return predictions


if __name__ == "__main__":
    print("CLIP Linear Probe Demo")
    
    try:
        import clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        
        print(f"CLIP loaded on {device}")
        print(f"Feature dim: 512")
        
    except ImportError:
        print("CLIP not installed, using random features demo")
    
    # Demo with random features
    print("\nTraining linear probe on random features:")
    features = torch.randn(1000, 512)
    labels = torch.randint(0, 10, (1000,))
    
    probe = train_linear_probe(features, labels, num_classes=10, epochs=50, lr=0.1)
    
    # Evaluate
    test_features = torch.randn(100, 512)
    test_labels = torch.randint(0, 10, (100,))
    
    acc = evaluate_probe(probe, test_features, test_labels)
    print(f"\nTest accuracy (random): {acc:.2%}")
