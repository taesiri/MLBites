"""
Autoencoder for Anomaly Detection - Starting Point

Implement an autoencoder and use it for anomaly detection.
Fill in the TODO sections to complete the implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    """Simple fully-connected autoencoder."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: list = [256, 128]):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # TODO: Build encoder
        # Input -> hidden layers -> latent
        
        # TODO: Build decoder
        # Latent -> hidden layers -> output
        
        pass
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        # TODO: Pass through encoder
        pass
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        # TODO: Pass through decoder
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        # TODO: Encode then decode
        pass


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for images."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # TODO: Build convolutional encoder
        # Conv layers with stride to downsample
        
        # TODO: Build convolutional decoder
        # ConvTranspose layers to upsample
        
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


def train_autoencoder(
    model: nn.Module,
    train_loader,
    epochs: int = 10,
    lr: float = 0.001
) -> list[float]:
    """
    Train autoencoder on data.
    
    Returns list of average losses per epoch.
    """
    # TODO: Define optimizer
    
    # TODO: Training loop
    # For each batch: forward, compute MSE loss, backward, step
    
    pass


def detect_anomalies(
    model: nn.Module,
    data: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """
    Detect anomalies based on reconstruction error.
    
    Args:
        model: Trained autoencoder
        data: Input data to check
        threshold: Error threshold above which = anomaly
        
    Returns:
        Boolean tensor (True = anomaly)
    """
    # TODO: Get reconstruction
    
    # TODO: Compute per-sample reconstruction error
    
    # TODO: Compare to threshold
    
    pass


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sample data (normal = low values, anomaly = high values)
    normal_data = torch.randn(100, 784) * 0.1
    anomaly_data = torch.randn(10, 784) * 2.0 + 5.0
    
    # Create and train autoencoder
    ae = Autoencoder(input_dim=784, latent_dim=32)
    
    print("Training autoencoder on normal data...")
    # Simple training loop
    optimizer = torch.optim.Adam(ae.parameters(), lr=0.001)
    
    for epoch in range(50):
        recon = ae(normal_data)
        loss = F.mse_loss(recon, normal_data)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Detect anomalies
    print("\nDetecting anomalies...")
    all_data = torch.cat([normal_data, anomaly_data], dim=0)
    anomalies = detect_anomalies(ae, all_data, threshold=0.5)
    
    print(f"Normal samples flagged: {anomalies[:100].sum()}/{100}")
    print(f"Anomaly samples flagged: {anomalies[100:].sum()}/{10}")
