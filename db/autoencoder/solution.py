"""
Autoencoder for Anomaly Detection - Solution

Complete implementation of autoencoder for anomaly detection.
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
        
        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU()
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass."""
        z = self.encode(x)
        return self.decode(z)


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder for images."""
    
    def __init__(self, in_channels: int = 1):
        super().__init__()
        
        # Encoder: downsample with strided convolutions
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Decoder: upsample with transposed convolutions
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


class VariationalAutoencoder(nn.Module):
    """Variational Autoencoder (VAE)."""
    
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Encoder
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.fc3(z))
        return self.fc4(h)
    
    def forward(self, x: torch.Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = Reconstruction + KL divergence."""
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss


def detect_anomalies(
    model: nn.Module,
    data: torch.Tensor,
    threshold: float
) -> torch.Tensor:
    """Detect anomalies based on reconstruction error."""
    model.eval()
    with torch.no_grad():
        recon = model(data)
        
        # Compute per-sample reconstruction error
        error = F.mse_loss(recon, data, reduction='none')
        error = error.view(data.size(0), -1).mean(dim=1)
        
        # Compare to threshold
        anomalies = error > threshold
    
    return anomalies


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create sample data (normal = low values, anomaly = high values)
    normal_data = torch.randn(100, 784) * 0.1
    anomaly_data = torch.randn(10, 784) * 2.0 + 5.0
    
    # Create and train autoencoder
    ae = Autoencoder(input_dim=784, latent_dim=32)
    
    print("Training autoencoder on normal data...")
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
