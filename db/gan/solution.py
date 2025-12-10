"""
GAN - Solution

Complete implementation of GAN.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network using transposed convolutions."""
    
    def __init__(self, latent_dim: int = 100, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.init_size = img_size // 4  # 7 for 28x28
        
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 128 * self.init_size * self.init_size),
            nn.BatchNorm1d(128 * self.init_size * self.init_size),
            nn.ReLU(True)
        )
        
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(x.size(0), 128, self.init_size, self.init_size)
        x = self.conv(x)
        return x


class Discriminator(nn.Module):
    """Discriminator network using convolutions."""
    
    def __init__(self, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        
        ds_size = img_size // 4
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * ds_size * ds_size, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc(x)
        return x


def train_gan(G, D, dataloader, epochs=10, latent_dim=100, device='cpu'):
    """Train GAN with alternating optimization."""
    G.to(device)
    D.to(device)
    
    opt_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)
            
            # Train Discriminator
            opt_D.zero_grad()
            
            real_pred = D(real_images)
            real_loss = criterion(real_pred, real_labels)
            
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = G(z).detach()
            fake_pred = D(fake_images)
            fake_loss = criterion(fake_pred, fake_labels)
            
            d_loss = real_loss + fake_loss
            d_loss.backward()
            opt_D.step()
            
            # Train Generator
            opt_G.zero_grad()
            
            z = torch.randn(batch_size, latent_dim, device=device)
            fake_images = G(z)
            fake_pred = D(fake_images)
            g_loss = criterion(fake_pred, real_labels)  # Fool D
            
            g_loss.backward()
            opt_G.step()
        
        print(f"Epoch {epoch+1}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")
    
    return G, D


if __name__ == "__main__":
    torch.manual_seed(42)
    
    latent_dim = 100
    G = Generator(latent_dim=latent_dim, img_channels=1, img_size=28)
    D = Discriminator(img_channels=1, img_size=28)
    
    z = torch.randn(16, latent_dim)
    fake_images = G(z)
    print(f"Generated images shape: {fake_images.shape}")
    
    scores = D(fake_images)
    print(f"Discriminator scores shape: {scores.shape}")
    
    print(f"\nG parameters: {sum(p.numel() for p in G.parameters()):,}")
    print(f"D parameters: {sum(p.numel() for p in D.parameters()):,}")
