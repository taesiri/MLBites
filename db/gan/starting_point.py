"""
GAN - Starting Point

Implement a Generative Adversarial Network from scratch.
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Generator network that creates images from noise."""
    
    def __init__(self, latent_dim: int = 100, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_size = img_size
        
        # TODO: Build generator network
        # Input: (batch, latent_dim)
        # Output: (batch, img_channels, img_size, img_size)
        
        # Hint: Use Linear -> Reshape -> ConvTranspose2d layers
        pass
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from latent noise."""
        # TODO: Forward pass
        pass


class Discriminator(nn.Module):
    """Discriminator network that classifies real vs fake."""
    
    def __init__(self, img_channels: int = 1, img_size: int = 28):
        super().__init__()
        
        # TODO: Build discriminator network
        # Input: (batch, img_channels, img_size, img_size)
        # Output: (batch, 1) - probability of being real
        
        # Hint: Use Conv2d -> LeakyReLU -> Flatten -> Linear
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify images as real or fake."""
        # TODO: Forward pass
        pass


def train_gan(G, D, dataloader, epochs=10, latent_dim=100, device='cpu'):
    """
    Train GAN with alternating optimization.
    """
    # TODO: Create optimizers for G and D
    
    # TODO: Define loss function (BCEWithLogitsLoss)
    
    for epoch in range(epochs):
        for real_images, _ in dataloader:
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # TODO: Train Discriminator
            # 1. Get D's predictions on real images
            # 2. Generate fake images
            # 3. Get D's predictions on fake images
            # 4. Compute D loss and update
            
            # TODO: Train Generator
            # 1. Generate fake images
            # 2. Get D's predictions
            # 3. Compute G loss (fool D) and update
            
            pass
    
    return G, D


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Create models
    latent_dim = 100
    G = Generator(latent_dim=latent_dim, img_channels=1, img_size=28)
    D = Discriminator(img_channels=1, img_size=28)
    
    # Test forward pass
    z = torch.randn(16, latent_dim)
    fake_images = G(z)
    print(f"Generated images shape: {fake_images.shape}")
    
    scores = D(fake_images)
    print(f"Discriminator scores shape: {scores.shape}")
