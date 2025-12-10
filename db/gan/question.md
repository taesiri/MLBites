# GAN (Generative Adversarial Network)

## Problem Statement

Implement a **Generative Adversarial Network (GAN)** from scratch. GANs consist of two networks—a Generator that creates fake samples and a Discriminator that tries to distinguish real from fake.

Your task is to:

1. Implement the Generator network
2. Implement the Discriminator network
3. Write the adversarial training loop
4. Handle the min-max optimization

## Requirements

- Generator takes noise and produces images
- Discriminator classifies real vs fake
- Implement proper GAN training (alternating optimization)
- Use appropriate loss functions (BCE or non-saturating)

## Function Signature

```python
class Generator(nn.Module):
    def __init__(self, latent_dim: int, img_channels: int, img_size: int):
        pass
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Generate images from noise."""
        pass

class Discriminator(nn.Module):
    def __init__(self, img_channels: int, img_size: int):
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify real (1) vs fake (0)."""
        pass
```

## GAN Training Loop

```python
for real_images in dataloader:
    # Train Discriminator
    D_loss = -[log(D(real)) + log(1 - D(G(z)))]
    
    # Train Generator
    G_loss = -log(D(G(z)))  # Non-saturating
```

## Example

```python
G = Generator(latent_dim=100, img_channels=1, img_size=28)
D = Discriminator(img_channels=1, img_size=28)

z = torch.randn(64, 100)
fake_images = G(z)  # (64, 1, 28, 28)

scores = D(fake_images)  # (64, 1)
```

## Hints

- Use transposed convolutions in Generator for upsampling
- Use LeakyReLU in Discriminator
- Label smoothing can help training stability
- Train D multiple times per G step if D is weak
