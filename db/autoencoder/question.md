# Autoencoder for Anomaly Detection

## Problem Statement

Implement an **Autoencoder** for anomaly detection using PyTorch. Autoencoders learn to compress and reconstruct data; anomalies can be detected by high reconstruction error.

Your task is to:

1. Build an encoder that compresses input to a latent representation
2. Build a decoder that reconstructs from the latent representation
3. Train on normal data and detect anomalies via reconstruction error

## Requirements

- Implement encoder and decoder networks
- Use reconstruction loss (MSE) for training
- Detect anomalies by thresholding reconstruction error
- Support both fully connected and convolutional autoencoders

## Function Signature

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        pass
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation."""
        pass
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to reconstruction."""
        pass
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full forward pass: encode then decode."""
        pass

def detect_anomalies(
    model: Autoencoder, 
    data: torch.Tensor, 
    threshold: float
) -> torch.Tensor:
    """Detect anomalies based on reconstruction error."""
    pass
```

## Example

```python
import torch

# Create autoencoder
ae = Autoencoder(input_dim=784, latent_dim=32)

# Train on normal data (e.g., MNIST digits 0-4)
for epoch in range(epochs):
    recon = ae(x_train)
    loss = F.mse_loss(recon, x_train)
    # ... backprop

# Detect anomalies (e.g., digit 9 should have high error)
anomalies = detect_anomalies(ae, x_test, threshold=0.1)
```

## Hints

- Train only on "normal" data (one class or clean data)
- Anomalies will have higher reconstruction error
- Choose threshold based on validation data
- Consider using a variational autoencoder (VAE) for better results
