# a Simple VAE

## Problem
A Variational Autoencoder (VAE) is a generative model that learns a latent representation of data. It consists of an encoder that maps inputs to a latent distribution, a reparameterization trick that enables backpropagation through sampling, and a decoder that reconstructs the input from the latent space.

## Task
Implement a simple VAE as a PyTorch `nn.Module` with:
1. An **encoder** that outputs `mu` (mean) and `logvar` (log variance) of the latent distribution
2. A **reparameterize** method that samples from the latent distribution using the reparameterization trick: `z = mu + std * epsilon` where `std = exp(0.5 * logvar)` and `epsilon ~ N(0, 1)`
3. A **decoder** that reconstructs the input from the latent vector `z`
4. A **forward** method that returns `(reconstructed, mu, logvar)`

Also implement a **vae_loss** function that computes the VAE loss: reconstruction loss (MSE) + KL divergence.

## Function Signature

```python
class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None: ...
    
    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor: ...
    
    def decode(self, z: torch.Tensor) -> torch.Tensor: ...
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


def vae_loss(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
) -> torch.Tensor: ...
```

## Inputs and Outputs

### VAE Class
- **Inputs**:
  - `input_dim`: dimension of input data (e.g., 784 for flattened 28x28 images)
  - `hidden_dim`: dimension of hidden layers in encoder/decoder
  - `latent_dim`: dimension of latent space
  - `x`: input tensor of shape `(batch_size, input_dim)`
- **Outputs**:
  - `encode(x)`: returns `(mu, logvar)` each of shape `(batch_size, latent_dim)`
  - `reparameterize(mu, logvar)`: returns `z` of shape `(batch_size, latent_dim)`
  - `decode(z)`: returns reconstructed tensor of shape `(batch_size, input_dim)`
  - `forward(x)`: returns `(recon_x, mu, logvar)`

### vae_loss Function
- **Inputs**:
  - `recon_x`: reconstructed input tensor `(batch_size, input_dim)`
  - `x`: original input tensor `(batch_size, input_dim)`
  - `mu`: mean of latent distribution `(batch_size, latent_dim)`
  - `logvar`: log variance of latent distribution `(batch_size, latent_dim)`
- **Output**: scalar tensor containing the total VAE loss (summed over batch)

## Constraints
- Must be solvable in 20–30 minutes.
- Interview-friendly: use simple MLP architecture (single hidden layer each for encoder/decoder).
- Encoder architecture: `input_dim -> hidden_dim -> (mu, logvar)`
- Decoder architecture: `latent_dim -> hidden_dim -> input_dim`
- Use ReLU activations in hidden layers; sigmoid on decoder output.
- KL divergence formula: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`
- Reconstruction loss: MSE (sum reduction, not mean)
- Allowed libs: PyTorch (`torch`, `torch.nn`, `torch.nn.functional`) and Python standard library.

## Examples

### Example 1 (forward pass shapes)
```python
vae = VAE(input_dim=784, hidden_dim=400, latent_dim=20)
x = torch.randn(32, 784)
recon_x, mu, logvar = vae(x)
# recon_x.shape == (32, 784)
# mu.shape == (32, 20)
# logvar.shape == (32, 20)
```

### Example 2 (loss computation)
```python
# For identical reconstruction and standard normal latent:
x = torch.ones(2, 4)
recon_x = torch.ones(2, 4)
mu = torch.zeros(2, 3)
logvar = torch.zeros(2, 3)  # std=1, so KL = 0
loss = vae_loss(recon_x, x, mu, logvar)
# loss == 0.0 (no reconstruction error, KL = 0)
```




