## Mathematical Background

### The VAE Objective

A Variational Autoencoder maximizes the **Evidence Lower Bound (ELBO)** on the log-likelihood of data:

$$
\log p(x) \geq \mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right] - D_{\text{KL}}\left(q_\phi(z|x) \| p(z)\right) = \mathcal{L}(\theta, \phi; x)
$$

where:
- \(p(z) = \mathcal{N}(0, I)\) is the prior over latent variables
- \(q_\phi(z|x)\) is the approximate posterior (encoder)
- \(p_\theta(x|z)\) is the likelihood (decoder)

Since we maximize the ELBO, we equivalently **minimize the negative ELBO**:

$$
\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}\left[\log p_\theta(x|z)\right]}_{\text{Reconstruction Loss}} + \underbrace{D_{\text{KL}}\left(q_\phi(z|x) \| p(z)\right)}_{\text{KL Divergence}}
$$

### Encoder: Approximate Posterior

The encoder parameterizes a diagonal Gaussian posterior:

$$
q_\phi(z|x) = \mathcal{N}(z; \mu_\phi(x), \text{diag}(\sigma^2_\phi(x)))
$$

We output \(\log \sigma^2\) (logvar) instead of \(\sigma^2\) for numerical stability, since:
- \(\sigma^2 > 0\) always, but neural networks output unbounded values
- \(\sigma = \exp(0.5 \cdot \log\sigma^2)\)

### The Reparameterization Trick

To backpropagate through stochastic sampling, we reparameterize:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

This moves the stochasticity to \(\epsilon\), making \(z\) a deterministic function of \(\mu\), \(\sigma\), and the noise \(\epsilon\). Gradients flow through \(\mu\) and \(\sigma\) normally.

### KL Divergence (Closed Form)

For two Gaussians \(q = \mathcal{N}(\mu, \sigma^2)\) and \(p = \mathcal{N}(0, 1)\), the KL divergence has a closed form:

$$
D_{\text{KL}}(q \| p) = \frac{1}{2}\left(\sigma^2 + \mu^2 - 1 - \log\sigma^2\right)
$$

Summing over all \(d\) latent dimensions:

$$
D_{\text{KL}} = \frac{1}{2} \sum_{j=1}^{d} \left(\sigma_j^2 + \mu_j^2 - 1 - \log\sigma_j^2\right) = -\frac{1}{2} \sum_{j=1}^{d} \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)
$$

In code with `logvar = log(σ²)`:

```python
kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
```

### Reconstruction Loss

Assuming Gaussian likelihood \(p_\theta(x|z) = \mathcal{N}(x; \hat{x}_\theta(z), I)\), the negative log-likelihood is proportional to MSE:

$$
-\log p_\theta(x|z) \propto \|x - \hat{x}\|^2
$$

We use sum reduction over all dimensions and batch elements:

```python
recon_loss = F.mse_loss(recon_x, x, reduction="sum")
```

---

## Approach

- **Encoder**: A two-layer MLP that maps input `x` to hidden representation, then projects to `mu` and `logvar` (mean and log-variance of the latent Gaussian).
- **Reparameterization trick**: Sample `z = mu + std * epsilon` where `std = exp(0.5 * logvar)` and `epsilon ~ N(0,1)`. This allows gradients to flow through the sampling operation.
- **Decoder**: A two-layer MLP that maps `z` back to reconstruction, with sigmoid output to bound values in [0, 1].
- **VAE loss** combines two terms:
  - Reconstruction loss (MSE): measures how well the decoder reconstructs the input.
  - KL divergence: regularizes the latent space to be close to a standard normal `N(0, I)`.
- The KL divergence for Gaussians has a closed form: `-0.5 * sum(1 + logvar - mu^2 - exp(logvar))`.
- Both losses use sum reduction to aggregate over all dimensions and batch elements.

## Correctness

- The reparameterization trick ensures that sampling is differentiable: randomness is introduced via `epsilon` which does not depend on learnable parameters.
- The encoder outputs `logvar` instead of `var` or `std` for numerical stability (log-space is unbounded and easier to optimize).
- KL divergence derivation assumes the prior is `N(0, I)` and the posterior is diagonal Gaussian `N(mu, diag(exp(logvar)))`.
- Sigmoid on decoder output ensures reconstruction values are in [0, 1], matching normalized image data.

## Complexity

- **Time**: O(batch_size × (input_dim × hidden_dim + hidden_dim × latent_dim)) per forward pass.
- **Space**: O(input_dim × hidden_dim + hidden_dim × latent_dim) for model parameters.

## Common Pitfalls

- Forgetting to use `logvar` (using variance directly leads to numerical issues with negative values).
- Using `0.5 * logvar` instead of `exp(0.5 * logvar)` for std—remember to exponentiate!
- Using mean reduction for losses instead of sum (changes the loss scale and KL weight).
- Not using `torch.randn_like(std)` which correctly samples on the same device/dtype.
- Forgetting sigmoid activation on decoder output.
- Sign error in KL divergence formula (it should be subtracted from the ELBO, so the loss term is positive).
