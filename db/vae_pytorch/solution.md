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

