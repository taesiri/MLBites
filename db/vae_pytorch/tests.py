from __future__ import annotations

from types import ModuleType

import torch


def _assert_shape(tensor: torch.Tensor, expected: tuple, name: str) -> None:
    if tensor.shape != expected:
        raise AssertionError(f"{name} shape mismatch: expected {expected}, got {tensor.shape}")


def _assert_close(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines VAE class and vae_loss function.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "VAE"):
        raise AssertionError("Candidate must define class `VAE`.")
    if not hasattr(candidate, "vae_loss"):
        raise AssertionError("Candidate must define function `vae_loss`.")

    torch.manual_seed(42)

    # --- Test 1: Output shapes ---
    vae = candidate.VAE(input_dim=784, hidden_dim=400, latent_dim=20)
    x = torch.randn(32, 784)
    recon_x, mu, logvar = vae(x)

    _assert_shape(recon_x, (32, 784), "recon_x")
    _assert_shape(mu, (32, 20), "mu")
    _assert_shape(logvar, (32, 20), "logvar")

    # --- Test 2: Encoder output shapes ---
    mu_enc, logvar_enc = vae.encode(x)
    _assert_shape(mu_enc, (32, 20), "encode mu")
    _assert_shape(logvar_enc, (32, 20), "encode logvar")

    # --- Test 3: Reparameterization output shape ---
    z = vae.reparameterize(mu_enc, logvar_enc)
    _assert_shape(z, (32, 20), "reparameterize z")

    # --- Test 4: Decoder output shape and range ---
    decoded = vae.decode(z)
    _assert_shape(decoded, (32, 784), "decode output")
    if decoded.min() < 0 or decoded.max() > 1:
        raise AssertionError(
            f"Decoder output should be in [0, 1] (sigmoid), got [{decoded.min():.4f}, {decoded.max():.4f}]"
        )

    # --- Test 5: vae_loss with perfect reconstruction and standard normal latent ---
    x_test = torch.ones(2, 4)
    recon_test = torch.ones(2, 4)
    mu_test = torch.zeros(2, 3)
    logvar_test = torch.zeros(2, 3)  # std=1, so KL = 0
    loss = candidate.vae_loss(recon_test, x_test, mu_test, logvar_test)

    _assert_close(
        loss,
        torch.tensor(0.0),
        atol=1e-6,
        rtol=1e-6,
        msg="Loss should be 0 for perfect reconstruction and standard normal latent",
    )

    # --- Test 6: vae_loss KL divergence component ---
    # Test with non-zero mu: KL = 0.5 * sum(mu^2) when logvar=0
    x_test2 = torch.zeros(1, 4)
    recon_test2 = torch.zeros(1, 4)  # perfect reconstruction
    mu_test2 = torch.tensor([[1.0, 2.0]])  # latent_dim=2
    logvar_test2 = torch.zeros(1, 2)
    loss_kl = candidate.vae_loss(recon_test2, x_test2, mu_test2, logvar_test2)

    # KL = -0.5 * sum(1 + 0 - mu^2 - 1) = -0.5 * sum(-mu^2) = 0.5 * sum(mu^2)
    # = 0.5 * (1 + 4) = 2.5
    expected_kl = torch.tensor(2.5)
    _assert_close(
        loss_kl,
        expected_kl,
        atol=1e-5,
        rtol=1e-5,
        msg="KL divergence computation incorrect",
    )

    # --- Test 7: vae_loss reconstruction component ---
    x_test3 = torch.tensor([[1.0, 2.0, 3.0]])
    recon_test3 = torch.tensor([[0.0, 1.0, 2.0]])  # diff of 1 for each
    mu_test3 = torch.zeros(1, 2)
    logvar_test3 = torch.zeros(1, 2)
    loss_recon = candidate.vae_loss(recon_test3, x_test3, mu_test3, logvar_test3)

    # MSE sum = (1-0)^2 + (2-1)^2 + (3-2)^2 = 1 + 1 + 1 = 3
    # KL = 0 (standard normal)
    expected_recon = torch.tensor(3.0)
    _assert_close(
        loss_recon,
        expected_recon,
        atol=1e-5,
        rtol=1e-5,
        msg="Reconstruction loss computation incorrect",
    )

    # --- Test 8: Reparameterization determinism with fixed seed ---
    torch.manual_seed(123)
    z1 = vae.reparameterize(mu_enc, logvar_enc)
    torch.manual_seed(123)
    z2 = vae.reparameterize(mu_enc, logvar_enc)
    _assert_close(z1, z2, atol=1e-7, rtol=1e-7, msg="Reparameterization should be deterministic with same seed")

    # --- Test 9: Gradient flow through reparameterization ---
    mu_grad = torch.randn(4, 8, requires_grad=True)
    logvar_grad = torch.randn(4, 8, requires_grad=True)
    
    vae_small = candidate.VAE(input_dim=16, hidden_dim=8, latent_dim=8)
    z_grad = vae_small.reparameterize(mu_grad, logvar_grad)
    loss_grad = z_grad.sum()
    loss_grad.backward()

    if mu_grad.grad is None:
        raise AssertionError("Gradients should flow through mu in reparameterization")
    if logvar_grad.grad is None:
        raise AssertionError("Gradients should flow through logvar in reparameterization")

    # --- Test 10: Full forward-backward pass ---
    torch.manual_seed(42)
    vae_train = candidate.VAE(input_dim=64, hidden_dim=32, latent_dim=8)
    x_train = torch.rand(16, 64)
    
    recon, mu_out, logvar_out = vae_train(x_train)
    loss_train = candidate.vae_loss(recon, x_train, mu_out, logvar_out)
    loss_train.backward()

    # Check that all parameters have gradients
    for name, param in vae_train.named_parameters():
        if param.grad is None:
            raise AssertionError(f"Parameter {name} has no gradient after backward pass")


