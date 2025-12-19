from __future__ import annotations

from types import ModuleType

import torch
import torch.nn as nn


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def _copy_weights_to_torch_gru(
    candidate_gru: nn.Module, torch_gru: nn.GRU
) -> None:
    """Copy weights from candidate SimpleGRU to torch.nn.GRU for comparison.
    
    PyTorch GRU weight layout: weight_ih_l0 and weight_hh_l0 each contain
    stacked weights for [reset, update, new] gates (in that order).
    """
    with torch.no_grad():
        H = candidate_gru.hidden_size
        
        # Build weight_ih: stack [r, z, n] from input linear layers
        weight_ih = torch.cat([
            candidate_gru.input_linear_rz.weight,  # [2*H, input_size] -> r, z
            candidate_gru.input_linear_n.weight,   # [H, input_size] -> n
        ], dim=0)
        torch_gru.weight_ih_l0.copy_(weight_ih)
        
        # Build weight_hh: stack [r, z, n] from hidden linear layers
        weight_hh = torch.cat([
            candidate_gru.hidden_linear_rz.weight,  # [2*H, hidden_size] -> r, z
            candidate_gru.hidden_linear_n.weight,   # [H, hidden_size] -> n
        ], dim=0)
        torch_gru.weight_hh_l0.copy_(weight_hh)
        
        # Copy biases if present
        if candidate_gru.input_linear_rz.bias is not None:
            bias_ih = torch.cat([
                candidate_gru.input_linear_rz.bias,  # [2*H] -> r, z
                candidate_gru.input_linear_n.bias,   # [H] -> n
            ], dim=0)
            torch_gru.bias_ih_l0.copy_(bias_ih)
            
            bias_hh = torch.cat([
                candidate_gru.hidden_linear_rz.bias,  # [2*H] -> r, z
                candidate_gru.hidden_linear_n.bias,   # [H] -> n
            ], dim=0)
            torch_gru.bias_hh_l0.copy_(bias_hh)


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "SimpleGRU"):
        raise AssertionError("Candidate must define class `SimpleGRU`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- test 1: output shapes ---
    input_size = 4
    hidden_size = 3
    B, T = 2, 5

    m = candidate.SimpleGRU(
        input_size=input_size, hidden_size=hidden_size, bias=True
    ).to(dtype=dtype)
    m.eval()

    x = torch.randn(B, T, input_size, dtype=dtype)
    output, h_n = m(x)

    assert output.shape == (B, T, hidden_size), (
        f"Expected output shape {(B, T, hidden_size)}, got {output.shape}"
    )
    assert h_n.shape == (B, hidden_size), (
        f"Expected h_n shape {(B, hidden_size)}, got {h_n.shape}"
    )

    # --- test 2: final hidden state equals last output ---
    _assert_allclose(
        output[:, -1, :], h_n,
        atol=1e-12, rtol=1e-12,
        msg="Final output timestep should equal h_n."
    )

    # --- test 3: match torch.nn.GRU output ---
    ref_gru = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
    ).to(dtype=dtype)
    ref_gru.eval()

    _copy_weights_to_torch_gru(m, ref_gru)

    x2 = torch.randn(B, T, input_size, dtype=dtype)
    out_candidate, h_candidate = m(x2)
    out_ref, h_ref = ref_gru(x2)

    # PyTorch GRU returns h_n with shape (num_layers, B, hidden_size)
    h_ref = h_ref.squeeze(0)

    _assert_allclose(
        out_candidate, out_ref,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch vs torch.nn.GRU."
    )
    _assert_allclose(
        h_candidate, h_ref,
        atol=1e-10, rtol=1e-10,
        msg="Final hidden state mismatch vs torch.nn.GRU."
    )

    # --- test 4: with initial hidden state ---
    h_0 = torch.randn(B, hidden_size, dtype=dtype)

    out_candidate2, h_candidate2 = m(x2, h0=h_0)
    out_ref2, h_ref2 = ref_gru(x2, h_0.unsqueeze(0))
    h_ref2 = h_ref2.squeeze(0)

    _assert_allclose(
        out_candidate2, out_ref2,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch with initial h0 vs torch.nn.GRU."
    )
    _assert_allclose(
        h_candidate2, h_ref2,
        atol=1e-10, rtol=1e-10,
        msg="h_n mismatch with initial h0 vs torch.nn.GRU."
    )

    # --- test 5: no bias ---
    m_nobias = candidate.SimpleGRU(
        input_size=input_size, hidden_size=hidden_size, bias=False
    ).to(dtype=dtype)
    m_nobias.eval()

    ref_nobias = nn.GRU(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=False,
        batch_first=True,
    ).to(dtype=dtype)
    ref_nobias.eval()

    _copy_weights_to_torch_gru(m_nobias, ref_nobias)

    x3 = torch.randn(B, T, input_size, dtype=dtype)
    out_nb, h_nb = m_nobias(x3)
    out_ref_nb, h_ref_nb = ref_nobias(x3)
    h_ref_nb = h_ref_nb.squeeze(0)

    _assert_allclose(
        out_nb, out_ref_nb,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch (no bias) vs torch.nn.GRU."
    )

    # --- test 6: gradients flow back to input ---
    xg = torch.randn(B, T, input_size, dtype=dtype, requires_grad=True)
    yg = m(xg)[0].sum()
    yg.backward()
    if xg.grad is None:
        raise AssertionError("Expected non-None gradient for input.")
    if not torch.isfinite(xg.grad).all():
        raise AssertionError("Found non-finite values in input gradient.")

    # --- test 7: single timestep ---
    x_single = torch.randn(B, 1, input_size, dtype=dtype)
    out_single, h_single = m(x_single)
    assert out_single.shape == (B, 1, hidden_size), (
        f"Single timestep output shape wrong: {out_single.shape}"
    )
    _assert_allclose(
        out_single[:, 0, :], h_single,
        atol=1e-12, rtol=1e-12,
        msg="Single timestep: output should equal h_n."
    )


