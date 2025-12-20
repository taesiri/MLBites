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


def _copy_weights_to_torch_lstm(
    candidate_lstm: nn.Module, torch_lstm: nn.LSTM
) -> None:
    """Copy weights from candidate SimpleLSTM to torch.nn.LSTM for comparison."""
    with torch.no_grad():
        # PyTorch LSTM uses weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        torch_lstm.weight_ih_l0.copy_(candidate_lstm.input_linear.weight)
        torch_lstm.weight_hh_l0.copy_(candidate_lstm.hidden_linear.weight)
        if candidate_lstm.input_linear.bias is not None:
            torch_lstm.bias_ih_l0.copy_(candidate_lstm.input_linear.bias)
            torch_lstm.bias_hh_l0.copy_(candidate_lstm.hidden_linear.bias)


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "SimpleLSTM"):
        raise AssertionError("Candidate must define class `SimpleLSTM`.")

    torch.manual_seed(42)
    dtype = torch.float64

    # --- test 1: output shapes ---
    input_size = 4
    hidden_size = 3
    B, T = 2, 5

    m = candidate.SimpleLSTM(
        input_size=input_size, hidden_size=hidden_size, bias=True
    ).to(dtype=dtype)
    m.eval()

    x = torch.randn(B, T, input_size, dtype=dtype)
    output, (h_n, c_n) = m(x)

    assert output.shape == (B, T, hidden_size), (
        f"Expected output shape {(B, T, hidden_size)}, got {output.shape}"
    )
    assert h_n.shape == (B, hidden_size), (
        f"Expected h_n shape {(B, hidden_size)}, got {h_n.shape}"
    )
    assert c_n.shape == (B, hidden_size), (
        f"Expected c_n shape {(B, hidden_size)}, got {c_n.shape}"
    )

    # --- test 2: final hidden state equals last output ---
    _assert_allclose(
        output[:, -1, :], h_n,
        atol=1e-12, rtol=1e-12,
        msg="Final output timestep should equal h_n."
    )

    # --- test 3: match torch.nn.LSTM output ---
    ref_lstm = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=True,
        batch_first=True,
    ).to(dtype=dtype)
    ref_lstm.eval()

    _copy_weights_to_torch_lstm(m, ref_lstm)

    x2 = torch.randn(B, T, input_size, dtype=dtype)
    out_candidate, (h_candidate, c_candidate) = m(x2)
    out_ref, (h_ref, c_ref) = ref_lstm(x2)

    # PyTorch LSTM returns h_n, c_n with shape (num_layers, B, hidden_size)
    h_ref = h_ref.squeeze(0)
    c_ref = c_ref.squeeze(0)

    _assert_allclose(
        out_candidate, out_ref,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch vs torch.nn.LSTM."
    )
    _assert_allclose(
        h_candidate, h_ref,
        atol=1e-10, rtol=1e-10,
        msg="Final hidden state mismatch vs torch.nn.LSTM."
    )
    _assert_allclose(
        c_candidate, c_ref,
        atol=1e-10, rtol=1e-10,
        msg="Final cell state mismatch vs torch.nn.LSTM."
    )

    # --- test 4: with initial hidden state ---
    h_0 = torch.randn(B, hidden_size, dtype=dtype)
    c_0 = torch.randn(B, hidden_size, dtype=dtype)

    out_candidate2, (h_candidate2, c_candidate2) = m(x2, hx=(h_0, c_0))
    out_ref2, (h_ref2, c_ref2) = ref_lstm(
        x2, (h_0.unsqueeze(0), c_0.unsqueeze(0))
    )
    h_ref2 = h_ref2.squeeze(0)
    c_ref2 = c_ref2.squeeze(0)

    _assert_allclose(
        out_candidate2, out_ref2,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch with initial hx vs torch.nn.LSTM."
    )
    _assert_allclose(
        h_candidate2, h_ref2,
        atol=1e-10, rtol=1e-10,
        msg="h_n mismatch with initial hx vs torch.nn.LSTM."
    )
    _assert_allclose(
        c_candidate2, c_ref2,
        atol=1e-10, rtol=1e-10,
        msg="c_n mismatch with initial hx vs torch.nn.LSTM."
    )

    # --- test 5: no bias ---
    m_nobias = candidate.SimpleLSTM(
        input_size=input_size, hidden_size=hidden_size, bias=False
    ).to(dtype=dtype)
    m_nobias.eval()

    ref_nobias = nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        bias=False,
        batch_first=True,
    ).to(dtype=dtype)
    ref_nobias.eval()

    _copy_weights_to_torch_lstm(m_nobias, ref_nobias)

    x3 = torch.randn(B, T, input_size, dtype=dtype)
    out_nb, (h_nb, c_nb) = m_nobias(x3)
    out_ref_nb, (h_ref_nb, c_ref_nb) = ref_nobias(x3)
    h_ref_nb = h_ref_nb.squeeze(0)
    c_ref_nb = c_ref_nb.squeeze(0)

    _assert_allclose(
        out_nb, out_ref_nb,
        atol=1e-10, rtol=1e-10,
        msg="Output mismatch (no bias) vs torch.nn.LSTM."
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
    out_single, (h_single, c_single) = m(x_single)
    assert out_single.shape == (B, 1, hidden_size), (
        f"Single timestep output shape wrong: {out_single.shape}"
    )
    _assert_allclose(
        out_single[:, 0, :], h_single,
        atol=1e-12, rtol=1e-12,
        msg="Single timestep: output should equal h_n."
    )




