from __future__ import annotations

from types import ModuleType

import torch


def _assert_allclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str
) -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        diff = (a - b).abs().max().item()
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def _copy_weights_to_nn_rnn(candidate_rnn, nn_rnn: torch.nn.RNN) -> None:
    """Copy weights from candidate RNN to torch.nn.RNN for comparison."""
    with torch.no_grad():
        nn_rnn.weight_ih_l0.copy_(candidate_rnn.W_ih)
        nn_rnn.weight_hh_l0.copy_(candidate_rnn.W_hh)
        nn_rnn.bias_ih_l0.copy_(candidate_rnn.b_ih)
        nn_rnn.bias_hh_l0.copy_(candidate_rnn.b_hh)


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "RNN"):
        raise AssertionError("Candidate must define class `RNN`.")

    torch.manual_seed(42)

    # --- test 1: output shapes are correct ---
    rnn = candidate.RNN(input_size=3, hidden_size=4)
    x = torch.randn(5, 2, 3)
    output, h_n = rnn(x)

    assert output.shape == (5, 2, 4), f"Expected output shape (5, 2, 4), got {output.shape}"
    assert h_n.shape == (1, 2, 4), f"Expected h_n shape (1, 2, 4), got {h_n.shape}"

    # --- test 2: h_n should equal the last timestep of output ---
    _assert_allclose(
        h_n,
        output[-1:],
        atol=1e-6,
        rtol=1e-6,
        msg="h_n should equal output[-1:] (last hidden state).",
    )

    # --- test 3: with initial hidden state h0 ---
    torch.manual_seed(123)
    rnn2 = candidate.RNN(input_size=4, hidden_size=5)
    x2 = torch.randn(3, 2, 4)
    h0 = torch.randn(1, 2, 5)
    output2, h_n2 = rnn2(x2, h0)

    assert output2.shape == (3, 2, 5), f"Expected output shape (3, 2, 5), got {output2.shape}"
    assert h_n2.shape == (1, 2, 5), f"Expected h_n shape (1, 2, 5), got {h_n2.shape}"
    _assert_allclose(
        h_n2,
        output2[-1:],
        atol=1e-6,
        rtol=1e-6,
        msg="h_n should equal output[-1:] when h0 is provided.",
    )

    # --- test 4: compare against torch.nn.RNN (using float64 for precision) ---
    torch.manual_seed(999)
    dtype = torch.float64
    input_size, hidden_size = 6, 8
    seq_len, batch = 7, 3

    candidate_rnn = candidate.RNN(input_size=input_size, hidden_size=hidden_size)

    # Convert weights to float64 for higher precision comparison
    candidate_rnn.W_ih = candidate_rnn.W_ih.to(dtype).requires_grad_(True)
    candidate_rnn.W_hh = candidate_rnn.W_hh.to(dtype).requires_grad_(True)
    candidate_rnn.b_ih = candidate_rnn.b_ih.to(dtype).requires_grad_(True)
    candidate_rnn.b_hh = candidate_rnn.b_hh.to(dtype).requires_grad_(True)

    nn_rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=False)
    nn_rnn = nn_rnn.to(dtype)
    _copy_weights_to_nn_rnn(candidate_rnn, nn_rnn)

    x3 = torch.randn(seq_len, batch, input_size, dtype=dtype)
    h0_3 = torch.randn(1, batch, hidden_size, dtype=dtype)

    output_candidate, h_n_candidate = candidate_rnn(x3, h0_3)
    output_nn, h_n_nn = nn_rnn(x3, h0_3)

    _assert_allclose(
        output_candidate,
        output_nn,
        atol=1e-10,
        rtol=1e-10,
        msg="Output does not match torch.nn.RNN.",
    )
    _assert_allclose(
        h_n_candidate,
        h_n_nn,
        atol=1e-10,
        rtol=1e-10,
        msg="Final hidden state does not match torch.nn.RNN.",
    )

    # --- test 5: single timestep (seq_len=1) ---
    torch.manual_seed(555)
    rnn3 = candidate.RNN(input_size=2, hidden_size=3)
    x_single = torch.randn(1, 4, 2)
    output_single, h_n_single = rnn3(x_single)

    assert output_single.shape == (1, 4, 3), f"Expected shape (1, 4, 3), got {output_single.shape}"
    _assert_allclose(
        output_single,
        h_n_single,
        atol=1e-6,
        rtol=1e-6,
        msg="For seq_len=1, output and h_n should be identical.",
    )

    # --- test 6: verify gradients flow through ---
    torch.manual_seed(777)
    rnn4 = candidate.RNN(input_size=3, hidden_size=4)
    x4 = torch.randn(4, 2, 3, requires_grad=True)
    output4, h_n4 = rnn4(x4)
    loss = output4.sum()
    loss.backward()

    assert x4.grad is not None, "Gradients should flow to input x."
    assert rnn4.W_ih.grad is not None, "Gradients should flow to W_ih."
    assert rnn4.W_hh.grad is not None, "Gradients should flow to W_hh."
    assert rnn4.b_ih.grad is not None, "Gradients should flow to b_ih."
    assert rnn4.b_hh.grad is not None, "Gradients should flow to b_hh."

    # --- test 7: batch size of 1 ---
    torch.manual_seed(888)
    rnn5 = candidate.RNN(input_size=5, hidden_size=3)
    x5 = torch.randn(10, 1, 5)
    output5, h_n5 = rnn5(x5)

    assert output5.shape == (10, 1, 3), f"Expected shape (10, 1, 3), got {output5.shape}"
    assert h_n5.shape == (1, 1, 3), f"Expected shape (1, 1, 3), got {h_n5.shape}"

