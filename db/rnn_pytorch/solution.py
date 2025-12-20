from __future__ import annotations

import math

import torch


class RNN:
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
    ) -> None:
        """A minimal vanilla RNN layer (Elman RNN) from scratch.

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size

        # PyTorch-style initialization: uniform in [-k, k] where k = 1/sqrt(hidden_size)
        k = 1.0 / math.sqrt(hidden_size)

        self.W_ih = torch.empty(hidden_size, input_size).uniform_(-k, k).requires_grad_(True)
        self.W_hh = torch.empty(hidden_size, hidden_size).uniform_(-k, k).requires_grad_(True)
        self.b_ih = torch.empty(hidden_size).uniform_(-k, k).requires_grad_(True)
        self.b_hh = torch.empty(hidden_size).uniform_(-k, k).requires_grad_(True)

    def __call__(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process input sequence through the RNN.

        Args:
            x: Input tensor of shape (seq_len, batch, input_size).
            h0: Optional initial hidden state of shape (1, batch, hidden_size).
                If None, defaults to zeros.

        Returns:
            output: Tensor of shape (seq_len, batch, hidden_size).
            h_n: Final hidden state of shape (1, batch, hidden_size).
        """
        seq_len, batch, _ = x.shape

        if h0 is None:
            h = torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h = h0.squeeze(0)

        outputs = []
        for t in range(seq_len):
            x_t = x[t]  # (batch, input_size)
            # h_t = tanh(x_t @ W_ih.T + b_ih + h @ W_hh.T + b_hh)
            h = torch.tanh(x_t @ self.W_ih.T + self.b_ih + h @ self.W_hh.T + self.b_hh)
            outputs.append(h)

        output = torch.stack(outputs, dim=0)  # (seq_len, batch, hidden_size)
        h_n = h.unsqueeze(0)  # (1, batch, hidden_size)

        return output, h_n




