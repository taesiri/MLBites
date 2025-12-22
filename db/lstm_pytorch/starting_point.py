from __future__ import annotations

import torch
import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        """A minimal single-layer LSTM module (batch-first).

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            bias: If False, the layer does not use bias weights.
        """
        super().__init__()
        # TODO: initialize the necessary layers
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run LSTM over the input sequence.

        Args:
            x: Input tensor of shape (B, T, input_size).
            hx: Optional tuple of (h_0, c_0), each of shape (B, hidden_size).
                If None, both are initialized to zeros.

        Returns:
            output: Tensor of shape (B, T, hidden_size) — hidden states at each timestep.
            (h_n, c_n): Tuple of final hidden state and cell state.
        """
        # TODO: implement the forward pass
        raise NotImplementedError




