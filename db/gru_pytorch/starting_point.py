from __future__ import annotations

import torch
import torch.nn as nn


class SimpleGRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
    ) -> None:
        """A minimal single-layer GRU module (batch-first).

        Args:
            input_size: The number of expected features in the input x.
            hidden_size: The number of features in the hidden state h.
            bias: If False, the layer does not use bias weights.
        """
        super().__init__()
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        h0: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GRU over the input sequence.

        Args:
            x: Input tensor of shape (B, T, input_size).
            h0: Optional initial hidden state of shape (B, hidden_size).
                If None, initialized to zeros.

        Returns:
            output: Tensor of shape (B, T, hidden_size) — hidden states at each timestep.
            h_n: Final hidden state of shape (B, hidden_size).
        """
        raise NotImplementedError


