from __future__ import annotations

import torch


def pad_sequences(
    sequences: list[torch.Tensor],
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad variable-length sequences to create a batch.

    Args:
        sequences: List of 1D tensors with varying lengths.
        pad_value: Value to use for padding.

    Returns:
        padded: Tensor of shape (B, T) with padded sequences.
        lengths: Tensor of shape (B,) with original sequence lengths.
    """
    raise NotImplementedError


def create_padding_mask(
    lengths: torch.Tensor,
    max_len: int,
) -> torch.Tensor:
    """Create a padding mask from sequence lengths.

    Args:
        lengths: 1D tensor of shape (B,) with sequence lengths.
        max_len: The padded sequence length T.

    Returns:
        Boolean tensor of shape (B, T) where True indicates padding positions.
    """
    raise NotImplementedError


def create_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Create a causal (future-blocking) mask for autoregressive attention.

    Args:
        seq_len: The sequence length T.
        device: Optional device for the output tensor.

    Returns:
        Boolean tensor of shape (T, T) where True indicates future positions to mask.
    """
    raise NotImplementedError



