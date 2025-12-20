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
    # Get the length of each sequence
    lengths = torch.tensor([seq.size(0) for seq in sequences], dtype=torch.long)

    # Find the maximum length for padding
    max_len = lengths.max().item()

    # Create output tensor filled with pad_value
    batch_size = len(sequences)
    padded = torch.full(
        (batch_size, max_len),
        pad_value,
        dtype=sequences[0].dtype,
        device=sequences[0].device,
    )

    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        padded[i, : seq.size(0)] = seq

    return padded, lengths


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
    # Create position indices [0, 1, 2, ..., max_len-1]
    positions = torch.arange(max_len, device=lengths.device)

    # Broadcast comparison: positions >= lengths means padding
    # lengths is (B,), positions is (T,)
    # lengths[:, None] is (B, 1), comparison gives (B, T)
    mask = positions >= lengths[:, None]

    return mask


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
    # Create row indices (query positions) and column indices (key positions)
    # Position (i, j) should be True if j > i (future position)
    row_indices = torch.arange(seq_len, device=device)
    col_indices = torch.arange(seq_len, device=device)

    # Broadcast: row_indices[:, None] is (T, 1), col_indices is (T,)
    # Result is (T, T) where mask[i, j] = (j > i)
    mask = col_indices > row_indices[:, None]

    return mask



