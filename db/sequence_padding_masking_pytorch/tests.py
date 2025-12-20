from __future__ import annotations

from types import ModuleType

import torch


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required function(s) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    torch.manual_seed(42)

    # Check that all required functions exist
    for fn_name in ["pad_sequences", "create_padding_mask", "create_causal_mask"]:
        if not hasattr(candidate, fn_name):
            raise AssertionError(f"Candidate must define function `{fn_name}`.")

    pad_sequences = candidate.pad_sequences
    create_padding_mask = candidate.create_padding_mask
    create_causal_mask = candidate.create_causal_mask

    # --- Test 1: pad_sequences basic functionality ---
    seqs = [
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([4.0, 5.0]),
        torch.tensor([6.0]),
    ]
    padded, lengths = pad_sequences(seqs, pad_value=0.0)

    expected_padded = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 0.0], [6.0, 0.0, 0.0]]
    )
    expected_lengths = torch.tensor([3, 2, 1])

    if padded.shape != (3, 3):
        raise AssertionError(f"Expected padded shape (3, 3), got {padded.shape}")
    if not torch.equal(padded, expected_padded):
        raise AssertionError(f"Padded values mismatch.\nExpected:\n{expected_padded}\nGot:\n{padded}")
    if not torch.equal(lengths, expected_lengths):
        raise AssertionError(f"Lengths mismatch.\nExpected: {expected_lengths}\nGot: {lengths}")

    # --- Test 2: pad_sequences with custom pad_value ---
    seqs2 = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    padded2, lengths2 = pad_sequences(seqs2, pad_value=-1.0)

    expected_padded2 = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    if not torch.equal(padded2, expected_padded2):
        raise AssertionError(f"Custom pad_value failed.\nExpected:\n{expected_padded2}\nGot:\n{padded2}")

    # --- Test 3: pad_sequences with single sequence (no actual padding needed) ---
    seqs3 = [torch.tensor([1.0, 2.0, 3.0])]
    padded3, lengths3 = pad_sequences(seqs3)
    if padded3.shape != (1, 3):
        raise AssertionError(f"Single sequence shape wrong: {padded3.shape}")
    if lengths3.item() != 3:
        raise AssertionError(f"Single sequence length wrong: {lengths3}")

    # --- Test 4: pad_sequences preserves dtype ---
    seqs_int = [torch.tensor([1, 2, 3]), torch.tensor([4])]
    padded_int, _ = pad_sequences(seqs_int, pad_value=0)
    if padded_int.dtype != torch.int64:
        raise AssertionError(f"Expected int64 dtype, got {padded_int.dtype}")

    # --- Test 5: create_padding_mask basic ---
    lengths = torch.tensor([3, 2, 1])
    mask = create_padding_mask(lengths, max_len=3)

    expected_mask = torch.tensor(
        [[False, False, False], [False, False, True], [False, True, True]]
    )
    if mask.shape != (3, 3):
        raise AssertionError(f"Expected mask shape (3, 3), got {mask.shape}")
    if not torch.equal(mask, expected_mask):
        raise AssertionError(f"Padding mask mismatch.\nExpected:\n{expected_mask}\nGot:\n{mask}")

    # --- Test 6: create_padding_mask with max_len > all lengths ---
    lengths2 = torch.tensor([2, 1])
    mask2 = create_padding_mask(lengths2, max_len=4)

    expected_mask2 = torch.tensor(
        [[False, False, True, True], [False, True, True, True]]
    )
    if not torch.equal(mask2, expected_mask2):
        raise AssertionError(f"Extended padding mask mismatch.\nExpected:\n{expected_mask2}\nGot:\n{mask2}")

    # --- Test 7: create_padding_mask with full-length sequence (no padding) ---
    lengths3 = torch.tensor([3])
    mask3 = create_padding_mask(lengths3, max_len=3)
    expected_mask3 = torch.tensor([[False, False, False]])
    if not torch.equal(mask3, expected_mask3):
        raise AssertionError(f"Full-length mask mismatch.\nExpected:\n{expected_mask3}\nGot:\n{mask3}")

    # --- Test 8: create_causal_mask basic ---
    causal = create_causal_mask(seq_len=4)

    expected_causal = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    if causal.shape != (4, 4):
        raise AssertionError(f"Expected causal mask shape (4, 4), got {causal.shape}")
    if not torch.equal(causal, expected_causal):
        raise AssertionError(f"Causal mask mismatch.\nExpected:\n{expected_causal}\nGot:\n{causal}")

    # --- Test 9: create_causal_mask size 1 ---
    causal1 = create_causal_mask(seq_len=1)
    expected_causal1 = torch.tensor([[False]])
    if not torch.equal(causal1, expected_causal1):
        raise AssertionError(f"Causal mask size 1 mismatch.\nExpected:\n{expected_causal1}\nGot:\n{causal1}")

    # --- Test 10: create_causal_mask size 2 ---
    causal2 = create_causal_mask(seq_len=2)
    expected_causal2 = torch.tensor([[False, True], [False, False]])
    if not torch.equal(causal2, expected_causal2):
        raise AssertionError(f"Causal mask size 2 mismatch.\nExpected:\n{expected_causal2}\nGot:\n{causal2}")

    # --- Test 11: Integration test - padding mask combined with attention simulation ---
    # Verify that padding mask correctly zeros out attention to padded positions
    seqs = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0])]
    padded, lengths = pad_sequences(seqs, pad_value=0.0)
    mask = create_padding_mask(lengths, max_len=3)

    # Simulate attention scores (simple dot product for demonstration)
    scores = padded @ padded.T  # (2, 3) @ (3, 2) = (2, 2) -- just for shape check
    # The key is that mask properly identifies the last position in second sequence
    if not mask[1, 2]:  # Position (1, 2) should be True (padding)
        raise AssertionError("Mask should mark position [1, 2] as padding")
    if mask[1, 1]:  # Position (1, 1) should be False (valid)
        raise AssertionError("Mask should mark position [1, 1] as valid")

    # --- Test 12: Verify device handling for causal mask ---
    if torch.cuda.is_available():
        causal_cuda = create_causal_mask(seq_len=3, device=torch.device("cuda"))
        if causal_cuda.device.type != "cuda":
            raise AssertionError(f"Expected CUDA device, got {causal_cuda.device}")



