from __future__ import annotations

from types import ModuleType

import numpy as np
import torch


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines the required class(es) for this question.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "CrossModalVisualizer"):
        raise AssertionError("Candidate must define class `CrossModalVisualizer`.")

    CrossModalVisualizer = candidate.CrossModalVisualizer

    # --- Test 1: Basic functionality with t-SNE ---
    torch.manual_seed(42)
    np.random.seed(42)

    image_emb = torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ])
    text_emb = torch.tensor([
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.9, 0.1],
    ])

    viz = CrossModalVisualizer(method="tsne", perplexity=1.5, random_state=42)
    coords, modalities = viz.fit_transform(image_emb, text_emb)

    # Check output shapes
    assert coords.shape == (5, 2), f"Expected coords shape (5, 2), got {coords.shape}"
    assert modalities.shape == (5,), f"Expected modalities shape (5,), got {modalities.shape}"

    # Check modality labels
    expected_modalities = np.array([0, 0, 0, 1, 1])
    assert np.array_equal(modalities, expected_modalities), (
        f"Expected modalities {expected_modalities}, got {modalities}"
    )

    # Check that coords are finite
    assert np.isfinite(coords).all(), "Coords contain NaN or Inf values"

    # --- Test 2: Larger dataset with t-SNE ---
    torch.manual_seed(123)
    image_emb_large = torch.randn(30, 64)
    text_emb_large = torch.randn(20, 64)

    viz_large = CrossModalVisualizer(method="tsne", perplexity=5.0, random_state=0)
    coords_large, modalities_large = viz_large.fit_transform(image_emb_large, text_emb_large)

    assert coords_large.shape == (50, 2), f"Expected shape (50, 2), got {coords_large.shape}"
    assert modalities_large.shape == (50,), f"Expected shape (50,), got {modalities_large.shape}"
    assert (modalities_large[:30] == 0).all(), "First 30 should be images (0)"
    assert (modalities_large[30:] == 1).all(), "Last 20 should be texts (1)"

    # --- Test 3: Return types ---
    assert isinstance(coords, np.ndarray), f"coords should be np.ndarray, got {type(coords)}"
    assert isinstance(modalities, np.ndarray), f"modalities should be np.ndarray, got {type(modalities)}"

    # --- Test 4: Invalid method raises error ---
    try:
        CrossModalVisualizer(method="invalid")
        raise AssertionError("Should raise ValueError for invalid method")
    except ValueError:
        pass

    # --- Test 5: Reproducibility with same random_state ---
    viz1 = CrossModalVisualizer(method="tsne", perplexity=1.5, random_state=999)
    viz2 = CrossModalVisualizer(method="tsne", perplexity=1.5, random_state=999)

    img = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    txt = torch.tensor([[0.0, 0.0, 1.0]])

    coords1, _ = viz1.fit_transform(img, txt)
    coords2, _ = viz2.fit_transform(img, txt)

    assert np.allclose(coords1, coords2, atol=1e-5), (
        "Same random_state should produce identical results"
    )

    # --- Test 6: GPU tensor handling (simulated) ---
    # Even if not on GPU, .cpu() should work fine
    img_cpu = torch.randn(5, 16)
    txt_cpu = torch.randn(3, 16)

    viz_cpu = CrossModalVisualizer(method="tsne", perplexity=2.0, random_state=42)
    coords_cpu, mod_cpu = viz_cpu.fit_transform(img_cpu, txt_cpu)

    assert coords_cpu.shape == (8, 2), f"Expected shape (8, 2), got {coords_cpu.shape}"
    assert len(mod_cpu) == 8, f"Expected 8 modality labels, got {len(mod_cpu)}"

    # --- Test 7: Embeddings are normalized (check by verifying output is reasonable) ---
    # Create embeddings with very different magnitudes
    img_varied = torch.tensor([[100.0, 0.0], [0.01, 0.0]])
    txt_varied = torch.tensor([[0.0, 50.0]])

    viz_varied = CrossModalVisualizer(method="tsne", perplexity=1.0, random_state=42)
    coords_varied, _ = viz_varied.fit_transform(img_varied, txt_varied)

    # After normalization, all should be on unit sphere, so t-SNE should work fine
    assert np.isfinite(coords_varied).all(), "Coords should be finite after normalization"

    # --- Test 8: Single sample per modality ---
    img_single = torch.randn(1, 8)
    txt_single = torch.randn(1, 8)

    viz_single = CrossModalVisualizer(method="tsne", perplexity=0.5, random_state=42)
    coords_single, mod_single = viz_single.fit_transform(img_single, txt_single)

    assert coords_single.shape == (2, 2), f"Expected shape (2, 2), got {coords_single.shape}"
    assert list(mod_single) == [0, 1], f"Expected [0, 1], got {list(mod_single)}"

