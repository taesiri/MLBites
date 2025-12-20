from __future__ import annotations

from types import ModuleType

import numpy as np


def _assert_allclose(
    a: np.ndarray, b: np.ndarray, *, atol: float, rtol: float, msg: str
) -> None:
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        diff = float(np.max(np.abs(a - b)))
        raise AssertionError(f"{msg}\nmax_abs_diff={diff}\na={a}\nb={b}")


def run_tests(candidate: ModuleType) -> None:
    """Run all tests against a candidate solution module.

    Args:
        candidate: a Python module that defines conv2d_forward.

    Raises:
        AssertionError: if any test fails.
    """
    if not hasattr(candidate, "conv2d_forward"):
        raise AssertionError("Candidate must define function `conv2d_forward`.")

    conv2d_forward = candidate.conv2d_forward

    rng = np.random.default_rng(42)
    atol = 1e-6
    rtol = 1e-6

    # --- Test 1: Simple 2x2 kernel on 4x4 input, stride=1, no padding ---
    x1 = np.array([[[[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]]]]).astype(np.float64)
    weight1 = np.array([[[[1, 0],
                          [0, 1]]]]).astype(np.float64)
    out1 = conv2d_forward(x1, weight1)
    expected1 = np.array([[[[7, 9, 11],
                            [15, 17, 19],
                            [23, 25, 27]]]]).astype(np.float64)
    
    if out1.shape != (1, 1, 3, 3):
        raise AssertionError(f"Test 1: Expected shape (1, 1, 3, 3), got {out1.shape}")
    _assert_allclose(out1, expected1, atol=atol, rtol=rtol, msg="Test 1: Simple 2x2 kernel failed")

    # --- Test 2: With padding ---
    x2 = np.ones((1, 1, 3, 3), dtype=np.float64)
    weight2 = np.ones((1, 1, 3, 3), dtype=np.float64)
    out2 = conv2d_forward(x2, weight2, padding=1)
    
    if out2.shape != (1, 1, 3, 3):
        raise AssertionError(f"Test 2: Expected shape (1, 1, 3, 3), got {out2.shape}")
    
    expected2_center = 9.0
    if not np.isclose(out2[0, 0, 1, 1], expected2_center, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 2: Center value should be 9, got {out2[0, 0, 1, 1]}")
    
    expected2_corner = 4.0
    if not np.isclose(out2[0, 0, 0, 0], expected2_corner, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 2: Corner value should be 4, got {out2[0, 0, 0, 0]}")

    # --- Test 3: With stride ---
    x3 = np.ones((1, 1, 4, 4), dtype=np.float64)
    weight3 = np.ones((1, 1, 2, 2), dtype=np.float64)
    out3 = conv2d_forward(x3, weight3, stride=2)
    
    expected_shape3 = (1, 1, 2, 2)
    if out3.shape != expected_shape3:
        raise AssertionError(f"Test 3: Expected shape {expected_shape3}, got {out3.shape}")
    
    expected3 = np.full((1, 1, 2, 2), 4.0, dtype=np.float64)
    _assert_allclose(out3, expected3, atol=atol, rtol=rtol, msg="Test 3: Stride=2 failed")

    # --- Test 4: With bias ---
    x4 = np.ones((1, 1, 3, 3), dtype=np.float64)
    weight4 = np.ones((2, 1, 2, 2), dtype=np.float64)
    bias4 = np.array([1.0, -1.0])
    out4 = conv2d_forward(x4, weight4, bias=bias4)
    
    if out4.shape != (1, 2, 2, 2):
        raise AssertionError(f"Test 4: Expected shape (1, 2, 2, 2), got {out4.shape}")
    
    expected4_ch0 = 4.0 + 1.0
    expected4_ch1 = 4.0 - 1.0
    if not np.allclose(out4[0, 0], expected4_ch0, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 4: Channel 0 should be {expected4_ch0}, got {out4[0, 0]}")
    if not np.allclose(out4[0, 1], expected4_ch1, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 4: Channel 1 should be {expected4_ch1}, got {out4[0, 1]}")

    # --- Test 5: Multiple input channels ---
    x5 = np.ones((1, 3, 4, 4), dtype=np.float64)
    weight5 = np.ones((1, 3, 2, 2), dtype=np.float64)
    out5 = conv2d_forward(x5, weight5)
    
    if out5.shape != (1, 1, 3, 3):
        raise AssertionError(f"Test 5: Expected shape (1, 1, 3, 3), got {out5.shape}")
    
    expected5 = np.full((1, 1, 3, 3), 12.0, dtype=np.float64)
    _assert_allclose(out5, expected5, atol=atol, rtol=rtol, msg="Test 5: Multiple input channels failed")

    # --- Test 6: Batch processing ---
    x6 = np.ones((4, 2, 5, 5), dtype=np.float64)
    x6[1] *= 2
    x6[2] *= 3
    x6[3] *= 4
    weight6 = np.ones((3, 2, 3, 3), dtype=np.float64)
    out6 = conv2d_forward(x6, weight6)
    
    if out6.shape != (4, 3, 3, 3):
        raise AssertionError(f"Test 6: Expected shape (4, 3, 3, 3), got {out6.shape}")
    
    expected_val_b0 = 18.0
    expected_val_b1 = 36.0
    expected_val_b2 = 54.0
    expected_val_b3 = 72.0
    
    if not np.allclose(out6[0], expected_val_b0, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 6: Batch 0 should be {expected_val_b0}")
    if not np.allclose(out6[1], expected_val_b1, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 6: Batch 1 should be {expected_val_b1}")
    if not np.allclose(out6[2], expected_val_b2, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 6: Batch 2 should be {expected_val_b2}")
    if not np.allclose(out6[3], expected_val_b3, atol=atol, rtol=rtol):
        raise AssertionError(f"Test 6: Batch 3 should be {expected_val_b3}")

    # --- Test 7: Compare with reference for random input ---
    x7 = rng.normal(size=(2, 3, 8, 8)).astype(np.float64)
    weight7 = rng.normal(size=(4, 3, 3, 3)).astype(np.float64)
    bias7 = rng.normal(size=4).astype(np.float64)
    
    out7 = conv2d_forward(x7, weight7, bias=bias7, stride=2, padding=1)
    
    expected_shape7 = (2, 4, 4, 4)
    if out7.shape != expected_shape7:
        raise AssertionError(f"Test 7: Expected shape {expected_shape7}, got {out7.shape}")
    
    H_out = (8 + 2 * 1 - 3) // 2 + 1
    W_out = (8 + 2 * 1 - 3) // 2 + 1
    if H_out != 4 or W_out != 4:
        raise AssertionError("Test 7: Output dimension calculation error")
    
    x7_pad = np.pad(x7, ((0, 0), (0, 0), (1, 1), (1, 1)), mode='constant')
    ref7 = np.zeros((2, 4, 4, 4), dtype=np.float64)
    for n in range(2):
        for c_out in range(4):
            for h in range(4):
                for w in range(4):
                    h_start = h * 2
                    w_start = w * 2
                    window = x7_pad[n, :, h_start:h_start+3, w_start:w_start+3]
                    ref7[n, c_out, h, w] = np.sum(window * weight7[c_out]) + bias7[c_out]
    
    _assert_allclose(out7, ref7, atol=1e-5, rtol=1e-5, msg="Test 7: Random input comparison failed")

    # --- Test 8: Output should be finite ---
    x8 = rng.normal(size=(4, 8, 16, 16)).astype(np.float64)
    weight8 = rng.normal(size=(16, 8, 3, 3)).astype(np.float64)
    bias8 = rng.normal(size=16).astype(np.float64)
    out8 = conv2d_forward(x8, weight8, bias=bias8, stride=1, padding=1)
    
    if not np.all(np.isfinite(out8)):
        raise AssertionError("Test 8: Output contains inf or nan")
    
    if out8.shape != (4, 16, 16, 16):
        raise AssertionError(f"Test 8: Expected shape (4, 16, 16, 16), got {out8.shape}")

    # --- Test 9: Edge case - 1x1 kernel ---
    x9 = rng.normal(size=(2, 4, 5, 5)).astype(np.float64)
    weight9 = rng.normal(size=(3, 4, 1, 1)).astype(np.float64)
    out9 = conv2d_forward(x9, weight9)
    
    if out9.shape != (2, 3, 5, 5):
        raise AssertionError(f"Test 9: Expected shape (2, 3, 5, 5), got {out9.shape}")
    
    ref9 = np.einsum('nchw,oc->nohw', x9, weight9[:, :, 0, 0])
    _assert_allclose(out9, ref9, atol=1e-5, rtol=1e-5, msg="Test 9: 1x1 kernel failed")

    # --- Test 10: Combined padding and stride ---
    x10 = np.arange(16).reshape(1, 1, 4, 4).astype(np.float64)
    weight10 = np.ones((1, 1, 2, 2), dtype=np.float64)
    out10 = conv2d_forward(x10, weight10, stride=2, padding=1)
    
    if out10.shape != (1, 1, 3, 3):
        raise AssertionError(f"Test 10: Expected shape (1, 1, 3, 3), got {out10.shape}")



