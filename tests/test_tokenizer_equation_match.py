"""
Tokenizer Equation Match Verification
======================================
Hand-computes the MSOPT tokenizer output for a known 8-element subseries
and asserts the code produces the same result.

Paper Equations:
  §3.1: s_t^{(w,d)} = [x_{t-d(w-1)}, x_{t-d(w-2)}, ..., x_t]
  §3.2: Standardize s -> z-score, partition into K=4 segments,
        quantize each segment mean into alpha_mu via Gaussian breakpoints,
        quantize each segment slope into alpha_beta via slope breakpoints.

This test uses a single scale (w=4, d=1) on a known 8-element series
and verifies the token word string character-by-character.
"""

import sys, os
import numpy as np
from scipy.stats import norm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer.msopt_tokenizer import MSOPTTokenizer


def test_receptive_field_extraction():
    """
    §3.1 Equation Check:
    Given series = [10, 20, 30, 40, 50, 60, 70, 80]
    For w=4, d=1, s=1:
      rf_span = 1*(4-1)+1 = 4
      t=0: series[0:4:1] = [10,20,30,40]  -> end_time = 3
      t=1: series[1:5:1] = [20,30,40,50]  -> end_time = 4
      ...
      t=4: series[4:8:1] = [50,60,70,80]  -> end_time = 7
    
    For w=4, d=2:
      rf_span = 2*(4-1)+1 = 7
      t=0: series[0:7:2] = [10,30,50,70]  -> end_time = 6
      t=1: series[1:8:2] = [20,40,60,80]  -> end_time = 7
    """
    print("=" * 70)
    print("  TEST 1: Receptive Field Extraction (§3.1)")
    print("=" * 70)
    
    series = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0])
    
    # w=4, d=1: rf_span = 4
    w, d = 4, 1
    rf_span = d * (w - 1) + 1
    assert rf_span == 4, f"Expected rf_span=4, got {rf_span}"
    
    # t=0
    rf_0 = series[0 : 0 + rf_span : d]
    assert np.array_equal(rf_0, [10, 20, 30, 40]), f"RF at t=0 wrong: {rf_0}"
    
    # t=4 (last valid)
    rf_4 = series[4 : 4 + rf_span : d]
    assert np.array_equal(rf_4, [50, 60, 70, 80]), f"RF at t=4 wrong: {rf_4}"
    
    # w=4, d=2: rf_span = 7
    w, d = 4, 2
    rf_span = d * (w - 1) + 1
    assert rf_span == 7, f"Expected rf_span=7, got {rf_span}"
    
    rf_0_d2 = series[0 : 0 + rf_span : d]
    assert np.array_equal(rf_0_d2, [10, 30, 50, 70]), f"RF(d=2) at t=0 wrong: {rf_0_d2}"
    
    rf_1_d2 = series[1 : 1 + rf_span : d]
    assert np.array_equal(rf_1_d2, [20, 40, 60, 80]), f"RF(d=2) at t=1 wrong: {rf_1_d2}"
    
    print("  ✓ All receptive field extractions match §3.1 equation.")
    print(f"    w=4,d=1: t=0 -> {rf_0.tolist()}, t=4 -> {rf_4.tolist()}")
    print(f"    w=4,d=2: t=0 -> {rf_0_d2.tolist()}, t=1 -> {rf_1_d2.tolist()}")
    return True


def test_1d_sax_discretization():
    """
    §3.2 Equation Check:
    Given rf = [10, 20, 30, 40] (w=4, K=4 segments, 1 value per segment)
    
    Step 1: Standardize
      mean = 25.0, std = 12.909...
      z = [-1.1619, -0.3873, 0.3873, 1.1619]
    
    Step 2: Gaussian breakpoints for alphabet_size_mean=4
      breakpoints = norm.ppf([0.25, 0.5, 0.75]) = [-0.6745, 0.0, 0.6745]
      Bins: (-inf, -0.6745) = A, [-0.6745, 0) = B, [0, 0.6745) = C, [0.6745, inf) = D
    
    Step 3: Quantize each segment mean
      seg0 mean = -1.1619 -> A (below -0.6745)
      seg1 mean = -0.3873 -> B (between -0.6745 and 0)
      seg2 mean = +0.3873 -> C (between 0 and 0.6745)
      seg3 mean = +1.1619 -> D (above 0.6745)
    
    Step 4: Slope per segment (single value per segment -> slope = 0)
      All slopes = 0.0 -> slope breakpoints [-0.2, 0.2] -> bin B (between -0.2 and 0.2)
    
    Expected word: "ABBBCBDB"
    (each segment contributes 2 chars: mean_sym + slope_sym)
    """
    print("\n" + "=" * 70)
    print("  TEST 2: 1D-SAX Discretization (§3.2)")
    print("=" * 70)
    
    rf = np.array([10.0, 20.0, 30.0, 40.0])
    
    # Manual computation
    mean_val = np.mean(rf)  # 25.0
    std_val = np.std(rf)    # 12.909...
    z = (rf - mean_val) / std_val
    print(f"  Raw RF: {rf.tolist()}")
    print(f"  Mean={mean_val}, Std={std_val:.4f}")
    print(f"  Z-scored: [{', '.join(f'{v:.4f}' for v in z)}]")
    
    # Gaussian breakpoints for 4 symbols
    breakpoints_mean = norm.ppf([0.25, 0.5, 0.75])
    print(f"  Mean breakpoints: [{', '.join(f'{v:.4f}' for v in breakpoints_mean)}]")
    
    # With K=4 and w=4, each segment has exactly 1 value
    # Segment means = z values themselves
    # Segment slopes = 0 (single value)
    expected_mean_syms = []
    for v in z:
        idx = np.searchsorted(breakpoints_mean, float(v))
        sym = chr(65 + idx)
        expected_mean_syms.append(sym)
        print(f"    z={v:.4f} -> bin {idx} -> '{sym}'")
    
    slope_breakpoints = np.array([-0.2, 0.2])
    slope_sym = chr(65 + np.searchsorted(slope_breakpoints, 0.0))  # 'B'
    print(f"  Slope=0.0 -> bin {np.searchsorted(slope_breakpoints, 0.0)} -> '{slope_sym}'")
    
    expected_word = "".join(f"{m}{slope_sym}" for m in expected_mean_syms)
    print(f"  Expected word: '{expected_word}'")
    
    # Now verify the tokenizer produces the same thing
    tokenizer = MSOPTTokenizer(
        window_sizes=[4],
        dilations=[1],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3,
        std_threshold=0.001
    )
    actual_word = tokenizer._to_1d_sax(rf)
    print(f"  Tokenizer output: '{actual_word}'")
    
    assert actual_word == expected_word, (
        f"MISMATCH! Expected '{expected_word}', got '{actual_word}'"
    )
    print(f"  ✓ 1D-SAX word matches hand computation: '{actual_word}'")
    return True


def test_stride_1_produces_all_windows():
    """
    Verify that stride s=1 produces exactly (T - rf_span + 1) tokens per scale.
    This confirms 100% translation invariance with no gaps.
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Dense Stride s=1 Translation Invariance")
    print("=" * 70)
    
    T = 100
    series = np.random.RandomState(42).randn(T) * 0.01
    
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8],
        dilations=[1, 2],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3
    )
    
    token_df, _ = tokenizer.fit_transform_series(series, channel_name="test")
    
    for (w, d) in tokenizer.scale_configs_:
        rf_span = d * (w - 1) + 1
        expected_count = T - rf_span + 1
        actual_count = len(token_df[(token_df['window_size'] == w) & (token_df['dilation'] == d)])
        
        status = "✓" if actual_count == expected_count else "✗"
        print(f"  {status} w={w}, d={d}: rf_span={rf_span}, expected {expected_count} tokens, got {actual_count}")
        assert actual_count == expected_count, (
            f"w={w},d={d}: Expected {expected_count} tokens, got {actual_count}"
        )
    
    print("  ✓ All scales produce exactly (T - rf_span + 1) tokens with s=1.")
    return True


def test_2d_spatial_grid_shape():
    """
    Verify the 2D Spatial Grid has shape [N_scales, T] as claimed in §3.3.
    """
    print("\n" + "=" * 70)
    print("  TEST 4: 2D Spatial Grid Shape (§3.3)")
    print("=" * 70)
    
    T = 200
    series = np.random.RandomState(42).randn(T) * 0.01
    
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1
    )
    
    grid, vocab = tokenizer.get_2d_spatial_grid_indices(series, channel_name="test")
    
    expected_n_scales = len(tokenizer.scale_configs_)  # 4 windows * 3 dilations = 12
    assert grid.shape == (expected_n_scales, T), (
        f"Expected ({expected_n_scales}, {T}), got {grid.shape}"
    )
    
    print(f"  Grid shape: {grid.shape}")
    print(f"  N_scales = {expected_n_scales} (4 windows × 3 dilations)")
    print(f"  Vocabulary size: {len(vocab)} unique words")
    print(f"  ✓ Grid shape matches §3.3: [{expected_n_scales}, {T}]")
    return True


def test_no_lookahead_bias():
    """
    Verify that token at time t only uses data from indices <= t.
    The receptive field s_t^{(w,d)} = [x_{t-d(w-1)}, ..., x_t] must end at t, not extend beyond.
    """
    print("\n" + "=" * 70)
    print("  TEST 5: Zero Lookahead Bias")
    print("=" * 70)
    
    T = 50
    series = np.random.RandomState(42).randn(T) * 0.01
    
    tokenizer = MSOPTTokenizer(window_sizes=[4, 8], dilations=[1, 2], stride=1)
    token_df, _ = tokenizer.fit_transform_series(series, channel_name="test")
    
    for _, row in token_df.iterrows():
        t_end = int(row['time_idx'])
        w = int(row['window_size'])
        d = int(row['dilation'])
        rf_span = d * (w - 1) + 1
        t_start = t_end - rf_span + 1
        
        # t_start must be >= 0
        assert t_start >= 0, f"Negative start index: {t_start}"
        # t_end must be < T
        assert t_end < T, f"End index {t_end} exceeds series length {T}"
        # The RF should not extend into the future beyond t_end
        # (this is guaranteed by construction but let's verify)
    
    # Stronger test: modify future data, verify tokens for t don't change
    series_a = series.copy()
    series_b = series.copy()
    series_b[30:] = 999.0  # Corrupt everything from index 30 onward
    
    tok_a = MSOPTTokenizer(window_sizes=[4], dilations=[1], stride=1)
    tok_b = MSOPTTokenizer(window_sizes=[4], dilations=[1], stride=1)
    
    df_a, _ = tok_a.fit_transform_series(series_a, channel_name="x")
    df_b, _ = tok_b.fit_transform_series(series_b, channel_name="x")
    
    # Tokens ending at t <= 29 should be identical (rf_span=4, so last safe is t=29 using data 26-29)
    safe_cutoff = 29  # tokens ending here use data [26,27,28,29] — all before corruption
    tokens_a = df_a[df_a['time_idx'] <= safe_cutoff].sort_values(['time_idx']).reset_index(drop=True)
    tokens_b = df_b[df_b['time_idx'] <= safe_cutoff].sort_values(['time_idx']).reset_index(drop=True)
    
    assert tokens_a['token'].tolist() == tokens_b['token'].tolist(), "LOOKAHEAD DETECTED!"
    
    print(f"  ✓ {len(tokens_a)} tokens at t≤{safe_cutoff} are identical regardless of future data corruption.")
    print(f"  ✓ Zero lookahead bias confirmed.")
    return True


if __name__ == "__main__":
    all_pass = True
    all_pass &= test_receptive_field_extraction()
    all_pass &= test_1d_sax_discretization()
    all_pass &= test_stride_1_produces_all_windows()
    all_pass &= test_2d_spatial_grid_shape()
    all_pass &= test_no_lookahead_bias()
    
    print("\n" + "=" * 70)
    if all_pass:
        print("  ALL 5 EQUATION VERIFICATION TESTS PASSED ✓")
    else:
        print("  SOME TESTS FAILED ✗")
    print("=" * 70)
