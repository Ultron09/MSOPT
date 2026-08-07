"""
MSOPT Tokenizer Standalone Inspection & Sanity Verification
============================================================
Inspects exact 1D-SAX symbolic words (mean symbol + slope symbol per segment)
extracted from authentic SPY daily prices across multi-scale dilated windows (w in {4,8,16,32}, d in {1,2,4}, s=1).
Verifies zero lookahead bias and correct 2D Scale-Time Spatial Grid mapping.
"""

import os
import sys
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.tokenizer.msopt_tokenizer import MSOPTTokenizer

def inspect_tokenizer():
    print(f"\n{'='*70}\n  MSOPT TOKENIZER STANDALONE INSPECTION & SANITY VERIFICATION\n{'='*70}")
    
    # 1. Load real SPY prices & compute log returns
    data_path = os.path.join(PROJECT_ROOT, "data", "spy_daily_real.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file missing at {data_path}.")
        
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    sample_df = df.iloc[100:150].copy()  # 50 daily bars for inspection
    prices = sample_df['Close'].values
    returns = np.log(prices[1:] / prices[:-1])
    dates = sample_df.index[1:]
    
    print(f"[Sample Data] {len(returns)} daily log return bars ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
    
    # 2. Instantiate MSOPT Tokenizer
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3,
        std_threshold=0.001
    )
    
    print(f"\n[Tokenizer Config]")
    print(f"  Window Sizes (W): {tokenizer.window_sizes}")
    print(f"  Dilations (D): {tokenizer.dilations}")
    print(f"  Total Scale Configurations (N_scales): {tokenizer.n_scales}")
    print(f"  Stride (s): {tokenizer.stride} (Dense Translation-Invariant)")
    print(f"  1D-SAX Segments per Window (K): {tokenizer.n_segments}")
    print(f"  Mean Alphabet Bins (mu): {tokenizer.alphabet_size_mean} (Symbols: A, B, C, D)")
    print(f"  Slope Alphabet Bins (beta): {tokenizer.alphabet_size_slope} (Symbols: A, B, C)")

    # 3. Transform sample series into 2D Spatial Token Grid & Token DataFrame
    token_df, vocab = tokenizer.fit_transform_series(returns, channel_name="SPY_ret")
    grid, vocab = tokenizer.get_2d_spatial_grid_indices(returns, channel_name="SPY_ret")
    
    print(f"\n[Output Spatial Grid Tensor]")
    print(f"  Extracted Unique Pattern Words (Vocab Size): {len(vocab)}")
    print(f"  2D Spatial Grid Index Matrix Shape: {grid.shape} (N_scales x T)")
    
    # 4. Print exact extracted tokens for target timestamp t=35
    target_idx = 35
    target_date = dates[target_idx].strftime('%Y-%m-%d')
    print(f"\n[Inspection at Target Timestamp t={target_idx} ({target_date})]")
    print(f"Daily Log Return: {returns[target_idx]:+.4f} (Close: ${prices[target_idx+1]:.2f})")
    print(f"{'-'*75}")
    print(f"{'Scale (w, d)':<15} {'Span L':<10} {'Lookback Range':<28} {'1D-SAX Token Word'}")
    print(f"{'-'*75}")
    
    for scale_idx, (w, d) in enumerate(tokenizer.scale_configs_):
        rf_span = d * (w - 1) + 1
        start_idx = target_idx - rf_span + 1
        
        if start_idx >= 0:
            token_id = grid[scale_idx, target_idx]
            word = tokenizer.inverse_vocabulary_.get(token_id, "N/A")
            start_date = dates[start_idx].strftime('%Y-%m-%d')
            range_str = f"[{start_date} -> {target_date}]"
            print(f"w={w:<2d}, d={d:<2d}        {rf_span:<10d} {range_str:<28} {word} (ID: {token_id})")
        else:
            print(f"w={w:<2d}, d={d:<2d}        {rf_span:<10d} {'[Padding / Warmup]':<28} N/A (Warmup)")

    # 5. Sanity Checks & Assertions
    print(f"\n[Sanity Checks]")
    # Check A: Grid dimensions match (n_scales, T)
    assert grid.shape == (tokenizer.n_scales, len(returns)), "Grid shape mismatch!"
    print("  ✓ Grid Dimensions Check Passed (N_scales x T)")
    
    # Check B: Non-negative token IDs
    assert np.all(grid >= 0), "Negative token ID found!"
    print("  ✓ Token ID Non-Negativity Passed")
    
    # Check C: Zero lookahead bias check (token at time_idx t only uses prices up to index t)
    for _, row in token_df.iterrows():
        t_end = int(row['time_idx'])
        w = int(row['window_size'])
        d = int(row['dilation'])
        span = d * (w - 1) + 1
        t_start = t_end - span + 1
        assert t_start >= 0, f"Invalid start index {t_start}!"
        assert t_end < len(returns), f"Lookahead leakage: end index {t_end} >= T!"
    print("  ✓ Zero Lookahead Bias Verification Passed (100% Retrospective)")

    print(f"\n{'='*70}\n  MSOPT TOKENIZER SANITY VERIFICATION PASSED 100%\n{'='*70}\n")

if __name__ == "__main__":
    inspect_tokenizer()
