"""
VALMOD / Variable-Length Motif Diagnostic
==========================================
Uses STUMPY's matrix profile to measure motif structure across
multiple window lengths on real SPY daily returns.

Purpose: Confirm that multi-scale repeating patterns actually exist
in financial time series, justifying the MSOPT multi-scale approach.

Reports:
- Motif distance statistics per window size
- Normalized distance (distance / sqrt(2*w)) to compare across scales
- Number of motifs below a significance threshold
"""

import os, sys
import numpy as np
import pandas as pd
import stumpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_valmod_diagnostic():
    print("=" * 80)
    print("  VALMOD MOTIF DIAGNOSTIC — SPY Daily Returns")
    print("=" * 80)
    
    # Load cached SPY data
    csv_path = os.path.join(PROJECT_ROOT, "data", "spy_daily_real.csv")
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    
    # Use daily log returns (same input as tokenizer)
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
    print(f"  Series length: {len(returns)} daily return bars")
    print(f"  Date range: {df.index[1].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    
    window_sizes = [4, 8, 16, 32, 64]
    results = []
    
    print(f"\n{'Window':>8} {'Min Dist':>10} {'Norm Dist':>10} {'Mean Dist':>10} {'Std Dist':>10} {'<0.1 Norm':>10} {'<0.5 Norm':>10}")
    print("-" * 70)
    
    for w in window_sizes:
        # Compute matrix profile
        mp = stumpy.stump(returns, m=w)
        distances = mp[:, 0].astype(float)
        
        # Normalized distance: divide by sqrt(2*w) for cross-scale comparability
        # sqrt(2*w) is the expected distance between two independent random walks of length w
        norm_factor = np.sqrt(2 * w)
        norm_dist = distances / norm_factor
        
        min_d = distances.min()
        mean_d = distances.mean()
        std_d = distances.std()
        norm_min = norm_dist.min()
        
        # Count motifs below significance thresholds
        n_below_01 = (norm_dist < 0.1).sum()
        n_below_05 = (norm_dist < 0.5).sum()
        
        print(f"{w:>8} {min_d:>10.4f} {norm_min:>10.4f} {mean_d:>10.4f} {std_d:>10.4f} {n_below_01:>10d} {n_below_05:>10d}")
        
        results.append({
            'window_size': w,
            'min_distance': min_d,
            'norm_min_distance': norm_min,
            'mean_distance': mean_d,
            'std_distance': std_d,
            'n_motifs_norm_lt_01': n_below_01,
            'n_motifs_norm_lt_05': n_below_05,
            'n_profiles': len(distances)
        })
    
    # Save results
    res_df = pd.DataFrame(results)
    csv_out = os.path.join(RESULTS_DIR, "valmod_diagnostic.csv")
    res_df.to_csv(csv_out, index=False)
    print(f"\n  Saved diagnostic CSV to: {csv_out}")
    
    # Interpretation
    print("\n─── INTERPRETATION ───")
    short_norm = res_df[res_df['window_size'] <= 16]['norm_min_distance'].mean()
    long_norm = res_df[res_df['window_size'] >= 32]['norm_min_distance'].mean()
    
    if short_norm < 0.3:
        print(f"  ✓ Short-scale motifs (w≤16) show strong structure (avg norm dist = {short_norm:.4f})")
    else:
        print(f"  ⚠ Short-scale motifs (w≤16) show weak structure (avg norm dist = {short_norm:.4f})")
    
    if long_norm > 0.5:
        print(f"  → Long-scale patterns (w≥32) approach noise (avg norm dist = {long_norm:.4f})")
        print(f"  → Multi-scale extraction captures real structure at short scales")
        print(f"     but tokens at w=32+ may be adding noise, not signal.")
    else:
        print(f"  ✓ Long-scale patterns also show structure (avg norm dist = {long_norm:.4f})")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar([str(w) for w in res_df['window_size']], res_df['norm_min_distance'], 
                color=['#2B6CB0' if v < 0.3 else '#C53030' for v in res_df['norm_min_distance']])
    axes[0].set_xlabel('Window Size')
    axes[0].set_ylabel('Normalized Min Motif Distance')
    axes[0].set_title('Motif Strength by Scale (lower = stronger)')
    axes[0].axhline(y=0.3, color='gray', linestyle='--', alpha=0.5, label='Significance threshold')
    axes[0].legend()
    
    axes[1].bar([str(w) for w in res_df['window_size']], res_df['n_motifs_norm_lt_05'],
                color='#2B6CB0')
    axes[1].set_xlabel('Window Size')
    axes[1].set_ylabel('Count of Motifs (norm dist < 0.5)')
    axes[1].set_title('Number of Significant Motifs per Scale')
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "valmod_diagnostic.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved diagnostic figure to: {fig_path}")
    
    print("\n" + "=" * 80)
    print("  VALMOD DIAGNOSTIC COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    run_valmod_diagnostic()
