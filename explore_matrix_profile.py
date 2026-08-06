"""
Real Matrix Profile & Motif Discovery Diagnostic for SPY
=========================================================
Uses STUMPY to compute 1D matrix profiles across multiple window sizes (m in {5, 10, 20, 30, 50}).
Identifies authentic recurring chart motif pairs and normalized Euclidean distances on SPY daily log returns.
"""

import os
import sys
import numpy as np
import pandas as pd
import stumpy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "spy_daily_real.csv")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_matrix_profile_diagnostic():
    print(f"\n{'='*70}\n  REAL MATRIX PROFILE MOTIF DIAGNOSTIC (SPY DAILY DATA)\n{'='*70}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file missing at {DATA_PATH}. Run src/data/preprocessing.py first.")

    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    returns = np.log(df['Close'] / df['Close'].shift(1)).dropna().values
    dates = df.index[1:]
    
    print(f"[Data] Analyzed {len(returns)} authentic daily SPY return bars ({dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')})")
    
    window_sizes = [5, 10, 20, 30, 50]
    motif_summary = []
    
    fig, axes = plt.subplots(len(window_sizes), 2, figsize=(12, 10), dpi=300)
    plt.suptitle("STUMPY Matrix Profile & Top Motif Pairs on SPY Daily Log Returns", fontsize=12, fontweight='bold', y=0.99)
    
    for idx, m in enumerate(window_sizes):
        print(f"\n[STUMPY] Computing 1D Matrix Profile for window size m = {m}...")
        mp = stumpy.stump(returns, m=m)
        
        matrix_distances = mp[:, 0].astype(float)
        nearest_neighbor_indices = mp[:, 1].astype(int)
        
        # Top motif pair (global minimum distance)
        motif_idx1 = int(np.argmin(matrix_distances))
        motif_idx2 = int(nearest_neighbor_indices[motif_idx1])
        min_dist = float(matrix_distances[motif_idx1])
        
        # Subseries normalized Euclidean distance is d_norm = sqrt(2 * m * (1 - Pearson_corr))
        # Normalized distance per bar: d_norm / sqrt(m)
        scaled_dist = min_dist / np.sqrt(m)
        
        date1 = dates[motif_idx1].strftime('%Y-%m-%d')
        date2 = dates[motif_idx2].strftime('%Y-%m-%d')
        
        print(f"  → Window m={m:2d}: Min Distance = {min_dist:.4f} (Scaled = {scaled_dist:.4f})")
        print(f"     Motif 1 Start: {date1} (Index {motif_idx1})")
        print(f"     Motif 2 Start: {date2} (Index {motif_idx2})")
        
        motif_summary.append({
            'Window_m': m,
            'Min_Distance': min_dist,
            'Scaled_Distance': scaled_dist,
            'Motif1_Date': date1,
            'Motif2_Date': date2,
            'Motif1_Index': motif_idx1,
            'Motif2_Index': motif_idx2
        })
        
        # Plot Matrix Profile distance curve
        ax_mp = axes[idx, 0]
        ax_mp.plot(matrix_distances, color='#2B6CB0', linewidth=0.8)
        ax_mp.axvline(motif_idx1, color='#C53030', linestyle='--', alpha=0.7, label=f"Motif 1: {date1}")
        ax_mp.axvline(motif_idx2, color='#2F855A', linestyle='--', alpha=0.7, label=f"Motif 2: {date2}")
        ax_mp.set_ylabel(f"m={m}\nDist", fontsize=8)
        ax_mp.grid(True, linestyle=':', alpha=0.5)
        ax_mp.legend(fontsize=7, loc='upper right')
        if idx == 0:
            ax_mp.set_title("Matrix Profile Distance P_x", fontsize=9, fontweight='bold')
            
        # Plot Overlay of Top Motif Pair Subseries
        ax_sub = axes[idx, 1]
        sub1 = returns[motif_idx1 : motif_idx1 + m]
        sub2 = returns[motif_idx2 : motif_idx2 + m]
        
        # z-normalize subseries for visual comparison
        norm_sub1 = (sub1 - np.mean(sub1)) / (np.std(sub1) + 1e-8)
        norm_sub2 = (sub2 - np.mean(sub2)) / (np.std(sub2) + 1e-8)
        
        ax_sub.plot(norm_sub1, color='#C53030', label=f"Motif 1 ({date1})", linewidth=1.5)
        ax_sub.plot(norm_sub2, color='#2F855A', linestyle='--', label=f"Motif 2 ({date2})", linewidth=1.5)
        ax_sub.set_ylabel("z-score", fontsize=8)
        ax_sub.grid(True, linestyle=':', alpha=0.5)
        ax_sub.legend(fontsize=7, loc='upper right')
        if idx == 0:
            ax_sub.set_title("z-Normalized Motif Shape Overlay", fontsize=9, fontweight='bold')

    axes[-1, 0].set_xlabel("Time Index t", fontsize=8)
    axes[-1, 1].set_xlabel("Window Offset (bars)", fontsize=8)
    
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "matrix_profile_motifs.png")
    plt.savefig(fig_path)
    print(f"\n[Output] Saved motif diagnostic figure to: {fig_path}")
    
    res_df = pd.DataFrame(motif_summary)
    summary_csv = os.path.join(RESULTS_DIR, "matrix_profile_summary.csv")
    res_df.to_csv(summary_csv, index=False)
    print(f"[Output] Saved motif summary CSV to: {summary_csv}")
    
    print(f"\n--- Matrix Profile Motif Summary Table ---")
    print(res_df.to_string(index=False))
    return res_df

if __name__ == "__main__":
    run_matrix_profile_diagnostic()
