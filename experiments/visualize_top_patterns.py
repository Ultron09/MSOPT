"""
Pillar 4: MSOPT Pattern Codebook Interpretability & Visualization
==================================================================
Identifies top predictive 1D-SAX pattern tokens and plots actual historical 
price chart subseries matching those token words.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import prepare_benchmark_dataset
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def generate_pattern_visualizations(ticker: str = "SPY"):
    print(f"\n{'='*70}\n  PILLAR 4: PATTERN CODEBOOK INTERPRETABILITY ({ticker})\n{'='*70}")
    df = prepare_benchmark_dataset(ticker)
    returns = df['Return'].values
    prices = df['Close'].values
    
    tokenizer = MSOPTTokenizer(window_sizes=[4, 8, 16, 32], dilations=[1, 2], stride=1)
    tokens_df, vocab = tokenizer.fit_transform_series(returns, channel_name=ticker)
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=ticker)
    bow_df.index = df.index[:len(bow_df)]
    
    combined = pd.concat([df[['Target_Dir']], bow_df], axis=1).dropna()
    token_cols = [c for c in combined.columns if c != 'Target_Dir']
    freq_cols = [c for c in token_cols if combined[c].sum() >= 10]
    
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.03, random_state=42, verbose=-1)
    clf.fit(combined[freq_cols], combined['Target_Dir'])
    
    importances = clf.feature_importances_
    sorted_indices = np.argsort(importances)[::-1]
    
    top_token_ids = [int(str(freq_cols[idx]).replace('token_', '')) for idx in sorted_indices[:4]]
    top_words = [tokenizer.inverse_vocabulary_[tid] for tid in top_token_ids]
    
    print(f"  Top-4 Most Predictive Pattern Tokens:")
    for i, w in enumerate(top_words):
        print(f"   #{i+1}: {w} (Importance Score: {importances[sorted_indices[i]]})")
        
    # Plot top 4 pattern subseries
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    axes = axes.flatten()
    
    plt.suptitle(f"MSOPT Learned Pattern Codebook Primitives ({ticker})", fontsize=12, fontweight='bold', y=0.98)
    
    for i, word in enumerate(top_words):
        matches = tokens_df[tokens_df['token'] == word]
        ax = axes[i]
        
        # Plot up to 5 actual price subseries matching this token
        sample_matches = matches.head(5)
        for _, row in sample_matches.iterrows():
            t_end = int(row['time_idx'])
            w_len = int(row['window_size'])
            d_len = int(row['dilation'])
            span = d_len * (w_len - 1) + 1
            t_start = max(0, t_end - span + 1)
            
            subseries = prices[t_start : t_end + 1 : d_len]
            norm_subseries = (subseries - np.mean(subseries)) / (np.std(subseries) + 1e-8)
            ax.plot(norm_subseries, alpha=0.7, linewidth=1.5)
            
        ax.set_title(f"Motif #{i+1}: {word[:25]}...", fontsize=9, fontweight='bold', color='#1A365D')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylabel("Normalized Price", fontsize=8)
        ax.set_xlabel("Receptive Field Index", fontsize=8)
        
    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "top_pattern_motifs.png")
    plt.savefig(fig_path)
    print(f"  → Saved motif visualization figure to: {fig_path}")

if __name__ == "__main__":
    generate_pattern_visualizations("SPY")
