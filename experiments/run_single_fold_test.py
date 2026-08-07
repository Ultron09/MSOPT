"""
Single-Fold Walk-Forward Baseline Test (SPY 2016)
=================================================
Executes a single, transparent walk-forward fold:
- Train Split: Jan 4, 2010 to Dec 31, 2015 (1,509 daily bars)
- Test Split:  Jan 4, 2016 to Dec 30, 2016 (252 daily bars)

Target: Fork B Volatility-Scaled Directional Move y_dir in {-1, 0, +1} (H=5 days, delta=0.5 sigma).
Evaluates Baseline (Technical Lags + Parkinson Vol) vs MSOPT 1D-SAX Tokens post 5 bps transaction costs.
"""

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import fetch_and_clean_ticker, build_real_features_and_targets
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from tests.test_backtest_metrics import calculate_backtest_metrics

def run_single_fold_spy():
    print(f"\n{'='*75}\n  SINGLE-FOLD WALK-FORWARD BASELINE RUN: SPY (TEST YEAR 2016)\n{'='*75}")
    
    # 1. Load authentic SPY daily data
    csv_path = os.path.join(PROJECT_ROOT, "data", "spy_daily_real.csv")
    df = fetch_and_clean_ticker("SPY")
    df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
    
    print(f"[Data] Loaded {len(df)} authentic daily SPY bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    
    # 2. Extract MSOPT 1D-SAX tokens
    df = df.dropna().copy()
    returns = df['Return'].values
    
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3
    )
    
    token_df, vocab = tokenizer.fit_transform_series(returns, channel_name="SPY_ret")
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name="SPY_ret")
    
    # Align bow_df and df exactly
    common_len = min(len(df), len(bow_df))
    df = df.iloc[-common_len:].copy()
    bow_df = bow_df.iloc[-common_len:].copy()
    returns = df['Return'].values
    
    # Filter frequent tokens (min 10 occurrences)
    token_counts = (bow_df > 0).sum(axis=0)
    freq_cols = token_counts[token_counts >= 5].index
    bow_df = bow_df[freq_cols]
    print(f"[Tokenizer] Extracted {len(freq_cols)} frequent 1D-SAX pattern token features.")
    
    # 3. Define Train & Test Splits
    train_mask = (df.index >= '2010-01-04') & (df.index <= '2015-12-31')
    test_mask = (df.index >= '2016-01-04') & (df.index <= '2016-12-30')
    
    # Baseline Feature Matrix (Technical Lags + Parkinson Volatility)
    tech_cols = ['Return', 'Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
    X_tech = df[tech_cols].values
    X_tokens = bow_df.values
    y = df['Target_Dir'].values
    
    print(f"[Splits] Train Bars: {train_mask.sum()} (2010–2015) | Test Bars: {test_mask.sum()} (2016)")
    print(f"[Target Class Distribution - Train]: -1: {(y[train_mask] == -1).sum()}, 0: {(y[train_mask] == 0).sum()}, +1: {(y[train_mask] == 1).sum()}")
    print(f"[Target Class Distribution - Test ]: -1: {(y[test_mask] == -1).sum()}, 0: {(y[test_mask] == 0).sum()}, +1: {(y[test_mask] == 1).sum()}")
    
    # 4. Train Models
    # A. Technical Baseline
    clf_base = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    clf_base.fit(X_tech[train_mask], y[train_mask])
    pred_base = clf_base.predict(X_tech[test_mask])
    
    # B. MSOPT Pattern Tokens
    clf_msopt = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    clf_msopt.fit(X_tokens[train_mask], y[train_mask])
    pred_msopt = clf_msopt.predict(X_tokens[test_mask])
    
    # 5. Evaluate Backtests post 5 bps costs
    test_returns = returns[test_mask]
    test_dates = df.index[test_mask]
    
    df_steps_base, metrics_base = calculate_backtest_metrics(test_returns, pred_base, fee_bps=5.0)
    df_steps_msopt, metrics_msopt = calculate_backtest_metrics(test_returns, pred_msopt, fee_bps=5.0)
    
    # 6. Print Out-of-Sample Results
    acc_base = (pred_base == y[test_mask]).mean()
    acc_msopt = (pred_msopt == y[test_mask]).mean()
    
    print(f"\n{'-'*75}")
    print(f"  2016 OUT-OF-SAMPLE BENCHMARK COMPARISON (SPY POST 5 BPS COSTS)")
    print(f"{'-'*75}")
    print(f"{'Metric':<25} {'Baseline (Tech Lags)':<25} {'MSOPT Tokens (Ours)'}")
    print(f"{'-'*75}")
    print(f"{'OOS Accuracy':<25} {acc_base:<25.2%} {acc_msopt:.2%}")
    print(f"{'Total Return':<25} {metrics_base['Total_Return']:<+25.2%} {metrics_msopt['Total_Return']:+.2%}")
    print(f"{'Sharpe Ratio':<25} {metrics_base['Sharpe_Ratio']:<25.4f} {metrics_msopt['Sharpe_Ratio']:.4f}")
    print(f"{'Sortino Ratio':<25} {metrics_base['Sortino_Ratio']:<25.4f} {metrics_msopt['Sortino_Ratio']:.4f}")
    print(f"{'Max Drawdown':<25} {metrics_base['Max_Drawdown']:<25.2%} {metrics_msopt['Max_Drawdown']:.2%}")
    print(f"{'Total Position Flips':<25} {metrics_base['Total_Trades']:<25d} {metrics_msopt['Total_Trades']:d}")
    print(f"{'Total Fee Cost Paid':<25} {metrics_base['Total_Fee_Cost']:<25.4%} {metrics_msopt['Total_Fee_Cost']:.4%}")
    print(f"{'-'*75}\n")
    
    # 7. Plot 2016 Cumulative Wealth Curves
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.plot(test_dates, df_steps_base['Wealth_W'], label=f"Baseline (Sharpe={metrics_base['Sharpe_Ratio']:.2f}, DD={metrics_base['Max_Drawdown']:.1%})", color='#718096', linewidth=1.5)
    ax.plot(test_dates, df_steps_msopt['Wealth_W'], label=f"MSOPT Tokens (Sharpe={metrics_msopt['Sharpe_Ratio']:.2f}, DD={metrics_msopt['Max_Drawdown']:.1%})", color='#2B6CB0', linewidth=2.0)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.6)
    ax.set_title("SPY 2016 Walk-Forward Out-of-Sample Wealth Curve (Post 5 Bps Costs)", fontsize=11, fontweight='bold')
    ax.set_ylabel("Wealth W_t (Starting at 1.0)", fontsize=9)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.legend(fontsize=9, loc='upper left')
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    fig_path = os.path.join(results_dir, "single_fold_spy_2016.png")
    plt.savefig(fig_path)
    print(f"[Output] Saved 2016 wealth curve figure to: {fig_path}")

if __name__ == "__main__":
    run_single_fold_spy()
