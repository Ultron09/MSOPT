"""
Full 10-Year Multi-Asset Expanding Walk-Forward Benchmark (2016–2025)
=======================================================================
Evaluates MSOPT pattern tokens vs Technical Baseline across SPY, QQQ, AAPL, and TLT.

Protocol:
- 10 Annual Expanding Walk-Forward Test Folds (2016, 2017, ..., 2025)
- Initial Train Window: Jan 2010 to Dec 2015
- Target: Fork B Directional Move y_dir in {-1, 0, +1} (H=5 days, delta=0.5 sigma)
- Slippage: Strict 5 bps (0.05%) deduction per position flip
"""

import os
import sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import fetch_and_clean_ticker, build_real_features_and_targets
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from tests.test_backtest_metrics import calculate_backtest_metrics

def run_asset_walk_forward(ticker: str) -> pd.DataFrame:
    print(f"\n{'='*70}\n  RUNNING 10-YEAR WALK-FORWARD BENCHMARK: {ticker}\n{'='*70}")
    
    csv_path = os.path.join(PROJECT_ROOT, "data", f"{ticker.lower()}_daily_real.csv")
    df = fetch_and_clean_ticker(ticker)
    df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
    df = df.dropna().copy()
    
    returns = df['Return'].values
    
    print(f"  [Tokenizer] Extracting multi-scale 1D-SAX tokens for {ticker}...")
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3
    )
    
    token_df, vocab = tokenizer.fit_transform_series(returns, channel_name=f"{ticker}_ret")
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=f"{ticker}_ret")
    
    # Align bow_df and df
    common_len = min(len(df), len(bow_df))
    df = df.iloc[-common_len:].copy()
    bow_df = bow_df.iloc[-common_len:].copy()
    returns = df['Return'].values
    
    # Filter frequent tokens (min 5 occurrences)
    token_counts = (bow_df > 0).sum(axis=0)
    freq_cols = token_counts[token_counts >= 5].index
    bow_df = bow_df[freq_cols]
    print(f"  → Retained {len(freq_cols)} frequent pattern token features for {ticker}.")
    
    tech_cols = ['Return', 'Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
    X_tech = df[tech_cols].values
    X_tokens = bow_df.values
    y = df['Target_Dir'].values
    dates = df.index
    
    # Annual test folds: 2016 to 2025
    test_years = list(range(2016, 2026))
    
    pred_base_all = []
    pred_msopt_all = []
    y_test_all = []
    ret_test_all = []
    
    for test_yr in test_years:
        train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
        test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
        
        if test_mask.sum() == 0:
            continue
            
        # Fit Baseline
        clf_base = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        clf_base.fit(X_tech[train_mask], y[train_mask])
        pred_base = clf_base.predict(X_tech[test_mask])
        
        # Fit MSOPT Tokens
        clf_msopt = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        clf_msopt.fit(X_tokens[train_mask], y[train_mask])
        pred_msopt = clf_msopt.predict(X_tokens[test_mask])
        
        pred_base_all.extend(pred_base)
        pred_msopt_all.extend(pred_msopt)
        y_test_all.extend(y[test_mask])
        ret_test_all.extend(returns[test_mask])
        
    pred_base_all = np.array(pred_base_all)
    pred_msopt_all = np.array(pred_msopt_all)
    y_test_all = np.array(y_test_all)
    ret_test_all = np.array(ret_test_all)
    
    # Backtest Metrics post 5 bps costs
    _, m_base = calculate_backtest_metrics(ret_test_all, pred_base_all, fee_bps=5.0)
    _, m_msopt = calculate_backtest_metrics(ret_test_all, pred_msopt_all, fee_bps=5.0)
    
    acc_base = (pred_base_all == y_test_all).mean()
    acc_msopt = (pred_msopt_all == y_test_all).mean()
    
    summary = [
        {
            'Asset': ticker,
            'Model_Paradigm': 'Baseline (Tech Lags)',
            'OOS_Accuracy': f"{acc_base:.2%}",
            'Total_Return': f"{m_base['Total_Return']:+.2%}",
            'Sharpe_Ratio': f"{m_base['Sharpe_Ratio']:.4f}",
            'Sortino_Ratio': f"{m_base['Sortino_Ratio']:.4f}",
            'Max_Drawdown': f"{m_base['Max_Drawdown']:.2%}",
            'Position_Flips': m_base['Total_Trades'],
            'Tx_Fee_Cost': f"{m_base['Total_Fee_Cost']:.2%}"
        },
        {
            'Asset': ticker,
            'Model_Paradigm': 'MSOPT Tokens (Ours)',
            'OOS_Accuracy': f"{acc_msopt:.2%}",
            'Total_Return': f"{m_msopt['Total_Return']:+.2%}",
            'Sharpe_Ratio': f"{m_msopt['Sharpe_Ratio']:.4f}",
            'Sortino_Ratio': f"{m_msopt['Sortino_Ratio']:.4f}",
            'Max_Drawdown': f"{m_msopt['Max_Drawdown']:.2%}",
            'Position_Flips': m_msopt['Total_Trades'],
            'Tx_Fee_Cost': f"{m_msopt['Total_Fee_Cost']:.2%}"
        }
    ]
    
    res_df = pd.DataFrame(summary)
    print(f"\n--- {ticker} 10-Year Walk-Forward Summary (2016–2025) ---")
    print(res_df.to_string(index=False))
    return res_df

def run_master_benchmark():
    all_res = []
    tickers = ["SPY", "QQQ", "AAPL", "TLT"]
    
    for ticker in tickers:
        df_res = run_asset_walk_forward(ticker)
        all_res.append(df_res)
        
    master_df = pd.concat(all_res, ignore_index=True)
    
    print(f"\n{'='*80}\n  MASTER 10-YEAR WALK-FORWARD BENCHMARK SUMMARY (2016–2025 POST 5 BPS COSTS)\n{'='*80}")
    print(master_df.to_string(index=False))
    
    results_dir = os.path.join(PROJECT_ROOT, "results")
    out_csv = os.path.join(results_dir, "master_walkforward_summary.csv")
    master_df.to_csv(out_csv, index=False)
    print(f"\n[Output] Saved master summary table to: {out_csv}")
    return master_df

if __name__ == "__main__":
    run_master_benchmark()
