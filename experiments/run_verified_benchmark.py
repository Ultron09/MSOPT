"""
Verified Walk-Forward Benchmark
================================
The single source of truth for all numbers in the paper.

Design principles:
1. SHA-256 verification of input data at load time
2. Deterministic seeding (numpy, random, lightgbm random_state)
3. Per-day prediction log — every number traceable to a specific day
4. Buy-and-hold benchmark included for every asset
5. 'Return' REMOVED from baseline features (confirmed autocorrelation leakage)
6. Self-check: runs twice, verifies output hash matches

Models evaluated:
  - Buy-and-Hold: trivial long-only benchmark
  - Technical Baseline (GBDT): Parkinson_Vol, Vol_MA30, Rel_Volume (NO Return)
  - MSOPT Tokens (GBDT): Rolling BoW histogram of multi-scale 1D-SAX tokens
"""

import os, sys, hashlib, random
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import build_real_features_and_targets
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from tests.test_backtest_metrics import calculate_backtest_metrics

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# SHA-256 hashes of the pinned data files
EXPECTED_HASHES = {
    "spy_daily_real.csv": "180d357d0847e391f3a8fc3f3cac3d108a3024b9a907d412eb9e96238f36b78a",
    "qqq_daily_real.csv": "934a3e2b807f80d1dc5f06b93cf381d73f0316a43ef133f3bec1528f1a065c08",
    "aapl_daily_real.csv": "fc8934204489ae6f75f9fcd540a1395c4bd8f43646655dd137d6c3dd6dccae95",
    "tlt_daily_real.csv": "3ce42cd2a3f38b06200458b2f6f9b2e4a416e1d05a79767c8a54bc4ae07654c9",
}

SEED = 42
TICKERS = ["SPY", "QQQ", "AAPL", "TLT"]
TEST_YEARS = list(range(2016, 2026))
FEE_BPS = 5.0

# Baseline features — 'Return' REMOVED after investigation confirmed autocorrelation leakage
BASELINE_FEATURES = ['Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']


def verify_data_hash(filepath: str, expected_hash: str):
    """Abort if data file doesn't match pinned SHA-256 hash."""
    with open(filepath, 'rb') as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            f"DATA INTEGRITY FAILURE: {filepath}\n"
            f"  Expected SHA-256: {expected_hash}\n"
            f"  Actual SHA-256:   {actual_hash}\n"
            f"  The data file has changed since it was pinned. Results are not reproducible."
        )
    return True


def set_deterministic_seed(seed: int):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)


def load_and_verify_data(ticker: str) -> pd.DataFrame:
    """Load cached CSV with SHA-256 verification."""
    filename = f"{ticker.lower()}_daily_real.csv"
    filepath = os.path.join(PROJECT_ROOT, "data", filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Missing data file: {filepath}")
    
    verify_data_hash(filepath, EXPECTED_HASHES[filename])
    print(f"  ✓ SHA-256 verified: {filename}")
    
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
    df = df.dropna().copy()
    return df


def run_single_asset(ticker: str) -> pd.DataFrame:
    """Run walk-forward benchmark for one asset, returning per-day prediction log."""
    print(f"\n{'='*70}")
    print(f"  ASSET: {ticker}")
    print(f"{'='*70}")
    
    set_deterministic_seed(SEED)
    df = load_and_verify_data(ticker)
    
    returns = df['Return'].values
    dates = df.index
    y = df['Target_Dir'].values
    
    # Prepare baseline features (NO 'Return')
    X_base = df[BASELINE_FEATURES].values
    
    # Prepare MSOPT token features
    print(f"  Extracting MSOPT tokens...")
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1, n_segments=4,
        alphabet_size_mean=4, alphabet_size_slope=3
    )
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=f"{ticker}_ret")
    
    # Align
    common_len = min(len(df), len(bow_df))
    df = df.iloc[-common_len:].copy()
    bow_df = bow_df.iloc[-common_len:].copy()
    returns = df['Return'].values
    dates = df.index
    y = df['Target_Dir'].values
    X_base = df[BASELINE_FEATURES].values
    
    # Filter infrequent tokens
    token_counts = (bow_df > 0).sum(axis=0)
    freq_cols = token_counts[token_counts >= 5].index
    bow_df = bow_df[freq_cols]
    X_tokens = bow_df.values
    print(f"  Retained {len(freq_cols)} frequent token features")
    
    # Per-day prediction log
    daily_log = []
    
    for test_yr in TEST_YEARS:
        train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
        test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
        n_test = test_mask.sum()
        if n_test == 0:
            continue
        
        # Train baseline GBDT
        clf_base = LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=SEED, verbose=-1
        )
        clf_base.fit(X_base[train_mask], y[train_mask])
        pred_base = clf_base.predict(X_base[test_mask])
        
        # Train MSOPT token GBDT
        clf_msopt = LGBMClassifier(
            n_estimators=100, learning_rate=0.05, max_depth=3,
            random_state=SEED, verbose=-1
        )
        clf_msopt.fit(X_tokens[train_mask], y[train_mask])
        pred_msopt = clf_msopt.predict(X_tokens[test_mask])
        
        # Log each day
        test_dates = dates[test_mask]
        test_returns = returns[test_mask]
        test_y = y[test_mask]
        
        for i, (date, ret, actual) in enumerate(zip(test_dates, test_returns, test_y)):
            daily_log.append({
                'date': date.strftime('%Y-%m-%d'),
                'asset': ticker,
                'fold_year': test_yr,
                'actual_return': float(ret),
                'actual_target': int(actual),
                'pred_bnh': 1,  # Buy-and-hold = always long
                'pred_baseline': int(pred_base[i]),
                'pred_msopt': int(pred_msopt[i]),
            })
    
    log_df = pd.DataFrame(daily_log)
    print(f"  Logged {len(log_df)} daily predictions across {len(TEST_YEARS)} folds")
    return log_df


def compute_strategy_metrics(log_df: pd.DataFrame, asset: str, model_col: str, model_name: str) -> dict:
    """Compute backtest metrics from daily prediction log."""
    mask = log_df['asset'] == asset
    returns = log_df.loc[mask, 'actual_return'].values
    signals = log_df.loc[mask, model_col].values
    actuals = log_df.loc[mask, 'actual_target'].values
    
    _, metrics = calculate_backtest_metrics(returns, signals, fee_bps=FEE_BPS)
    acc = float((signals == actuals).mean())
    
    return {
        'Asset': asset,
        'Model': model_name,
        'OOS_Accuracy': f"{acc:.2%}",
        'Total_Return': f"{metrics['Total_Return']:+.2%}",
        'Sharpe': f"{metrics['Sharpe_Ratio']:.4f}",
        'Sortino': f"{metrics['Sortino_Ratio']:.4f}",
        'Max_DD': f"{metrics['Max_Drawdown']:.2%}",
        'Flips': metrics['Total_Trades'],
        'Fee_Cost': f"{metrics['Total_Fee_Cost']:.2%}",
    }


def run_full_benchmark():
    print("=" * 80)
    print("  VERIFIED WALK-FORWARD BENCHMARK")
    print(f"  Seed: {SEED} | Fee: {FEE_BPS} bps | Folds: {TEST_YEARS[0]}-{TEST_YEARS[-1]}")
    print(f"  Baseline features: {BASELINE_FEATURES} (NO 'Return' — leakage confirmed)")
    print("=" * 80)
    
    # Run all assets
    all_logs = []
    for ticker in TICKERS:
        log_df = run_single_asset(ticker)
        all_logs.append(log_df)
    
    master_log = pd.concat(all_logs, ignore_index=True)
    
    # Save per-day log
    daily_log_path = os.path.join(RESULTS_DIR, "verified_benchmark_daily_log.csv")
    master_log.to_csv(daily_log_path, index=False)
    print(f"\n  Saved per-day prediction log: {daily_log_path}")
    
    # Compute summary metrics
    summary_rows = []
    for asset in TICKERS:
        summary_rows.append(compute_strategy_metrics(master_log, asset, 'pred_bnh', 'Buy-and-Hold'))
        summary_rows.append(compute_strategy_metrics(master_log, asset, 'pred_baseline', 'Baseline (Vol Features)'))
        summary_rows.append(compute_strategy_metrics(master_log, asset, 'pred_msopt', 'MSOPT Tokens (Ours)'))
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(RESULTS_DIR, "verified_benchmark_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    print(f"\n{'='*80}")
    print("  VERIFIED BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(summary_df.to_string(index=False))
    
    # Compute hash of output for determinism check
    with open(daily_log_path, 'rb') as f:
        output_hash = hashlib.sha256(f.read()).hexdigest()
    print(f"\n  Output SHA-256: {output_hash}")
    
    return master_log, summary_df, output_hash


def determinism_self_check():
    """Run the benchmark twice and verify outputs are identical."""
    print("\n" + "=" * 80)
    print("  DETERMINISM SELF-CHECK: Running benchmark twice")
    print("=" * 80)
    
    _, _, hash1 = run_full_benchmark()
    print(f"\n  Run 1 hash: {hash1}")
    
    _, _, hash2 = run_full_benchmark()
    print(f"  Run 2 hash: {hash2}")
    
    if hash1 == hash2:
        print(f"\n  ✓ DETERMINISM CONFIRMED — both runs produced identical output")
        print(f"    SHA-256: {hash1}")
    else:
        print(f"\n  ✗ DETERMINISM FAILED — outputs differ!")
        print(f"    Run 1: {hash1}")
        print(f"    Run 2: {hash2}")
        raise ValueError("Non-deterministic benchmark output!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-determinism", action="store_true",
                        help="Run twice and verify outputs match")
    args = parser.parse_args()
    
    if args.check_determinism:
        determinism_self_check()
    else:
        run_full_benchmark()
