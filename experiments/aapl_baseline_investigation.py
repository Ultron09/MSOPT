"""
AAPL Baseline +2354% Investigation
====================================
The AAPL technical baseline returned +2354% while AAPL's buy-and-hold was +1044%.
This script investigates whether signal leakage explains the anomaly.

Hypotheses:
1. Using 'Return' (today's log return) as a feature creates autocorrelation 
   leakage on a trending stock — the model learns "if up today, probably up 
   this week" which isn't prediction, it's momentum autocorrelation.
2. The 3-class target (up/flat/down) with volatility-scaled thresholds may 
   create a class imbalance that accidentally biases toward long positions.
"""

import os, sys
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import fetch_and_clean_ticker, build_real_features_and_targets
from tests.test_backtest_metrics import calculate_backtest_metrics


def investigate():
    print("=" * 80)
    print("  AAPL BASELINE +2354% INVESTIGATION")
    print("=" * 80)
    
    df = fetch_and_clean_ticker("AAPL")
    df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
    df = df.dropna().copy()
    
    returns = df['Return'].values
    dates = df.index
    
    # ─────────────────────────────────────────────────────
    # CHECK 1: Target class distribution
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 1: Target Class Distribution ───")
    y = df['Target_Dir'].values
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    for u, c in zip(unique, counts):
        label = {-1: "Down", 0: "Flat", 1: "Up"}[u]
        print(f"  {label:5s} ({u:+d}): {c:5d} ({c/total:.1%})")
    
    # ─────────────────────────────────────────────────────
    # CHECK 2: Feature autocorrelation with target
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 2: Feature Correlation with 5-Day Forward Return ───")
    tech_cols = ['Return', 'Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
    forward_ret = df['Forward_Return_H'].values
    
    for col in tech_cols:
        feat = df[col].values
        corr = np.corrcoef(feat, forward_ret)[0, 1]
        print(f"  Corr({col:15s}, Forward_Return_H) = {corr:+.4f}")
    
    # ─────────────────────────────────────────────────────
    # CHECK 3: Year-by-year fold breakdown
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 3: Year-by-Year Fold Breakdown ───")
    print(f"{'Year':>6} {'Acc':>8} {'B&H Ret':>10} {'Base Ret':>10} {'Base Sharpe':>12} {'Flips':>6} {'Fees':>8} {'Pred +1':>8} {'Pred 0':>8} {'Pred -1':>8}")
    print("-" * 100)
    
    test_years = list(range(2016, 2026))
    X_tech = df[tech_cols].values
    
    for test_yr in test_years:
        train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
        test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
        if test_mask.sum() == 0:
            continue
        
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        clf.fit(X_tech[train_mask], y[train_mask])
        pred = clf.predict(X_tech[test_mask])
        
        ret_test = returns[test_mask]
        acc = (pred == y[test_mask]).mean()
        
        # Buy and hold for this year
        bnh_ret = np.exp(ret_test.sum()) - 1
        
        # Strategy return
        _, m = calculate_backtest_metrics(ret_test, pred, fee_bps=5.0)
        
        # Signal distribution
        n_up = (pred == 1).sum()
        n_flat = (pred == 0).sum()
        n_down = (pred == -1).sum()
        
        print(f"{test_yr:>6} {acc:>7.1%} {bnh_ret:>+9.1%} {m['Total_Return']:>+9.1%} {m['Sharpe_Ratio']:>11.4f} {m['Total_Trades']:>6d} {m['Total_Fee_Cost']:>7.2%} {n_up:>8d} {n_flat:>8d} {n_down:>8d}")
    
    # ─────────────────────────────────────────────────────
    # CHECK 4: Run WITHOUT 'Return' feature
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 4: Baseline WITHOUT 'Return' Feature ───")
    tech_cols_no_ret = ['Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
    X_tech_no_ret = df[tech_cols_no_ret].values
    
    pred_all_noret = []
    ret_all = []
    y_all = []
    
    for test_yr in test_years:
        train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
        test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
        if test_mask.sum() == 0:
            continue
        
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        clf.fit(X_tech_no_ret[train_mask], y[train_mask])
        pred = clf.predict(X_tech_no_ret[test_mask])
        
        pred_all_noret.extend(pred)
        ret_all.extend(returns[test_mask])
        y_all.extend(y[test_mask])
    
    pred_all_noret = np.array(pred_all_noret)
    ret_all = np.array(ret_all)
    y_all = np.array(y_all)
    
    acc_noret = (pred_all_noret == y_all).mean()
    _, m_noret = calculate_backtest_metrics(ret_all, pred_all_noret, fee_bps=5.0)
    
    print(f"  WITH 'Return':    Acc={0.4320:.1%}, Return=+2354.43%, Sharpe=1.8004")
    print(f"  WITHOUT 'Return': Acc={acc_noret:.1%}, Return={m_noret['Total_Return']:+.2%}, Sharpe={m_noret['Sharpe_Ratio']:.4f}")
    print(f"  Flips: {m_noret['Total_Trades']}, Fees: {m_noret['Total_Fee_Cost']:.2%}")
    
    drop_pct = (2354.43 - m_noret['Total_Return'] * 100) / 2354.43 * 100
    print(f"\n  Removing 'Return' changes the result by {drop_pct:+.1f}%")
    if abs(m_noret['Total_Return']) < 10.44:  # Less than buy-and-hold
        print(f"  → Without 'Return', baseline underperforms buy-and-hold (+1044%)")
        print(f"  → CONFIRMS: 'Return' feature is the primary driver of the anomalous +2354%")
    
    # ─────────────────────────────────────────────────────
    # CHECK 5: Buy-and-hold compounded return
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 5: Buy-and-Hold Benchmark ───")
    bnh_returns = ret_all.copy()
    bnh_wealth = np.cumprod(1 + bnh_returns)
    bnh_total = bnh_wealth[-1] - 1
    bnh_sharpe = np.mean(bnh_returns) / (np.std(bnh_returns, ddof=1) + 1e-8) * np.sqrt(252)
    bnh_dd = np.min((bnh_wealth - np.maximum.accumulate(bnh_wealth)) / np.maximum.accumulate(bnh_wealth))
    
    print(f"  Buy-and-Hold (OOS period only):")
    print(f"    Total Return: {bnh_total:+.2%}")
    print(f"    Sharpe:       {bnh_sharpe:.4f}")
    print(f"    Max Drawdown: {bnh_dd:.2%}")
    
    # ─────────────────────────────────────────────────────
    # CHECK 6: Feature importance
    # ─────────────────────────────────────────────────────
    print("\n─── CHECK 6: LightGBM Feature Importance (Full Training Set) ───")
    full_train_mask = dates < '2025-01-01'
    clf_full = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    clf_full.fit(X_tech[full_train_mask], y[full_train_mask])
    
    importances = clf_full.feature_importances_
    for col, imp in sorted(zip(tech_cols, importances), key=lambda x: -x[1]):
        pct = imp / importances.sum() * 100
        bar = "█" * int(pct / 2)
        print(f"  {col:20s}: {imp:5d} ({pct:5.1f}%) {bar}")
    
    print("\n" + "=" * 80)
    print("  INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    investigate()
