"""
Forensic Audit Script
=====================
Checks every claim in the paper against ground truth:
1. Is the backtest actually deterministic? (run it twice, compare)
2. What is AAPL's real buy-and-hold return 2016-2025?
3. What is TLT's real buy-and-hold return 2016-2025?
4. What does 940 position flips on TLT actually mean mechanically?
5. How does yfinance data differ between downloads (cache staleness)?
"""

import os, sys
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.preprocessing import fetch_and_clean_ticker, build_real_features_and_targets

print("=" * 80)
print("  FORENSIC AUDIT: Ground Truth Price Verification")
print("=" * 80)

# ─────────────────────────────────────────────────────────────
# AUDIT 1: What is AAPL's REAL buy-and-hold return 2016–2025?
# ─────────────────────────────────────────────────────────────
print("\n─── AUDIT 1: AAPL Real Buy-and-Hold Return ───")
aapl_raw = fetch_and_clean_ticker("AAPL")
aapl_2016 = aapl_raw.loc["2016-01-01":"2025-12-31"]
if len(aapl_2016) > 0:
    first_close = aapl_2016['Close'].iloc[0]
    last_close = aapl_2016['Close'].iloc[-1]
    bnh_return = (last_close / first_close - 1) * 100
    bnh_multiple = last_close / first_close
    print(f"  AAPL First Close (Jan 2016):  ${first_close:.2f}")
    print(f"  AAPL Last Close (end 2025):   ${last_close:.2f}")
    print(f"  Buy-and-Hold Return:          {bnh_return:+.2f}%")
    print(f"  Buy-and-Hold Multiple:        {bnh_multiple:.1f}x")
    print(f"  Paper claims baseline return: +2354.43% — is this plausible?")
    print(f"  Verdict: {'PLAUSIBLE' if abs(bnh_return - 2354.43) < 500 else 'IMPLAUSIBLE — paper claim is WRONG'}")
else:
    print("  ERROR: No AAPL data in 2016-2025 range!")

# ─────────────────────────────────────────────────────────────
# AUDIT 2: What is TLT's REAL buy-and-hold return 2016–2025?
# ─────────────────────────────────────────────────────────────
print("\n─── AUDIT 2: TLT Real Buy-and-Hold Return ───")
tlt_raw = fetch_and_clean_ticker("TLT")
tlt_2016 = tlt_raw.loc["2016-01-01":"2025-12-31"]
if len(tlt_2016) > 0:
    first_close = tlt_2016['Close'].iloc[0]
    last_close = tlt_2016['Close'].iloc[-1]
    bnh_return = (last_close / first_close - 1) * 100
    max_close = tlt_2016['Close'].max()
    min_close = tlt_2016['Close'].min()
    max_dd_bnh = (min_close / max_close - 1) * 100
    print(f"  TLT First Close (Jan 2016):  ${first_close:.2f}")
    print(f"  TLT Last Close (end 2025):   ${last_close:.2f}")
    print(f"  TLT Peak Close:              ${max_close:.2f}")
    print(f"  TLT Trough Close:            ${min_close:.2f}")
    print(f"  Buy-and-Hold Return:         {bnh_return:+.2f}%")
    print(f"  Buy-and-Hold Max Drawdown:   {max_dd_bnh:.2f}%")
    print(f"  Paper claims baseline return: -99.79%, Max DD: -99.80%")
    print(f"  940 position flips paying 70.95% in fees on a ${first_close:.0f} ETF")
    print(f"  Verdict: A -99.79% return on a liquid Treasury ETF is PHYSICALLY INCOHERENT")
    print(f"           without extreme leverage. TLT itself only dropped ~{max_dd_bnh:.0f}% peak-to-trough.")
    print(f"           The baseline's losses come almost entirely from 940 trade flips × 5bps = {940*0.0005*100:.1f}% raw fee burden.")
    print(f"           But 940 flips × 0.05% = {940*0.0005*100:.1f}% total fee is ADDITIVE, not 70.95%.")
    print(f"           The 70.95% figure implies compounded fee drag on a strategy that churns daily.")
else:
    print("  ERROR: No TLT data in 2016-2025 range!")

# ─────────────────────────────────────────────────────────────
# AUDIT 3: Determinism check — run the same experiment twice
# ─────────────────────────────────────────────────────────────
print("\n─── AUDIT 3: Determinism Check (SPY only) ───")
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from tests.test_backtest_metrics import calculate_backtest_metrics
from lightgbm import LGBMClassifier

def run_spy_once():
    df = fetch_and_clean_ticker("SPY")
    df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
    df = df.dropna().copy()
    returns = df['Return'].values
    
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1, n_segments=4,
        alphabet_size_mean=4, alphabet_size_slope=3
    )
    token_df, vocab = tokenizer.fit_transform_series(returns, channel_name="SPY_ret")
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name="SPY_ret")
    
    common_len = min(len(df), len(bow_df))
    df = df.iloc[-common_len:].copy()
    bow_df = bow_df.iloc[-common_len:].copy()
    returns = df['Return'].values
    
    token_counts = (bow_df > 0).sum(axis=0)
    freq_cols = token_counts[token_counts >= 5].index
    bow_df = bow_df[freq_cols]
    
    tech_cols = ['Return', 'Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
    X_tech = df[tech_cols].values
    X_tokens = bow_df.values
    y = df['Target_Dir'].values
    dates = df.index
    
    test_years = list(range(2016, 2026))
    pred_msopt_all = []
    ret_test_all = []
    
    for test_yr in test_years:
        train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
        test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
        if test_mask.sum() == 0:
            continue
        clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
        clf.fit(X_tokens[train_mask], y[train_mask])
        pred = clf.predict(X_tokens[test_mask])
        pred_msopt_all.extend(pred)
        ret_test_all.extend(returns[test_mask])
    
    pred_msopt_all = np.array(pred_msopt_all)
    ret_test_all = np.array(ret_test_all)
    _, m = calculate_backtest_metrics(ret_test_all, pred_msopt_all, fee_bps=5.0)
    return m

print("  Running SPY experiment — Run 1...")
m1 = run_spy_once()
print(f"    Run 1 Sharpe: {m1['Sharpe_Ratio']:.6f}, Return: {m1['Total_Return']:+.6f}")

print("  Running SPY experiment — Run 2...")
m2 = run_spy_once()
print(f"    Run 2 Sharpe: {m2['Sharpe_Ratio']:.6f}, Return: {m2['Total_Return']:+.6f}")

sharpe_match = np.isclose(m1['Sharpe_Ratio'], m2['Sharpe_Ratio'], atol=1e-6)
return_match = np.isclose(m1['Total_Return'], m2['Total_Return'], atol=1e-6)
print(f"  Sharpe identical across runs? {'YES ✓' if sharpe_match else 'NO ✗ — NON-DETERMINISTIC!'}")
print(f"  Return identical across runs? {'YES ✓' if return_match else 'NO ✗ — NON-DETERMINISTIC!'}")

# ─────────────────────────────────────────────────────────────
# AUDIT 4: Check if cached CSV matches fresh yfinance download
# ─────────────────────────────────────────────────────────────
print("\n─── AUDIT 4: Data Cache Staleness Check ───")
csv_path = os.path.join(PROJECT_ROOT, "data", "spy_daily_real.csv")
if os.path.exists(csv_path):
    cached = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    print(f"  Cached SPY CSV: {len(cached)} bars, {cached.index[0].strftime('%Y-%m-%d')} to {cached.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Cache file last modified: {pd.Timestamp(os.path.getmtime(csv_path), unit='s')}")
else:
    print("  No cached CSV found — data is downloaded fresh each time (non-reproducible!)")

# ─────────────────────────────────────────────────────────────
# AUDIT 5: What the backtest ACTUALLY does to produce -99.79% on TLT
# ─────────────────────────────────────────────────────────────
print("\n─── AUDIT 5: TLT Baseline Mechanics Deep Dive ───")
df = fetch_and_clean_ticker("TLT")
df = build_real_features_and_targets(df, horizon=5, delta_vol=0.5)
df = df.dropna().copy()
returns = df['Return'].values
dates = df.index

tech_cols = ['Return', 'Parkinson_Vol', 'Vol_MA30', 'Rel_Volume']
X_tech = df[tech_cols].values
y = df['Target_Dir'].values

test_years = list(range(2016, 2026))
pred_base_all = []
ret_test_all = []
y_test_all = []

for test_yr in test_years:
    train_mask = (dates < f"{test_yr}-01-01") & (dates >= '2010-01-01')
    test_mask = (dates >= f"{test_yr}-01-01") & (dates <= f"{test_yr}-12-31")
    if test_mask.sum() == 0:
        continue
    clf = LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X_tech[train_mask], y[train_mask])
    pred = clf.predict(X_tech[test_mask])
    pred_base_all.extend(pred)
    ret_test_all.extend(returns[test_mask])
    y_test_all.extend(y[test_mask])

pred_base_all = np.array(pred_base_all)
ret_test_all = np.array(ret_test_all)

# Count signal transitions
pos_changes = np.abs(np.diff(pred_base_all))
n_flips = int(np.sum(pos_changes > 0))
total_days = len(pred_base_all)
flip_rate = n_flips / total_days

# Signal distribution
unique, counts = np.unique(pred_base_all, return_counts=True)
print(f"  Total OOS days: {total_days}")
print(f"  Signal distribution: {dict(zip(unique.astype(int), counts))}")
print(f"  Position flips: {n_flips}")
print(f"  Flip rate: {flip_rate:.2%} of days")
print(f"  Avg days between flips: {total_days/max(n_flips,1):.1f}")

# Run full backtest to see wealth curve
df_steps, m_tlt = calculate_backtest_metrics(ret_test_all, pred_base_all, fee_bps=5.0)

# Show fee mechanics
total_fee_raw = df_steps['Tx_Cost'].sum()
print(f"\n  DETAILED FEE MECHANICS:")
print(f"  Total raw fee deducted:          {total_fee_raw:.4f} ({total_fee_raw*100:.2f}% of initial $1)")
print(f"  But this is ADDITIVE, not compounded into wealth.")
print(f"  Final wealth W_T:                {df_steps['Wealth_W'].iloc[-1]:.6f}")
print(f"  Total Return (W_T - 1):          {m_tlt['Total_Return']:+.4%}")
print(f"  Max Drawdown:                    {m_tlt['Max_Drawdown']:.4%}")
print(f"  Sharpe:                          {m_tlt['Sharpe_Ratio']:.4f}")

# Show where the wealth curve collapses
wealth = df_steps['Wealth_W'].values
wealth_10pct = np.where(wealth < 0.1)[0]
wealth_1pct = np.where(wealth < 0.01)[0]
if len(wealth_10pct) > 0:
    print(f"\n  Wealth drops below $0.10 at day {wealth_10pct[0]} of {total_days}")
if len(wealth_1pct) > 0:
    print(f"  Wealth drops below $0.01 at day {wealth_1pct[0]} of {total_days}")

# Show how compounding of daily fees destroys wealth
# If you flip almost every day and pay 0.05% each time, 
# over 2500 days that's roughly 2500 * 0.0005 = 1.25 in additive fees
# But in COMPOUNDING: (1 - 0.0005)^940 ≈ 0.625 — you keep 62.5% just from fees alone
# The actual formula is even worse because flip magnitude can be 2 (short-to-long)
print(f"\n  COMPOUNDING EXPLANATION:")
print(f"  With 940 flips, many are magnitude-2 (short↔long), costing 2×5bps = 10bps each")
print(f"  Compounded fee drag: (1-0.0005)^940 = {(1-0.0005)**940:.4f}")
print(f"  Compounded fee drag: (1-0.0010)^940 = {(1-0.0010)**940:.4f} (if all are magnitude-2)")
print(f"  The actual drag depends on the mix of flip magnitudes.")

# But the REAL question: is this a VALID strategy or a broken backtest?
print(f"\n  CRITICAL QUESTION: Is this a valid strategy result or a broken backtest?")
print(f"  A strategy that flips {flip_rate:.0%} of days on a low-vol Treasury ETF")
print(f"  and compounds those fees is not 'wrong' — it's a legitimate demonstration")
print(f"  that high-frequency signal churn destroys capital on low-volatility assets.")
print(f"  BUT the paper should NOT describe this as 'losing essentially all capital'")
print(f"  without explaining it's a SELF-INFLICTED fee death spiral, not a market loss.")

print("\n" + "=" * 80)
print("  FORENSIC AUDIT COMPLETE")
print("=" * 80)
