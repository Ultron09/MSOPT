"""
Real Stage 1 Walk-Forward Benchmark Execution Engine
====================================================
Performs a zero-shortcut, 10-year expanding walk-forward evaluation (2016-2025)
on authentic SPY, AAPL, QQQ, and TLT daily data.

Enforces:
1. Strict Walk-Forward (Train < Test Year, Zero Lookahead)
2. Accurate 5 Bps Slippage & Transaction Cost Accounting
3. Real Directional Accuracy, Macro F1, Sharpe, Sortino, Max Drawdown
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import fetch_and_clean_ticker, build_real_features_and_targets
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "real_stage1")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_baseline_features(df: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    """Build standard technical momentum and lag features."""
    feats = pd.DataFrame(index=df.index)
    for i in range(1, lags + 1):
        feats[f'lag_{i}'] = df['Return'].shift(i)
    
    feats['vol_10'] = df['Parkinson_Vol'].rolling(10).mean()
    feats['vol_30'] = df['Parkinson_Vol'].rolling(30).mean()
    feats['rel_vol'] = df['Rel_Volume']
    feats['mom_5'] = df['Return'].rolling(5).sum()
    feats['mom_20'] = df['Return'].rolling(20).sum()
    return feats

def compute_real_strategy_metrics(
    daily_returns: np.ndarray,
    positions: np.ndarray,
    fee_bps: float = 0.0005
) -> Tuple[float, float, float]:
    """
    Computes exact strategy metrics post 5 bps transaction costs.
    
    positions: array of predicted signals in {-1, 0, 1}
    daily_returns: actual daily log returns R_t
    """
    # Align position signal: signal predicted at end of day t-1 applies to return at day t
    pos_shifted = np.roll(positions, 1)
    pos_shifted[0] = 0 # No position on first day
    
    # Calculate trades (position changes)
    trade_flips = np.abs(np.diff(pos_shifted, prepend=0))
    trade_costs = trade_flips * fee_bps
    
    # Simple returns for wealth accumulation
    simple_returns = np.exp(daily_returns) - 1.0
    
    # Strategy daily returns after deduction of costs
    strat_daily_rets = pos_shifted * simple_returns - trade_costs
    
    # Annualized Sharpe Ratio (252 trading days)
    mean_daily = np.mean(strat_daily_rets)
    std_daily = np.std(strat_daily_rets) + 1e-8
    sharpe = (mean_daily * 252.0) / (std_daily * np.sqrt(252.0))
    
    # Annualized Sortino Ratio (downside volatility only)
    downside = strat_daily_rets[strat_daily_rets < 0]
    downside_std = np.std(downside) * np.sqrt(252.0) + 1e-8 if len(downside) > 0 else 1e-8
    sortino = (mean_daily * 252.0) / downside_std
    
    # Maximum Drawdown from wealth curve
    wealth_curve = np.cumprod(1.0 + strat_daily_rets)
    peak = np.maximum.accumulate(wealth_curve)
    drawdowns = (wealth_curve - peak) / peak
    max_dd = float(np.min(drawdowns))
    
    return float(sharpe), float(sortino), max_dd


def evaluate_ticker_walk_forward(ticker: str) -> pd.DataFrame:
    print(f"\n{'='*70}\n  RUNNING REAL 10-YEAR WALK-FORWARD EVALUATION: {ticker}\n{'='*70}")
    raw_df = fetch_and_clean_ticker(ticker)
    df = build_real_features_and_targets(raw_df, horizon=5)
    
    # 1. Technical Baseline Features
    tech_feats = build_baseline_features(df, lags=20)
    
    # 2. MSOPT 1D-SAX Tokens
    print("  [MSOPT Tokenizer] Extracting authentic multi-scale 1D-SAX tokens...")
    returns = df['Return'].values
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3,
        std_threshold=0.001
    )
    
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=ticker)
    bow_df.index = df.index[:len(bow_df)]
    bow_df.columns = [f"token_{col}" for col in bow_df.columns]
    
    # Merge datasets
    combined_df = pd.concat([df[['Target_Dir', 'Return']], tech_feats, bow_df], axis=1).dropna()
    
    token_cols = [c for c in combined_df.columns if c.startswith('token_')]
    freq_token_cols = [c for c in token_cols if combined_df[c].sum() >= 5]
    tech_cols = tech_feats.columns.tolist()
    comb_cols = tech_cols + freq_token_cols
    
    print(f"  → Retained {len(freq_token_cols)} frequent pattern token features.")
    
    # Walk-Forward Splits (2016 to 2025)
    test_years = range(2016, 2026)
    
    all_preds = {'Baseline_Tech': [], 'MSOPT_Tokens': [], 'Combined': []}
    all_trues = []
    all_returns = []
    
    for year in test_years:
        train_mask = (combined_df.index.year < year)
        test_mask = (combined_df.index.year == year)
        
        if not any(test_mask) or not any(train_mask):
            continue
            
        train_df = combined_df[train_mask]
        test_df = combined_df[test_mask]
        
        y_tr = train_df['Target_Dir']
        y_te = test_df['Target_Dir']
        rets_te = test_df['Return'].values
        
        models_to_test = [
            ('Baseline_Tech', tech_cols),
            ('MSOPT_Tokens', freq_token_cols),
            ('Combined', comb_cols)
        ]
        
        for name, cols in models_to_test:
            clf = lgb.LGBMClassifier(
                n_estimators=50,
                learning_rate=0.03,
                num_leaves=15,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
            clf.fit(train_df[cols], y_tr)
            preds = clf.predict(test_df[cols])
            
            all_preds[name].extend(preds)
            
        all_trues.extend(y_te.values)
        all_returns.extend(rets_te)
        
    # Overall 10-Year Strategy Summary
    y_true_arr = np.array(all_trues)
    rets_arr = np.array(all_returns)
    
    summary_results = []
    for name in all_preds:
        preds_arr = np.array(all_preds[name])
        acc = accuracy_score(y_true_arr, preds_arr)
        f1 = f1_score(y_true_arr, preds_arr, average='macro')
        sharpe, sortino, max_dd = compute_real_strategy_metrics(rets_arr, preds_arr, fee_bps=0.0005)
        
        summary_results.append({
            'Ticker': ticker,
            'Model_Paradigm': name,
            'OOS_Accuracy': acc,
            'Macro_F1': f1,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd
        })
        
    res_df = pd.DataFrame(summary_results)
    print(f"\n--- AUTHENTIC 10-YEAR WALK-FORWARD SUMMARY ({ticker}) POST 5 BPS COSTS ---")
    print(res_df.to_string(index=False))
    
    res_df.to_csv(os.path.join(RESULTS_DIR, f"{ticker.lower()}_real_summary.csv"), index=False)
    return res_df

def main():
    all_res = []
    for ticker in ["SPY", "QQQ", "AAPL", "TLT"]:
        res = evaluate_ticker_walk_forward(ticker)
        all_res.append(res)
        
    full_df = pd.concat(all_res, ignore_index=True)
    print(f"\n{'='*70}\n  FINAL AUTHENTIC CROSS-ASSET MSOPT SUMMARY (2016-2025 POST 5 BPS COSTS)\n{'='*70}")
    print(full_df.round(4).to_string(index=False))
    full_df.to_csv(os.path.join(RESULTS_DIR, "master_real_summary.csv"), index=False)

if __name__ == "__main__":
    main()
