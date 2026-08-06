"""
Master Benchmark Pipeline: Walk-Forward Evaluation of MSOPT Tokens vs Baseline
=============================================================================
Evaluates MSOPT Multi-Scale Overlapping Tokens on SPY, AAPL, QQQ, TLT (2010–2026).

Enforces:
1. Expanding Walk-Forward Validation (2016–2026, zero lookahead)
2. High-SNR Task Targets (Fork B: Volatility-scaled Directional Threshold Moves)
3. Transaction Cost Awareness (5 bps slippage & execution fees per trade)
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add src to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import prepare_benchmark_dataset, TICKERS
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "msopt_benchmarks")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_baseline_features(df: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    """Build standard fixed-window baseline features."""
    feats = pd.DataFrame(index=df.index)
    for i in range(1, lags + 1):
        feats[f'lag_{i}'] = df['Return'].shift(i)
    
    feats['vol_10'] = df['Parkinson_Vol'].rolling(10).mean()
    feats['vol_30'] = df['Parkinson_Vol'].rolling(30).mean()
    feats['rel_vol'] = df['Rel_Volume']
    feats['mom_5'] = df['Return'].rolling(5).sum()
    feats['mom_20'] = df['Return'].rolling(20).sum()
    return feats

def calculate_strategy_sharpe(y_true_ret: np.ndarray, y_pred_signal: np.ndarray, fee_bps: float = 0.0005) -> Tuple[float, float, float]:
    """
    Calculate Out-of-Sample Sharpe Ratio, Sortino Ratio, and Max Drawdown 
    incorporating 5 bps transaction costs per position flip.
    """
    # Signal: +1 (Long), -1 (Short), 0 (Cash)
    positions = y_pred_signal.copy()
    
    # Calculate position flips to apply transaction fees
    trades = np.abs(np.diff(positions, prepend=0))
    costs = trades * fee_bps
    
    # Strategy daily log returns
    strat_returns = positions * y_true_ret - costs
    
    # Annualized Sharpe (assuming 252 trading days)
    mean_ret = np.mean(strat_returns) * 252
    std_ret = np.std(strat_returns) * np.sqrt(252) + 1e-8
    sharpe = mean_ret / std_ret
    
    # Annualized Sortino (downside volatility only)
    downside_returns = strat_returns[strat_returns < 0]
    downside_std = np.std(downside_returns) * np.sqrt(252) + 1e-8 if len(downside_returns) > 0 else 1e-8
    sortino = mean_ret / downside_std
    
    # Max Drawdown
    cum_returns = np.exp(np.cumsum(strat_returns))
    running_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - running_max) / running_max
    max_dd = np.min(drawdowns)
    
    return float(sharpe), float(sortino), float(max_dd)

def run_walk_forward_benchmark(ticker: str):
    print(f"\n{'='*70}\n  RUNNING WALK-FORWARD BENCHMARK FOR TICKER: {ticker}\n{'='*70}")
    df = prepare_benchmark_dataset(ticker)
    
    # 1. Baseline Features
    baseline_feats = build_baseline_features(df, lags=20)
    
    # 2. MSOPT Pattern Tokens
    print("  [MSOPT Tokenizer] Extracting multi-scale 1D-SAX pattern tokens...")
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
    
    # Rolling 30-day bag of words histogram
    bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=ticker)
    bow_df.index = df.index[:len(bow_df)]
    bow_df.columns = [f"token_{col}" for col in bow_df.columns]
    
    # Combine datasets
    combined_df = pd.concat([df[['Target_Dir', 'Return']], baseline_feats, bow_df], axis=1).dropna()
    
    # Prune rare tokens appearing < 5 times to prevent overfitting
    token_cols = [c for c in combined_df.columns if c.startswith('token_')]
    frequent_token_cols = [c for c in token_cols if combined_df[c].sum() >= 5]
    
    base_cols = baseline_feats.columns.tolist()
    comb_cols = base_cols + frequent_token_cols
    
    print(f"  → Retained {len(frequent_token_cols)} frequent pattern tokens.")
    
    # Targets & Features
    target_col = 'Target_Dir'
    
    # Walk-Forward Validation (Expanding Window: 2016 to 2026)
    test_years = range(2016, 2026)
    year_results = []
    
    all_preds = {'Baseline_LGBM': [], 'MSOPT_LGBM': [], 'Combined_LGBM': [], 'MSOPT_Ridge': []}
    all_trues = []
    all_rets = []
    
    for test_year in test_years:
        train_mask = (combined_df.index.year < test_year)
        test_mask = (combined_df.index.year == test_year)
        
        if not any(test_mask):
            continue
            
        train_data = combined_df[train_mask]
        test_data = combined_df[test_mask]
        
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        y_test_ret = test_data['Return'].values
        
        model_configs = [
            ('Baseline_LGBM', base_cols, 'lgbm'),
            ('MSOPT_LGBM', frequent_token_cols, 'lgbm'),
            ('Combined_LGBM', comb_cols, 'lgbm'),
            ('MSOPT_Ridge', frequent_token_cols, 'ridge')
        ]
        
        row_res = {'Year': test_year, 'N_Test': len(y_test)}
        
        for name, feats, mtype in model_configs:
            X_tr, X_te = train_data[feats], test_data[feats]
            
            if mtype == 'lgbm':
                clf = lgb.LGBMClassifier(
                    n_estimators=60,
                    learning_rate=0.03,
                    num_leaves=15,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    verbose=-1
                )
            else:
                clf = RidgeClassifier(alpha=1.0)
                
            clf.fit(X_tr, y_train)
            preds = clf.predict(X_te)
            
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='macro')
            sharpe, sortino, max_dd = calculate_strategy_sharpe(y_test_ret, preds, fee_bps=0.0005)
            
            row_res[f'{name}_Acc'] = acc
            row_res[f'{name}_Sharpe'] = sharpe
            
            all_preds[name].extend(preds)
            
        all_trues.extend(y_test.values)
        all_rets.extend(y_test_ret)
        year_results.append(row_res)
        
    res_df = pd.DataFrame(year_results)
    
    print(f"\n--- Walk-Forward Out-of-Sample Performance ({ticker}) ---")
    print(res_df.to_string(index=False))
    
    # Calculate Overall Out-of-Sample Strategy Metrics across entire 2016-2025 period
    overall_metrics = {}
    all_trues_arr = np.array(all_trues)
    all_rets_arr = np.array(all_rets)
    
    for name in all_preds:
        preds_arr = np.array(all_preds[name])
        acc = accuracy_score(all_trues_arr, preds_arr)
        f1 = f1_score(all_trues_arr, preds_arr, average='macro')
        sharpe, sortino, max_dd = calculate_strategy_sharpe(all_rets_arr, preds_arr, fee_bps=0.0005)
        
        overall_metrics[name] = {
            'OOS_Accuracy': acc,
            'Macro_F1': f1,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd
        }
        
    overall_df = pd.DataFrame(overall_metrics).T
    print(f"\n--- Overall 10-Year OOS Strategy Summary ({ticker}) Post 5 bps Costs ---")
    print(overall_df.round(4).to_string())
    
    res_df.to_csv(os.path.join(RESULTS_DIR, f"{ticker.lower()}_yearly_results.csv"), index=False)
    overall_df.to_csv(os.path.join(RESULTS_DIR, f"{ticker.lower()}_overall_summary.csv"))
    return overall_df

def main():
    all_summaries = {}
    for ticker in TICKERS:
        summ = run_walk_forward_benchmark(ticker)
        all_summaries[ticker] = summ
        
    print(f"\n{'='*70}\n  FINAL CROSS-ASSET MSOPT BENCHMARK SUMMARY (POST 5 BPS COSTS)\n{'='*70}")
    for ticker, summ in all_summaries.items():
        print(f"\nAsset: {ticker}")
        print(summ.round(4).to_string())

if __name__ == "__main__":
    main()
