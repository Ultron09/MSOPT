"""
Pillar 3: Overlapping Stride Invariance Ablation
================================================
Proves why dense overlapping stride (s=1) outperforms non-overlapping partitions (s=w).
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import prepare_benchmark_dataset
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from experiments.benchmark_pipeline import calculate_strategy_sharpe

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def run_stride_ablation(ticker: str = "SPY") -> pd.DataFrame:
    print(f"\n{'='*70}\n  PILLAR 3: OVERLAPPING STRIDE INVARIANCE ABLATION ({ticker})\n{'='*70}")
    df = prepare_benchmark_dataset(ticker)
    returns = df['Return'].values
    
    stride_configs = {
        'Dense_Overlapping_s1': 1,
        'Partial_Overlapping_s2': 2,
        'Non_Overlapping_s4': 4
    }
    
    ablation_results = []
    
    for name, s_val in stride_configs.items():
        tokenizer = MSOPTTokenizer(window_sizes=[4, 8, 16], dilations=[1, 2], stride=s_val)
        bow_df = tokenizer.get_rolling_bow_histogram(returns, rolling_window=30, channel_name=ticker)
        bow_df.index = df.index[:len(bow_df)]
        
        combined = pd.concat([df[['Target_Dir', 'Return']], bow_df], axis=1).dropna()
        token_cols = [c for c in combined.columns if c not in ['Target_Dir', 'Return']]
        freq_cols = [c for c in token_cols if combined[c].sum() >= 5]
        if not freq_cols:
            freq_cols = token_cols
            
        test_years = range(2016, 2026)
        all_preds, all_trues, all_rets = [], [], []
        
        for test_year in test_years:
            train_mask = (combined.index.year < test_year)
            test_mask = (combined.index.year == test_year)
            if not any(test_mask) or not any(train_mask):
                continue
                
            tr, te = combined[train_mask], combined[test_mask]
            clf = lgb.LGBMClassifier(n_estimators=50, learning_rate=0.03, random_state=42, verbose=-1)
            clf.fit(tr[freq_cols], tr['Target_Dir'])
            preds = clf.predict(te[freq_cols])
            
            all_preds.extend(preds)
            all_trues.extend(te['Target_Dir'].values)
            all_rets.extend(te['Return'].values)
            
        acc = accuracy_score(all_trues, all_preds)
        sharpe, sortino, max_dd = calculate_strategy_sharpe(np.array(all_rets), np.array(all_preds), fee_bps=0.0005)
        
        ablation_results.append({
            'Ticker': ticker,
            'Stride_Configuration': name,
            'Stride_s': s_val,
            'OOS_Accuracy': acc,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd
        })
        
    res_df = pd.DataFrame(ablation_results)
    print(f"\n--- Stride Invariance Ablation Results ({ticker}) ---")
    print(res_df.round(4).to_string(index=False))
    res_df.to_csv(os.path.join(RESULTS_DIR, f"ablation_stride_{ticker.lower()}.csv"), index=False)
    return res_df

if __name__ == "__main__":
    for ticker in ["SPY", "QQQ"]:
        run_stride_ablation(ticker)
