"""
Pillar 1: PyTorch MSOPT Deep Conv-Transformer Walk-Forward Benchmark
===================================================================
Evaluates the full PyTorch MSOPTDeepNeuralClassifier on 2D Spatial Token Grids
across 10 walk-forward years (2016–2026) post 5 bps transaction costs.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import prepare_benchmark_dataset, TICKERS
from src.tokenizer.msopt_tokenizer import MSOPTTokenizer
from src.models.msopt_engine import MSOPTDeepNeuralClassifier
from experiments.benchmark_pipeline import calculate_strategy_sharpe

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_eval_pytorch_msopt(
    ticker: str,
    seq_len: int = 60,
    epochs: int = 12,
    batch_size: int = 32
) -> pd.DataFrame:
    print(f"\n{'='*70}\n  PILLAR 1: PYTORCH DEEP MSOPT CONV-TRANSFORMER BENCHMARK ({ticker})\n{'='*70}")
    df = prepare_benchmark_dataset(ticker)
    returns = df['Return'].values
    
    tokenizer = MSOPTTokenizer(
        window_sizes=[4, 8, 16, 32],
        dilations=[1, 2, 4],
        stride=1
    )
    
    print("  [MSOPT Tokenizer] Constructing 2D Spatial Grid Matrices...")
    grid, vocab = tokenizer.get_2d_spatial_grid_indices(returns, channel_name=ticker)
    vocab_size = len(vocab) + 1
    n_scales = grid.shape[0]
    
    # Target values: map {-1, 0, 1} -> {0, 1, 2}
    y_raw = df['Target_Dir'].values[:grid.shape[1]]
    y_mapped = y_raw + 1
    
    # Create rolling 2D spatial grid sequence samples of shape [N, N_scales, seq_len]
    X_samples, y_samples, rets_samples, dates_samples = [], [], [], []
    dates = df.index[:grid.shape[1]]
    
    for t in range(seq_len, grid.shape[1]):
        X_samples.append(grid[:, t - seq_len : t])
        y_samples.append(y_mapped[t])
        rets_samples.append(returns[t])
        dates_samples.append(dates[t])
        
    X_arr = np.array(X_samples)
    y_arr = np.array(y_samples)
    rets_arr = np.array(rets_samples)
    dates_arr = pd.DatetimeIndex(dates_samples)
    
    # Walk-Forward Validation (2016 to 2025)
    test_years = range(2016, 2026)
    year_results = []
    
    all_preds, all_trues, all_rets = [], [], []
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using Compute Device: {device}")
    
    for test_year in test_years:
        train_mask = (dates_arr.year < test_year)
        test_mask = (dates_arr.year == test_year)
        
        if not any(test_mask) or not any(train_mask):
            continue
            
        X_tr, y_tr = torch.tensor(X_arr[train_mask], dtype=torch.long), torch.tensor(y_arr[train_mask], dtype=torch.long)
        X_te, y_te = torch.tensor(X_arr[test_mask], dtype=torch.long), torch.tensor(y_arr[test_mask], dtype=torch.long)
        
        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=batch_size, shuffle=False)
        
        model = MSOPTDeepNeuralClassifier(
            vocab_size=vocab_size,
            n_scales=n_scales,
            num_classes=3,
            embed_dim=32,
            n_heads=2,
            n_layers=2
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.train()
        for epoch in range(epochs):
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits = model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
                
        # Inference
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                fold_preds.extend(preds)
                
        fold_preds_arr = np.array(fold_preds) - 1 # Map {0,1,2} back to {-1,0,1}
        fold_trues_arr = y_arr[test_mask] - 1
        fold_rets_arr = rets_arr[test_mask]
        
        acc = float(np.mean(fold_preds_arr == fold_trues_arr))
        sharpe, sortino, max_dd = calculate_strategy_sharpe(fold_rets_arr, fold_preds_arr, fee_bps=0.0005)
        
        year_results.append({
            'Year': test_year,
            'OOS_Accuracy': acc,
            'Sharpe_Ratio': sharpe,
            'Sortino_Ratio': sortino,
            'Max_Drawdown': max_dd
        })
        
        all_preds.extend(fold_preds_arr)
        all_trues.extend(fold_trues_arr)
        all_rets.extend(fold_rets_arr)
        
    overall_sharpe, overall_sortino, overall_max_dd = calculate_strategy_sharpe(
        np.array(all_rets), np.array(all_preds), fee_bps=0.0005
    )
    overall_acc = float(np.mean(np.array(all_preds) == np.array(all_trues)))
    
    res_df = pd.DataFrame(year_results)
    print(f"\n--- Overall 10-Year PyTorch Conv-Transformer Results ({ticker}) ---")
    print(f"  OOS Accuracy:  {overall_acc*100:.2f}%")
    print(f"  Sharpe Ratio:  {overall_sharpe:.4f}")
    print(f"  Sortino Ratio: {overall_sortino:.4f}")
    print(f"  Max Drawdown:  {overall_max_dd*100:.2f}%")
    
    summary_df = pd.DataFrame([{
        'Ticker': ticker,
        'Model': 'PyTorch_MSOPT_ConvTransformer',
        'OOS_Accuracy': overall_acc,
        'Sharpe_Ratio': overall_sharpe,
        'Sortino_Ratio': overall_sortino,
        'Max_Drawdown': overall_max_dd
    }])
    summary_df.to_csv(os.path.join(RESULTS_DIR, f"pytorch_msopt_{ticker.lower()}.csv"), index=False)
    return summary_df

def main():
    summaries = []
    for ticker in ["SPY", "QQQ"]:
        res = train_eval_pytorch_msopt(ticker)
        summaries.append(res)
    full_df = pd.concat(summaries, ignore_index=True)
    full_df.to_csv(os.path.join(RESULTS_DIR, "pytorch_msopt_all.csv"), index=False)

if __name__ == "__main__":
    main()
