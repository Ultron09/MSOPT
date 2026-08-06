"""
Week 2 Experiment: Predictive Power of BORF Multi-Scale Tokens vs. Baseline
===========================================================================
Strict Walk-Forward Evaluation on SPY, AAPL, QQQ daily returns (2010–2026).

Compares:
1. Baseline: Fixed-window raw returns + rolling volatility + momentum
2. BORF Tokenizer: Multi-scale 1D-SAX Bag-of-Words rolling token histograms
3. Combination: Baseline + BORF Tokens

Models evaluated: LightGBM Classifier & Ridge Classifier
Validation: Walk-Forward expanding window with 5-year initial train, 1-year step.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.tokenizer.borf_tokenizer import FinancialBORFTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "week2_borf_predictive")
os.makedirs(RESULTS_DIR, exist_ok=True)

TICKERS = ["SPY", "AAPL", "QQQ"]

def load_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df['Return'] = np.log(df['Close'] / df['Close'].shift(1))
    df['Target_NextDay'] = (df['Return'].shift(-1) > 0).astype(int)
    df['Target_5Day'] = (df['Close'].shift(-5) / df['Close'] - 1 > 0).astype(int)
    return df.dropna()

def build_baseline_features(df: pd.DataFrame, lags: int = 20) -> pd.DataFrame:
    """Build standard fixed-window financial baseline features."""
    feats = pd.DataFrame(index=df.index)
    for i in range(1, lags + 1):
        feats[f'lag_{i}'] = df['Return'].shift(i)
    
    feats['vol_10'] = df['Return'].rolling(10).std()
    feats['vol_30'] = df['Return'].rolling(30).std()
    feats['mom_5'] = df['Return'].rolling(5).sum()
    feats['mom_20'] = df['Return'].rolling(20).sum()
    return feats

def run_experiment_for_ticker(ticker: str):
    print(f"\n{'='*60}\n  Evaluating Ticker: {ticker}\n{'='*60}")
    df = load_data(ticker)
    
    # 1. Baseline Features
    baseline_feats = build_baseline_features(df, lags=20)
    
    # 2. BORF Tokens
    print("  [BORF] Extracting multi-scale pattern tokens...")
    returns = df['Return'].values
    tokenizer = FinancialBORFTokenizer(
        window_sizes=[5, 10, 20, 40],
        dilations=[1, 2],
        stride=1,
        n_segments=4,
        alphabet_size_mean=4,
        alphabet_size_slope=3,
        std_threshold=0.001
    )
    
    token_df, vocab = tokenizer.fit_transform_series(returns)
    print(f"    → Discretized into {len(vocab)} unique pattern words.")
    
    # Rolling 30-day bag of words histogram
    bow_df = tokenizer.get_bag_of_words_histogram(returns, rolling_window=30)
    bow_df.index = df.index[:len(bow_df)]
    bow_df.columns = [f"token_{col}" for col in bow_df.columns]
    
    # Combine datasets
    combined_df = pd.concat([df[['Target_NextDay']], baseline_feats, bow_df], axis=1).dropna()
    
    feature_cols_base = baseline_feats.columns.tolist()
    feature_cols_borf = bow_df.columns.tolist()
    feature_cols_comb = feature_cols_base + feature_cols_borf
    
    target_col = 'Target_NextDay'
    
    # Walk-Forward Validation (Expanding Window)
    # Start train at 2010-2015, test on 2016, expand by 1 year until 2026
    test_years = range(2016, 2026)
    
    results = []
    
    for test_year in test_years:
        train_mask = (combined_df.index.year < test_year)
        test_mask = (combined_df.index.year == test_year)
        
        if not any(test_mask):
            continue
            
        train_data = combined_df[train_mask]
        test_data = combined_df[test_mask]
        
        y_train = train_data[target_col]
        y_test = test_data[target_col]
        
        # Models
        model_names = ['Baseline_LGBM', 'BORF_LGBM', 'Combined_LGBM', 'BORF_Ridge']
        feat_sets = [feature_cols_base, feature_cols_borf, feature_cols_comb, feature_cols_borf]
        
        year_res = {'Year': test_year, 'N_Test': len(y_test)}
        
        for name, feats in zip(model_names, feat_sets):
            X_tr, X_te = train_data[feats], test_data[feats]
            
            if 'LGBM' in name:
                clf = lgb.LGBMClassifier(
                    n_estimators=50,
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
            
            year_res[f'{name}_Acc'] = acc
            
        results.append(year_res)
        
    res_df = pd.DataFrame(results)
    
    print(f"\n--- Walk-Forward Out-of-Sample Accuracy Results ({ticker}) ---")
    print(res_df.to_string(index=False))
    
    avg_accs = {col: res_df[col].mean() for col in res_df.columns if col.endswith('_Acc')}
    print(f"\n--- Overall Mean Accuracy across 2016–2025 ---")
    for k, v in avg_accs.items():
        print(f"  {k:20s}: {v*100:.2f}%")
        
    res_df.to_csv(os.path.join(RESULTS_DIR, f"{ticker.lower()}_walk_forward_results.csv"), index=False)
    return res_df, avg_accs

def main():
    all_summary = {}
    for ticker in TICKERS:
        _, avg_accs = run_experiment_for_ticker(ticker)
        all_summary[ticker] = avg_accs
        
    summary_df = pd.DataFrame(all_summary).T
    print(f"\n{'='*60}\n  FINAL CROSS-ASSET PREDICTIVE SUMMARY\n{'='*60}")
    print((summary_df * 100).round(2).to_string())
    
    summary_path = os.path.join(RESULTS_DIR, "summary_predictive_power.csv")
    summary_df.to_csv(summary_path)
    print(f"\nSaved summary to {summary_path}")

if __name__ == "__main__":
    main()
