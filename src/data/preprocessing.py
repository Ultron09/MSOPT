"""
Real Financial Data Preprocessor & Verification Engine
======================================================
Downloads authentic OHLCV data via yfinance for SPY, AAPL, QQQ, TLT.
Computes real log returns, Parkinson volatility, and Fork B directional targets.
Enforces zero synthetic data, zero fallbacks, and zero data leakage.
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["SPY", "QQQ", "AAPL", "TLT"]

def fetch_and_clean_ticker(ticker: str, start: str = "2010-01-01", end: str = "2026-01-01") -> pd.DataFrame:
    """Fetch raw daily OHLCV from Yahoo Finance and clean column names."""
    csv_path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily_real.csv")
    
    if os.path.exists(csv_path):
        print(f"[Data] Loading cached authentic CSV for {ticker} from {csv_path}")
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"[Data] Downloading authentic daily OHLCV for {ticker} ({start} to {end})...")
        df = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(csv_path)

    # Clean & validate columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column {col} in {ticker} download!")

    df = df[required_cols].astype(float).dropna()
    df = df[df['Volume'] > 0] # Remove zero volume days
    
    print(f"[Data] {ticker}: Loaded {len(df)} authentic daily bars ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
    return df


def build_real_features_and_targets(df: pd.DataFrame, horizon: int = 5, delta_vol: float = 0.5) -> pd.DataFrame:
    """
    Build real log returns, Parkinson volatility, relative volume, and Fork B targets.
    
    Target 1: y_dir in {-1, 0, 1}
      +1 if H-day forward return > +0.5 * sigma_parkinson * sqrt(H)
      -1 if H-day forward return < -0.5 * sigma_parkinson * sqrt(H)
       0 otherwise
    """
    out = df.copy()

    # 1. Daily Log Returns (r_t = ln(P_t / P_{t-1}))
    out['Return'] = np.log(out['Close'] / out['Close'].shift(1))

    # 2. Parkinson Volatility estimator: sigma_P = sqrt( (ln(H/L))^2 / (4 * ln(2)) )
    log_hl = np.log(out['High'] / out['Low'])
    out['Parkinson_Vol'] = np.sqrt((log_hl ** 2) / (4 * np.log(2)))
    out['Vol_MA30'] = out['Parkinson_Vol'].rolling(30).mean()

    # 3. Relative Volume (Volume_t / MA20(Volume))
    out['Rel_Volume'] = out['Volume'] / (out['Volume'].rolling(20).mean() + 1e-8)

    # 4. Forward Return over H days: ln(Close_{t+H} / Close_t)
    out['Forward_Return_H'] = np.log(out['Close'].shift(-horizon) / out['Close'])

    # 5. Volatility-scaled Threshold: delta * Parkinson_Vol_t * sqrt(H)
    threshold = delta_vol * out['Parkinson_Vol'] * np.sqrt(horizon)

    # Directional Target (No lookahead in features, target is strictly future horizon)
    dir_target = np.zeros(len(out), dtype=int)
    dir_target[out['Forward_Return_H'] > threshold] = 1
    dir_target[out['Forward_Return_H'] < -threshold] = -1
    out['Target_Dir'] = dir_target

    # Volatility Expansion Target: 1 if future max vol over H days exceeds 1.5x rolling 30-day mean
    future_max_vol = out['Parkinson_Vol'].shift(-horizon).rolling(horizon).max()
    out['Target_Vol'] = (future_max_vol > 1.5 * out['Vol_MA30']).astype(int)

    # Drop NaNs created by rolling windows and forward shifts
    clean_out = out.dropna()
    return clean_out


build_high_snr_dataset = build_real_features_and_targets


def prepare_benchmark_dataset(ticker: str, start: str = "2010-01-01", end: str = "2026-01-01", horizon: int = 5) -> pd.DataFrame:
    """Convenience loader and preprocessor for ticker benchmark dataset."""
    raw_df = fetch_and_clean_ticker(ticker, start, end)
    return build_real_features_and_targets(raw_df, horizon=horizon)


if __name__ == "__main__":
    for ticker in ["SPY", "AAPL", "QQQ", "TLT"]:
        raw_df = fetch_and_clean_ticker(ticker)
        data = build_real_features_and_targets(raw_df)
        print(f"  -> {ticker} Clean Feature Matrix: Shape {data.shape}")
        print(f"     Target_Dir Counts: Down(-1)={sum(data['Target_Dir']==-1)}, Neutral(0)={sum(data['Target_Dir']==0)}, Up(+1)={sum(data['Target_Dir']==1)}")
        print(f"     Buy-and-Hold Total Return: {(np.exp(data['Return'].sum()) - 1)*100:.2f}%\n")
