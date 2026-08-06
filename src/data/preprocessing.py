"""
High-SNR Financial Data Preprocessor & Target Builder
=====================================================
Preplaces standard financial raw price data with high-SNR classification targets
(Fork B: Volatility-scaled Directional Threshold Moves & Volatility Regime Shifts).
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Tuple, List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = ["SPY", "AAPL", "QQQ", "TLT"]

def load_or_download_symbol(ticker: str, start: str = "2008-01-01") -> pd.DataFrame:
    """Load cached OHLCV CSV or download from Yahoo Finance."""
    csv_path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    else:
        print(f"[Data] Downloading {ticker} from {start}...")
        df = yf.download(ticker, start=start, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.to_csv(csv_path)
    return df

def build_high_snr_dataset(
    df: pd.DataFrame,
    horizon: int = 5,
    vol_scale_delta: float = 0.5
) -> pd.DataFrame:
    """
    Build high-signal feature set and Fork B classification targets.
    """
    out = df.copy()
    
    # 1. Log Returns
    out['Return'] = np.log(out['Close'] / out['Close'].shift(1))
    
    # 2. Parkinson Volatility
    high_low_ratio = np.log(out['High'] / out['Low'])
    out['Parkinson_Vol'] = np.sqrt((high_low_ratio ** 2) / (4 * np.log(2)))
    out['Vol_MA30'] = out['Parkinson_Vol'].rolling(30).mean()
    
    # 3. Relative Volume
    out['Rel_Volume'] = out['Volume'] / (out['Volume'].rolling(20).mean() + 1e-8)
    
    # 4. Target 1: Directional Threshold Move (y_dir in {-1, 0, +1})
    # Future H-day return
    future_return = np.log(out['Close'].shift(-horizon) / out['Close'])
    vol_threshold = vol_scale_delta * out['Parkinson_Vol'] * np.sqrt(horizon)
    
    # Map to classes: +1 (Up), -1 (Down), 0 (Neutral)
    dir_target = np.zeros(len(out), dtype=int)
    dir_target[future_return > vol_threshold] = 1
    dir_target[future_return < -vol_threshold] = -1
    out['Target_Dir'] = dir_target
    
    # 5. Target 2: Volatility Expansion Target (y_vol in {0, 1})
    future_max_vol = out['Parkinson_Vol'].shift(-horizon).rolling(horizon).max()
    out['Target_Vol'] = (future_max_vol > 1.5 * out['Vol_MA30']).astype(int)
    
    return out.dropna()

def prepare_benchmark_dataset(ticker: str) -> pd.DataFrame:
    """Load ticker and return fully prepared high-SNR dataset."""
    raw_df = load_or_download_symbol(ticker)
    return build_high_snr_dataset(raw_df)

if __name__ == "__main__":
    for ticker in TICKERS:
        df = prepare_benchmark_dataset(ticker)
        print(f"\n[Data Check] {ticker} ({len(df)} bars):")
        print(f"  Target_Dir Distribution: Up={sum(df['Target_Dir']==1)}, Down={sum(df['Target_Dir']==-1)}, Neutral={sum(df['Target_Dir']==0)}")
        print(f"  Target_Vol Expansion Ratio: {df['Target_Vol'].mean()*100:.1f}%")
