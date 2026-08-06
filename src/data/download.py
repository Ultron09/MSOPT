"""
Data acquisition module for the pattern tokenization research.
Downloads historical OHLCV data from Yahoo Finance.
"""
import os
import pandas as pd
import yfinance as yf
from datetime import datetime


DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def download_ohlcv(ticker: str, start: str = "2010-01-01", end: str = None) -> pd.DataFrame:
    """Download daily OHLCV data for a ticker."""
    if end is None:
        end = datetime.now().strftime("%Y-%m-%d")
    
    print(f"Downloading {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    # Flatten multi-level columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"  → {len(df)} rows, {df.columns.tolist()}")
    return df


def save_data(df: pd.DataFrame, ticker: str):
    """Save DataFrame to CSV in the data directory."""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily.csv")
    df.to_csv(path)
    print(f"  → Saved to {path}")
    return path


def load_data(ticker: str) -> pd.DataFrame:
    """Load previously saved data."""
    path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data found for {ticker}. Run download first.")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def download_all(tickers: list = None, start: str = "2010-01-01"):
    """Download and save data for all target tickers."""
    if tickers is None:
        tickers = ["SPY", "AAPL", "QQQ"]
    
    results = {}
    for ticker in tickers:
        df = download_ohlcv(ticker, start=start)
        save_data(df, ticker)
        results[ticker] = df
    
    return results


if __name__ == "__main__":
    download_all()
