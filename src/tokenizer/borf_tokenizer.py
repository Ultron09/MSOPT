"""
Bag-Of-Receptive-Fields (BORF) Tokenizer for Financial Time Series
==================================================================
Based on Spinnato et al. (IEEE Access 2024, arXiv:2311.18029).

Implements multi-scale receptive field extraction, 1D-SAX discretization 
(mean + trend slope per segment), and sparse token vocabulary building.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm

class FinancialBORFTokenizer:
    """
    Financial BORF Tokenizer extracting multi-scale pattern tokens from financial time series.
    
    Parameters:
    -----------
    window_sizes : list of int
        Receptive field window lengths (default: [4, 8, 16, 32])
    dilations : list of int
        Receptive field dilation factors (default: [1, 2, 4])
    stride : int
        Step size between sliding receptive fields (default: 1 for dense overlapping)
    n_segments : int
        Number of segments per receptive field for 1D-SAX (default: 4)
    alphabet_size_mean : int
        Number of discrete symbols for segment mean value (default: 4)
    alphabet_size_slope : int
        Number of discrete symbols for segment trend slope (default: 3: Down, Flat, Up)
    std_threshold : float
        Standard deviation ratio threshold below which a segment is marked constant (default: 0.01)
    """
    
    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        stride: int = 1,
        n_segments: int = 4,
        alphabet_size_mean: int = 4,
        alphabet_size_slope: int = 3,
        std_threshold: float = 0.01
    ):
        self.window_sizes = window_sizes or [4, 8, 16, 32]
        self.dilations = dilations or [1, 2, 4]
        self.stride = stride
        self.n_segments = n_segments
        self.alphabet_size_mean = alphabet_size_mean
        self.alphabet_size_slope = alphabet_size_slope
        self.std_threshold = std_threshold
        
        # Precompute SAX Gaussian breakpoints for mean discretization
        self.mean_breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size_mean + 1)[1:-1])
        # Slope breakpoints (negative, neutral, positive)
        self.slope_breakpoints = np.array([-0.25, 0.25]) if alphabet_size_slope == 3 else norm.ppf(np.linspace(0, 1, self.alphabet_size_slope + 1)[1:-1])
        
        self.vocabulary_: Dict[str, int] = {}
        self.inverse_vocabulary_: Dict[int, str] = {}
        self.is_fitted = False

    def _discretize_value(self, val: float, breakpoints: np.ndarray) -> str:
        """Map continuous value to discrete symbol (A, B, C, ...)."""
        idx = np.searchsorted(breakpoints, val)
        return chr(65 + idx)  # 'A', 'B', 'C', ...

    def _extract_receptive_field(self, series: np.ndarray, start_idx: int, w: int, d: int) -> Optional[np.ndarray]:
        """Extract dilated subseries: [x_j, x_{j+d}, ..., x_{j+d(w-1)}]."""
        end_idx = start_idx + d * (w - 1) + 1
        if end_idx > len(series):
            return None
        return series[start_idx:end_idx:d]

    def _to_1d_sax(self, rf: np.ndarray) -> str:
        """
        Convert a Receptive Field into a 1D-SAX word string.
        Each segment yields 2 symbols: (Mean_Symbol, Slope_Symbol).
        """
        w = len(rf)
        std_rf = np.std(rf)
        if std_rf < self.std_threshold:
            # Low volatility / flat subseries token
            return "FLAT_" + str(w)

        # Normalize receptive field (Z-score)
        norm_rf = (rf - np.mean(rf)) / (std_rf + 1e-8)

        # Split into n_segments
        seg_size = w / self.n_segments
        word_parts = []

        for s in range(self.n_segments):
            idx_start = int(np.round(s * seg_size))
            idx_end = int(np.round((s + 1) * seg_size))
            seg = norm_rf[idx_start:idx_end]
            
            if len(seg) == 0:
                continue
                
            # Segment mean
            seg_mean = np.mean(seg)
            mean_sym = self._discretize_value(seg_mean, self.mean_breakpoints)
            
            # Segment slope (linear regression fit)
            if len(seg) > 1:
                t = np.arange(len(seg))
                slope = np.polyfit(t, seg, 1)[0]
            else:
                slope = 0.0
            slope_sym = self._discretize_value(slope, self.slope_breakpoints)
            
            word_parts.append(f"{mean_sym}{slope_sym}")

        return "".join(word_parts)

    def fit_transform_series(self, series: np.ndarray) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Extract all BORF tokens across time for a single time series.
        
        Returns:
        --------
        token_df : pd.DataFrame
            DataFrame with columns: ['time_idx', 'token', 'window_size', 'dilation', 'token_id']
        vocab : dict
            Mapping from token_str -> token_id
        """
        tokens_extracted = []
        
        for w in self.window_sizes:
            for d in self.dilations:
                rf_span = d * (w - 1) + 1
                for t in range(0, len(series) - rf_span + 1, self.stride):
                    rf = self._extract_receptive_field(series, t, w, d)
                    if rf is None:
                        continue
                    
                    word = self._to_1d_sax(rf)
                    full_token = f"w{w}_d{d}_{word}"
                    
                    if full_token not in self.vocabulary_:
                        token_id = len(self.vocabulary_)
                        self.vocabulary_[full_token] = token_id
                        self.inverse_vocabulary_[token_id] = full_token
                    else:
                        token_id = self.vocabulary_[full_token]
                        
                    tokens_extracted.append({
                        'time_idx': t + rf_span - 1, # Token assigned to end-timestamp of the receptive field
                        'start_idx': t,
                        'token': full_token,
                        'token_id': token_id,
                        'window_size': w,
                        'dilation': d
                    })
                    
        df = pd.DataFrame(tokens_extracted)
        self.is_fitted = True
        return df, self.vocabulary_

    def get_bag_of_words_histogram(self, series: np.ndarray, rolling_window: int = 60) -> pd.DataFrame:
        """
        Transform time series into rolling Bag-of-Words (token count histograms).
        Ideal for downstream GBDT / linear classification.
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer must be fitted first!")
            
        token_df, _ = self.fit_transform_series(series)
        
        # Pivot table: time_idx vs token_id counts
        pivot = pd.crosstab(token_df['time_idx'], token_df['token_id'])
        
        # Rolling sum over specified window
        rolling_bow = pivot.rolling(window=rolling_window, min_periods=1).sum().fillna(0)
        return rolling_bow


if __name__ == "__main__":
    # Quick sanity test on synthetic data
    np.random.seed(42)
    sample_returns = np.random.randn(500) * 0.01
    
    tokenizer = FinancialBORFTokenizer(window_sizes=[4, 8, 16], dilations=[1, 2])
    token_df, vocab = tokenizer.fit_transform_series(sample_returns)
    
    print(f"Extraction Test Complete:")
    print(f"  Total tokens extracted: {len(token_df)}")
    print(f"  Vocabulary size: {len(vocab)} unique pattern words")
    print(f"  Sample tokens:\n{token_df.head(10)}")
