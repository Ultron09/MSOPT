"""
Production MSOPT Tokenizer & 2D Spatial Grid Builder
===================================================
Implements multi-scale dilated receptive field extraction, 1D-SAX discretization
(mean + slope), codebook vocabulary building, and 2D spatial grid indexing.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from scipy.stats import norm

class MSOPTTokenizer:
    """
    Multi-Scale Overlapping Pattern Tokenizer (MSOPT).
    
    Parameters:
    -----------
    window_sizes : list of int
        Receptive field window lengths (default: [4, 8, 16, 32])
    dilations : list of int
        Subsampling dilation strides (default: [1, 2, 4])
    stride : int
        Sliding step size (default: 1 for dense translation invariance)
    n_segments : int
        Number of equal segments per receptive field for 1D-SAX (default: 4)
    alphabet_size_mean : int
        Number of discrete symbols for segment mean value (default: 4)
    alphabet_size_slope : int
        Number of discrete symbols for segment trend slope (default: 3)
    std_threshold : float
        Variance ratio threshold below which a segment is marked constant (default: 0.001)
    """
    
    def __init__(
        self,
        window_sizes: Optional[List[int]] = None,
        dilations: Optional[List[int]] = None,
        stride: int = 1,
        n_segments: int = 4,
        alphabet_size_mean: int = 4,
        alphabet_size_slope: int = 3,
        std_threshold: float = 0.001
    ):
        self.window_sizes = window_sizes or [4, 8, 16, 32]
        self.dilations = dilations or [1, 2, 4]
        self.stride = stride
        self.n_segments = n_segments
        self.alphabet_size_mean = alphabet_size_mean
        self.alphabet_size_slope = alphabet_size_slope
        self.std_threshold = std_threshold
        
        # Precompute Gaussian breakpoints for 1D-SAX mean
        self.mean_breakpoints = norm.ppf(np.linspace(0, 1, self.alphabet_size_mean + 1)[1:-1])
        # Slope breakpoints (negative, neutral, positive)
        self.slope_breakpoints = np.array([-0.2, 0.2]) if alphabet_size_slope == 3 else norm.ppf(np.linspace(0, 1, self.alphabet_size_slope + 1)[1:-1])
        
        self.vocabulary_: Dict[str, int] = {}
        self.inverse_vocabulary_: Dict[int, str] = {}
        self.scale_configs_: List[Tuple[int, int]] = []
        
        for w in self.window_sizes:
            for d in self.dilations:
                self.scale_configs_.append((w, d))
                
        self.n_scales = len(self.scale_configs_)

    def _discretize(self, val: float, breakpoints: np.ndarray) -> str:
        idx = np.searchsorted(breakpoints, float(val))
        return chr(65 + idx)  # 'A', 'B', 'C'...

    def _to_1d_sax(self, rf: np.ndarray) -> str:
        """Convert a single receptive field subseries into a 1D-SAX word string."""
        w = len(rf)
        std_rf = np.std(rf)
        if std_rf < self.std_threshold:
            return f"FLAT_{w}"

        norm_rf = (rf - np.mean(rf)) / (std_rf + 1e-8)
        seg_size = w / self.n_segments
        word_parts = []

        for s in range(self.n_segments):
            idx_start = int(np.round(s * seg_size))
            idx_end = int(np.round((s + 1) * seg_size))
            seg = norm_rf[idx_start:idx_end]
            
            if len(seg) == 0:
                continue
                
            # Mean symbol
            m_sym = self._discretize(np.mean(seg), self.mean_breakpoints)
            
            # Slope symbol
            if len(seg) > 1:
                t = np.arange(len(seg))
                slope = np.polyfit(t, seg, 1)[0]
            else:
                slope = 0.0
            s_sym = self._discretize(slope, self.slope_breakpoints)
            
            word_parts.append(f"{m_sym}{s_sym}")

        return "".join(word_parts)

    def fit_transform_series(self, series: np.ndarray, channel_name: str = "ret") -> Tuple[pd.DataFrame, Dict[str, int]]:
        """
        Extract multi-scale tokens for a 1D continuous sequence across time.
        """
        tokens_extracted = []
        
        for scale_idx, (w, d) in enumerate(self.scale_configs_):
            rf_span = d * (w - 1) + 1
            for t in range(0, len(series) - rf_span + 1, self.stride):
                rf = series[t : t + rf_span : d]
                word = self._to_1d_sax(rf)
                full_token = f"{channel_name}_w{w}_d{d}_{word}"
                
                if full_token not in self.vocabulary_:
                    token_id = len(self.vocabulary_) + 1  # 0 reserved for padding
                    self.vocabulary_[full_token] = token_id
                    self.inverse_vocabulary_[token_id] = full_token
                else:
                    token_id = self.vocabulary_[full_token]
                    
                tokens_extracted.append({
                    'time_idx': t + rf_span - 1, # End timestamp of receptive field
                    'scale_idx': scale_idx,
                    'token': full_token,
                    'token_id': token_id,
                    'window_size': w,
                    'dilation': d
                })
                
        df = pd.DataFrame(tokens_extracted)
        return df, self.vocabulary_

    def get_rolling_bow_histogram(self, series: np.ndarray, rolling_window: int = 30, channel_name: str = "ret") -> pd.DataFrame:
        """
        Build rolling Bag-of-Words token frequency count histogram.
        """
        token_df, _ = self.fit_transform_series(series, channel_name=channel_name)
        pivot = pd.crosstab(token_df['time_idx'], token_df['token_id'])
        rolling_bow = pivot.rolling(window=rolling_window, min_periods=1).sum().fillna(0)
        return rolling_bow

    def get_2d_spatial_grid_indices(self, series: np.ndarray, channel_name: str = "ret") -> Tuple[np.ndarray, Dict[str, int]]:
        """
        Construct 2D Spatial Grid Index Matrix H of shape [N_Scales, Time_Steps].
        Each entry H[scale_idx, t] contains the token_id active at timestamp t.
        """
        token_df, vocab = self.fit_transform_series(series, channel_name=channel_name)
        T = len(series)
        grid = np.zeros((self.n_scales, T), dtype=int)
        
        for _, row in token_df.iterrows():
            grid[int(row['scale_idx']), int(row['time_idx'])] = int(row['token_id'])
            
        return grid, vocab


if __name__ == "__main__":
    np.random.seed(42)
    sample_series = np.random.randn(300) * 0.01
    
    tokenizer = MSOPTTokenizer(window_sizes=[4, 8, 16, 32], dilations=[1, 2])
    grid, vocab = tokenizer.get_2d_spatial_grid_indices(sample_series, channel_name="SPY")
    
    print(f"[MSOPT Tokenizer Check]")
    print(f"  Vocabulary Size: {len(vocab)} unique pattern words")
    print(f"  2D Spatial Grid Shape: {grid.shape} (N_Scales={grid.shape[0]}, Time_Steps={grid.shape[1]})")
