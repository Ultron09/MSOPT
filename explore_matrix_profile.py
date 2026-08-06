"""
Week 1 Exploration: Matrix Profile Analysis on Financial Data
=============================================================
This script answers the fundamental question:
"Do repeating multi-scale patterns actually exist in liquid equity prices?"

Steps:
1. Download 10+ years of daily OHLCV for SPY, AAPL, QQQ
2. Run Matrix Profile at multiple window lengths (10, 20, 50, 100)
3. Visualize top motifs at each scale
4. Run multi-length analysis to find optimal pattern scales
5. Generate a summary report of findings
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# ─── Configuration ───────────────────────────────────────────────
TICKERS = ["SPY", "AAPL", "QQQ"]
DATA_START = "2010-01-01"
WINDOW_LENGTHS = [10, 20, 50, 100]       # bars (trading days)
MULTI_LENGTH_RANGE = range(5, 105, 5)     # for sweep analysis
TOP_K_MOTIFS = 3                          # motifs to visualize per scale

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "week1_matrix_profile")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── Step 1: Data Acquisition ───────────────────────────────────
def download_data():
    """Download OHLCV data for all tickers."""
    import yfinance as yf
    
    data = {}
    for ticker in TICKERS:
        cache_path = os.path.join(DATA_DIR, f"{ticker.lower()}_daily.csv")
        
        if os.path.exists(cache_path):
            print(f"[DATA] Loading cached {ticker} from {cache_path}")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        else:
            print(f"[DATA] Downloading {ticker} from {DATA_START}...")
            df = yf.download(ticker, start=DATA_START, auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(cache_path)
            print(f"  → {len(df)} rows saved to {cache_path}")
        
        data[ticker] = df
        print(f"  → {ticker}: {len(df)} rows, {df.index[0].date()} to {df.index[-1].date()}")
    
    return data


# ─── Step 2: Matrix Profile at Multiple Scales ──────────────────
def compute_matrix_profiles(close_prices: np.ndarray, ticker: str):
    """Compute Matrix Profile at multiple window lengths."""
    import stumpy
    
    profiles = {}
    for m in WINDOW_LENGTHS:
        if len(close_prices) <= m:
            print(f"  [SKIP] {ticker} too short for window={m}")
            continue
        
        print(f"  [MP] Computing Matrix Profile for {ticker}, window={m}...")
        mp = stumpy.stump(close_prices, m=m)
        profiles[m] = {
            'distances': mp[:, 0].astype(float),
            'indices': mp[:, 1].astype(int),
            'left_indices': mp[:, 2].astype(int),
            'right_indices': mp[:, 3].astype(int),
        }
        
        min_dist = np.min(profiles[m]['distances'])
        mean_dist = np.mean(profiles[m]['distances'])
        median_dist = np.median(profiles[m]['distances'])
        print(f"    → min_dist={min_dist:.4f}, mean={mean_dist:.4f}, median={median_dist:.4f}")
    
    return profiles


# ─── Step 3: Extract and Visualize Top Motifs ────────────────────
def extract_motifs(close_prices: np.ndarray, profiles: dict, ticker: str):
    """Extract top-K motifs at each scale and visualize them."""
    import stumpy
    
    all_motifs = {}
    
    for m, mp_data in profiles.items():
        print(f"  [MOTIF] Extracting top-{TOP_K_MOTIFS} motifs for {ticker}, window={m}...")
        
        distances = mp_data['distances']
        indices = mp_data['indices']
        
        # Find motif pairs (lowest distance subsequence pairs)
        motifs = []
        used_indices = set()
        
        sorted_idx = np.argsort(distances)
        for idx in sorted_idx:
            if idx in used_indices:
                continue
            nn_idx = indices[idx]
            if nn_idx in used_indices:
                continue
            # Check no overlap
            if abs(idx - nn_idx) < m:
                continue
            
            motifs.append({
                'idx1': idx,
                'idx2': nn_idx,
                'distance': distances[idx],
                'subseq1': close_prices[idx:idx+m].copy(),
                'subseq2': close_prices[nn_idx:nn_idx+m].copy(),
            })
            
            # Mark neighborhood as used to avoid near-duplicates
            for offset in range(-m//2, m//2 + 1):
                used_indices.add(idx + offset)
                used_indices.add(nn_idx + offset)
            
            if len(motifs) >= TOP_K_MOTIFS:
                break
        
        all_motifs[m] = motifs
    
    return all_motifs


def plot_motifs(close_prices: np.ndarray, profiles: dict, motifs: dict, 
                ticker: str, dates: pd.DatetimeIndex):
    """Create a comprehensive visualization of Matrix Profile results."""
    
    n_windows = len(profiles)
    fig = plt.figure(figsize=(20, 5 * n_windows + 4))
    gs = gridspec.GridSpec(n_windows + 1, 3, figure=fig, hspace=0.4, wspace=0.3)
    
    # Top row: full price series
    ax_price = fig.add_subplot(gs[0, :])
    ax_price.plot(dates, close_prices, color='#2196F3', linewidth=0.8, alpha=0.9)
    ax_price.set_title(f'{ticker} Close Price', fontsize=14, fontweight='bold')
    ax_price.set_ylabel('Price ($)')
    ax_price.grid(True, alpha=0.3)
    
    # Subsequent rows: one per window length
    for row_idx, m in enumerate(sorted(profiles.keys())):
        mp_data = profiles[m]
        m_motifs = motifs.get(m, [])
        
        # Column 1: Matrix Profile
        ax_mp = fig.add_subplot(gs[row_idx + 1, 0])
        ax_mp.plot(mp_data['distances'], color='#FF5722', linewidth=0.5, alpha=0.8)
        ax_mp.set_title(f'Matrix Profile (window={m})', fontsize=11, fontweight='bold')
        ax_mp.set_ylabel('Distance')
        ax_mp.grid(True, alpha=0.3)
        
        # Mark motif locations
        for i, motif in enumerate(m_motifs):
            ax_mp.axvline(motif['idx1'], color='#4CAF50', alpha=0.5, linestyle='--', linewidth=1)
            ax_mp.axvline(motif['idx2'], color='#4CAF50', alpha=0.5, linestyle='--', linewidth=1)
        
        # Column 2: Top motif pairs (normalized)
        ax_motif = fig.add_subplot(gs[row_idx + 1, 1])
        colors = ['#4CAF50', '#2196F3', '#FF9800']
        for i, motif in enumerate(m_motifs):
            # Z-normalize for shape comparison
            s1 = (motif['subseq1'] - np.mean(motif['subseq1'])) / (np.std(motif['subseq1']) + 1e-8)
            s2 = (motif['subseq2'] - np.mean(motif['subseq2'])) / (np.std(motif['subseq2']) + 1e-8)
            ax_motif.plot(s1, color=colors[i % len(colors)], linewidth=2, 
                         label=f'Motif {i+1}a (d={motif["distance"]:.3f})', alpha=0.8)
            ax_motif.plot(s2, color=colors[i % len(colors)], linewidth=2, 
                         linestyle='--', label=f'Motif {i+1}b', alpha=0.6)
        ax_motif.set_title(f'Top Motif Pairs (z-normalized, w={m})', fontsize=11, fontweight='bold')
        ax_motif.legend(fontsize=7, loc='upper right')
        ax_motif.grid(True, alpha=0.3)
        
        # Column 3: Distance distribution histogram
        ax_hist = fig.add_subplot(gs[row_idx + 1, 2])
        ax_hist.hist(mp_data['distances'], bins=80, color='#9C27B0', alpha=0.7, edgecolor='white', linewidth=0.3)
        ax_hist.axvline(np.median(mp_data['distances']), color='red', linestyle='--', 
                       label=f'Median={np.median(mp_data["distances"]):.2f}')
        ax_hist.axvline(np.percentile(mp_data['distances'], 5), color='green', linestyle='--',
                       label=f'5th pct={np.percentile(mp_data["distances"], 5):.2f}')
        ax_hist.set_title(f'Distance Distribution (w={m})', fontsize=11, fontweight='bold')
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)
    
    plt.suptitle(f'Matrix Profile Multi-Scale Analysis — {ticker}', 
                fontsize=16, fontweight='bold', y=1.01)
    
    save_path = os.path.join(RESULTS_DIR, f'{ticker.lower()}_matrix_profile.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [PLOT] Saved to {save_path}")
    return save_path


# ─── Step 4: Multi-Length Sweep ──────────────────────────────────
def multi_length_sweep(close_prices: np.ndarray, ticker: str):
    """Sweep across window lengths to find optimal pattern scales."""
    import stumpy
    
    print(f"  [SWEEP] Running multi-length sweep for {ticker}...")
    results = []
    
    for m in MULTI_LENGTH_RANGE:
        if len(close_prices) <= m + 10:
            continue
        
        mp = stumpy.stump(close_prices, m=m)
        distances = mp[:, 0].astype(float)
        
        results.append({
            'window_length': int(m),
            'min_distance': float(np.min(distances)),
            'p5_distance': float(np.percentile(distances, 5)),
            'p10_distance': float(np.percentile(distances, 10)),
            'median_distance': float(np.median(distances)),
            'mean_distance': float(np.mean(distances)),
            'std_distance': float(np.std(distances)),
            # "Motif density" = fraction of subsequences with very low distance
            'motif_density_p10': float(np.mean(distances < np.percentile(distances, 10))),
            # Normalized by window length for cross-scale comparison
            'min_dist_per_point': float(np.min(distances) / np.sqrt(m)),
            'p5_dist_per_point': float(np.percentile(distances, 5) / np.sqrt(m)),
        })
        
        if m % 20 == 0:
            print(f"    → window={m}: min_dist={results[-1]['min_distance']:.4f}, "
                  f"normalized_min={results[-1]['min_dist_per_point']:.4f}")
    
    df = pd.DataFrame(results)
    return df


def plot_multi_length_sweep(sweep_results: dict):
    """Plot the multi-length sweep results for all tickers."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'SPY': '#2196F3', 'AAPL': '#4CAF50', 'QQQ': '#FF5722'}
    
    for ticker, df in sweep_results.items():
        color = colors.get(ticker, '#9C27B0')
        
        # Min distance vs window length
        axes[0, 0].plot(df['window_length'], df['min_distance'], 
                       color=color, linewidth=2, marker='o', markersize=4, label=ticker)
        
        # Normalized min distance (per sqrt(window_length))
        axes[0, 1].plot(df['window_length'], df['min_dist_per_point'],
                       color=color, linewidth=2, marker='o', markersize=4, label=ticker)
        
        # 5th percentile distance
        axes[1, 0].plot(df['window_length'], df['p5_distance'],
                       color=color, linewidth=2, marker='o', markersize=4, label=ticker)
        
        # Median distance
        axes[1, 1].plot(df['window_length'], df['median_distance'],
                       color=color, linewidth=2, marker='o', markersize=4, label=ticker)
    
    titles = [
        'Min Distance vs Window Length\n(lower = stronger motif)',
        'Normalized Min Distance (÷√window)\n(cross-scale comparison)',
        '5th Percentile Distance vs Window Length\n(top 5% most similar pairs)',
        'Median Distance vs Window Length\n(overall similarity landscape)',
    ]
    ylabels = ['Min Distance', 'Normalized Min Distance', '5th Pct Distance', 'Median Distance']
    
    for ax, title, ylabel in zip(axes.flat, titles, ylabels):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Window Length (trading days)')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Multi-Scale Pattern Existence Analysis\nDo repeating patterns exist? At what scales?', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(RESULTS_DIR, 'multi_length_sweep.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  [PLOT] Multi-length sweep saved to {save_path}")
    return save_path


# ─── Step 5: Generate Summary Report ────────────────────────────
def generate_report(data: dict, all_profiles: dict, all_motifs: dict, 
                   sweep_results: dict, plot_paths: dict):
    """Generate a markdown summary report of findings."""
    
    lines = [
        "# Week 1: Matrix Profile Exploration — Findings",
        "",
        f"> **Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"> **Tickers analyzed**: {', '.join(TICKERS)}",
        f"> **Window lengths**: {WINDOW_LENGTHS}",
        f"> **Multi-length sweep range**: {list(MULTI_LENGTH_RANGE)[0]}–{list(MULTI_LENGTH_RANGE)[-1]} (step {list(MULTI_LENGTH_RANGE)[1] - list(MULTI_LENGTH_RANGE)[0]})",
        "",
        "---",
        "",
        "## Key Question: Do repeating multi-scale patterns exist in equity prices?",
        "",
    ]
    
    for ticker in TICKERS:
        close = data[ticker]['Close'].dropna().values
        profiles = all_profiles[ticker]
        motifs = all_motifs[ticker]
        
        lines.append(f"### {ticker}")
        lines.append(f"- **Period**: {data[ticker].index[0].date()} to {data[ticker].index[-1].date()}")
        lines.append(f"- **Data points**: {len(close)}")
        lines.append("")
        lines.append("| Window | Min Dist | 5th Pct | Median | Top Motif Distance |")
        lines.append("|--------|----------|---------|--------|-------------------|")
        
        for m in sorted(profiles.keys()):
            mp_data = profiles[m]
            m_motifs = motifs.get(m, [])
            top_motif_dist = m_motifs[0]['distance'] if m_motifs else 'N/A'
            if isinstance(top_motif_dist, float):
                top_motif_dist = f"{top_motif_dist:.4f}"
            
            lines.append(
                f"| {m} | {np.min(mp_data['distances']):.4f} | "
                f"{np.percentile(mp_data['distances'], 5):.4f} | "
                f"{np.median(mp_data['distances']):.4f} | {top_motif_dist} |"
            )
        
        lines.append("")
    
    # Multi-length sweep summary
    lines.extend([
        "---",
        "",
        "## Multi-Length Sweep: Optimal Pattern Scales",
        "",
    ])
    
    for ticker, df in sweep_results.items():
        best_norm = df.loc[df['min_dist_per_point'].idxmin()]
        best_raw = df.loc[df['min_distance'].idxmin()]
        
        lines.append(f"### {ticker}")
        lines.append(f"- **Best scale (normalized)**: window={int(best_norm['window_length'])} "
                     f"(normalized min dist={best_norm['min_dist_per_point']:.4f})")
        lines.append(f"- **Best scale (raw)**: window={int(best_raw['window_length'])} "
                     f"(raw min dist={best_raw['min_distance']:.4f})")
        
        # Find scales where patterns are particularly strong (low normalized distance)
        vals = df['min_dist_per_point'].values.astype(float)
        threshold = float(np.percentile(vals, 25))
        strong_scales = df.loc[vals <= threshold, 'window_length'].tolist()
        lines.append(f"- **Strong pattern scales** (bottom 25% normalized distance): "
                     f"{[int(s) for s in strong_scales]}")
        lines.append("")
    
    # Conclusions
    lines.extend([
        "---",
        "",
        "## Preliminary Conclusions",
        "",
        "### Do repeating patterns exist?",
        "",
        "*(To be filled after reviewing the plots and numbers above)*",
        "",
        "### At what scales?",
        "",
        "*(To be filled after reviewing multi-length sweep results)*",
        "",
        "### Go/No-Go for Week 2?",
        "",
        "*(To be determined based on above findings)*",
        "",
    ])
    
    report_path = os.path.join(RESULTS_DIR, "week1_findings.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    print(f"  [REPORT] Saved to {report_path}")
    return report_path


# ─── Main Pipeline ───────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  WEEK 1: MATRIX PROFILE EXPLORATION")
    print("  Do repeating multi-scale patterns exist in equity prices?")
    print("=" * 70)
    
    # Step 1: Download data
    print("\n[STEP 1] Data Acquisition")
    print("-" * 40)
    data = download_data()
    
    # Step 2-3: Matrix Profile + Motif extraction per ticker
    all_profiles = {}
    all_motifs = {}
    plot_paths = {}
    
    for ticker in TICKERS:
        print(f"\n[STEP 2-3] Matrix Profile Analysis: {ticker}")
        print("-" * 40)
        
        close = data[ticker]['Close'].dropna().values.astype(np.float64)
        dates = data[ticker]['Close'].dropna().index
        
        # Compute on returns instead of raw prices for better pattern matching
        # (prices have trend, returns are more stationary)
        returns = np.diff(np.log(close))  # log returns
        dates_ret = dates[1:]
        
        print(f"  Using log returns: {len(returns)} data points")
        
        profiles = compute_matrix_profiles(returns, ticker)
        motifs = extract_motifs(returns, profiles, ticker)
        
        plot_path = plot_motifs(returns, profiles, motifs, ticker, dates_ret)
        
        all_profiles[ticker] = profiles
        all_motifs[ticker] = motifs
        plot_paths[ticker] = plot_path
    
    # Step 4: Multi-length sweep
    print(f"\n[STEP 4] Multi-Length Sweep Analysis")
    print("-" * 40)
    sweep_results = {}
    for ticker in TICKERS:
        close = data[ticker]['Close'].dropna().values.astype(np.float64)
        returns = np.diff(np.log(close))
        sweep_df = multi_length_sweep(returns, ticker)
        sweep_results[ticker] = sweep_df
        
        # Save raw sweep data
        sweep_df.to_csv(os.path.join(RESULTS_DIR, f'{ticker.lower()}_sweep.csv'), index=False)
    
    sweep_plot_path = plot_multi_length_sweep(sweep_results)
    
    # Step 5: Generate report
    print(f"\n[STEP 5] Generating Summary Report")
    print("-" * 40)
    report_path = generate_report(data, all_profiles, all_motifs, sweep_results, plot_paths)
    
    print("\n" + "=" * 70)
    print("  EXPLORATION COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Report: {report_path}")
    print(f"\nPlots generated:")
    for ticker, path in plot_paths.items():
        print(f"  {ticker}: {path}")
    print(f"  Multi-length sweep: {sweep_plot_path}")
    
    return data, all_profiles, all_motifs, sweep_results


if __name__ == "__main__":
    main()
