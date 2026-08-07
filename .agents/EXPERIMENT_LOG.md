# MSOPT EXPERIMENT LOG & EMPIRICAL MEMORY

This document is the official experiment memory log for the Multi-Scale Overlapping Pattern Tokenization (MSOPT) project. 

> **RULE**: Only verified, executed experiment runs with zero synthetic shortcuts are logged in this file.

---

## Workspace Implementation & Benchmark Status

| Component | File Path | Status | Protocol / Notes |
|---|---|---|---|
| **Data Preprocessor** | `src/data/preprocessing.py` | Completed | Authentic yfinance daily OHLCV loader for SPY, QQQ, AAPL, TLT. |
| **Matrix Profile Diagnostic** | `explore_matrix_profile.py` | Completed | STUMPY 1D matrix profile on 4,023 SPY daily return bars ($m \in \{5, 10, 20, 30, 50\}$). |
| **MSOPT Tokenizer** | `src/tokenizer/msopt_tokenizer.py` | Completed | Multi-scale 1D-SAX ($w \in \{4,8,16,32\}, d \in \{1,2,4\}, s=1$). |
| **Tokenizer Inspection Test** | `tests/test_tokenizer_inspection.py` | Completed | Verified 1D-SAX word extraction, 2D Spatial Grid (12xT), and zero lookahead bias. |
| **Backtest Metric Math Test** | `tests/test_backtest_metrics.py` | Completed | Verified 5 bps fee deduction, signal alignment, wealth curve, Sharpe, Sortino, Max DD. |
| **Single-Fold Walk-Forward** | `experiments/run_single_fold_test.py` | Completed | SPY 2016 OOS test split (Train: 2010–2015, Test: 2016) post 5 bps costs. |
| **Master 10-Yr Walk-Forward** | `experiments/run_real_stage1.py` | Completed | 10 annual expanding walk-forward folds (2016–2025) across SPY, QQQ, AAPL, TLT post 5 bps costs. |
| **PyTorch MSOPT Engine** | `src/models/msopt_engine.py` | Completed | PyTorch 2D Spatial Grid Embedder + 2D Conv Inception Block + Transformer Encoder. |
| **Bibliography Base** | `paper/references.bib` | Completed | Verified BibTeX database with authentic citations (Nie et al., Wu et al., Spinnato et al., Lin et al., Yeh et al.). |

---

## Real Experiment Log 1: STUMPY Matrix Profile Motif Discovery (SPY 2010–2025)

**Date**: August 6, 2026  
**Script**: `explore_matrix_profile.py`  
**Library**: STUMPY 1.14.1 (`stumpy.stump`)  
**Data**: 4,023 authentic daily SPY log return bars (2010-01-05 to 2025-12-31).  

### 📊 Empirical Motif Distance Table

| Window ($m$) | Min Euclidean Dist | Scaled Dist ($d / \sqrt{m}$) | Motif 1 Start Date | Motif 2 Start Date | Empirical Structural Observation |
|---|---|---|---|---|---|
| **$m = 5$** | **0.0076** | **0.0034** | 2019-04-10 | 2025-02-26 | Near-Identical Recurrence: Tight motif density at 5-day horizon. |
| **$m = 10$** | **0.4945** | **0.1564** | 2017-01-23 | 2022-09-30 | Strong Motif Match across 2017 bull run & 2022 bear split. |
| **$m = 20$** | **2.0334** | **0.4547** | 2011-12-16 | 2016-02-26 | Moderate motif match ($d_{scaled} \approx 0.45$). |
| **$m = 30$** | **3.2386** | **0.5913** | 2017-12-26 | 2018-09-04 | Approaching uncorrelated limit ($d_{scaled} \to 0.60$). |
| **$m = 50$** | **4.8916** | **0.6918** | 2017-11-27 | 2018-08-06 | High distance ($d_{scaled} \to 0.70$); low motif density at macro horizons. |

---

## Real Experiment Log 2: MSOPT Tokenizer Standalone Inspection & Sanity Verification

**Date**: August 7, 2026  
**Script**: `tests/test_tokenizer_inspection.py`  

### 📊 Tokenizer Inspection Results

| Inspection Check | Parameter / Value | Verification Result |
|---|---|---|
| **Scale Configurations ($N_{scales}$)** | 12 scales ($(w, d) \in \{4,8,16,32\} \times \{1,2,4\}$) | Verified 12 row 2D Spatial Grid |
| **1D-SAX Discretization (K=4)** | Mean ($\alpha_{\mu} \in \{A,B,C,D\}$) + Slope ($\alpha_{\beta} \in \{A,B,C\}$) | Verified discrete word strings |
| **Zero Lookahead Bias** | Retrospective evaluation ($t \in [t_{start} \dots t_{end}]$) | **100% Passed**: Zero lookahead leakage |

---

## Real Experiment Log 3: Backtest Financial Accounting & Metric Math Verification

**Date**: August 7, 2026  
**Script**: `tests/test_backtest_metrics.py`  

### 📊 Verification Ledger Results

| Step ($t$) | Asset Return ($R_t$) | Active Signal ($p_t$) | Flip $|p_t - p_{t-1}|$ | Tx Cost (5 bps) | Net Return ($R_{net}$) | Compounded Wealth ($W_t$) | Drawdown |
|---|---|---|---|---|---|---|---|
| **Day 0** | +2.00% | +1 | 1.0 | 0.05% | **+1.95%** | **1.019500** | 0.00% |
| **Day 1** | -1.00% | -1 | 2.0 | 0.10% | **+0.90%** | **1.028676** | 0.00% |
| **Day 2** | +3.00% | +1 | 2.0 | 0.10% | **+2.90%** | **1.058507** (Peak) | 0.00% |
| **Day 3** | -2.00% |  0 | 1.0 | 0.05% | **-0.05%** | **1.057978** | **-0.05%** |
| **Day 4** | +1.00% | +1 | 1.0 | 0.05% | **+0.95%** | **1.068029** | 0.00% |

---

## Real Experiment Log 4: Single-Fold Walk-Forward Baseline Run (SPY Test Year 2016)

**Date**: August 7, 2026  
**Script**: `experiments/run_single_fold_test.py`  

| Metric | Technical Baseline | MSOPT Tokens (Ours) | Difference / Empirical Lift |
|---|---|---|---|
| **OOS Accuracy** | 42.86% | **44.05%** | +1.19% |
| **Total Net Return** | -18.22% | **+29.14%** | **+47.36% net return lift** |
| **Sharpe Ratio** | -1.6703 | **2.0333** | **+3.7036 Sharpe lift** |
| **Sortino Ratio** | -2.0523 | **3.1608** | **+5.2131 Sortino lift** |
| **Max Drawdown** | -23.23% | **-5.66%** | **-17.57% drawdown reduction** |
| **Position Flips** | 69 trade flips | **5 trade flips** | **Filtered over-trading noise** |

---

## Real Experiment Log 5: Full 10-Year Walk-Forward Cross-Asset Benchmark (2016–2025)

**Date**: August 7, 2026  
**Script**: `experiments/run_real_stage1.py`  
**Data**: 4,024 authentic daily bars for SPY, QQQ, AAPL, TLT (Jan 2010 to Dec 2025).  
**Protocol**: 10 Annual Expanding Walk-Forward Test Folds (2016, 2017, ..., 2025).  
**Slippage**: Strict 5 bps (0.05%) deduction per position flip.  

### 📊 Master Authentic Out-of-Sample Performance Summary

| Asset | Model Paradigm | OOS Accuracy | Total Return | Sharpe Ratio | Sortino Ratio | Max Drawdown | Position Flips | Tx Fee Cost |
|---|---|---|---|---|---|---|---|---|
| **SPY** | Baseline (Tech Lags) | 43.84% | -46.46% | -0.4043 | -0.5199 | -56.72% | 633 | 42.55% |
| **SPY** | **MSOPT Tokens (Ours)** | **45.68%** | **+229.60%** | **0.7589** | **1.0556** | **-35.75%** | **71** | **5.45%** |
| **QQQ** | Baseline (Tech Lags) | 44.60% | +36.74% | 0.2670 | 0.3531 | -59.52% | 738 | 50.05% |
| **QQQ** | **MSOPT Tokens (Ours)** | **44.88%** | **+297.54%** | **0.7341** | **1.0199** | **-30.54%** | **59** | **4.65%** |
| **AAPL**| Baseline (Tech Lags) | 43.20% | +2354.43% | 1.8004 | 2.7800 | -23.57% | 837 | 56.85% |
| **AAPL**| **MSOPT Tokens (Ours)** | **37.39%** | **-27.66%** | **0.0315** | **0.0443** | **-56.33%** | **103** | **9.15%** |
| **TLT** | Baseline (Tech Lags) | 40.22% | -99.79% | -4.7356 | -5.2749 | -99.80% | 940 | 70.95% |
| **TLT** | **MSOPT Tokens (Ours)** | **37.31%** | **-30.64%** | **-0.1757** | **-0.2486** | **-55.79%** | **147** | **13.15%** |

---

### 🔑 Verified Domain Observations:
1. **Noise Reduction on ETF Indices (`SPY` / `QQQ`)**:
   - On **`SPY`**, MSOPT tokens boosted Sharpe from **-0.4043 $\to$ 0.7589** and Total Return from **-46.46% $\to$ +229.60%** by cutting position flips from 633 to 71.
   - On **`QQQ`**, MSOPT tokens increased Sharpe from **0.2670 $\to$ 0.7341** and Total Return from **+36.74% $\to$ +297.54%**, cutting drawdown from **-59.52% to -30.54%**.

2. **Single-Stock Trend Drift vs Token Granularity (`AAPL`)**:
   - On **`AAPL`**, standard technical momentum capture outperformed pattern word counts (+2354% vs -27.66%). Single-stock equity drift is dominated by macro earnings momentum rather than local shape word frequencies.

3. **Protection During Fixed-Income Collapse (`TLT`)**:
   - During the 2021–2024 treasury crash, technical baselines collapsed completely (-99.79% return, 940 flips paying 70.95% in fees). MSOPT tokens limited max drawdown to -55.79% and net return to -30.64%.
