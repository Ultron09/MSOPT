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
| **PyTorch MSOPT Engine** | `src/models/msopt_engine.py` | Completed | PyTorch 2D Spatial Grid Embedder + 2D Conv Inception Block + Transformer Encoder. |
| **Bibliography Base** | `paper/references.bib` | Completed | Verified BibTeX database with authentic citations (Nie et al., Wu et al., Spinnato et al., Lin et al., Yeh et al.). |

---

## Real Experiment Log 1: STUMPY Matrix Profile Motif Discovery (SPY 2010–2025)

**Date**: August 6, 2026  
**Script**: `explore_matrix_profile.py`  
**Library**: STUMPY 1.14.1 (`stumpy.stump`)  
**Data**: 4,023 authentic daily SPY log return bars (2010-01-05 to 2025-12-31).  
**Figure**: `results/matrix_profile_motifs.png`  

### 📊 Empirical Motif Distance Table

| Window ($m$) | Min Euclidean Dist | Scaled Dist ($d / \sqrt{m}$) | Motif 1 Start Date | Motif 2 Start Date | Empirical Structural Observation |
|---|---|---|---|---|---|
| **$m = 5$** | **0.0076** | **0.0034** | 2019-04-10 | 2025-02-26 | **Near-Identical Recurrence**: Extremely tight motif density at 5-day horizon. |
| **$m = 10$** | **0.4945** | **0.1564** | 2017-01-23 | 2022-09-30 | **Strong Motif Match**: High shape similarity across 2017 bull run & 2022 bear split. |
| **$m = 20$** | **2.0334** | **0.4547** | 2011-12-16 | 2016-02-26 | Moderate motif match ($d_{scaled} \approx 0.45$). |
| **$m = 30$** | **3.2386** | **0.5913** | 2017-12-26 | 2018-09-04 | Approaching uncorrelated limit ($d_{scaled} \to 0.60$). |
| **$m = 50$** | **4.8916** | **0.6918** | 2017-11-27 | 2018-08-06 | High distance ($d_{scaled} \to 0.70$); low motif density at macro horizons. |

---

## Real Experiment Log 3: Backtest Financial Accounting & Metric Math Verification

**Date**: August 7, 2026  
**Script**: `tests/test_backtest_metrics.py`  
**Protocol**: Hand-calculated 5-day deterministic return and signal sequence verification.  

### 📊 Verification Ledger Results

| Step ($t$) | Asset Return ($R_t$) | Signal ($p_t$) | Flip $|p_t - p_{t-1}|$ | Tx Cost (5 bps) | Gross Return | Net Return ($R_{net}$) | Compounded Wealth ($W_t$) | Drawdown |
|---|---|---|---|---|---|---|---|---|
| **Day 0** | +2.00% | +1 | 1.0 | 0.05% (0.0005) | +2.00% | **+1.95%** | **1.019500** | 0.00% |
| **Day 1** | -1.00% | -1 | 2.0 | 0.10% (0.0010) | +1.00% | **+0.90%** | **1.028676** | 0.00% |
| **Day 2** | +3.00% | +1 | 2.0 | 0.10% (0.0010) | +3.00% | **+2.90%** | **1.058507** (Peak) | 0.00% |
| **Day 3** | -2.00% |  0 | 1.0 | 0.05% (0.0005) |  0.00% | **-0.05%** | **1.057978** | **-0.05%** |
| **Day 4** | +1.00% | +1 | 1.0 | 0.05% (0.0005) | +1.00% | **+0.95%** | **1.068029** | 0.00% |

---

### 🔑 Accounting Verification Findings:
1. **Signal Shift Alignment**: Position held during day $t$ is strictly determined at end of day $t-1$.
2. **Transaction Cost Deduction**: Fee of 5 bps ($0.05\% = 0.0005$) per unit of position change is deducted on the day the flip occurs.
3. **Wealth & Drawdown**: Wealth compounds multiplicatively $W_t = \prod (1 + R_{net, t})$; Max Drawdown measures peak-to-trough percentage drop relative to running maximum.



---

### 🔑 Empirical Finding for MSOPT Architecture:
- Motifs decay rapidly as scale increases beyond $m \ge 30$ days ($d_{scaled}$ rises from $0.0034 \to 0.6918$).
- This confirms that localized visual subseries patterns reside primarily in **short-to-medium scales ($w \in [4, 16]$)**, justifying multi-scale dilated receptive field extraction ($w \in \{4,8,16,32\}$) over giant single-scale macro windows.

