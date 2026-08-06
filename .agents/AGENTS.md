# PROJECT GUIDELINES & EVOLVING AGENT MEMORY CONSTRAINTS

**Project**: Multi-Scale Overlapping Pattern Tokenization (MSOPT) for Non-Stationary Financial Time Series  
**Scope**: Workspace-level rules for AI Pair Programmer and Subagents.

---

## 🚨 MANDATORY CONTINUOUS LEARNING & MEMORY RULE

> **AGENTS MUST AUTOMATICALLY UPDATE AGENT FILES & EXPERIMENT LOGS AS NEW CODE, STAGES, AND EXPERIMENTS ARE IMPLEMENTED.**
>
> 1. **No Memory Loss**: Every experiment, code implementation stage, or parameter test MUST be logged immediately in `.agents/EXPERIMENT_LOG.md`.
> 2. **Never Repeat Mistakes**: Before designing or modifying code, agents MUST read `.agents/EXPERIMENT_LOG.md` to verify what approaches failed and why (e.g., pandas scalar comparison bugs, fixed patch boundary clipping, unweighted BoW temporal loss).
> 3. **Log Empirical Results**: Record exact Out-of-Sample metrics (Accuracy %, AUC-ROC, Sharpe Ratio, Max Drawdown, Slippage impact).
> 4. **Update Code Maps**: When new modules or components are created, update Section 4 (Codebase State & Implementations Map) in this document.

---

## 1. Core Architectural & Philosophical Rules

1. **Research First, Code Second**: No experimental code execution or major refactoring shall take place until theoretical architecture and validation protocols are formally aligned.
2. **Never Treat Time Series as 1D Scalars**: Always respect local 2D visual chart primitives (micro-spikes, consolidations, regime shifts). Avoid point-by-point processing ($x_t$) without localized receptive field or patch context.
3. **Dense Overlapping Receptive Fields ($s=1$)**: All pattern tokenizers must maintain dense stride ($s=1$) to guarantee **translation invariance**. Rigid uniform non-overlapping boundaries ($S=P$) and non-overlapping adaptive partitions (TS-BPE, PATK) are prohibited for primary feature extraction due to boundary clipping distortion.
4. **Discretization via 1D-SAX**: Continuous subseries must be discretized using 1D-SAX (segment mean $\mu_k$ + segment trend slope $\beta_k$) to reduce noise and enforce structural shape quantization.
5. **High-SNR Task Target (Fork B)**: Evaluate on Directional Threshold Moves ($y_{dir} \in \{-1, 0, +1\}$) or Volatility Regime Shifts ($y_{vol} \in \{0, 1\}$) rather than raw point return regression.

---

## 2. Validation & Backtesting Discipline (Zero-Overfitting Policy)

1. **Strict Walk-Forward Validation**: All empirical models must be evaluated using expanding or rolling walk-forward windows. Random $K$-fold cross-validation is strictly forbidden due to lookahead temporal leakage.
2. **Transaction Cost Enforcement**: All financial backtests must deduct a minimum of **5 bps (0.05%)** per trade for slippage and execution costs.
3. **Cross-Asset Testing**: Models must be evaluated across diverse asset classes (e.g., `/SPY`, `/AAPL`, `/QQQ`, `/TLT`) across distinct historical market regimes (2008 Crash, 2020 COVID shock, 2022 Inflationary bear, 2023–2026 Tech bull).

---

## 3. Codebase Implementation Map & Live Status

| Module / Component | File Path | Status | Key Findings / Insights |
|---|---|---|---|
| **Data Downloader** | `src/data/download.py` | Completed | Downloads 10+ yrs OHLCV for SPY/AAPL/QQQ with CSV caching. |
| **Matrix Profile Diagnostic** | `explore_matrix_profile.py` | Completed | Confirmed motif density peaks at short scales ($w \in [5, 20]$). $d_{norm} \to 1.0$ at $w \ge 50$. |
| **BORF Tokenizer** | `src/tokenizer/borf_tokenizer.py` | Prototype | Implements 1D-SAX (mean+slope), dilations ($w, d$), and sparse BoW count histograms. |
| **Literature Base** | `research_papers/` | Completed | 8 deep-dive papers & master proposal (`01`–`08`, `INDEX.md`, 2026 BPE/DPR/PATK frontier). |
| **Research Proposal** | `RESEARCH_PROPOSAL.md` | Completed | Formal departmental submit proposal with opening Core Problem Statement. |
| **MSOPT Engine** | `src/models/msopt_engine.py` | Planned (Stage 2) | 2D Scale-Time Spatial Grid & Hybrid Conv-Transformer Backbone. |

---

## 4. Key Lessons & Known Failure Modes (Do Not Repeat)

- **Failure Mode 1: Pandas/Numpy Scalar Dtype Mismatch**: Using raw numpy scalars inside pandas condition indexing on Windows triggers `TypeError: len() of unsized object`. Always cast to plain Python `float()` or `int()` before dataframe comparison.
- **Failure Mode 2: Unicode Windows Codec Error**: Characters like `→` trigger `UnicodeEncodeError: 'charmap'`. Always set `$env:PYTHONUTF8="1"` or stick to ASCII text in stdout prints.
- **Failure Mode 3: Unweighted Bag-of-Words Temporal Loss**: Raw BORF histograms lose the time order of tokens. Downstream models must use positional encodings or sequential n-gram tokens.
- **Failure Mode 4: Non-Overlapping Boundary Clipping Distortion**: TS-BPE (May 2025) and PATK force non-overlapping partitions to preserve batching, but clip candlestick patterns when timing shifts by 1 bar. MSOPT enforces dense overlapping stride $s=1$.
