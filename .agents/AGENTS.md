# PROJECT GUIDELINES & EVOLVING AGENT MEMORY CONSTRAINTS

**Project**: Multi-Scale Overlapping Pattern Tokenization (MSOPT) for Non-Stationary Financial Time Series  
**Scope**: Workspace-level rules for AI Pair Programmer and Subagents.

---

## 🚨 MANDATORY CONTINUOUS LEARNING & MEMORY RULE

> **AGENTS MUST AUTOMATICALLY UPDATE AGENT FILES & EXPERIMENT LOGS AS NEW CODE, STAGES, AND EXPERIMENTS ARE IMPLEMENTED.**
>
> 1. **No Memory Loss**: Every experiment, code implementation stage, or parameter test MUST be logged immediately in `.agents/EXPERIMENT_LOG.md`.
> 2. **Never Repeat Mistakes**: Before designing or modifying code, agents MUST read `.agents/EXPERIMENT_LOG.md` to verify what approaches failed and why.
> 3. **Log Empirical Results**: Record exact Out-of-Sample metrics (Accuracy %, AUC-ROC, Sharpe Ratio, Max Drawdown, Slippage impact).
> 4. **Update Code Maps**: When new modules or components are created, update Section 3 (Codebase State & Implementations Map) in this document.

---

## 1. Core Architectural & Philosophical Rules

1. **Research First, Code Second**: Theoretical architecture and validation protocols are formally aligned.
2. **Never Treat Time Series as 1D Scalars**: Respect local 2D visual chart primitives.
3. **Dense Overlapping Receptive Fields ($s=1$)**: Maintain dense stride $s=1$ for **translation invariance**.
4. **Discretization via 1D-SAX**: Discretize subseries using 1D-SAX (mean $\mu_k$ + slope $\beta_k$).
5. **High-SNR Task Target (Fork B)**: Evaluate on Directional Threshold Moves ($y_{dir} \in \{-1, 0, +1\}$) or Volatility Regimes ($y_{vol} \in \{0, 1\}$).

---

## 2. Validation & Backtesting Discipline (Zero-Overfitting Policy)

1. **Strict Walk-Forward Validation**: Evaluate using expanding walk-forward windows (2016–2026, zero lookahead).
2. **Transaction Cost Enforcement**: Deduct **5 bps (0.05%)** per trade for slippage and execution.
3. **Cross-Asset Testing**: Evaluate across `/SPY`, `/AAPL`, `/QQQ`, `/TLT`.

---

## 3. Codebase Implementation Map & Live Status

| Module / Component | File Path | Status | Key Findings / Insights |
|---|---|---|---|
| **Data Preprocessor** | `src/data/preprocessing.py` | Completed | OHLCV loader, Parkinson volatility, and Fork B high-SNR directional/volatility targets. |
| **Matrix Profile Diagnostic** | `explore_matrix_profile.py` | Completed | Confirmed motif density peaks at short scales ($w \in [5, 20]$). $d_{norm} \to 1.0$ at $w \ge 50$. |
| **MSOPT Tokenizer** | `src/tokenizer/msopt_tokenizer.py` | Completed | Multi-scale 1D-SAX ($w \in \{4,8,16,32\}, d \in \{1,2,4\}, s=1$), rolling BoW & 2D spatial grid indexing. |
| **MSOPT PyTorch Engine** | `src/models/msopt_engine.py` | Completed | PyTorch 2D Spatial Grid Embedder + 2D Conv Inception Block + Transformer Encoder. |
| **Benchmark Pipeline** | `experiments/benchmark_pipeline.py` | Completed | 10-year walk-forward evaluation (2016–2026) post 5 bps transaction costs. |
| **Literature Base** | `research_papers/` | Completed | 8 deep-dive papers & master proposal (`01`–`08`, `INDEX.md`). |
| **Research Proposal** | `RESEARCH_PROPOSAL.md` | Completed | Official proposal for Universal AI University (Suryaansh Singh & Prof. Shivaji Pawar). |

---

## 4. Key Lessons & Known Failure Modes (Do Not Repeat)

- **Failure Mode 1: Pandas/Numpy Scalar Dtype Mismatch**: Cast numpy scalars to plain Python `float()` or `int()` before dataframe comparison.
- **Failure Mode 2: Unicode Windows Codec Error**: Always set `$env:PYTHONUTF8="1"`.
- **Failure Mode 3: Unweighted Bag-of-Words Temporal Loss**: Raw BORF histograms lose sequence order. MSOPT PyTorch Engine uses 2D spatial grid + time embeddings.
- **Failure Mode 4: Non-Overlapping Boundary Clipping Distortion**: TS-BPE and PATK force non-overlapping partitions; MSOPT enforces dense overlapping stride $s=1$.
