# DEPARTMENT RESEARCH PROPOSAL

**Project Title**: Multi-Scale Overlapping Pattern Tokenization (MSOPT) for Non-Stationary Financial Time Series  
**Lead Researcher / Author**: **Suryaansh Prithvijit Singh**  
**Submitted To**: **Prof. Shivaji Pawar**, Head of Department (HOD), Department of Future Tech  
**Institution**: **Universal AI University**  
**Date**: August 2026  

---

# 🔴 CORE PROBLEM STATEMENT

> **EXISTING TIME SERIES DEEP LEARNING ARCHITECTURES SUFFER FROM A FUNDAMENTAL PARADIGM FAILURE WHEN APPLIED TO NON-STATIONARY FINANCIAL MARKETS:**
>
> 1. **THE 1D SERIAL POINTWISE BLINDSPOT**: Serial models (LSTMs, vanilla Transformers) evaluate scalar price points $x_t$ sequentially. This destroys local visual shape context (inflections, wedges, double bottoms) and causes quadratic computational explosion $O(T^2)$, preventing long historical context processing.
> 2. **THE RIGID UNIFORM PATCHING BOTTLENECK**: Modern patch transformers (PatchTST) force rigid uniform sequence slicing ($P=16$). In non-stationary markets, fixed boundaries clip pattern inflections arbitrarily, enforce single-scale rigidity, and lack translation invariance—rendering them fragile during market regime shifts.
> 3. **THE QUANTITATIVE VISUAL GAP**: Human quantitative traders analyze market dynamics visually as **multi-scale 2D spatial chart primitives** (micro-spikes, daily consolidations, macro regimes). Existing machine learning paradigms force 1D serial vectors or rigid 1D uniform grids, failing to extract localized multi-scale visual tokens.

---

## 1. Executive Summary & Abstract

Financial price series (equities, indices, derivatives) present one of the most challenging domains for deep learning due to extreme non-stationarity, low signal-to-noise ratios, and regime-shifting dynamics. Current models force an unviable tradeoff between point-wise serial processing ($O(T^2)$ computational complexity with zero local shape context) and fixed-patch uniform slicing (boundary clipping distortion with zero multi-scale adaptation).

This research proposes **Multi-Scale Overlapping Pattern Tokenization (MSOPT)**—a novel paradigm that reformulates financial time series modeling. Rather than treating market data as a 1D scalar sequence, MSOPT transforms time series into a **2D Scale-Time Spatial Tensor of multi-scale overlapping receptive fields**. By combining dilated receptive field extraction ($w \in [4, 32], d \in [1, 4]$) with 1D-SAX symbolic discretization (mean + trend slope quantization), MSOPT constructs an interpretable, translation-invariant vocabulary of market pattern tokens ("words"). 

This proposal outlines the theoretical motivation, literature gap, mathematical framework, validation protocol, and expected departmental contributions of the MSOPT research program at **Universal AI University**.

---

## 2. 2025–2026 Literature Landscape & Research Gap Analysis

| Model Family | Published / Venue | Core Mechanism | Critical Limitation in Financial Series |
|---|---|---|---|
| **PatchTST** | Nie et al. (ICLR 2023) | Uniform fixed patching ($P=16, S=8$) + ViT | Rigid patch length; boundary clipping; zero multi-scale adaptation. |
| **TimesNet** | Wu et al. (ICLR 2023) | 1D $\to$ 2D Tensor Reshaping via FFT periods | Assumes stationary periodicity; fails on non-periodic, regime-shifting financial markets. |
| **BORF** | Spinnato et al. (IEEE 2024) | Dilated Receptive Fields + 1D-SAX Discretization | Bag-of-Words loss of temporal sequence order; no end-to-end gradient learning. |
| **TS-BPE** | Götz et al. (May 2025/2026) | Byte Pair Encoding subseries merge | **Non-overlapping partition**; boundary clipping; not finance-tuned; black-box vocabulary. |
| **DPR** | Zhong et al. (May 2026) | Dynamic Pattern Recalibration soft-routing | Recalibration layer on top of tokens; **not a token discovery mechanism**. |
| **PATK** | AAAI-26 (March 2026) | Physics-aware HMM elastic tokenization | **Non-overlapping partition**; general-purpose; not human-legible. |

### The Critical Gap
**No existing framework unifies multi-scale dilated receptive field extraction, dense translation-invariant overlapping tokenization ($s=1$), 1D-SAX human-legible codebooks, and 2D scale-time spatial tensor representations within a non-stationary financial modeling paradigm.**

---

## 3. Proposed Solution: Multi-Scale Overlapping Pattern Tokenization (MSOPT)

MSOPT introduces a 4-stage pipeline designed specifically to solve the non-stationary financial pattern problem:

```
                          1. RAW FINANCIAL MULTIVARIATE SERIES
                      (Log Returns, Parkinson Volatility, Relative Volume)
                                       │
                                       ▼
                   2. MULTI-SCALE DILATED RECEPTIVE FIELD SAMPLING
                     (Window w in {4,8,16,32}, Dilation d in {1,2,4}, Stride s=1)
                                       │
                                       ▼
                     3. THRESHOLDED 1D-SAX SYMBOLIC DISCRETIZATION
               (Segment Mean Quantization a_mu + Segment Slope Quantization a_beta)
                                       │
                                       ▼
                       4. 2D SCALE-TIME SPATIAL TENSOR MAPPING
                 [Y-axis = Receptive Field Scale (w,d), X-axis = Time Index t]
                                       │
                                       ▼
                     5. 2D SPATIAL CONVOLUTION & TRANSFORMER ENCODER
                  (Position + Scale + Volatility Multi-Dimensional Embeddings)
                                       │
                                       ▼
                  6. DIRECTIONAL THRESHOLD & REGIME CLASSIFICATION
```

---

## 4. Task Formulation: High-Signal Classification Targets (Fork B Alignment)

To avoid the low signal-to-noise ratio trap of raw point return regression ($\hat{y}_{t+1} \in \mathbb{R}$, where MSE loss forces trivial "flat" predictions), MSOPT evaluates two robust classification tasks:

1. **Directional Move Threshold Classification ($y_{dir}$)**:
   $$y_{dir, t} = \begin{cases} +1 & \text{if } R_{t:t+H} > +\delta \cdot \sigma_t \\ -1 & \text{if } R_{t:t+H} < -\delta \cdot \sigma_t \\ 0 & \text{otherwise (Neutral/Noise)} \end{cases}$$
   where $\delta = 0.5$ volatility units over horizon $H \in \{1, 5, 20\}$ days.

2. **Volatility Regime Shift Classification ($y_{vol}$)**:
   $$y_{vol, t} = \mathbb{I}\left(\sigma_{t:t+H} > 1.5 \cdot \bar{\sigma}_{30}\right)$$
   Predicting upcoming market volatility expansions and regime breakdowns.

---

## 5. Primary Research Questions & Testable Hypotheses

- **RQ1**: *Does dense overlapping receptive field tokenization ($s=1$) preserve predictive pattern signal better than single-scale fixed patching (PatchTST) and non-overlapping adaptive partitions (TS-BPE)?*
  - **H1**: MSOPT will achieve statistically significant out-of-sample Sharpe ratio improvement over PatchTST and TS-BPE due to 100% translation invariance.
- **RQ2**: *Does 1D-SAX symbolic discretization improve signal-to-noise ratio in financial returns compared to continuous vector embeddings?*
  - **H2**: Quantizing local subseries into discrete mean+slope symbols eliminates noise spikes while preserving structural shape primitives.
- **RQ3**: *Can a 2D Scale-Time Spatial Grid effectively model inter-scale pattern composition?*
  - **H3**: 2D spatial convolutions across scale and time dimensions will isolate cross-scale market interactions (e.g., micro-spikes triggering macro-regime breaks) better than 1D temporal convolutions.

---

## 6. Rigorous Methodology & Validation Protocol

To guarantee scientific validity and prevent backtest overfitting, MSOPT will be evaluated under a strict protocol:

1. **Dataset**: 15+ years of high-liquidity market data (SP500 ETF `/SPY`, Apple `/AAPL`, Nasdaq ETF `/QQQ`, Treasury ETF `/TLT`) spanning multiple market regimes (2008 Financial Crisis, 2020 COVID shock, 2022 Inflationary bear market, 2023–2026 Tech bull market).
2. **Walk-Forward Expanding Window Evaluation**: Initial 5-year training window, expanding annually. Zero lookahead bias.
3. **Transaction-Cost Awareness**: All backtests enforce 5 bps slippage and commission costs per trade.
4. **Benchmark Comparison**:
   - *Baseline 1*: Standard Technical Features + LightGBM / XGBoost
   - *Baseline 2*: PatchTST (Fixed Patch Transformer)
   - *Baseline 3*: TimesNet (FFT 2D Transformer)
   - *Baseline 4*: TS-BPE (Byte Pair Encoding Transformer)
   - *Baseline 5*: Native BORF + Bag-of-Words Classifier
5. **Evaluation Metrics**:
   - Statistical: Directional Accuracy (%), AUC-ROC, Macro F1-Score.
   - Financial: Out-of-Sample Sharpe Ratio, Sortino Ratio, Maximum Drawdown (Calmar Ratio).

---

## 7. Project Timeline & Milestones

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Literature Synthesis & Departmental Blueprint (COMPLETED)               │
│ - Deep-dive knowledge base across 8 core literature domains                      │
│ - Departmental Research Proposal compilation                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Formal Architectural Specification & Codebook Design (Weeks 1-2)       │
│ - Mathematical formalization of 1D-SAX codebook & 2D spatial tensor grid          │
│ - Baseline benchmark environment setup (Walk-forward backtest protocol)          │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: Framework Construction & Controlled Experiments (Weeks 3-5)             │
│ - Implementation of MSOPT Tokenizer & 2D Scale-Time Spatial Embedder             │
│ - Controlled ablation studies (MSOPT vs PatchTST vs TS-BPE vs TimesNet vs BORF)  │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Empirical Validation, Paper Writing & Department Defense (Weeks 6-8)   │
│ - Cross-asset generalization experiments & transaction-cost backtesting          │
│ - Drafting academic manuscript for submission to peer-reviewed venue             │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Expected Deliverables & Departmental Impact

1. **Academic Paper**: A formal research paper targeted for top-tier conference/journal submission (*IEEE PAMI*, *ICLR*, *ACM KDD*, or *Journal of Financial Data Science*).
2. **Open-Source Python Framework**: A robust, well-documented time series tokenization toolkit (`msopt-ts`) bridging traditional quantitative finance and modern deep learning.
3. **Novel Intellectual Property**: A benchmarked, state-of-the-art multi-scale pattern recognition architecture tailored for high-noise non-stationary domains.

---

## 9. Departmental Approval & Sign-Off Request

We respectfully request departmental review and approval to proceed with the formal research program outlined above.

| Role | Name | Signature | Date |
|---|---|---|---|
| **Lead Researcher / Author** | **Suryaansh Prithvijit Singh** (Universal AI University) | ____________________ | ____/____/2026 |
| **HOD, Future Tech** | **Prof. Shivaji Pawar** (Universal AI University) | ____________________ | ____/____/2026 |

