# Dynamic vs. Fixed Patching: Cautionary Results & 2025-2026 Literature Frontier

> **Context**: Analysis of recent benchmarking literature (Götz et al., BPE 2025/2026; DPR Zhong et al., 2026; PATK AAAI-26; arXiv:2606.04074).

---

## 1. Executive Summary & The "Dynamic Patching Paradox"

Intuition strongly suggests that **dynamic, variable-length patching** (segmenting time series at natural inflection points or structural breaks) should strictly outperform **fixed, uniform patching** (e.g., rigid $P=16$ windows).

However, rigorous empirical benchmarks on standard time series forecasting datasets (ETTh, Weather, Electricity, Traffic) reveal a surprising result known as the **Dynamic Patching Paradox**:

> *Across standard benchmarks, a properly tuned uniform-patch model (PatchTST) equals or beats most dynamic patching algorithms.*

Understanding **why** dynamic patching fails in naive implementations—and how 2025/2026 competitors attempt to address it—is critical so that our research design avoids these traps and delivers genuine, verifiable gains.

---

## 2. 2025–2026 Literature Frontier: Recent Competitors & Limitations

| Competitor Paper | Core Mechanism | Key Strength | Critical Limitation / Gap vs. MSOPT |
|---|---|---|---|
| **BPE for Time Series** (Götz et al., arXiv:2505.14411, May 2025/2026) | Byte Pair Encoding merge of raw subseries into adaptive-length tokens | Adaptive token length; outperforms foundation models (Chronos, MOMENT) | **Non-overlapping partition** (every point belongs to 1 token); destroys translation invariance; general-purpose, not finance-tuned. |
| **DPR — Dynamic Pattern Recalibration** (Zhong et al., arXiv:2605.06310, May 2026) | Soft-routed pattern response recalibration layer on top of tokens | Shifts behavior at volatility spikes and regime boundaries | Operates on top of existing tokens; **not a token discovery mechanism**. |
| **PATK — Physics-Aware Elastic Tokenization** (AAAI-26, March 2026) | HMM-driven elastic tokens via Rate-of-Variation & Spectral Energy | Physics-conditioned boundary placement | **Non-overlapping partition**; general-purpose; not human-legible. |
| **Dynamic TMoE** (arXiv:2605.20678, May 2026) | Drift-aware Mixture-of-Experts with temporal memory router | Designed for non-stationary series | Model routing level, not tokenization. |

---

## 3. The 4 Root Causes of Dynamic Patching Failure

### Failure Mode 1: Over-Segmentation in High-Noise Environments
In noisy series (such as financial log returns), local noise spikes trigger premature boundary splits. The dynamic tokenizer generates dozens of tiny, noisy fragments, creating erratic token sequences that overwhelm downstream attention layers.

### Failure Mode 2: Loss of Uniform Batching Efficiency & Non-Overlapping Partitions
Recent competitors (BPE, PATK, TimeMosaic) force **non-overlapping partitions** to preserve GPU batching. However, non-overlapping partitions introduce **boundary clipping**: a 3-bar candlestick pattern shifting by 1 bar gets sliced in half across patch boundaries.

### Failure Mode 3: Disconnect Between Segmentation Objective and Downstream Loss
Most dynamic patchers split boundaries using unsupervised statistical criteria (e.g., change-point detection, variance shifts). However, **points of high variance shift do not necessarily correspond to points where forecasting loss is reducible** (arXiv:2606.04074).

### Failure Mode 4: Token Distribution Shift
In rigid uniform patching, token $i$ and token $i+1$ have fixed temporal step distance $S$. In dynamic patching, token durations fluctuate (e.g., Token 1 = 3 days, Token 2 = 25 days). Without explicit temporal duration encodings, attention mechanisms confuse fast micro-events with long macro-regimes.

---

## 4. The MSOPT Differentiator: Why Overlapping Multi-Scale Receptive Fields Win

MSOPT solves all 4 failure modes by combining:
1. **Dense Overlapping Stride ($s=1$)**: Eliminates boundary clipping distortion and guarantees **100% translation invariance**.
2. **1D-SAX Discretization**: Quantizes both segment mean AND trend slope, eliminating high-frequency noise while retaining shape primitives (`w8_d1_ACBD`).
3. **Classification & Volatility Targets (Fork B)**: Evaluates directional move thresholds ($\pm \delta$) and volatility regime shifts rather than raw return regression, maximizing signal-to-noise ratio.
4. **2D Scale-Time Spatial Grid**: Maps multi-scale tokens into a 2D matrix ($Y=\text{Scales}, X=\text{Time}$), providing explicit scale positional embeddings $\mathbf{E}_{scale}(w, d)$.
