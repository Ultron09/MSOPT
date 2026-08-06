# Multi-Scale Overlapping Pattern Tokenization (MSOPT)

> **Research Focus**: Non-linear, non-stationary financial time series pattern tokenization and spatial representation learning.

---

# 🔴 CORE PROBLEM STATEMENT

> **EXISTING TIME SERIES DEEP LEARNING ARCHITECTURES SUFFER FROM A FUNDAMENTAL PARADIGM FAILURE WHEN APPLIED TO NON-STATIONARY FINANCIAL MARKETS:**
>
> 1. **THE 1D SERIAL POINTWISE BLINDSPOT**: Serial models (LSTMs, vanilla Transformers) evaluate scalar price points $x_t$ sequentially. This destroys local visual shape context (inflections, wedges, double bottoms) and causes quadratic computational explosion $O(T^2)$, preventing long historical context processing.
> 2. **THE RIGID UNIFORM PATCHING BOTTLENECK**: Modern patch transformers (PatchTST) force rigid uniform sequence slicing ($P=16$). In non-stationary markets, fixed boundaries clip pattern inflections arbitrarily, enforce single-scale rigidity, and lack translation invariance—rendering them fragile during market regime shifts.
> 3. **THE QUANTITATIVE VISUAL GAP**: Human quantitative traders analyze market dynamics visually as **multi-scale 2D spatial chart primitives** (micro-spikes, daily consolidations, macro regimes). Existing machine learning paradigms force 1D serial vectors or rigid 1D uniform grids, failing to extract localized multi-scale visual tokens.

---

## 🔍 1. WHAT IS THERE (Current SOTA & Literature Base)

Existing state-of-the-art models in time series machine learning include:
- **PatchTST** (Nie et al., ICLR 2023): Slices univariate time series into uniform non-overlapping patches ($P=16, S=8$) and applies Vision Transformer self-attention.
- **TimesNet** (Wu et al., ICLR 2023): Discovers top-$k$ dominant frequencies via FFT, reshapes 1D series into 2D period tensors, and applies 2D Inception convolutions.
- **BORF** (Spinnato et al., IEEE 2024): Extracts dilated receptive fields ($w, d, s$), quantizes subseries via 1D-SAX (mean + trend slope), and constructs Bag-of-Words histograms.
- **VALMOD** (Linardi et al., SIGMOD 2018): Computes variable-length Matrix Profiles across window lengths $m \in [m_{min}, m_{max}]$ using distance lower-bounding.

---

## 🎯 2. WHY IT IS THERE (Design Intent & Strengths of Prior Art)

- **PatchTST exists** to solve the quadratic attention bottleneck of vanilla Transformers by reducing token count from $T$ to $T/P$, speeding up compute by $P^2$.
- **TimesNet exists** to exploit powerful 2D Convolutional Neural Network (CNN) feature extractors by converting 1D sequences into 2D intra-period and inter-period matrices.
- **BORF exists** to provide an interpretable, highly effective symbolic dictionary classifier that captures non-contiguous dilated sub-patterns.
- **VALMOD exists** to discover exact repeating shape motifs across arbitrary scales without manual window tuning.

---

## 🚫 3. WHAT IS NOT THERE (The Research Gap & Missing Capabilities)

Despite these advances, **critical capabilities are completely missing for financial time series**:

1. **No Shift-Invariant Overlapping Tokenization**: PatchTST's rigid stride $S=8$ clips patterns arbitrarily based on start index. If a chart pattern shifts by 2 days, token embeddings change completely.
2. **No Multi-Scale Spatial Grid**: TimesNet relies on **FFT Fourier periods**, assuming market dynamics are stationary and periodic. Financial markets are non-periodic and regime-shifting—FFT period detection fails completely.
3. **No Sequence Order Retention in BORF**: BORF compresses pattern frequencies into an **unweighted Bag-of-Words histogram**, discarding the crucial temporal sequence order of tokens ($A \to B \to C$).
4. **No Unified Framework**: No existing solution combines multi-scale dilated receptive field sampling, 1D-SAX codebook quantization, 2D Scale-Time Spatial Grid tensors, and deep Transformer attention.

---

## 💡 4. WHAT WE ARE PROPOSING (The MSOPT Breakthrough Solution)

We propose **Multi-Scale Overlapping Pattern Tokenization (MSOPT)**—a 4-stage pipeline built specifically for non-stationary financial markets:

```
                          1. RAW FINANCIAL MULTIVARIATE SERIES
                      (Log Returns, Parkinson Volatility, Relative Volume)
                                       │
                                       ▼
                   2. DENSE MULTI-SCALE DILATED RECEPTIVE FIELDS
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
                        6. DIRECTIONAL ALPHA PREDICTION & ASSET ALLOCATION
```

### Core Key Innovations:
- **Dense Stride ($s=1$)**: Guarantees complete translation invariance across all pattern scales.
- **1D-SAX Codebook**: Quantizes both segment mean level AND trend slope direction, eliminating noise while preserving shape.
- **2D Scale-Time Spatial Tensor**: Maps multi-scale tokens into a 2D matrix ($Y=\text{Scales}, X=\text{Time}$), enabling 2D spatial convolutions to capture inter-scale pattern composition (e.g., micro-spikes triggering macro regime breaks).

---

## 🛠️ 5. HOW WE ARE GOING TO DELIVER (Execution & Delivery Strategy)

Our delivery roadmap follows a rigorous 4-phase execution framework:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: Literature Synthesis & Departmental Approval (COMPLETED)               │
│ - 8 paper knowledge base in research_papers/ + INDEX.md                          │
│ - Formal Department Submission Proposal in RESEARCH_PROPOSAL.md                  │
│ - Workspace agent memory & continuous learning system in .agents/                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 2: Empirical Diagnostics & Baseline Infrastructure (COMPLETED)            │
│ - Downloaded 15+ years daily OHLCV for SPY, AAPL, QQQ                             │
│ - Matrix Profile analysis (explore_matrix_profile.py) confirmed motif density    │
│   concentrates at short scales (w in [5, 20], d_norm ≈ 0.42-0.45)                │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 3: Framework Construction & Controlled Ablation (WEEKS 3-5)               │
│ - Implement MSOPT Tokenizer & 2D Scale-Time Spatial Embedder (src/tokenizer/)    │
│ - Build 2D Conv-Transformer Neural Backbone (src/models/)                        │
│ - Controlled walk-forward ablation vs PatchTST, TimesNet, BORF, and LightGBM     │
├──────────────────────────────────────────────────────────────────────────────────┤
│ PHASE 4: Cross-Asset Backtesting & Academic Publication (WEEKS 6-8)             │
│ - Walk-forward expanding window backtesting with explicit 5 bps transaction cost │
│ - Out-of-Sample evaluation across SPY, AAPL, QQQ, TLT                            │
│ - Drafting academic manuscript for top-tier submission                           │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Navigation

- 📄 **Official Department Proposal**: [`RESEARCH_PROPOSAL.md`](RESEARCH_PROPOSAL.md)
- 📚 **Literature Research Library**: [`research_papers/INDEX.md`](research_papers/INDEX.md)
- 📐 **Master Architecture Spec**: [`research_papers/08_Proposed_MultiScale_Dynamic_Token_Architecture.md`](research_papers/08_Proposed_MultiScale_Dynamic_Token_Architecture.md)
- 🤖 **Agent Memory & Guidelines**: [`.agents/AGENTS.md`](.agents/AGENTS.md)
- 📊 **Empirical Experiment Log**: [`.agents/EXPERIMENT_LOG.md`](.agents/EXPERIMENT_LOG.md)
#   M S O P T  
 