# Literature Summary — Key Papers

> Compiled from deep-dive reading of the 5 must-read papers for this research.

---

## 1. BORF — Bag-Of-Receptive-Fields (Spinnato et al., 2024)

**Paper**: "A Bag of Receptive Fields for Time Series Extrinsic Predictions"  
**Published**: IEEE Access 2024 | [arXiv:2311.18029](https://arxiv.org/abs/2311.18029)  
**Code**: [github.com/fspinna/borf](https://github.com/fspinna/borf) + `aeon` toolkit

### Core Algorithm
1. **Windowing**: Extract receptive fields with parameters (w=length, d=dilation, s=stride)
   - Receptive field: `r = [x_j, x_{j+d}, ..., x_{j+d(w-1)}]`
   - Dilation expands temporal coverage without increasing window size
   - This IS the CNN receptive field concept applied to time series
2. **Normalization**: Thresholded z-standardization per receptive field
   - If sigma_r/sigma_x < theta, set to 0 (constant segment detection)
   - Handles missing values via complete case analysis
3. **Approximation**: 1D-SAX discretization to symbolic words
   - Segments each RF into `l` parts, computes mean + slope per segment
   - Quantizes into alphabet symbols, concatenates into a "word"
4. **Transform**: Count word frequencies as sparse bag-of-words matrix
   - Multiple configurations run in parallel, stacked horizontally

### Key Parameters (Default Heuristic)
- Window sizes: `w = [4, 8, 16, ..., 2^floor(log2(m))]` (powers of 2)
- Dilations: `d = [1, 2, ..., 2^floor(log2(log2(m)))]`
- Word lengths: `l = [1, 2, 4, 8]`
- Stride: `s = 1` (maximum overlap by default!)
- Alphabet: `a_mu = 2-3`, `a_beta = 1-3`
- Complexity: O(g*n*c*m^2) time, effectively linear space

### Results
- **TSC**: 2nd place overall (tied with ROCKET), best dictionary-based
- **TSER**: **1st place overall** — beats even black-box methods
- Deterministic, interpretable, handles variable-length + missing values

### Relevance to Our Work
- **Closest prior art** — literally implements multi-scale receptive field to symbolic token vocabulary
- Gap: stride-based overlap only, not content-aware; not optimized for finance; no learned boundaries
- Our starting point for Stage 1 experiments

---

## 2. PatchTST (Nie et al., 2023)

**Paper**: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"  
**Published**: ICLR 2023 | [arXiv:2211.14730](https://arxiv.org/abs/2211.14730)  
**Code**: [github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

### Core Architecture
1. **Patching**: Segment time series into fixed-length subseries patches
   - Patch length P, stride S (overlap when S < P)
   - Each patch becomes one input token to the Transformer
   - Directly inspired by Vision Transformer (ViT) image patches
2. **Channel-independence**: Each variate treated as separate univariate series
   - All channels share same embedding + Transformer weights
   - Dramatically improves generalization
3. **Transformer Encoder**: Standard self-attention on patch tokens

### Key Benefits
- Local semantic info preserved within patches (vs single-point tokens)
- Quadratic reduction in attention computation (N/P tokens instead of N)
- Can attend to longer history with same compute budget
- Self-supervised pre-training via masked patch prediction

### Results
- SOTA on 7 long-term forecasting benchmarks
- Excellent transfer learning performance

### Relevance to Our Work
- **THE baseline we must beat** — simple, strong, well-understood
- Our dynamic tokenizer must outperform PatchTST with same compute budget
- Key limitation: fixed patch length, no content-aware boundaries

---

## 3. TimesNet (Wu et al., 2023)

**Paper**: "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis"  
**Published**: ICLR 2023 | [arXiv:2210.02186](https://arxiv.org/abs/2210.02186)  
**Code**: [github.com/thuml/TimesNet](https://github.com/thuml/TimesNet)

### Core Idea
1. **Multi-periodicity detection**: Use FFT to find top-K dominant periods
2. **1D to 2D reshaping**: For each period, reshape 1D series into 2D tensor
   - Rows = intra-period variation (what happens within one cycle)
   - Columns = inter-period variation (how cycles evolve over time)
3. **2D Convolution**: Apply Inception-style 2D conv block
   - Multiple kernel sizes in parallel (like our multi-scale idea)
4. **Adaptive aggregation**: Weighted sum of multi-period representations

### Results
- SOTA across 5 tasks: forecasting, imputation, classification, anomaly detection
- Task-general backbone (same architecture, different heads)

### Relevance to Our Work
- Validates the **2D representation is powerful** for time series
- FFT-based period detection is elegant but rigid
- Our approach: data-driven pattern boundaries instead of FFT periods
- Could combine: use TimesNet-style 2D conv within our token encoder

---

## 4. VALMOD — Variable-Length Matrix Profile (Linardi et al., 2018)

**Paper**: "Matrix Profile X: VALMOD — Scalable Discovery of Variable-Length Motifs in Data Series"  
**Published**: SIGMOD 2018

### Core Algorithm
- Standard Matrix Profile: for fixed length m, compute nearest-neighbor distance for every subsequence
- **VALMOD extension**: compute this across a range of lengths simultaneously
- Uses lower-bounding techniques to prune unnecessary comparisons
- Up to 20x faster than brute-force multi-length search

### Relevance to Our Work
- **Answers "at what scales do patterns exist?" cheaply** — before building any ML
- If VALMOD shows no repeating motifs at any scale, the whole premise fails
- Used in our Week 1 exploration as diagnostic tool

---

## 5. "Adaptive Patching Is Harder Than It Looks" (2026)

**Paper**: Benchmark study on dynamic vs. fixed patching for time series forecasting

### Key Finding
- On standard benchmarks, a **properly tuned uniform-patch baseline is competitive with dynamic patching**
- No consistent directional advantage for dynamic patching when aggregated across datasets
- Gains are method- and dataset-specific

### Implications
1. **Dynamic does not automatically equal better** — we cannot assume variable-length tokens will win
2. **Fair baseline is non-negotiable** — must compare against well-tuned fixed-patch PatchTST
3. **Financial data may be different** — the paper tests on standard benchmarks, not financial data which has regime changes and fat tails

---

## Summary: Where We Stand vs. Literature

| Capability | BORF | PatchTST | TimesNet | VALMOD | Our Proposal |
|---|:---:|:---:|:---:|:---:|:---:|
| Variable-length tokens | Yes | No | No | Yes | Yes |
| Multi-scale | Yes | No | Yes (FFT) | Yes | Yes |
| Overlapping | Yes (stride) | Partial | No | N/A | Yes (content-aware) |
| Learned boundaries | No | No | No | No | Yes |
| 2D representation | No | No | Yes | No | Yes (optional) |
| Interpretable vocabulary | Yes | No | No | Yes (motifs) | Yes |
| Financial evaluation | No | No | No | Partial | Yes |
| Transformer backbone | No | Yes | No (CNN) | No | Yes (Stage 2) |
