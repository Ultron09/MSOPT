# TimesNet: Temporal 2D-Variation Modeling for Time Series

> **Paper Title**: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis  
> **Authors**: Haixu Wu, Tengge Hu, Yong Liu, Hang Zhou, Jianmin Wang, Mingsheng Long  
> **Publication**: ICLR 2023 | arXiv:2210.02186  
> **Official Implementation**: [github.com/thuml/TimesNet](https://github.com/thuml/TimesNet)

---

## 1. Executive Summary & Core Philosophical Insight

Traditional time series models analyze sequences purely in 1D space (point-by-point or patch-by-patch). However, real-world time series exhibit **multi-periodicity**—nested cyclic dynamics operating simultaneously across different temporal scales (e.g., daily intraday patterns + weekly cycles + quarterly seasonal trends).

**TimesNet** introduces a revolutionary transform: **1D time series to 2D tensor reshaping** based on dominant frequency periods detected via Fast Fourier Transform (FFT). By transforming 1D sequences into 2D matrices, TimesNet enables standard **2D Convolutional Neural Networks (CNNs)** (such as Inception blocks) to simultaneously capture:
1. **Intra-period variation** (variations within one period cycle, represented across columns).
2. **Inter-period variation** (variations across consecutive cycles, represented across rows).

---

## 2. Mathematical Formalization & Algorithmic Pipeline

Given a 1D univariate time series $X \in \mathbb{R}^T$:

### Step 1: Multi-Periodicity Detection via FFT
1. Compute the Fast Fourier Transform (FFT) of sequence $X$:
$$A = \text{FFT}(X) \in \mathbb{C}^T$$
2. Calculate the amplitude spectrum $A_{amp} = |A| \in \mathbb{R}^{\lfloor T/2 \rfloor}$.
3. Identify top-$k$ dominant frequencies $f_1, f_2, \dots, f_k$ with highest amplitude values.
4. Calculate corresponding fundamental periods $\{p_1, p_2, \dots, p_k\}$:
$$p_i = \left\lfloor \frac{T}{f_i} \right\rfloor$$

### Step 2: 1D to 2D Reshaping
For each identified period $p_i$, reshape the 1D series $X \in \mathbb{R}^T$ into a 2D matrix $X^{2D}_i \in \mathbb{R}^{f_i \times p_i}$:
$$X^{2D}_i(r, c) = X\left((r - 1) \cdot p_i + c\right)$$
where $r \in \{1, \dots, f_i\}$ represents period index (inter-period) and $c \in \{1, \dots, p_i\}$ represents intra-period time step.

- Zero-padding is applied if $T$ is not evenly divisible by $p_i$.

### Step 3: 2D Convolutional Inception Block (`TimesBlock`)
Pass each 2D tensor $X^{2D}_i$ through a 2D Inception Conv block containing multi-scale 2D convolution kernels (e.g., $1\times 1, 3\times 3, 5\times 5$):
$$\hat{X}^{2D}_i = \text{Inception2D}\left(X^{2D}_i\right) \in \mathbb{R}^{C \times f_i \times p_i}$$

The 2D spatial convolution naturally captures local 2D spatial correlations between adjacent intra-period time steps and adjacent inter-period cycles!

### Step 4: 2D to 1D Truncated Reshaping & Adaptive Aggregation
1. Reshape the processed 2D feature map $\hat{X}^{2D}_i$ back into 1D sequence $\hat{X}_i \in \mathbb{R}^T$.
2. Compute softmax weights based on period amplitudes:
$$w_i = \frac{\exp(A_{amp}[f_i])}{\sum_{j=1}^k \exp(A_{amp}[f_j])}$$
3. Weighted aggregation across all $k$ period scales:
$$X_{out} = \sum_{i=1}^k w_i \cdot \hat{X}_i$$

---

## 3. Structural Comparison: TimesBlock vs. Standard 1D Models

```
1D Input Series X(T)
       │
       ▼
   FFT Spectrum
       │
  ┌────┴──────────────────────────┐
  ▼ (Period p1)                  ▼ (Period p2)
Reshape into 2D(f1 x p1)     Reshape into 2D(f2 x p2)
  │                              │
2D Inception Conv              2D Inception Conv
  │                              │
Reshape back to 1D(T)          Reshape back to 1D(T)
  └────┬──────────────────────────┘
       ▼
Adaptive Amplitude Weighting Sum -> Output 1D Series
```

---

## 4. Strengths & Limitations

### Strengths
1. **Unified Multi-Task Backbone**: Performs state-of-the-art across short-term forecasting, long-term forecasting, classification, anomaly detection, and imputation.
2. **Effective 2D Representation**: 2D conv filters excel at capturing localized 2D patterns (e.g., 2D patches/filters).
3. **Multi-Scale Period Fusion**: Automatically selects prominent frequencies without manual tuning.

### Limitations in Financial Context (Why We Need a Better Tokenizer)
1. **FFT Assumes Stationary Periodicity**: Financial markets are non-stationary and non-periodic. Market regimes do not repeat with fixed calendar frequency (e.g., earnings spikes occur quarterly, but volatility crashes occur stochastically). FFT period detection fails when patterns are episodic rather than periodic.
2. **Fixed Tensor Dimensions**: Reshaping into $f_i \times p_i$ forces rigid row/column alignment. Market patterns vary in duration across cycles (one consolidation lasts 7 days, the next lasts 12 days).
3. **Global Spectral Leakage**: Fourier transform over long windows mixes historical regimes with current state.

---

## 5. Takeaway for Our Research Architecture

TimesNet proves that **2D representation of 1D time series enables powerful multi-scale feature extraction via 2D convolutions / patches**.

However, instead of relying on **FFT periodic reshaping** (which breaks on non-periodic financial data), our proposal constructs **2D pattern token maps** using **content-aware variable-length receptive fields (BORF-like or dynamic patches)** stacked horizontally and vertically across scale and time!
