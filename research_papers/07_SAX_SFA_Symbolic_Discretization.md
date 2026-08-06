# Symbolic Time Series Discretization: SAX, 1D-SAX, SFA, & Dictionary Models

> **Foundational Literature**: Survey of Symbolic Representation Methods for Time Series Mining.

---

## 1. Executive Summary & Why Symbolic Discretization Matters

Continuous time series are noisy, high-dimensional, and computationally expensive to index or compare. **Symbolic Discretization** transforms continuous sequences into strings of discrete symbols (words) from a small alphabet (e.g., `'A', 'B', 'C', 'D'`).

Symbolic representations unlock three massive advantages:
1. **Dimensionality Reduction & Denoising**: Smooths out high-frequency micro-noise while preserving global structural morphology.
2. **Sub-linear Hashing & Lookup**: Enables constant-time $O(1)$ dictionary lookups, inverted indexing, and exact frequency count histograms.
3. **Distance Bounding**: Distances in the discrete symbolic space lower-bound true Euclidean distances in continuous space.

---

## 2. Foundations: Classic SAX (Symbolic Aggregate approXimation)

Proposed by Lin et al. (DMKD 2007), **SAX** is the gold standard for symbolic time series discretization.

### Step 1: PAA (Piecewise Aggregate Approximation)
A z-normalized time series $\hat{X} \in \mathbb{R}^n$ is reduced to $w$ equal-sized segments. The PAA representation $\bar{X} = (\bar{x}_1, \dots, \bar{x}_w)$ computes the mean of each segment:
$$\bar{x}_i = \frac{w}{n} \sum_{j=\frac{n}{w}(i-1)+1}^{\frac{n}{w}i} \hat{x}_j$$

### Step 2: Gaussian Breakpoint Quantization
Under z-normalization, time series subseries values approximate a Standard Normal distribution $\mathcal{N}(0, 1)$.

For an alphabet of size $\alpha$, equiprobable region breakpoints $\mathbf{\mathcal{B}} = \{\beta_1, \beta_2, \dots, \beta_{\alpha-1}\}$ are selected using the Inverse Gaussian CDF ($\Phi^{-1}$):
$$P(X < \beta_i) = \frac{i}{\alpha}$$

Each PAA coefficient $\bar{x}_i$ is assigned a symbol corresponding to its Gaussian region:
$$\text{Symbol}(\bar{x}_i) = \text{chr}(65 + \text{index of region})$$

Example: $w=4, \alpha=3 \implies \text{Word} = \text{"ABCA"}$.

---

## 3. Advanced Evolution: 1D-SAX (Mean + Slope Quantization)

Classic SAX only captures segment **mean** values ($\bar{x}_i$). If a segment drops sharply from $+2.0$ to $-2.0$, its mean is $0.0$—identical to a flat segment at $0.0$!

**1D-SAX** (Minnen et al., 2011; extended in BORF 2024) fixes this by representing each segment with **two orthogonal descriptors**:
1. **Segment Mean ($\mu_k$)**: Quantized via Gaussian breakpoints $A_{\mu}$.
2. **Segment Trend Slope ($\beta_k$)**: Linear regression slope $y = \beta_k \cdot t + \alpha$ fit to segment $k$, quantized via slope breakpoints $A_{\beta}$.

```
1D-SAX Segment = (Mean Symbol, Slope Symbol)
Example: Segment 1 (High Mean 'D', Down Slope 'C') -> "DC"
Full 4-segment Word -> "DC-CB-BA-AA"
```

1D-SAX retains **both height and direction**, dramatically improving pattern discrimination!

---

## 4. Spectral Alternative: SFA (Symbolic Fourier Approximation)

Used in **BOSS** (Bag-of-SFA-Symbols) and **WEASEL** (Schäfer et al., 2017):
1. Instead of PAA, apply Discrete Fourier Transform (DFT) to each sliding subseries window.
2. Retain the top-$l$ real and imaginary Fourier coefficients (low-frequency trend + main oscillation).
3. Quantize Fourier coefficients using Multiple Coefficient Binning (MCB).

**Comparison: SAX vs. SFA**:
- **SAX (Time-Domain)**: Superior at capturing sharp temporal spikes, step functions, and localized inflection points.
- **SFA (Frequency-Domain)**: Superior at capturing smooth stationary periodic waves.
- **Financial Context**: Financial markets exhibit sharp jump-diffusions and structural breaks $\implies$ **1D-SAX (Time-Domain)** is significantly more effective than spectral SFA!

---

## 5. Architectural Synthesis for Our Research

Our tokenizer leverages **1D-SAX Multi-Scale Receptive Fields**:
- **Mean Alphabet $a_{\mu}=4$**: Captures 4 price level buckets (Overbought, High, Low, Oversold).
- **Slope Alphabet $a_{\beta}=3$**: Captures 3 trend momentum states (Down, Flat, Up).
- **Receptive Fields**: Multi-scale window sizes $w \in \{4, 8, 16, 32\}$ and dilations $d \in \{1, 2, 4\}$.

This generates an interpretable, highly predictive vocabulary of multi-scale market pattern tokens!
