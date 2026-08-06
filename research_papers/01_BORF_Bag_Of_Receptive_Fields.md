# BORF: Bag-of-Receptive-Fields for Time Series Classification & Regression

> **Paper Title**: A Bag of Receptive Fields for Time Series Extrinsic Predictions  
> **Authors**: Francesco Spinnato, Lorenzo Nardi, et al.  
> **Publication**: IEEE Access 2024 | arXiv:2311.18029  
> **Official Implementation**: [github.com/fspinna/borf](https://github.com/fspinna/borf) | Integrated into `aeon` toolkit

---

## 1. Executive Summary & Core Paradigm Shift

Traditional time series representation methods suffer from a fundamental trade-off:
- **Global sequence models** (Transformers, LSTMs): Process the entire sequence point-by-point or in rigid uniform blocks, missing dynamic multi-scale sub-patterns.
- **Local shapelet / dictionary methods** (BOSS, WEASEL): Extract contiguous subsequences at fixed window lengths, failing to capture dilated (non-contiguous) sub-patterns that span wider temporal contexts.

**BORF** bridges this gap by introducing **Receptive Fields (RF)**—a concept borrowed directly from Convolutional Neural Networks (CNNs)—into symbolic dictionary-based time series modeling. By parameterizing subseries extraction with **window length ($w$)**, **dilation ($d$)**, and **stride ($s$)**, BORF generates a rich vocabulary of multi-scale, overlapping, symbolic pattern tokens.

---

## 2. Mathematical Formalization & Algorithmic Steps

Given an input univariate time series $X = (x_1, x_2, \dots, x_T) \in \mathbb{R}^T$:

### Step 1: Receptive Field Extraction
A receptive field $r$ starting at index $j$ with window size $w$ and dilation factor $d$ is defined as:
$$r_{j, w, d} = \left[ x_j, x_{j+d}, x_{j+2d}, \dots, x_{j+(w-1)d} \right]$$

- **Effective Temporal Coverage (Receptive Field Span)**: $S = (w - 1) \cdot d + 1$
- **Stride ($s$)**: Step size between consecutive starting indices $j$. Setting $s=1$ yields maximal overlap between receptive fields.

### Step 2: Thresholded Z-Standardization
Financial and physical time series often contain flat or constant subseries. Standard z-normalization amplifies noise in near-zero variance windows. BORF applies a variance-thresholded normalization:

$$\hat{r} = \begin{cases} 
0_{w} & \text{if } \frac{\sigma_r}{\sigma_X} < \theta \\
\frac{r - \mu_r}{\sigma_r} & \text{otherwise}
\end{cases}$$

where $\mu_r$ and $\sigma_r$ are the mean and standard deviation of $r$, $\sigma_X$ is the global standard deviation, and $\theta$ is the flat-window threshold.

### Step 3: 1D-SAX Discretization (Symbolic Word Generation)
Each normalized receptive field $\hat{r}$ of length $w$ is discretized into a symbolic word using **1D-SAX**:
1. Divide $\hat{r}$ into $l$ equal temporal segments.
2. For each segment $k \in \{1, \dots, l\}$, extract two features:
   - **Segment Mean ($\mu_k$)**: Average value in segment $k$.
   - **Segment Trend Slope ($\beta_k$)**: Linear regression slope $y = \beta_k \cdot t + \alpha$ fit to segment $k$.
3. Quantize $\mu_k$ using Gaussian breakpoints $\mathbf{\mathcal{B}}_{\mu}$ into alphabet $A_{\mu}$ of size $a_{\mu}$.
4. Quantize $\beta_k$ using slope breakpoints $\mathbf{\mathcal{B}}_{\beta}$ into alphabet $A_{\beta}$ of size $a_{\beta}$.
5. Concatenate pairs: $\text{Word} = (\mu_1, \beta_1) \circ (\mu_2, \beta_2) \circ \dots \circ (\mu_l, \beta_l)$.

### Step 4: Bag-of-Words (BoW) Histogram Construction
For a given configuration set $C = (w, d, l, a_{\mu}, a_{\beta})$, count the frequency of each unique word in the sequence $X$. Stacking frequency vectors across all configurations yields a high-dimensional sparse histogram vector $\mathbf{h}_X$.

---

## 3. Hyperparameter Configuration & Default Heuristics

| Parameter | Notation | Description | Default Heuristic / Search Space |
|---|---|---|---|
| **Window Lengths** | $w$ | Receptive field point count | $w \in \{4, 8, 16, \dots, 2^{\lfloor \log_2(T) \rfloor}\}$ |
| **Dilations** | $d$ | Subsampling stride within RF | $d \in \{1, 2, \dots, 2^{\lfloor \log_2(\log_2(T)) \rfloor}\}$ |
| **Word Lengths** | $l$ | Number of segments per RF | $l \in \{1, 2, 4, 8\}$ such that $l \le w$ |
| **Mean Alphabet** | $a_{\mu}$ | Symbols for segment mean | $a_{\mu} \in \{2, 3, 4\}$ |
| **Slope Alphabet**| $a_{\beta}$ | Symbols for segment trend | $a_{\beta} \in \{1, 2, 3\}$ |
| **Stride** | $s$ | Sliding window step size | $s = 1$ (Maximum overlapping subseries) |

---

## 4. Strengths, Limitations, and Empirical Performance

### Empirical Performance
- **Time Series Extrinsic Regression (TSER)**: Ranked **#1 overall** across standard benchmarks (UCR/UEA repository), outperforming deep learning models (ResNet, InceptionTime) and non-dictionary methods.
- **Time Series Classification (TSC)**: Ranked **#2 overall** (tied with ROCKET), making it the top-performing dictionary-based classifier.

### Key Strengths
1. **Multi-Scale Receptive Fields**: Dilated sampling captures both micro-structure (small $w, d=1$) and macro-trends (small $w$, large $d$) without increasing word length.
2. **Computational Efficiency**: Sub-quadratic complexity $O(C \cdot T)$ per sequence, dramatically faster than Matrix Profile or Deep Transformers.
3. **Interpretability**: Words directly map back to human-understandable shape primitives (e.g., "Sharp drop followed by flat consolidation").

### Key Limitations
1. **Fixed Grid Discretization**: 1D-SAX relies on rigid segment boundaries within each RF, ignoring sub-segment inflection points.
2. **Unweighted Bag-of-Words**: Order of tokens in time is lost inside the histogram unless positional windows or sequential n-grams are added.
3. **No Learned Representation**: Features are engineered via SAX quantization rather than learned end-to-end via gradient descent.

---

## 5. Direct Relevance to Financial Market Research

Financial asset prices (returns, volatility, volume ratio) exhibit non-stationary multi-scale patterns:
- **High-frequency / Short-term**: Micro-spikes, bid-ask spread rebounds ($w=4, d=1$).
- **Medium-term**: intraday momentum, multi-day flag patterns ($w=16, d=2$).
- **Macro-term**: Volatility compression, multi-week regime transitions ($w=32, d=4$).

BORF's dilated receptive fields allow us to build a **dictionary of financial market tokens** that capture non-contiguous, multi-scale patterns (e.g., "high volatility spike followed 4 days later by a sharp trend reversal") without requiring massive training datasets.
