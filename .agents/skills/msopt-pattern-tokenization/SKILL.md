---
name: msopt-pattern-tokenization
description: Specialized guidelines and mathematical specification for building Multi-Scale Overlapping Pattern Tokenization (MSOPT) systems for non-stationary financial time series.
---

# Multi-Scale Overlapping Pattern Tokenization (MSOPT) Skill

This skill provides expert patterns, mathematical specifications, and algorithmic steps for constructing and operating **Multi-Scale Overlapping Pattern Tokenization (MSOPT)** systems.

---

## 1. Core Architecture Overview

MSOPT converts non-stationary 1D financial series into an interpretable **2D Scale-Time Spatial Grid** of symbolic pattern tokens:

```
[Raw Financial Series] 
        │
        ▼ (Dilated Sampling w in {4,8,16,32}, d in {1,2,4}, s=1)
[Multi-Scale Receptive Fields]
        │
        ▼ (1D-SAX Discretization: Mean + Trend Slope)
[Symbolic Codebook Vocabulary ("w8_d1_ACBD")]
        │
        ▼ (2D Matrix Assembly)
[2D Scale-Time Spatial Tensor (Scale x Time x Embedding)]
        │
        ▼ (Spatial Convolutions + Transformer Attention)
[Predictive Directional Signal]
```

---

## 2. Step-by-Step Mathematical Specification

### Step 1: Receptive Field Extraction
Given series $X = (x_1, \dots, x_T)$:
For window sizes $w \in \{4, 8, 16, 32\}$ and dilations $d \in \{1, 2, 4\}$ with stride $s=1$:
$$r_{t, w, d} = [x_t, x_{t+d}, x_{t+2d}, \dots, x_{t+(w-1)d}]$$

### Step 2: Thresholded Normalization
$$\hat{r} = \begin{cases} \mathbf{0}_w & \text{if } \sigma_r / \sigma_X < \theta \\ \frac{r - \mu_r}{\sigma_r} & \text{otherwise} \end{cases}$$

### Step 3: 1D-SAX Word Discretization
For each segment $k \in \{1, \dots, l\}$:
- Quantize segment mean $\mu_k \to A_{\mu} \in \{\text{A, B, C, D}\}$ via Gaussian breakpoints.
- Quantize segment slope $\beta_k \to A_{\beta} \in \{\text{Down, Flat, Up}\}$ via slope breakpoints.
- Format token: `w{w}_d{d}_{Word}` (e.g., `w16_d2_ACBD`).

### Step 4: 2D Spatial Grid Assembly
Construct tensor $\mathbf{H} \in \mathbb{R}^{(K \cdot J) \times T \times D_{emb}}$ where row index corresponds to scale $(w_k, d_j)$, column index corresponds to timestamp $t$, and depth stores multi-dimensional token embeddings:
$$\mathbf{E}(t, w, d) = \mathbf{e}_{token} + \mathbf{E}_{pos}(t) + \mathbf{E}_{scale}(w, d) + \mathbf{E}_{vol}(\sigma_t)$$

---

## 3. Best Practices & Design Constraints

1. **Always Enforce Stride $s=1$**: Prevents boundary-clipping artifacts and guarantees translation invariance.
2. **Dynamic Vocabulary Pruning**: Filter out rare words appearing $< 5$ times in training window to prevent codebook bloat.
3. **Multi-Channel Independence**: Apply tokenization independently to returns, volatility, and volume, then concatenate or fuse in embedding space.
