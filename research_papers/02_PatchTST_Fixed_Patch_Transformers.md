# PatchTST: Fixed-Patch Time Series Transformers

> **Paper Title**: A Time Series is Worth 64 Words: Long-term Forecasting with Transformers  
> **Authors**: Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam  
> **Publication**: ICLR 2023 | arXiv:2211.14730  
> **Official Implementation**: [github.com/yuqinie98/PatchTST](https://github.com/yuqinie98/PatchTST)

---

## 1. Executive Summary & Core Contribution

Prior to PatchTST, applying Vision Transformer (ViT) architecture directly to time series treated individual scalar time points as input tokens. This approach caused two severe bottlenecks:
1. **Loss of Local Semantic Context**: A single scalar data point lacks local directional, trend, or curvature context.
2. **Quadratic Attention Complexity**: Processing sequence length $L$ requires $O(L^2)$ computation, limiting context windows to short time horizons ($L \le 96$).

**PatchTST** solves both issues by segmenting time series into **subseries patches** (tokens), reducing sequence length from $L$ to $N \approx L/P$ (where $P$ is patch length) and reducing self-attention complexity to $O(N^2) \approx O((L/P)^2)$.

---

## 2. Mathematical Formalization & Architecture

Given a multivariate time series $X \in \mathbb{R}^{M \times L}$ with $M$ channels and history length $L$:

### Step 1: Channel-Independence (CI)
Instead of processing all $M$ channels jointly, PatchTST splits the multivariate series into $M$ independent univariate series $x^{(m)} \in \mathbb{R}^L$:
- All channels share the **same Transformer backbone and parameters**.
- Prevents cross-channel overfitting and allows zero-shot transfer across datasets.

### Step 2: Patching (Fixed Windowing)
For a univariate sequence $x \in \mathbb{R}^L$, given patch length $P$ and stride $S$:
$$x = \left[ x_1, x_2, \dots, x_L \right]$$

The sequence is converted into $N$ overlapping patches $x^{(p)} \in \mathbb{R}^{P \times N}$:
$$N = \left\lfloor \frac{L - P}{S} \right\rfloor + 2$$

Each patch $x_i^{(p)} = [x_{(i-1)S + 1}, \dots, x_{(i-1)S + P}]^\top \in \mathbb{R}^P$.

### Step 3: Linear Projection & Positional Embedding
Each $P$-dimensional patch vector is projected into a $D$-dimensional embedding space via a linear matrix $W_P \in \mathbb{R}^{D \times P}$:
$$u_i = W_P x_i^{(p)} + W_{pos, i}, \quad u_i \in \mathbb{R}^D$$

where $W_{pos} \in \mathbb{R}^{D \times N}$ is a learnable or sinusoidal 1D positional embedding matrix.

### Step 4: Vanilla Transformer Encoder
The patch tokens $U = [u_1, u_2, \dots, u_N] \in \mathbb{R}^{D \times N}$ pass through a standard Multi-Head Self-Attention (MHSA) encoder:
$$Q_i = U W_Q, \quad K_i = U W_K, \quad V_i = U W_V$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{D_k}}\right) V$$

### Step 5: Linear Forecasting / Classification Head
The output tokens from the Transformer encoder $Z \in \mathbb{R}^{D \times N}$ are flattened or pooled and mapped to prediction horizon $H$ via a linear layer:
$$\hat{Y} = W_{head} \cdot \text{Flatten}(Z) \in \mathbb{R}^H$$

---

## 3. Key Hyperparameters & Trade-offs

| Parameter | Symbol | Typical Values | Impact on Performance |
|---|---|---|---|
| **Patch Length** | $P$ | 16, 24, 64 | Larger $P$ smooths noise, smaller $P$ captures fine grain |
| **Stride** | $S$ | 8, 16 ($S \le P$) | $S < P$ creates patch overlap; $S=P$ yields non-overlapping patches |
| **Lookback Horizon** | $L$ | 336, 512, 720 | Patching enables $L \ge 512$ without memory overflow |
| **Embedding Dim** | $D$ | 64, 128, 256 | Model capacity |
| **Head Layers** | $H_l$ | 3, 4 | Transformer encoder depth |

---

## 4. Strengths & Limitations

### Strengths
1. **Substantial Compute Reduction**: Patching by factor $P$ speeds up attention computation by $P^2$.
2. **Local Semantic Retention**: Each patch encapsulates local momentum, slope, and variance.
3. **Channel-Independence Robustness**: Consistently outperforms channel-dependent models on benchmark datasets.

### Limitations & Gaps (Critical to Our Research)
1. **Rigid Fixed Patch Length ($P$)**: PatchTST forces all patches to have exact length $P$ regardless of market dynamics. A sudden volatility crash spans 2 days, while a consolidation wedge spans 15 days; a fixed $P=16$ distorts both patterns.
2. **No Multi-Scale Dilation**: PatchTST only looks at contiguous contiguous slice $[t, t+P]$. It cannot capture dilated patterns ($x_t, x_{t+2}, x_{t+4}$).
3. **No Content-Aware Boundaries**: Boundaries are placed at fixed index steps $S, 2S, 3S \dots$ rather than at natural price inflection points (pivots, peaks, troughs).

---

## 5. Synthesis & Comparison with Our Research Goal

| Dimension | PatchTST (Baseline) | Our Proposed Multi-Scale Tokenizer |
|---|---|---|
| **Patch Length** | Fixed ($P=16$) | Dynamic & Variable ($w \in \{4, 8, 16, 32\}$) |
| **Boundary Placement** | Uniform index stride $S$ | Content-aware inflection & multi-scale overlapping |
| **Receptive Field** | Contiguous only | Multi-scale & Dilated ($d \in \{1, 2, 4\}$) |
| **Vocabulary** | Continuous linear projection | Discretized symbolic dictionary / Learned codebook |
| **Financial Fit** | Sub-optimal for regime shifts | Tailored for non-stationary market regimes |
