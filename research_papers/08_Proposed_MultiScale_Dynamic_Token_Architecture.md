# Master Theoretical Proposal: Multi-Scale Overlapping Pattern Tokenization for Financial Time Series

> **Author**: Airborne Research Initiative  
> **Target Context**: Non-linear, non-stationary financial market pattern discovery  
> **Synthesis**: Combining BORF Receptive Fields, PatchTST Transformers, TimesNet 2D Reshaping, VALMOD Multi-Scale Analysis, and 1D-SAX Discretization.

---

## 1. Executive Summary & Paradigm Shift

Standard time series machine learning forces a false choice:
1. **Serial Point-by-Point Models** (LSTMs, vanilla Transformers): Process individual scalar prices ($x_t$), ignoring local 2D visual structure and suffering quadratic compute bottlenecks.
2. **Rigid Fixed-Patch Models** (PatchTST): Segment series into uniform non-overlapping blocks ($P=16$), obscuring dynamic multi-scale patterns and breaking at arbitrary index boundaries.

### Our Solution: **Multi-Scale Overlapping Pattern Tokenization (MSOPT)**
Instead of linear sequence modeling or fixed uniform patching, **MSOPT** views time series as a **2D dynamic grid of multi-scale overlapping receptive fields**. Market dynamics are discretized into an interpretable, spatial-temporal vocabulary of pattern tokens ("words") spanning multiple scales ($w \in [4, 32]$) and dilations ($d \in [1, 4]$) with maximum overlap ($s=1$).

```
Raw Financial Series (Returns, Volatility, Volume)
                        │
                        ▼
   Multi-Scale Overlapping Receptive Field Extraction
       (w in {4,8,16,32}, d in {1,2,4}, Stride s=1)
                        │
                        ▼
            1D-SAX Discretization Engine
     (Segment Mean a_mu=4, Segment Slope a_beta=3)
                        │
                        ▼
          Sparse Symbolic Vocabulary Codebook
      (Unique Pattern Tokens: "w4_d1_ACBD", "w16_d2_DDCB")
                        │
                        ▼
            2D Scale-Time Spatial Tensor
   [Rows = Scales/Dilations, Columns = Time Index t]
                        │
                        ▼
 2D Spatial Conv / Token-Transformer + Scale Positional Encodings
                        │
                        ▼
        Walk-Forward Predictive Alpha Signal
```

---

## 2. Mathematical Formalization of MSOPT

### Component 1: Multi-Scale Dilated Receptive Fields
For multivariate financial series $\mathbf{X} \in \mathbb{R}^{M \times T}$ (Channels: Log Return, Parkinson Volatility, Relative Volume):

At time $t$, extract a set of receptive fields across scales $\mathbf{W} = \{w_1, w_2, \dots, w_K\}$ and dilations $\mathbf{D} = \{d_1, d_2, \dots, d_J\}$:
$$r_{t, w, d} = \left[ x_t, x_{t+d}, x_{t+2d}, \dots, x_{t+(w-1)d} \right] \in \mathbb{R}^w$$

- **Receptive Field Temporal Coverage**: $S(w, d) = (w - 1)d + 1$
- **Dense Overlap**: Evaluated at every time step $t \in \{1, \dots, T - S(w,d) + 1\}$ with stride $s=1$.

### Component 2: Thresholded 1D-SAX Discretization
To prevent noise amplification during flat market consolidation, calculate receptive field standard deviation $\sigma_r$:

$$\hat{r} = \begin{cases} 
\mathbf{0}_w & \text{if } \frac{\sigma_r}{\sigma_{\mathbf{X}}} < \theta \\
\frac{r - \mu_r}{\sigma_r} & \text{otherwise}
\end{cases}$$

Divide $\hat{r}$ into $l$ temporal segments. For segment $k \in \{1, \dots, l\}$:
1. **Segment Mean ($\mu_k$)**: Quantized via Gaussian breakpoints $\mathcal{B}_{\mu} \to A_{\mu} \in \{\text{A, B, C, D}\}$.
2. **Segment Slope ($\beta_k$)**: Fitted via OLS regression $y = \beta_k \cdot \tau + \alpha$, quantized via slope breakpoints $\mathcal{B}_{\beta} \to A_{\beta} \in \{\text{Down, Flat, Up}\}$.

Token Word $V_{t, w, d} = \text{Concat}\left( (\mu_1, \beta_1), (\mu_2, \beta_2), \dots, (\mu_l, \beta_l) \right)$.

### Component 3: 2D Scale-Time Spatial Grid
For each time step $t$, assemble all extracted multi-scale tokens into a **2D Scale-Time Tensor** $\mathbf{H} \in \mathbb{R}^{K \cdot J \times T \times D_{emb}}$:
- **Vertical Axis ($Y$)**: Scale & Dilation configurations $(w_1 d_1, w_1 d_2, \dots, w_K d_J)$.
- **Horizontal Axis ($X$)**: Time index $t$.
- **Depth ($Z$)**: Dense token embedding vector $\mathbf{e}(V_{t, w, d}) \in \mathbb{R}^{D_{emb}}$.

This transforms the 1D financial series into a **2D Spatial Pattern Image**, where 2D convolutions (TimesNet-style) or 2D spatial attention can simultaneously extract:
- **Intra-scale temporal dynamics** (horizontal convolutions across time $t$).
- **Inter-scale hierarchical composition** (vertical convolutions across receptive field scales $w$).

---

## 4. 2D Positional + Scale + Volatility Embedding Scheme

To ensure downstream Transformers understand both temporal placement and scale resolution:

$$\mathbf{E}(t, w, d) = \mathbf{e}_{token}(V_{t, w, d}) + \mathbf{E}_{pos}(t) + \mathbf{E}_{scale}(w, d) + \mathbf{E}_{vol}(\sigma_t)$$

where:
- $\mathbf{e}_{token}$: Learnable codebook embedding for discrete pattern token $V$.
- $\mathbf{E}_{pos}(t)$: Standard 1D temporal positional encoding (sinusoidal or learned).
- $\mathbf{E}_{scale}(w, d)$: Receptive field scale embedding, informing the model of the token's temporal span.
- $\mathbf{E}_{vol}(\sigma_t)$: Local volatility regime embedding, providing macro context.

---

## 5. Comparative Taxonomy Matrix across All Literature

| Dimension | BORF | PatchTST | TimesNet | VALMOD | **MSOPT (Our Proposal)** |
|---|:---:|:---:|:---:|:---:|:---:|
| **Token Boundaries** | Uniform stride $s$ | Fixed stride $S=P$ | Period-aligned | Exact match | **Multi-Scale Overlapping ($s=1$)** |
| **Receptive Fields** | Dilated $w, d$ | Contiguous $P$ | 2D Fourier grid | Subsequence $m$ | **Dilated Multi-Scale Pyramid** |
| **Symbolic Codebook**| 1D-SAX | None (Linear) | None (Conv) | None (Continuous) | **1D-SAX + Learned Codebook** |
| **2D Spatial Tensor**| No (1D BoW) | No (1D Patch) | Yes (FFT Period) | No | **Yes (Scale $\times$ Time Grid)** |
| **Translation Invariance**| High | Low | Low | High | **Maximum** |
| **Non-Stationary Market Fit**| Medium | Low | Low | Medium | **High (Tailored for Finance)** |

---

## 6. Next Steps: Research Review & Validation Protocol

Now that our theoretical solution is completely formulated:
1. Review all 8 research documents in `research_papers/` to verify mathematical consistency and architectural soundness.
2. Align on the exact design parameters before initiating implementation.
