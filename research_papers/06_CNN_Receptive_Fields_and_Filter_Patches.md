# CNN Receptive Fields, Overlapping Filters, & 2D Financial Patches

> **Theoretical Foundations**: Connecting Computer Vision (2D Convolutional Filters) to Financial Quantitative Pattern Recognition.

---

## 1. The Core Metaphor: Quant Visual Intuition

Human quantitative traders rarely look at market prices as raw 1D numbers ($100.1, 100.3, 100.2 \dots$). Instead, quants view market charts **visually as 2D spatial images**:
- A chart shows price on the vertical Y-axis and time on the horizontal X-axis.
- Human eyes spot **2D visual primitives**: "head and shoulders", "double bottoms", "cup and handle", "volatility squeeze", "bullish engulfing".

In Computer Vision, **2D Convolutional Filters** process images by sliding small spatial receptive fields (e.g., $3 \times 3$ or $7 \times 7$ kernels) across the image tensor:

```
[2D Conv Filter]  ->  Detects Edges, Corners, Curves  ->  Higher-level Object Features
```

Our core thesis translates this to financial time series:
**Can we treat time series subsequences as 2D overlapping patches / filters that detect structural financial primitives?**

---

## 2. Receptive Field Theory: 1D vs. 2D

### 1D Signal Receptive Field
In a 1D sequence $X \in \mathbb{R}^T$, a 1D convolution kernel $k \in \mathbb{R}^w$ with stride $s$ and dilation $d$ has an effective receptive field:
$$\text{RF}_{1D} = (w - 1) \cdot d + 1$$

- **Dilation ($d > 1$)**: Allows the filter to look at wide temporal windows without filling every intermediate sample, preserving parameter count while expanding receptive context.

### 2D Time-Feature Matrix Representation
To unlock full 2D CNN capabilities, we transform univariate or multivariate financial series into a 2D Feature-Time Matrix $M \in \mathbb{R}^{F \times T}$:
- **Row dimension ($F$)**: Multiple feature representations (e.g., Row 1 = Returns, Row 2 = Log Volatility, Row 3 = Volume Change, Row 4 = High-Low Spread).
- **Column dimension ($T$)**: Time steps.

A 2D Convolutional Filter $K \in \mathbb{R}^{f \times w}$ sliding across $M$ operates simultaneously over **feature channels ($f$)** and **temporal window ($w$)**.

---

## 3. Mathematical Equivalence: Overlapping Patches = Convolutional Stride

Consider two patch tokenization approaches:

1. **Non-Overlapping Patching** (PatchTST with $S = P$):
   - Patches: $[x_1 \dots x_{16}], [x_{17} \dots x_{32}], [x_{33} \dots x_{48}]$
   - Equivalent to a 1D Conv layer with kernel size $w=16$ and **stride $s=16$**.
   - **Problem**: Shifts in pattern timing by just 1 bar completely change patch token assignments!

2. **Dense Overlapping Patching** (BORF / Stride $s=1$):
   - Patches: $[x_1 \dots x_{16}], [x_2 \dots x_{17}], [x_3 \dots x_{18}]$
   - Equivalent to a 1D Conv layer with kernel size $w=16$ and **stride $s=1$**.
   - **Advantage**: Perfect **translation invariance**. A head-and-shoulders pattern is recognized regardless of whether it begins on Monday or Tuesday.

---

## 4. Multi-Scale Receptive Fields: The Feature Pyramid Network (FPN) Analogy

In computer vision, Feature Pyramid Networks (FPN) detect objects of varying sizes (small birds vs. huge airplanes) by applying filters across multiple resolution scales.

In financial series:
- **Small Receptive Fields ($w=4, d=1$)**: Detect micro-structure spikes, order flow imbalances, intra-hour reversals.
- **Medium Receptive Fields ($w=16, d=1$)**: Detect daily swing patterns, momentum consolidation.
- **Dilated Receptive Fields ($w=16, d=4$)**: Detect multi-week macro regimes, business cycle transitions.

By extracting **multi-scale overlapping receptive fields**, our tokenizer creates a **Financial Feature Pyramid**—providing downstream models with simultaneous micro and macro visual primitives!
