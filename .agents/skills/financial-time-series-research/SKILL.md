---
name: financial-time-series-research
description: Specialized research methodology and benchmarking guidelines for financial pattern discovery, Matrix Profile motif diagnostics, and walk-forward backtesting protocols.
---

# Financial Time Series Research & Benchmarking Skill

This skill provides comprehensive instructions for conducting rigorous quantitative research on financial time series models.

---

## 1. Literature Benchmark Reference Matrix

When evaluating novel tokenization ideas, benchmark against the 5 canonical literature baselines:

| Baseline | Paper Reference | Key Mechanism | Primary Failure Mode in Finance |
|---|---|---|---|
| **PatchTST** | Nie et al. (ICLR 2023) | Uniform fixed patches ($P=16$) + ViT | Boundary clipping; zero multi-scale adaptation. |
| **TimesNet** | Wu et al. (ICLR 2023) | FFT 2D Period Reshaping + 2D Conv | Assumes stationary periodicity (fails on market regime shifts). |
| **BORF** | Spinnato et al. (IEEE 2024) | Receptive Fields + 1D-SAX BoW | Unweighted Bag-of-Words loses temporal sequence order. |
| **VALMOD** | Linardi et al. (SIGMOD 2018) | Multi-length Matrix Profile | Exact distance search; non-predictive diagnostic. |
| **Dynamic Patching** | Mid-2026 Benchmarks | Adaptive boundary discovery | Noise over-segmentation & loss of uniform batching. |

---

## 2. Matrix Profile & VALMOD Motif Diagnostics

Before building machine learning models on a new asset:
1. Compute Matrix Profile distances $P$ across window scales $m \in [5, 100]$.
2. Compute normalized mean distance per point: $d_{norm} = d / \sqrt{m}$.
3. Verify motif existence:
   - $d_{norm} < 0.5 \implies$ Strong repeating motif present.
   - $d_{norm} > 0.9 \implies$ Near-random walk (no exact shape motif).
4. Observation: Equity returns consistently display strong motif density at short scales ($m \in [5, 20]$), confirming that tokenization must focus on micro/medium receptive fields.

---

## 3. Strict Walk-Forward Evaluation Protocol

All financial experiments MUST adhere to the following 4 rules:

```
[2010 - 2015 Train] ──> Test 2016
[2010 - 2016 Train] ─────────> Test 2017
[2010 - 2017 Train] ──────────────> Test 2018
... (Expanding Window until 2026)
```

1. **Expanding Window**: Train model on $[T_0, T_{test}-1]$, evaluate on $T_{test}$.
2. **Zero Lookahead**: Scaling parameters (Z-score mean/std, SAX breakpoints) must be fitted **only on the training split**.
3. **Transaction Costs**: Enforce 5 bps (0.05%) execution slippage per trade.
4. **Metrics**: Report Directional Accuracy (%), AUC-ROC, Out-of-Sample Sharpe Ratio, Sortino Ratio, and Max Drawdown.
