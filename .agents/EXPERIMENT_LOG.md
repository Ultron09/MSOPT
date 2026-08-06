# MSOPT EXPERIMENT LOG & EMPIRICAL MEMORY

This document is the official experiment memory log for the Multi-Scale Overlapping Pattern Tokenization (MSOPT) project. 

> **RULE**: Only verified, executed experiment runs with zero synthetic shortcuts are logged in this file.

---

## Workspace Implementation & Benchmark Status

| Component | File Path | Status | Protocol / Notes |
|---|---|---|---|
| **Data Preprocessor** | `src/data/preprocessing.py` | Completed | Authentic yfinance daily OHLCV loader for SPY, QQQ, AAPL, TLT. |
| **MSOPT Tokenizer** | `src/tokenizer/msopt_tokenizer.py` | Completed | Multi-scale 1D-SAX ($w \in \{4,8,16,32\}, d \in \{1,2,4\}, s=1$). |
| **PyTorch MSOPT Engine** | `src/models/msopt_engine.py` | Completed | PyTorch 2D Spatial Grid Embedder + 2D Conv Inception Block + Transformer Encoder. |
| **Bibliography Base** | `paper/references.bib` | Completed | Verified BibTeX database with authentic citations (Nie et al., Wu et al., Spinnato et al., Lin et al., Yeh et al.). |

---

## Experiments Log

*(No experimental runs logged yet. Real walk-forward evaluations will be appended here as they complete.)*
