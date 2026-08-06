---
name: continuous-learning
description: Automated protocol for self-updating agent memory files, experiment logs, codebase implementation maps, and anti-hallucination ledgers as new code stages are completed.
---

# Continuous Learning & Self-Evolution Skill

This skill enforces continuous memory updating and self-evolution for AI agents working on this project.

---

## 1. When to Trigger Self-Updates

An agent MUST trigger a self-update of `.agents/AGENTS.md` and `.agents/EXPERIMENT_LOG.md` upon any of the following events:

1. **New File / Module Created**: When a new script or package module (e.g., `src/models/msopt_engine.py`) is added to the codebase.
2. **Experiment Completed**: When an empirical experiment, walk-forward backtest, or ablation study finishes.
3. **Bug / Failure Mode Discovered**: When a silent bug, data leakage issue, or library pitfall is identified.
4. **Architectural Change**: When design parameters (e.g., window sizes $w$, 1D-SAX alphabets $a_{\mu}, a_{\beta}$) are altered based on empirical evidence.

---

## 2. Update Checklist for Agents

### Step 1: Update `.agents/AGENTS.md`
- Locate Section 3 (**Codebase Implementation Map & Live Status**).
- Update the status and key findings for existing modules or add new rows for new modules.
- If a bug or pitfall was encountered, append it to Section 4 (**Key Lessons & Known Failure Modes**).

### Step 2: Update `.agents/EXPERIMENT_LOG.md`
- Assign a sequential ID (`EXP-001`, `EXP-002`, `EXP-003`...).
- Log the Date, Tickers, Method, Parameters, Metrics (OOS Accuracy, Sharpe Ratio, Drawdown), and Action Items.
- Add a detailed sub-section outlining the hypothesis and outcome.

### Step 3: Anti-Hallucination Check
Before answering user prompts or writing code:
- Read `.agents/EXPERIMENT_LOG.md` to ensure past failed approaches are not re-suggested.
- Verify active codebase state in `.agents/AGENTS.md`.
