# Fishery Gov Sim - Full Chat Summary (Plain-Speak Version)

## TL;DR in one paragraph
I started with a minimal fishery commons sim, then moved it toward the actual thesis gap: not just "can cooperation emerge?", but "does cooperation survive repeated adversarial strategy invasion?" We now have a working invasion loop, governance controls, train/test regime split, fixed harsh benchmark packs, LLM-to-policy adapter (replay + live OpenAI client path), CI-based governance ablations, a showcase report generator, and cleaned results folders so experiments are reproducible and easy to present.

---

## What the project is trying to answer
Main question:

**Are cooperative norms in a renewable commons evolutionarily stable under adversarial strategy generation pressure, and which governance signals make them more resistant?**

So this is not a basic "cooperation appears in one setup" project.  
It is a **pressure-test** project under distribution shift + invasion.

---

## What we found when auditing the repo
Initial diagnosis:
- Good: minimal fishery environment + collapse dynamics + baseline sweeps already worked.
- Not good: the core novelty piece (`evolutionary invasion`) was missing/incomplete at first.
- Not good: some scripts were brittle, configs incomplete, and outputs messy in folder structure.

We fixed those directly.

---

## What has been built/fixed so far

### 1) Core environment + governance
- Fishery dynamics already existed and were kept minimal (good for causal interpretation).
- Added governance knobs and wired them through runs:
  - monitoring probability
  - quota fraction
  - sanctions (base fine + growth)
- Enforcement now impacts extraction pressure (not just accounting), so governance has real ecological effect.

### 2) Evolutionary invasion engine
- Implemented repeated selection + replacement loop:
  - evaluate current population
  - remove lower-fitness strategies
  - inject new invader strategies
  - continue for many generations
- Tracks collapse, welfare, stock, inequality, sanctions, diversity, aggressiveness trends.

### 3) LLM strategy adapter
- Added adapter path: **prompt -> strict JSON policy -> executable strategy**.
- Supports:
  - replay file client (deterministic, offline testing)
  - live OpenAI Responses API client (online generation path)
- This keeps the injector interface stable while swapping strategy source.

### 4) Train/test generalization split
- Selection happens on train regime.
- Evaluation includes held-out test regime(s), per generation.
- Supports multi-regime held-out packs (not just one test condition).

### 5) Fixed held-out benchmark packs
- Added built-in harsh packs (e.g. `harsh_v1`, `harsh_v2`) with named regimes.
- Also supports loading custom pack YAML files.
- Invasion outputs include per-regime metrics like `test_<regime>_collapse_rate`.

### 6) Governance ablation with confidence intervals
- New ablation script compares:
  - `none`
  - `monitoring`
  - `monitoring_sanctions`
- Runs multiple independent runs and reports **95% CI** per key metric.
- Generates:
  - per-run CSV
  - summary table CSV
  - markdown table for reports/papers/slides

### 7) Showcase/report script
- Auto-generates a markdown narrative:
  - what problem we’re solving
  - why design choices are made
  - current evidence + ablation table
- Written in direct, explainable language for meetings.

### 8) Notebook progression
- Added `02_invasion_benchmark_pack_and_ci.ipynb`
- It documents phase-2 workflow and plotting/reporting patterns.

### 9) Results folder cleanup
- Added organizer script to move loose CSV/MD outputs into structured paths:
  - `results/runs/invasion`
  - `results/runs/ablation`
  - `results/runs/baselines`
  - `results/runs/showcase`
- Applied cleanup so outputs are now structured.

---

## What is going right
- Project is now aligned with the actual novelty claim (invasion stability, not just emergence).
- Reproducibility is much better (standardized scripts + structured outputs + benchmark packs).
- We can run deterministic offline mode and a live LLM mode via same interface.
- Ablations now produce CI-backed tables, so claims are less anecdotal.

---

## What is still weak / what to be careful about
- Some harsh packs currently drive near-certain collapse across conditions; good as stress test, but weak for distinguishing mechanisms.
- Need calibrated benchmark difficulty tiers so you can show meaningful separation between governance options.
- Live LLM path is implemented, but full-scale live runs depend on API setup/cost control and should be logged carefully.
- Statistical reporting still mostly CI-only; likely need hypothesis tests/effect sizes later depending on reviewer/supervisor expectations.

---

## "How do I explain this in a meeting?" (talk track)
Use this:

> "I’ve moved the project from basic cooperation emergence to stability under invasion pressure. We now have an evolutionary strategy-turnover loop, train/test regime split, and fixed harsh benchmark packs. Governance is treated as modular intervention, and we evaluate none vs monitoring vs monitoring+sanctions with repeated runs and 95% CIs. I also built an LLM-to-policy adapter so invaders can be generated from live model outputs, not just hand-coded mutations."

Then if they ask "what’s the contribution?" say:

> "The contribution is a pressure-test protocol for commons cooperation: adversarial strategy injection + held-out regime evaluation + governance robustness measurement with uncertainty estimates."

---

## Quick command cheatsheet (the useful ones)

```bash
conda activate fishery
```

Run invasion with benchmark pack:

```bash
python -m experiments.run_invasion \
  --injector-mode mutation \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_baseline
```

Run LLM injection (live, requires key):

```bash
export OPENAI_API_KEY=...
python -m experiments.run_invasion \
  --injector-mode llm_json \
  --llm-model gpt-4.1-mini \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_live
```

Run governance ablation with CIs:

```bash
python -m experiments.run_governance_ablation \
  --benchmark-pack harsh_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --output-prefix results/runs/ablation/governance_ablation
```

Generate explainer report:

```bash
python -m experiments.showcase_project \
  --ablation-table results/runs/ablation/governance_ablation_table.csv \
  --invasion-generations results/runs/invasion/invasion_baseline_generations.csv \
  --output results/runs/showcase/showcase_report.md
```

---

## Next best steps (practical)
1. Add a **difficulty-calibrated benchmark pack** (`easy`, `medium`, `hard`) so governance comparisons separate clearly.
2. Run higher-powered sweeps (more `n_runs`, seeds, generations) for publishable confidence.
3. Add one compact analysis script to compute effect sizes and rank governance by robustness + welfare + fairness jointly.
4. If supervisors ask for novelty clarity: frame the method as a **stability-under-invasion evaluation protocol** for commons governance.

---
