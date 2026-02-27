# fishery-gov-sim

Minimal sequential fishery commons simulator for studying cooperation stability.

## Research Focus
This repo is scoped to the following question:

Are cooperative norms in a renewable commons evolutionarily stable under repeated adversarial strategy injection, and which governance signals improve resistance to invasion?

The project is intentionally "golf-sim" simple:
- one global renewable stock
- harvest actions by multiple agents
- collapse threshold with patience window
- optional noisy stock observations
- optional monitoring + quota enforcement + graduated sanctions

## Project Layout
- `fishery_sim/env.py`: environment dynamics and collapse logic
- `fishery_sim/agents.py`: baseline heuristic policies
- `fishery_sim/simulation.py`: episode runner
- `fishery_sim/evolution.py`: evolutionary invasion loop and strategy turnover
- `fishery_sim/llm_adapter.py`: prompt -> JSON policy adapter for LLM strategy injection
- `fishery_sim/benchmarks.py`: fixed held-out benchmark packs (`harsh_v1`, `harsh_v2`)
- `experiments/run_single.py`: one rollout sanity check
- `experiments/run_sweep.py`: seed sweep for fixed population metrics
- `experiments/run_greedy_sweep.py`: composition tipping-point scan
- `experiments/run_invasion.py`: multi-generation invasion with train/test regime split
- `experiments/run_governance_ablation.py`: publishable governance ablation with confidence intervals
- `experiments/showcase_project.py`: auto-generated narrative report + evidence summary
- `experiments/organize_results.py`: organizes CSV/MD artifacts into structured `results/runs/*`
- `notebooks/02_invasion_benchmark_pack_and_ci.ipynb`: phase-2 reproducibility and reporting notebook

## Quick Start
Activate environment first:

```bash
conda activate fishery
```

Run a single baseline episode:

```bash
python -m experiments.run_single
```

Run fixed-population sweep:

```bash
python -m experiments.run_sweep
```

Run greedy-composition tipping point sweep:

```bash
python -m experiments.run_greedy_sweep
```

Run evolutionary invasion pressure test:

```bash
python -m experiments.run_invasion \
  --generations 30 \
  --population-size 12 \
  --seeds-per-generation 64 \
  --replacement-fraction 0.3 \
  --adversarial-pressure 0.7 \
  --injector-mode mutation \
  --output-prefix results/runs/invasion/invasion_baseline
```

Run held-out train/test regime split:

```bash
python -m experiments.run_invasion \
  --train-regen-rate 1.5 \
  --train-obs-noise-std 10 \
  --test-regen-rate 1.1 \
  --test-obs-noise-std 25 \
  --output-prefix results/runs/invasion/invasion_regime_split
```

Run LLM JSON policy injection with a **live model** (requires `OPENAI_API_KEY`):

```bash
export OPENAI_API_KEY=...
python -m experiments.run_invasion \
  --injector-mode llm_json \
  --llm-model gpt-4.1-mini \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_live
```

Run LLM JSON policy injection using replay file (offline deterministic):

```bash
python -m experiments.run_invasion \
  --injector-mode llm_json \
  --llm-policy-replay-file experiments/configs/llm_policy_replay.jsonl \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_llm_json_replay
```

Governance defense run:

```bash
python -m experiments.run_invasion \
  --monitoring-prob 0.8 \
  --quota-fraction 0.08 \
  --base-fine-rate 1.5 \
  --fine-growth 0.7 \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_governance
```

Generate governance ablation table (`none` vs `monitoring` vs `monitoring+sanctions`) with confidence intervals:

```bash
python -m experiments.run_governance_ablation \
  --benchmark-pack harsh_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --output-prefix results/runs/ablation/governance_ablation
```

Generate showcase report:

```bash
python -m experiments.showcase_project \
  --ablation-table results/runs/ablation/governance_ablation_table.csv \
  --invasion-generations results/runs/invasion/invasion_baseline_generations.csv \
  --output results/runs/showcase/showcase_report.md
```

Organize/cleanup historical CSV/MD outputs:

```bash
python -m experiments.organize_results --apply
```

## Output Metrics
Generation-level outputs include:
- train and test collapse rate
- train and test mean stock / final stock
- train and test welfare (sum of payoffs)
- train and test payoff inequality (Gini)
- sanction and violation counts
- strategy diversity and aggressiveness trends
- per-regime held-out metrics when using benchmark packs

These metrics are written to:
- `results/runs/invasion/*_generations.csv`
- `results/runs/invasion/*_strategies.csv`
- `results/runs/ablation/*_runs.csv`
- `results/runs/ablation/*_table.csv`
- `results/runs/showcase/*.md`
