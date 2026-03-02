# fishery-gov-sim

Minimal sequential fishery commons simulator for studying cooperation stability.

## Research Focus
This repo is scoped to the following question:

Are cooperative norms in a renewable commons evolutionarily stable under repeated adversarial strategy injection, and which governance signals improve resistance to invasion?

The project is intentionally simple:
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
- `fishery_sim/benchmarks.py`: fixed held-out benchmark packs (`easy_v1`, `medium_v1`, `hard_v1`, `heldout_v1`, `mixed_v1`, `harsh_v1`, `harsh_v2`)
- `experiments/run_single.py`: one rollout sanity check
- `experiments/run_sweep.py`: seed sweep for fixed population metrics
- `experiments/run_greedy_sweep.py`: composition tipping-point scan
- `experiments/run_invasion.py`: multi-generation invasion with train/test regime split
- `experiments/run_governance_ablation.py`: publishable governance ablation with confidence intervals
- `experiments/showcase_project.py`: auto-generated narrative report + evidence summary
- `experiments/check_llm_setup.py`: backend check for replay/ollama/openai JSON policy output
- `experiments/make_invasion_gif.py`: animated invasion dashboard GIF generator
- `experiments/make_episode_gif.py`: animated per-step fishery episode GIF generator
- `experiments/make_governance_comparison_gif.py`: animated side-by-side none vs monitoring+sanctions comparison GIF
- `experiments/run_visual_governance_pair.py`: one-command matched-seed visual pipeline for slide-ready governance GIFs
- `experiments/organize_results.py`: buckets results into `curated`, `exploratory`, and `scratch` under each run type
- `experiments/summarize_tiered_ablation.py`: summarizes tiered (`easy/medium/hard`) governance tables into one report
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

Progress bars are enabled by default for long runs. Use `--no-progress` to disable.

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
  --llm-provider openai \
  --llm-model gpt-4.1-mini \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_live
```

Run LLM JSON policy injection with **free local Ollama** (no paid key):

```bash
# Install Ollama once from https://ollama.com/download
# Start server (if not already running from Ollama app)
ollama serve

ollama pull qwen2.5:3b-instruct

python -m experiments.check_llm_setup \
  --provider ollama \
  --model qwen2.5:3b-instruct

python -m experiments.run_invasion \
  --injector-mode llm_json \
  --llm-provider ollama \
  --llm-model qwen2.5:3b-instruct \
  --benchmark-pack harsh_v1 \
  --output-prefix results/runs/invasion/invasion_ollama

# Optional: stop server when done
pkill -f "ollama serve"
```

Compare matched `mutation` vs live Ollama `llm_json` runs:

```bash
python -m experiments.summarize_injector_comparison \
  --mutation-prefix results/runs/invasion/invasion_step4_match_mutation \
  --llm-prefix results/runs/invasion/invasion_step4_match_ollama_live \
  --output-prefix results/runs/invasion/invasion_step4_match_comparison
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
  --benchmark-pack heldout_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --output-prefix results/runs/ablation/governance_ablation
```

Run full-capacity tiered mutation ablations (`easy`, `medium`, `hard`):

```bash
python -m experiments.run_governance_ablation \
  --benchmark-pack easy_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --test-seeds-per-generation 64 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --injector-mode mutation \
  --output-prefix results/runs/ablation/tiered_mutation_easy_v1

python -m experiments.run_governance_ablation \
  --benchmark-pack medium_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --test-seeds-per-generation 64 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --injector-mode mutation \
  --output-prefix results/runs/ablation/tiered_mutation_medium_v1

python -m experiments.run_governance_ablation \
  --benchmark-pack hard_v1 \
  --n-runs 5 \
  --generations 30 \
  --seeds-per-generation 64 \
  --test-seeds-per-generation 64 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --injector-mode mutation \
  --output-prefix results/runs/ablation/tiered_mutation_hard_v1
```

Run live-Ollama medium-tier governance ablation:

```bash
python -m experiments.run_governance_ablation \
  --benchmark-pack medium_v1 \
  --n-runs 2 \
  --generations 10 \
  --seeds-per-generation 24 \
  --test-seeds-per-generation 24 \
  --replacement-fraction 0.2 \
  --adversarial-pressure 0.3 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --injector-mode llm_json \
  --llm-provider ollama \
  --llm-model qwen2.5:3b-instruct \
  --output-prefix results/runs/ablation/tiered_llm_medium_v1
```

Summarize tier runs:

```bash
python -m experiments.summarize_tiered_ablation \
  --ablation-dir results/runs/ablation \
  --output-prefix results/runs/ablation/tiered_ablation_summary
```

Run a matched governance-conditioned injector comparison (`none` vs `monitoring+sanctions`, mutation vs live Ollama):

```bash
python -m experiments.run_governance_ablation \
  --benchmark-pack heldout_v1 \
  --n-runs 2 \
  --generations 10 \
  --seeds-per-generation 24 \
  --test-seeds-per-generation 24 \
  --replacement-fraction 0.2 \
  --adversarial-pressure 0.3 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --run-seed-stride 1000 \
  --injector-mode mutation \
  --output-prefix results/runs/ablation/governance_match_step4_mutation

python -m experiments.run_governance_ablation \
  --benchmark-pack heldout_v1 \
  --n-runs 2 \
  --generations 10 \
  --seeds-per-generation 24 \
  --test-seeds-per-generation 24 \
  --replacement-fraction 0.2 \
  --adversarial-pressure 0.3 \
  --train-regen-rate 2.0 \
  --train-obs-noise-std 6 \
  --run-seed-stride 1000 \
  --injector-mode llm_json \
  --llm-provider ollama \
  --llm-model qwen2.5:3b-instruct \
  --output-prefix results/runs/ablation/governance_match_step4_ollama_live

python -m experiments.summarize_governance_injector_match \
  --mutation-table results/runs/ablation/governance_match_step4_mutation_table.csv \
  --llm-table results/runs/ablation/governance_match_step4_ollama_live_table.csv \
  --output-prefix results/runs/ablation/governance_match_step4_injector_split
```

Organize results after experiments (recommended):

```bash
python -m experiments.organize_results --results-dir results --apply
```

Result folder convention:
- `results/runs/invasion/curated`: slide/report-ready invasion artifacts
- `results/runs/invasion/exploratory`: useful but non-final runs
- `results/runs/invasion/scratch`: smoke/tmp/check runs
- same three buckets for `ablation`, `showcase`, and `baselines`

Generate showcase report:

```bash
python -m experiments.showcase_project \
  --ablation-table results/runs/ablation/governance_ablation_table.csv \
  --invasion-generations results/runs/invasion/invasion_baseline_generations.csv \
  --output results/runs/showcase/showcase_report.md
```

Generate animated GIFs for demos/slides:

```bash
# Optional: generate a less-saturated visual benchmark first
python -m experiments.run_invasion \
  --generations 10 \
  --seeds-per-generation 24 \
  --test-seeds-per-generation 24 \
  --train-regen-rate 1.5 \
  --train-obs-noise-std 10 \
  --benchmark-pack heldout_v1 \
  --output-prefix results/runs/invasion/invasion_visual_benchmark

python -m experiments.make_invasion_gif \
  --input results/runs/invasion/invasion_visual_benchmark_generations.csv \
  --output results/runs/showcase/invasion_train_vs_heldout_dynamics.gif

python -m experiments.make_episode_gif \
  --output results/runs/showcase/episode_stock_harvest_dynamics.gif
```

Interpretation tip: if every held-out bar is exactly `1.0`, the benchmark pack is saturated (all regimes collapse), which is a regime severity issue rather than a plotting bug.

Generate a SocialJax-style visual comparison pack (matched seeds, moving side-by-side):

```bash
python -m experiments.run_visual_governance_pair
```

This writes:
- `results/runs/showcase/invasion_none_baseline.gif`
- `results/runs/showcase/invasion_monitoring_sanctions.gif`
- `results/runs/showcase/invasion_none_vs_monitoring_sanctions.gif`

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
- robustness timing metrics:
  - `time_to_collapse` (first generation where test collapse reaches 1.0; right-censored at horizon)
  - `first_generation_test_collapse_ge_0_8` (first generation where test collapse reaches 0.8; right-censored at horizon)
  - `per_regime_survival_over_generations_*` (mean survival fraction `1 - collapse_rate`)

These metrics are written to:
- `results/runs/invasion/*_generations.csv`
- `results/runs/invasion/*_strategies.csv`
- `results/runs/ablation/*_runs.csv`
- `results/runs/ablation/*_table.csv`
- `results/runs/ablation/*_survival_curves.csv`
- `results/runs/showcase/*.md`
- `results/runs/showcase/*.gif`
