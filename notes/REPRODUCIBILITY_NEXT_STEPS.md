# Reproducibility Next Steps

## Core source files

The core source files for the current paper are:

- `fishery_sim/harvest.py`: Harvest environment, strategy format, and episode mechanics.
- `fishery_sim/harvest_evolution.py`: structured strategies, mutation, search-generated entrants, and population turnover support.
- `fishery_sim/harvest_benchmarks.py`: scenario presets, actor-capability presets, overseer-capability presets, and capability-gap definitions.
- `experiments/run_harvest_invasion.py`: single Harvest invasion run entry point.
- `experiments/run_harvest_invasion_matrix.py`: matrix runner for Stage A and related sweeps.
- `experiments/run_targeted_threshold_replay.py`: threshold replay runner.
- `experiments/analyze_threshold_replay_grid.py`: threshold replay analysis.
- `experiments/audit_threshold_sweep_completeness.py`: full threshold sweep completeness audit.
- `experiments/run_overseer_limit_ablation.py`: reduced/full overseer-limit ablation launcher.
- `experiments/analyze_overseer_limit_ablation.py`: overseer-limit ablation analysis.
- `experiments/audit_overseer_limit_ablation.py`: overseer-limit ablation completeness audit.
- `experiments/plot_scalable_oversight_paper_v5.py`: paper figure/table generator.
- `paper/paper_v5_scalable_oversight_commons/main.tex`: current manuscript.

## Generated files

The following are generated and should not be treated as hand-authored source:

- `paper/paper_v5_scalable_oversight_commons/main.pdf`;
- files under `paper/paper_v5_scalable_oversight_commons/figures/`;
- files under `paper/paper_v5_scalable_oversight_commons/tables/`;
- most files under `results/`;
- LaTeX auxiliary files such as `.aux`, `.bbl`, `.blg`, `.log`, `.out`.

Generated paper artifacts can be kept for supervisor review, but the repo should document how to rebuild them.

## Result files needed to reproduce current figures and tables

Stage A:

- `results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_stageA_ranking.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv`
- `results/runs/showcase/curated/harvest_oversight_gap_stageA_threshold_sensitivity.csv`
- `results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv`
- `results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_runs.csv`

Full threshold sweep:

- `results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv`
- `results/runs/threshold_replay/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered_runs.csv`
- `results/artifacts/github/harvest_oversight_gap_threshold_replay_full_grid_recovered-bundle.zip`
- `notes/THRESHOLD_SWEEP_COMPLETENESS_AUDIT.md`

LLM bridge:

- Qwen bank, summary, map summary, and map samples under `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_*`
- Llama bank, summary, map summary, and map samples under `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_*`

Overseer ablation:

- reduced raw runs: `results/runs/overseer_limit_ablation/reduced_ablation_runs.csv`
- reduced summary: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced.csv`
- reduced audit: `results/runs/showcase/curated/harvest_overseer_limit_ablation_reduced_audit.md`
- full files should use the same naming pattern with `_full` if the full ablation is run later.

## Commands to rebuild current paper artifacts

From the repository root:

```bash
python -m experiments.analyze_harvest_oversight_stageA \
  --table-csv results/runs/showcase/curated/harvest_oversight_gap_stageA_table.csv \
  --case-trace-csv results/runs/showcase/curated/harvest_oversight_gap_stageA_oversight_case_trace.csv \
  --strategy-history-csv results/runs/harvest_invasion/curated/harvest_oversight_gap_stageA_strategy_history.csv \
  --output-prefix results/runs/showcase/curated/harvest_oversight_gap_stageA

python -m experiments.analyze_threshold_replay_grid \
  --input-csv results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered.csv \
  --output-prefix results/runs/showcase/curated/harvest_oversight_gap_threshold_replay_full_grid_recovered

python -m experiments.audit_threshold_sweep_completeness

MPLCONFIGDIR=/private/tmp/mplconfig python -m experiments.plot_scalable_oversight_paper_v5
```

From `paper/paper_v5_scalable_oversight_commons/`:

```bash
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## What still needs repo cleanup

- Decide which notes are supervisor-facing and which are temporary working notes.
- Decide whether generated PDFs and paper figures should be committed or regenerated in release scripts.
- Add a top-level paper reproduction README once the final experiment set is fixed.
- Package the recovered result bundle in a release artifact or documented external storage location, since `results/` is ignored.
- Remove or archive obsolete proposal/interview files only after manual confirmation.
- Keep GitHub Actions workflow documentation separate from the scientific method; it is compute infrastructure.

## What should eventually be committed

Commit source and documentation needed for reproducibility:

- core experiment scripts;
- analysis scripts;
- audit scripts;
- current manuscript source;
- bibliography;
- paper README;
- supervisor-ready notes that explain direction, validation status, and reproducibility;
- lightweight generated paper artifacts if useful for review.

## What should remain ignored

Keep large or machine-generated files ignored unless deliberately packaged:

- full `results/` trees;
- raw large run histories unless curated as release artifacts;
- temporary LaTeX build files;
- local cloud/runpod payloads or credentials;
- local model downloads;
- personal interview/proposal drafts that are not part of the paper.

## Immediate cleanup priority

Do not aggressively clean the repo right now. The immediate priority is to keep the paper reproducible from known curated inputs and to make the next ablation command/audit path clear. A larger cleanup should happen after supervisor feedback decides whether the next deliverable is a paper submission, a benchmark release, or another experimental pass.
