# Repo Cleanup Report

Generated after inspecting `git status --short` and `git diff --stat`.

## Summary

The repository contains a mix of substantive source changes, paper source changes, paper deliverables, generated build artifacts, and notes from proposal/interview work. The safe cleanup policy is to remove only reproducible cache/build clutter and preserve all source files, paper source files, final PDFs, figures, tables, scripts, raw logs, and curated result CSVs.

## Classification

| Group | Files or examples | Decision | Rationale |
| --- | --- | --- | --- |
| Source code changes | `experiments/build_harvest_strategy_bank.py`, `experiments/extract_harvest_oversight_case.py`, `experiments/run_harvest_invasion.py`, `experiments/run_harvest_invasion_matrix.py`, `fishery_sim/harvest.py`, `fishery_sim/harvest_llm_population.py`, `fishery_sim/llm_adapter.py`, tests | Keep | These are substantive implementation changes for the scalable-oversight and LLM bridge work. |
| New analysis/plot scripts | `experiments/analyze_harvest_oversight_stageA.py`, `experiments/plot_scalable_oversight_paper_v5.py` | Keep | These are needed for reproducibility and paper figure generation. |
| Paper source changes | `paper/paper_v5_scalable_oversight_commons/main.tex`, `refs.bib`, `README.md` | Keep | These are the current paper source and bibliography. |
| Generated figures | `paper/paper_v5_scalable_oversight_commons/figures/fig01_*` through `fig06_*` | Keep | These are paper-facing outputs. They should be reproducible from the plotting script and listed in the manifest. |
| Generated tables | `paper/paper_v5_scalable_oversight_commons/tables/*.tex` | Keep | These are paper-facing table outputs generated from result CSVs. |
| Generated PDFs | `paper/paper_v5_scalable_oversight_commons/main.pdf`, `notes/Ameer-Alhashemi-Proposal-210426.pdf`, `notes/scalable_oversight_next_plan.pdf`, interview preview PDF | Keep or user decision | Current paper PDF should be kept. Proposal/interview PDFs are deliverables but may not belong in a paper-focused commit. |
| CSV/result artifacts | `results/runs/showcase/curated/*`, `results/runs/harvest_invasion/curated/*` | Keep; do not delete | These are ignored by git but required for reproducibility. Raw and curated result files must be preserved. |
| LaTeX build artifacts | `*.aux`, `*.bbl`, `*.blg`, `*.log`, `*.out` | Delete if untracked; user decision if tracked | Untracked build products are reproducible clutter. Tracked build files in older paper folders should only be removed through an explicit cleanup commit. |
| Python/cache artifacts | `__pycache__/`, `*.pyc`, `.pytest_cache/` | Delete as disposable artifact | Reproducible runtime cache. |
| Logs | LaTeX logs such as `paper/*/main.log`, `notes/*.log` | Delete if untracked | Build logs are disposable. Experimental/raw logs are not included in this deletion rule. |
| Ambiguous notes/proposal files | `notes/CLAIM_LEDGER.md`, `notes/CONTRIBUTION_REORIENTATION.md`, `notes/FIGURE_STYLE_GUIDE.md`, `notes/SUPERVISOR_NOTE.md`, `notes/*closeout.md`, `notes/interview_presentation/*`, `tmp_supervisor_meeting_brief_2026-03-31.md` | User decision needed for staging | These may be valuable project notes but are not all paper-reproducibility inputs. |
| Deleted tracked file | `notes/Ameer-Alhashemi-Proposal-140426.pdf` | User decision needed | This is a tracked deletion. I did not restore or remove further because it may reflect intentional user cleanup. |
| Cloud/run helper scripts | `scripts/*.sh`, `requirements-cloud.txt` | Keep; user decision for staging | Useful for cloud LLM generation, but separate from the core paper reproducibility path. |

## Safe Cleanup Actions Performed

Pending during report creation:

- Add `.gitignore` rules for Python/test caches and LaTeX build products.
- Remove untracked LaTeX build artifacts.
- Remove `__pycache__` directories and `.pyc` files.

## Preserved Because Ambiguous Or Important

- All result CSVs and raw/curated result logs under `results/`.
- All paper source files, bibliography files, figures, tables, and final PDFs.
- All substantive code changes.
- All project notes and proposal/interview artifacts, except disposable build products.

## Recommended Staging Scope Later

For a clean paper/reproducibility commit, stage:

- relevant source scripts under `experiments/`, `fishery_sim/`, `tests/`;
- `paper/paper_v5_scalable_oversight_commons/main.tex`, `refs.bib`, `README.md`, `figures/`, `tables/`;
- reproducibility notes: `REPO_CLEANUP_REPORT.md`, `PAPER_REPRODUCIBILITY_MANIFEST.md`, `FIGURE_TABLE_DECISION_LOG.md`, `VALIDATION_STATUS.md`, and uncertainty/literature TODOs.

Do not stage proposal/interview artifacts unless the commit is meant to archive broader PhD materials.
