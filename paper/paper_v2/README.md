# paper_v2 writing scaffold

This folder is the writing-first scaffold for converting `paper_v1` curated evidence into a thesis/paper draft.

## Build
Use your preferred LaTeX engine from this directory, for example:

```bash
cd paper/paper_v2
pdflatex main.tex
```

For bibliography + stable references:

```bash
cd paper/paper_v2
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Regenerate paper_v2 contextual/mechanism artifacts used in the manuscript:

```bash
cd /Users/ameerfiras/invasion-commons/invasion-commons
python -m experiments.generate_paper_v2_artifacts
```

Regenerate the Fishery RL validation figure used in the manuscript:

```bash
cd /Users/ameerfiras/invasion-commons/invasion-commons
python -m experiments.plot_fishery_rl_paper
```

## Canonical evidence sources
- `results/runs/showcase/curated/paper_v1_main_results.md`
- `results/runs/showcase/curated/paper_v1_methods.md`
- `results/runs/showcase/curated/paper_v1_gates.json`
- `results/runs/ablation/curated/paper_v1_table1_main_deltas.csv`
- `results/runs/ablation/curated/paper_v1_table3_medium_effect_size_ranking.csv`

## Scope discipline
- Avoid adding new mechanisms in this cycle.
- Keep hard tier in appendix.
- Update claims only when supported by CI-backed artifacts.
