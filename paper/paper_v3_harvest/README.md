# `paper_v3_harvest` working draft

This folder contains the Harvest-led follow-up draft. It is intentionally separate from `paper_v2`, which stays frozen as the completed two-study baseline paper.

## Purpose

This draft recenters the project around:

- institutional robustness in plural multi-agent systems
- governance architecture rather than only signal design
- capability escalation rather than only fixed cooperation

The current draft already integrates:

- the Harvest architecture-restoration pilot
- the high-power architecture confirmation
- the welfare-incidence analysis
- the capability-ladder analysis

## Build

```bash
cd paper/paper_v3_harvest
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
TEXMFVAR=/tmp/texmf-var pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

## Scope discipline

- Keep `paper_v2` frozen.
- Do not treat Harvest LLM as main evidence unless its reliability gates are cleared in a separate cycle.
- Keep Fishery Commons as the precursor signal-design study.
- Keep Harvest Commons as the main architecture study.
