# Figure Style Guide for the Scalable-Oversight Commons Paper

> Historical Stage A figure plan. For the current matched-reviewer manuscript,
> use `notes/research_review/PAPER_CLOSEOUT_20260924.md` instead. Its main
> empirical figure is the context-clustered inspection-budget comparison,
> not the rank-subtraction capability-gap plot below. The older figure decisions
> are retained to document the development path, not as current paper guidance.

## External Figure Standards Checked

I checked recent papers and paper pages in multi-agent reinforcement learning, sequential social dilemmas, scalable oversight, and LLM-agent evaluation. The useful pattern is consistent across them: main figures are usually multi-panel, use compact typography, state the experimental comparison directly, and pair plots with exact tables. Conceptual diagrams are sparse and hierarchical. Empirical plots usually include uncertainty, range, or clear notes when uncertainty is unavailable.

Reference examples consulted:

- Leibo et al., sequential social dilemmas: https://arxiv.org/abs/1702.03037
- Melting Pot 2.0: https://arxiv.org/abs/2211.13746
- SocialJax: https://arxiv.org/abs/2503.14576
- Measuring Progress on Scalable Oversight: https://arxiv.org/abs/2211.03540
- Scaling Laws for Scalable Oversight: https://arxiv.org/abs/2504.18530
- LLM-agent cooperation work: https://arxiv.org/abs/2501.16173
- LLM collective behavior work: https://arxiv.org/abs/2602.16662

## General Standards

Figures should answer one question each. A reader should understand the comparison from the panel titles and caption without reading the full paragraph.

Use one visual system across the paper:

- serif font to match the LaTeX manuscript;
- colorblind-safe condition palette;
- consistent condition names: None, Local, Global, Hybrid;
- consistent line widths and marker sizes;
- light gridlines only on quantitative axes;
- vector PDF/SVG for paper use and high-resolution PNG for quick review.

Avoid:

- default Matplotlib blue/orange/green/red;
- legends that cover data;
- winner maps without effect size;
- over-averaged LLM plots that hide stress-setting differences;
- captions that only describe the axes.

## Figure-Specific Decisions

### Figure 1: Study Logic

Decision: keep in main paper after redesign.

Purpose: show the whole research chain without making it look like a slide-deck flowchart.

Standard: conceptual diagram with numbered blocks, minimal text, and one row for variables/readouts.

### Figure 2: Benchmark Mechanism

Decision: keep in main paper after redesign.

Purpose: show how strategy artifacts enter the Harvest substrate and how local/global predicates are evaluated.

Standard: mechanism diagram, not a generic workflow. It must make the local/global safety split visible.

### Figure 3: Capability Gap

Decision: main empirical figure.

Purpose: show what happens as actor capability rises relative to overseer capability.

Standard: multi-panel quantitative figure. Lines show condition means. Bands show range across stress cells at the same capability gap. The bands are not confidence intervals and must be described that way.

Remaining weakness: full run-level uncertainty is not shown in this aggregated figure. Add confidence intervals if a future data product exposes run-level summaries by capability gap.

### Figure 4: Winner Map

Decision: keep as supporting figure, not the main evidence.

Purpose: summarize rankings across actor/overseer capability cells.

Standard: each cell shows the winning architecture and the winner's patch-health difference relative to the best non-winning patch-health architecture. Negative values are important because they reveal cases where the ranked winner is not the ecological winner.

Remaining weakness: winner maps still compress multiple metrics into one cell. Use alongside Tables 1-2 and Figure 3.

### Figure 5: Episode Trace

Decision: keep in main paper as explanatory case trace.

Purpose: make local-pass/global-fail concrete.

Standard: shaded failure moments, direct labels, and caption clearly stating that this is an example case.

Remaining weakness: one trace is qualitative evidence. The matrix-level local-pass/global-fail rate remains the quantitative evidence.

### Figure 6: LLM Bridge

Decision: keep in main paper as pilot figure.

Purpose: show that model-generated structured strategies can be evaluated in the same pipeline and reproduce the pressure pattern.

Standard: split by model and stress setting. Avoid averaging away stress effects. Caption must state that these are offline structured strategies, not live LLM agents.

Remaining weakness: no uncertainty intervals are available in the current LLM summary files. Add standard errors or confidence intervals if future LLM evaluation logs preserve per-population outcomes.

## Implementation Rules

The plotting script should:

- export `.pdf`, `.svg`, and `.png`;
- use the shared condition palette;
- avoid hard-coded default styling;
- use captions and panel titles that state the figure's claim;
- include visible uncertainty or range only when supported by available data;
- annotate limitations directly when a visual could overstate evidence.
