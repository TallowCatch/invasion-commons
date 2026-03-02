# paper_v2 writing-first cycle plan

## Objective
Convert frozen `paper_v1` evidence into a defendable thesis/paper draft with only targeted validation if requested.

## Scope rules
- No new mechanism implementation in this cycle.
- Use curated `paper_v1_*` artifacts as canonical sources.
- Hard tier remains appendix-only unless explicitly recalibrated.

## Workstream A: writing package (priority 1)
1. Draft LaTeX sections using files under `paper/paper_v2/sections`:
- introduction
- methods
- results
- limitations
- appendix
2. Pull numbers only from curated artifacts:
- `results/runs/ablation/curated/paper_v1_table1_main_deltas.csv`
- `results/runs/ablation/curated/paper_v1_table3_medium_effect_size_ranking.csv`
- `results/runs/showcase/curated/paper_v1_gates.json`
3. Include Figures A-D from curated showcase outputs.

## Workstream B: targeted validation (priority 2)
1. Run only if supervisor asks:
- `easy_v1` symmetry rerun at `n_runs=5`.
2. Do not add new environments/governance mechanisms.

## Workstream C: presentation assets (priority 3)
1. Build 3-slide core pack:
- slide 1: research question + invasion protocol
- slide 2: medium collapse reduction with CIs
- slide 3: ranking + hard-tier appendix note

## Acceptance criteria
- `paper_v1` gate logic still PASS/PASS/not-saturated for medium.
- First full LaTeX draft compiles with sections and figure placeholders.
- Claims remain aligned to CI-supported evidence.

## Meeting talk track
1. Problem shift: emergence -> invasion-stability.
2. Experimental discipline: matched matrix + held-out regimes + CIs.
3. Main result: medium-tier governance robustness holds across mutation and live LLM.
4. Caveat handling: hard tier is appendix stress-only.
5. Next step: writing-first cycle with minimal validation only.
