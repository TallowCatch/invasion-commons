# Harvest institutional alignment follow-up

This note freezes the current transition into the Harvest-led follow-up cycle.

## What stays frozen

- `paper_v2` remains the completed two-study baseline paper.
- The main Study 2 claim in `paper_v2` still rests on the narrowed `top_down_only` versus `hybrid` comparison under `search_mutation`.
- Harvest live `llm_json` remains capability-only evidence.

## What the new cycle is for

The next cycle is not another incremental paper-v2 patch. It is a Harvest-led follow-up centered on institutional robustness under escalating multi-agent strategic pressure.

The main question becomes:

> Which governance architectures remain robust as strategic pressure increases in a plural, interacting multi-agent system?

## Implemented repo changes at cycle start

- created a new branch target for the follow-up: `codex/harvest-institutional-alignment`
- added shared Harvest stage presets:
  - `architecture_stageB`
  - `architecture_stageC`
  - `capability_ladder_stageB`
- extended the Harvest invasion pipeline with per-agent episode logging
- extended the Harvest summarizer with:
  - multi-condition pairwise contrasts
  - capability-ladder outputs
  - welfare-incidence summaries
- added a new institutional follow-up plot script
- added a new writing scaffold in `paper/paper_v3_harvest/`

## Immediate next experiment order

1. Run `architecture_stageB` as the pilot restoration of the full architecture set.
2. If the pilot separates meaningfully, run `architecture_stageC` remotely through GitHub Actions.
3. Run `capability_ladder_stageB`.
4. Use the new per-agent logs to produce welfare-incidence tables.

## Interpretation discipline

- Fishery Commons remains the precursor signal-design study.
- Harvest Commons is the main architecture study.
- Hybrid governance should not be described as universal dominance.
- Welfare cost should now be treated as a distributional result to be decomposed, not just mentioned as a limitation.
