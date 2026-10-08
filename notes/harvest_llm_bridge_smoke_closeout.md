# Harvest LLM Bridge Smoke Closeout

## Purpose

This smoke test checks whether the LLM strategy-bank bridge can run end-to-end using a local model. It is not a publication-sized experiment.

The bridge uses language models to generate structured Harvest strategies. The strategies are validated, deduplicated, saved as a bank, sampled into populations, and evaluated through the existing Harvest governance-map runner.

## Strategy Bank Result

Model:

`ollama__qwen2_5_3b_instruct`

The bank builder targeted two unique strategies per attitude.

| Attitude | Accepted unique | Attempts | Parse failures | Target reached |
| --- | ---: | ---: | ---: | --- |
| cooperative | 2 | 2 | 0 | true |
| exploitative | 2 | 2 | 0 | true |

The resulting bank contains four valid unique strategy signatures. This is enough to test the pipeline, but too small for a substantive paper claim.

Files:

`results/runs/showcase/curated/harvest_llm_bridge_smoke_bank.csv`

`results/runs/showcase/curated/harvest_llm_bridge_smoke_summary.csv`

## Governance Map Smoke Result

The smoke governance map used the community-irrigation Harvest preset, ideal oversight, three conditions, exploitative shares of 0.0 and 1.0, three sampled populations per cell, and two evaluation seeds per population.

Under no oversight, exploitative-heavy populations performed worse than cooperative-heavy populations:

| Exploitative share | No-oversight patch health | No-oversight garden failure |
| ---: | ---: | ---: |
| 0.0 | 11.41 | 0.17 |
| 1.0 | 10.43 | 1.00 |

Across all conditions, exploitative-heavy populations had lower mean patch health and higher failure:

| Exploitative share | Mean patch health | Garden failure | Mean welfare |
| ---: | ---: | ---: | ---: |
| 0.0 | 14.84 | 0.06 | 14.86 |
| 1.0 | 14.26 | 0.33 | 15.45 |

This is the expected first signal: model-generated exploitative strategies can worsen the commons outcome, and governance can be evaluated against that pressure.

Files:

`results/runs/showcase/curated/harvest_llm_bridge_smoke_map_summary.csv`

`results/runs/showcase/curated/harvest_llm_bridge_smoke_map_samples.csv`

`results/runs/showcase/curated/harvest_llm_bridge_smoke_map_strategy_scores.csv`

`results/runs/showcase/curated/harvest_llm_bridge_smoke_map_attitudes.csv`

## Interpretation

The smoke test validates the mechanics of the LLM bridge. It shows that a local model can generate valid structured Harvest strategies, that cooperative and exploitative banks can be sampled into populations, and that the existing governance-map runner can evaluate those populations.

The smoke test should not be treated as evidence for the main paper. The sample is too small. Its value is that the next proper LLM bridge run is now technically justified.

## Next LLM Bridge Run

The next run should increase the bank to at least 32 valid unique strategies per attitude before moving to 128. It should use both moderate-coupling and high-coupling stress settings, include no oversight, local oversight, global signal, and hybrid oversight, and evaluate exploitative shares of 0.0, 0.5, and 1.0 before expanding to the full five-share grid.

The success gate is:

- valid unique strategies are generated for both attitudes;
- exploitative-heavy populations worsen at least one safety or resource metric under no oversight;
- governance conditions remain empirically distinguishable;
- the results can be processed without manual cleanup.
