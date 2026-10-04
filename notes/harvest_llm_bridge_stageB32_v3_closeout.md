# Harvest LLM Bridge Stage B32 v3 Closeout

## Purpose

This run tested the LLM bridge without relying on cloud infrastructure. RunPod was not usable from the current location, so the bridge was run locally with two small open-weight models already available through Ollama:

- `qwen2.5:3b-instruct`
- `llama3.2:3b`

The scientific purpose was not to claim frontier-model behavior. The purpose was to test whether model-generated strategies can be inserted into the existing Harvest oversight pipeline in a controlled and reproducible way.

## What Was Generated

Both models generated structured Harvest strategy banks using the same fixed strategy schema. Each model generated:

- `32` cooperative strategies
- `32` exploitative strategies

The generated strategies were validated, clamped to the Harvest strategy bounds, deduplicated, and saved as CSV rows. The LLMs did not act live inside the environment. They generated complete strategies that were then evaluated by the existing simulator.

## Bank Validity

### Qwen 2.5 3B Instruct

Qwen reached the full target cleanly:

- cooperative: `32/32` accepted in `32` attempts
- exploitative: `32/32` accepted in `32` attempts
- parse failures: `0`
- duplicates: `0`

Qwen produced strong parameter separation between cooperative and exploitative strategies:

| Attitude | Low harvest | Mid harvest | High harvest | Cap margin |
|---|---:|---:|---:|---:|
| Cooperative | 0.168 | 0.363 | 0.608 | 0.034 |
| Exploitative | 0.511 | 0.801 | 0.934 | 0.177 |

### Llama 3.2 3B

Llama also reached the full target:

- cooperative: `32/32` accepted in `34` attempts
- exploitative: `32/32` accepted in `33` attempts
- parse failures: `3` total
- duplicates: `0`

Llama also produced clear parameter separation:

| Attitude | Low harvest | Mid harvest | High harvest | Cap margin |
|---|---:|---:|---:|---:|
| Cooperative | 0.208 | 0.404 | 0.603 | 0.033 |
| Exploitative | 0.545 | 0.802 | 0.928 | 0.213 |

## Governance Map Settings

Both banks were evaluated with the same Harvest LLM governance map:

- scenarios: `community_irrigation`, `forest_co_management`
- public-facing interpretation: moderate-coupling and high-coupling commons
- conditions: no oversight, local oversight, global signal, hybrid oversight
- oversight regime: ideal
- exploitative strategy share: `0.0`, `0.5`, `1.0`
- sampled populations per cell: `40`
- population size: `6`
- evaluation seeds per sampled population: `8`

## Main Results

### No-Oversight Sanity Check

Both models passed the key sanity check. Increasing the exploitative share sharply reduced patch health and produced near-complete or complete garden failure under no oversight.

For Qwen under no oversight:

| Scenario | Exploitative share | Patch health | Garden failure |
|---|---:|---:|---:|
| Community irrigation | 0.0 | 15.02 | 0.00 |
| Community irrigation | 0.5 | 9.03 | 1.00 |
| Community irrigation | 1.0 | 4.14 | 1.00 |
| Forest co-management | 0.0 | 12.55 | 0.00 |
| Forest co-management | 0.5 | 7.50 | 1.00 |
| Forest co-management | 1.0 | 2.61 | 1.00 |

For Llama under no oversight:

| Scenario | Exploitative share | Patch health | Garden failure |
|---|---:|---:|---:|
| Community irrigation | 0.0 | 14.52 | 0.00 |
| Community irrigation | 0.5 | 8.82 | 0.99 |
| Community irrigation | 1.0 | 4.56 | 1.00 |
| Forest co-management | 0.0 | 12.68 | 0.00 |
| Forest co-management | 0.5 | 7.46 | 1.00 |
| Forest co-management | 1.0 | 2.94 | 1.00 |

This establishes that the generated exploitative strategies are behaviorally meaningful, not just differently labelled.

### Oversight Separation

For both models, local oversight helped only modestly, while global signal and hybrid oversight eliminated garden failure in this ideal-regime map.

Qwen condition means:

| Condition | Patch health | Garden failure | Welfare |
|---|---:|---:|---:|
| No oversight | 8.47 | 0.667 | 14.56 |
| Local oversight | 9.82 | 0.663 | 13.27 |
| Global signal | 15.55 | 0.000 | 13.49 |
| Hybrid oversight | 15.89 | 0.000 | 12.65 |

Llama condition means:

| Condition | Patch health | Garden failure | Welfare |
|---|---:|---:|---:|
| No oversight | 8.50 | 0.666 | 14.84 |
| Local oversight | 9.88 | 0.659 | 13.71 |
| Global signal | 15.54 | 0.000 | 13.64 |
| Hybrid oversight | 15.83 | 0.000 | 12.91 |

### Winner Pattern

Hybrid oversight ranked first by patch health in all six scenario and exploitative-share cells for both Qwen and Llama. The margin over global signal was small in several cells, so the paper should avoid claiming that hybrid universally dominates. The safer interpretation is that model-generated exploitative strategies reproduce the earlier pattern that architectures with a system-level signal are much more robust ecologically than local-only oversight.

## Interpretation

The bridge succeeded as a controlled pilot. It shows that the Harvest oversight benchmark can accept model-generated structured strategies and still produce meaningful governance comparisons. The model-generated exploitative banks create real ecological pressure under no oversight. Local-only oversight remains weak in the face of that pressure. Global signal and hybrid oversight preserve patch health and eliminate garden failure in this ideal-regime map.

This strengthens the link to LLM-agent population work without changing the project into a live LLM-agent benchmark. The main contribution remains governance and oversight evaluation under strategic pressure. The LLM bridge adds a strategy source that is closer to AI-generated populations.

## Limits

These are small local models, so the result should be framed as an open-model pilot, not as evidence about frontier model populations. The map also uses ideal oversight. The next step should test whether the same pattern survives constrained oversight and a slightly larger sampled-population count.

Gemma was not used locally because it was unstable on the laptop. It can be revisited only if a non-US cloud provider or accessible GPU option is available.

## Files

Qwen:

- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_bank.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_summary.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_summary.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_samples.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_strategy_scores.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_qwen2_5_3b_map_attitudes.csv`

Llama:

- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_bank.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_summary.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_summary.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_samples.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_strategy_scores.csv`
- `results/runs/showcase/curated/harvest_llm_bridge_stageB32_v3_local_llama3_2_3b_map_attitudes.csv`

## Constrained-Oversight Follow-Up

The constrained-oversight version of the same Qwen and Llama maps was also run. This used the same generated banks and the same exploitative-share grid, but changed the governance-friction regime from ideal to constrained.

The constrained regime adds imperfect detection, limited targeting capacity, one-round enforcement delay, and a per-intervention budget cost. This is the more relevant version for the scalable-oversight framing because it tests whether the model-generated strategy result survives when oversight itself is limited.

Qwen constrained condition means:

| Condition | Patch health | Garden failure | Welfare | Burden | Missed target rate |
|---|---:|---:|---:|---:|---:|
| No oversight | 8.59 | 0.666 | 14.51 | 0.00 | 0.000 |
| Local oversight | 9.81 | 0.665 | 13.34 | 0.00 | 0.000 |
| Global signal | 15.24 | 0.000 | 12.04 | 3.25 | 0.357 |
| Hybrid oversight | 15.53 | 0.001 | 11.50 | 2.65 | 0.289 |

Llama constrained condition means:

| Condition | Patch health | Garden failure | Welfare | Burden | Missed target rate |
|---|---:|---:|---:|---:|---:|
| No oversight | 8.54 | 0.658 | 14.80 | 0.00 | 0.000 |
| Local oversight | 9.83 | 0.652 | 13.70 | 0.00 | 0.000 |
| Global signal | 15.18 | 0.001 | 12.14 | 3.31 | 0.365 |
| Hybrid oversight | 15.42 | 0.001 | 11.70 | 2.74 | 0.298 |

Under constrained oversight, hybrid ranked first by patch health in `5/6` cells for both Qwen and Llama. Global signal ranked first in the remaining `1/6` cell for each model. This is a useful result because the bridge does not collapse into a simple claim that hybrid always wins. The more defensible interpretation is that model-generated exploitative strategies reproduce the core oversight pattern: local-only oversight remains weak under exploitative pressure, while architectures with a system-level signal are much more robust ecologically. Hybrid is usually strongest on patch health, but the margin is sometimes small and comes with lower welfare.

Constrained oversight reduced patch health slightly relative to ideal oversight for the governed architectures:

| Model | Condition | Patch-health change | Failure-rate change |
|---|---|---:|---:|
| Qwen | Hybrid | -0.357 | +0.001 |
| Qwen | Global signal | -0.308 | +0.000 |
| Llama | Hybrid | -0.408 | +0.001 |
| Llama | Global signal | -0.362 | +0.001 |

The result is therefore stable enough to write up as an LLM-strategy bridge pilot. It should still be framed as a pilot because the models are small local open-weight models, but the experimental logic is now sound.

## Recommended Next Step

The next useful step is manuscript integration, not more local model generation. The paper should add a short LLM bridge subsection showing:

1. model-generated strategies were valid and diverse;
2. exploitative banks created real ecological pressure under no oversight;
3. local-only oversight remained weak;
4. global signal and hybrid oversight kept failure near zero under both ideal and constrained regimes;
5. the result is a controlled bridge to model-generated populations, not a live LLM-agent benchmark.

If a third model becomes accessible later through a non-US cloud option, it can be added as replication. It is not required before writing the pilot result.
