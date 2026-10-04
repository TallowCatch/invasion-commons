# Harvest LLM Bridge Stage B32 Closeout

## Purpose

This pilot tests whether the scalable-oversight Harvest pipeline can be extended from hand-written and search-generated strategies to LLM-generated structured strategies without switching to live LLM agents.

The question is whether a local model can generate valid Harvest strategies, whether cooperative and exploitative prompts produce behaviorally different strategy populations, and whether local, global-signal, and hybrid oversight still separate when those strategies are evaluated in the existing Harvest benchmark.

## LLM generation path

Model: `qwen2.5:3b-instruct` through local Ollama.

Strategy format: the existing structured Harvest strategy schema. The model does not produce arbitrary code. It produces threshold and response parameters that are parsed, clamped, deduplicated, and compiled into the same Harvest strategy class used by the rest of the repository.

Fix applied before the run: the previous local run looked stuck because the output cap was too small, which truncated JSON and produced repeated silent parse failures. The builder now exposes `--max-output-tokens`, logs progress and parse failures, and writes partial bank files during generation.

## Strategy-bank result

Output prefix:

`results/runs/showcase/curated/harvest_llm_bridge_stageB32_local_qwen`

Summary:

| Attitude | Accepted unique | Attempts | Parse failures | Duplicates | Acceptance rate |
|---|---:|---:|---:|---:|---:|
| cooperative | 32 | 45 | 0 | 13 | 0.711 |
| exploitative | 32 | 52 | 0 | 20 | 0.615 |

This satisfies the basic validity requirement. Both banks reached the target size with no parse failures. The exploitative bank had more duplicates, which suggests the local model has a narrower exploitative-policy mode, but it still produced enough unique strategies for a controlled pilot.

Parameter differences were visible. Compared with cooperative strategies, exploitative strategies had higher mean high-resource harvest fractions and different response weights. The difference is not a full behavioral proof by itself, so the bank was evaluated in the governance map.

## Governance-map result

Output prefix:

`results/runs/showcase/curated/harvest_llm_bridge_stageB32_local_qwen_map`

Settings:

- stress settings: moderate-coupling commons and high-coupling commons
- conditions: no oversight, local oversight, global signal, hybrid oversight
- exploitative shares: 0.0, 0.5, 1.0
- sampled populations per cell: 40
- evaluation seeds per population: 8

### No-oversight degradation

Under no oversight, increasing the exploitative share worsened ecological outcomes.

In moderate-coupling commons, mean patch health declined from 10.24 at exploitative share 0.0 to 9.06 at exploitative share 1.0. Neighborhood overharvest increased from 2.20 to 2.74.

In high-coupling commons, mean patch health declined from 6.97 at exploitative share 0.0 to 5.87 at exploitative share 1.0. Neighborhood overharvest increased from 1.89 to 2.45.

This supports the first bridge requirement: LLM-generated exploitative strategy populations create stronger commons pressure than cooperative-only populations.

### Oversight separation

Averaged across stress settings and exploitative-share levels:

| Condition | Mean patch health | Garden failure rate | Mean welfare | Neighborhood overharvest |
|---|---:|---:|---:|---:|
| no oversight | 8.04 | 0.844 | 14.06 | 2.29 |
| local oversight | 11.95 | 0.272 | 12.99 | 1.44 |
| global signal | 15.50 | 0.000 | 14.03 | 2.58 |
| hybrid oversight | 16.00 | 0.000 | 13.36 | 1.90 |

The governance conditions are empirically distinguishable. Local oversight improves ecological outcomes relative to no oversight, but does not eliminate garden failure. Global signal and hybrid oversight eliminate garden failure in this pilot. Hybrid has the highest mean patch health and lower neighborhood overharvest than global signal, while global signal has higher mean welfare.

### Winner pattern

Using patch health first, then garden failure, then welfare, hybrid oversight ranks first in all six scenario-by-exploitative-share cells in this pilot.

This should be reported carefully. The result supports the LLM bridge, but it is still a local-model pilot, not a multi-model claim.

## Interpretation

Stage B32 shows that the LLM bridge is technically viable and behaviorally meaningful. The local model can generate valid structured strategies, exploitative prompts produce worse commons pressure under no oversight, and oversight architectures still separate under the generated strategy population.

The result is strong enough to justify scaling the bridge. It is not yet enough to claim a broad result about LLMs in general because it uses one local model and one fixed prompt family.

## Recommended next step

The next publication-grade run should keep this exact structured-strategy design and scale along two axes:

1. Increase strategy-bank size to 64 or 128 accepted unique strategies per attitude.
2. Add at least one stronger API model if access is available, while keeping the local model as the open baseline.

Live LLM agents should remain deferred. The current strategy-bank design is more reproducible and easier to inspect, which is better for a first paper-grade LLM bridge.
