# Local LLM Model Selection for the Harvest Bridge

## Hardware constraint

The local machine is an Apple M1 MacBook Air with 8 GB RAM. This rules out large open models for routine local experimentation. The practical local range is 3B to 4B parameter models, run one at a time through Ollama.

Disk space was also a constraint. Several large raw CSV outputs were compressed to make room for additional models. The compressed files remain available as `.gz` archives.

## Installed local models

The current local Ollama set is:

- `qwen2.5:3b-instruct`
- `gemma3:4b`
- `llama3.2:3b`

These models give three different open-model families: Qwen, Google Gemma, and Meta Llama. This is a stronger setup than relying on one local model, while still keeping the work reproducible and free of proprietary API access.

## Generation validity

All three models can generate structured Harvest strategies through the existing strategy-bank pipeline.

Qwen produced a 32 cooperative plus 32 exploitative bank with zero parse failures.

Gemma produced an 8 cooperative plus 8 exploitative bank with zero parse failures. It sometimes wrapped JSON in markdown fences, but the parser repaired this correctly.

Llama produced an 8 cooperative plus 8 exploitative bank with zero parse failures and direct JSON output.

## Behavioral usefulness

The model should not only produce valid JSON. It should also produce strategy populations that differ behaviorally across cooperative and exploitative prompts.

Qwen is currently the strongest bridge model. Under no oversight, increasing exploitative share worsened patch health and neighborhood overharvest. Oversight conditions separated clearly: local oversight improved outcomes, while global signal and hybrid oversight eliminated garden failure in the pilot.

Llama is also useful. It produced valid strategies quickly, and its exploitative strategies had lower restraint and weaker neighbor reciprocity. In the governance map, local oversight improved outcomes relative to no oversight, and global signal and hybrid oversight eliminated garden failure. The exploitative-share effect was less monotonic than Qwen, so it should be treated as a secondary model until scaled beyond 8 plus 8 strategies.

Gemma is valid but less useful as a stress model under the current prompt. Its exploitative strategies did not produce exploitative-action pressure in the map, and increasing exploitative share did not clearly worsen outcomes. This is an interesting model-behavior result, but Gemma should not be the main stress model unless the prompt is revised and revalidated.

## Recommended model plan

For the paper path, use:

1. `qwen2.5:3b-instruct` as the main local LLM bridge model.
2. `llama3.2:3b` as the second local model after scaling its bank to 32 plus 32 or 64 plus 64.
3. `gemma3:4b` as a diagnostic model showing that some models generate valid but less adversarial strategy populations.

The main paper claim should remain about governance evaluation under model-generated strategy populations. It should not claim that these local models represent all frontier LLM behavior.

## Cloud option

If stronger open models are needed, the next step should be a cheap cloud GPU run rather than a proprietary API. A rented GPU can run a 7B, 14B, or 32B open model through the same strategy-bank interface. This would preserve reproducibility better than closed API models, while giving a stronger capability comparison than the local MacBook can support.
