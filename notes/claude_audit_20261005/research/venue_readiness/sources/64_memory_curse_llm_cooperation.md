# 64 — The Memory Curse: How Expanded Recall Erodes Cooperative Intent in LLM Agents

- Authors: Jiayuan Liu, Tianqin Li, Shiyi Du, Xin Luo, Haoxuan Zeng, Emanuel Tewolde, Tai Sing Lee, Tonghan Wang, Carl Kingsford, Vincent Conitzer
- URL: https://arxiv.org/abs/2605.08060
- Posted: 2026-05-08
- Accessed: 2026-10-12
- Type: arXiv preprint (7 LLMs, 4 games, 500 rounds)
- Credibility 4/5 · Recency 4/5 · Bias 3/5

## Verbatim quotes
- Abstract: "Across 7 LLMs and 4 games over 500 rounds, expanding accessible history degrades cooperation in 18 of 28 model--game settings, a pattern we term the memory curse."
- Abstract: "memory sanitization holds prompt length fixed while replacing visible history with synthetic cooperative records, which restores cooperation substantially, proving the trigger is memory content, not length alone."

## Threat to claim 1: low — this is about AGENT memory, ours is REVIEWER memory. But a reviewer may conflate the two. Add one sentence: our "memory" is the reviewer's record of caught offences; agents' visible history is held fixed across arms (verify this is true in our code before saying it).
## Threat to claim 4: low-medium — shows LLM behaviour shifts with how much history is shown; our one-round-gain result could depend on history length in the prompt. Note as a limitation if history length was not varied.
