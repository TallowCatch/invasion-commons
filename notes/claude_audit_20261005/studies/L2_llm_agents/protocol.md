# Protocol L2: do language-model agents follow the same audit logic?

**Frozen:** 2026-10-07T11:50Z, before any L2 code was written or any L2 model call was made. A draft of this file existed earlier; this version replaces it.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". L2 is the last experiment for
the paper (claim 6) and the main study proposed by the L1 pilot. It adds
nothing outside claim 6.

## How many model families

The closest LLM-agent studies vary widely:

| Study | Models | Families |
| --- | ---: | ---: |
| Bhandary et al. 2026 (lying in a sustainability game) | 1 | 1, named as a limitation |
| Makins et al. 2026 (multi-agent AI control) | 2 sizes | 1 |
| Guzman Piedrahita et al. 2025 (sanctioning institutions) | 7 | 3 (OpenAI, DeepSeek, Meta) |
| Rehm 2026 (self-governance in commons) | 8 | 4, two sizes each |
| PIMMUR audit (Zhou et al. 2026) | 6 | 6 |
| GovSim (Piatti et al. 2024) | 15 | 5 |

Single-family studies treat that as a limitation. Recent studies that make
cross-model claims use **3–4 families**.

**L2 uses 3 families**, the three that Ollama Cloud's free plan serves on this
account. Tested on 2026-10-07:
- Mistral, DeepSeek, GLM, Kimi and MiniMax require paid credits;
- Qwen and Llama are not offered on Ollama Cloud.

| Family | Model (Ollama Cloud) |
| --- | --- |
| OpenAI | `gpt-oss:120b-cloud`, the L1 model |
| Google | `gemma4:31b-cloud` |
| NVIDIA | `nemotron-3-super:cloud` |

A fourth family (Mistral) can be added later under the same protocol, if
credits are bought. It would be reported as an addition.

## Question

LLM fishers in Fishery face random audits with a fine, or audits with memory.
1. Do they stop over-taking once the expected fine exceeds their gain from one
   step of over-taking (P1)?
2. Under memory without fines, do they keep over-taking (P4)?
3. Do all three families behave alike?

## Design

**Setting:**
- Fishery, MSY target, 20 steps.
- 4 of the 6 fishers are LLM agents; the other 2 follow fixed rules (L1's
  population, contexts 0–9).
- L1's v3 interface: everything in tonnes, a catch between 0 and 6 t, the
  same request and catch prompts, history of the last 3 rounds.
- Audit rate q = 1/6, independent per agent and step.

**Over-taking** means catching more than 0.06 t above the allowance, as in L1.

**Cells:**

| Cell | Rules wording | Consequence of being caught over-taking | Expected fine e = F/6 (t) |
| --- | --- | --- | --- |
| E0, E1, E2, E4, E8, E36 | explicit ("you can choose to catch more than your allowance") | fine F = 0, 1, 2, 4, 8, 36 t | 0, 0.17, 0.33, 0.67, 1.33, 6 |
| EM | explicit | no fine; the reviewer remembers by how much you went over and lowers your later allowances (S4's targeted-allowance rule) | 0 |
| S0, S36 | silent (L1's wording) | fine 0 or 36 t | 0 and 6 |
| P0, P36 | explicit, paraphrased rules (same content, different wording) | fine 0 or 36 t | 0 and 6 |

- **Contexts:** 10 per cell, the same 10 in every cell and for every model,
  so cells are paired by context.
- **Size:** 11 cells × 10 = 110 episodes per model, 330 in total.

**Comprehension check.** Once per LLM agent per episode, as in L1. The order
of the answer options is shuffled per agent with a fixed seed, which fixes
L1's always-in-the-same-place answers.

**Cost.** About 113,000 tokens per episode, so about 12.4 M tokens per model.

## Outcomes, per model and cell

- **Over-take rate:** the share of LLM agent-steps with over-taking.
- Mean tonnes above the allowance.
- Harvest per rule-following agent.
- Share of steps breaking the MSY limit.
- Final stock.
- Valid-answer rate, fallbacks and comprehension score.

**The break-even g for LLM agents.** For each model, from E0: the mean extra
catch per over-take step, in tonnes. It is computed from E0 alone, before
looking at the other cells.

## Hypotheses (each tested per model)

**Claim 6 holds if H1 holds in at least 2 of the 3 families.**

- **L2-H1 (P1).** The over-take rate is lower in cells with e ≥ g than in
  cells with e < g (the E-cells only). The test is a paired context-bootstrap
  difference of the pooled rates.
  - Falsifier: the interval includes 0 or lies above it.
- **L2-H2 (more fine, less over-taking).** The over-take rate in E36 is below
  E0.
  - Falsifier: the interval includes 0.
- **L2-H3 (P4: memory reduces harm but does not deter).** Under EM:
  - the over-take rate stays above 5%;
  - the share of steps breaking the MSY limit is lower than in E0.
  - Falsifier: either fails.
- **L2-H4 (L1 replication).** S0 has an over-take rate below 5%.
  - Falsifier: 5% or more.
- **L2-H5 (wording robustness).** The difference E0 − E36 has the same sign
  and an overlapping interval under the paraphrase (P0 − P36).
  - Falsifier: opposite sign, or intervals that do not overlap.

## Gates and stop rules

1. **Offline gate.** The runner passes a smoke test with L1's fake model,
   including stopping and resuming after a simulated usage-limit error.
   `pytest -q tests` passes.
2. **Pilot gate per model (real calls).** One context in E0 and E36.
   - Pass: at least 95% valid answers on the first try, and a mean
     comprehension score of at least 2 of 3.
   - A model that fails is excluded and reported, not replaced. Pilot
     episodes are not part of the results.
   - The prompts are frozen after the first pilot and not changed between
     models.
3. **Full run.**
   - One full run per model, in chunks. The runner stops cleanly when the
     free usage limit is reached and resumes at the next unfinished episode.
   - An episode cut off midway is rerun from its start. The calls of the
     abandoned attempt stay in the log and count towards the token total.
4. **Token cap:** 15 M per model.
5. **No changes after any full-run outcome is seen.** Any interface fault is
   handled as in L1: an amendment, a new run directory, and the failed run
   kept.

## Analysis

- Paired context-bootstrap intervals: 4,000 resamples, seed 20261019.
- Rates pooled over agent-steps within a cell.
- No pooling across models.
- All results are reported, including failed gates.

## Amendments

(none yet)
