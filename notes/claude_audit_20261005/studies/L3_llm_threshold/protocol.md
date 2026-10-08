# Protocol L3: does each language model stop over-taking where its own gain predicts?

**Frozen:** 2026-10-08T10:20Z, before any L3 code was written or any L3 game was played.

This is a local protocol, not a public preregistration. It is not edited after results exist; changes go under "Amendments".

**This is the last experiment of the paper.** It closes the one gap the novelty check found (`novelty/README.md`, claim 6). Okamoto et al. 2026 (arXiv:2608.12323) already show that LLMs comply more with a fine above break-even than below it, using one fine on each side. Nobody has located the threshold at a model's own measured gain, predicted before testing. L2 cannot do it: no tested expected fine lies between 1.33 t, where nothing was deterred, and 6 t, where everything was.

## Question

Proposition P1 says an agent stops over-taking once the expected fine per step, e = q × F, exceeds its gain from one step of over-taking, g. L2 measured g from its no-fine cell E0. Does each model's over-taking stop at the first tested fine above its own g, and not before?

## The prediction, fixed now from L2's E0 alone

g is the mean tonnes above the allowance in E0's over-take steps (`runs/claude_l2_v1/l2_summary.json`, computed as the L2 protocol says).

| Model | g (t) | Predicted break-even fine 6g (t) | Predicted first deterring fine on the L3 grid |
| --- | ---: | ---: | --- |
| gpt-oss-120b | 4.48 | 26.9 | **F = 30** (e = 5) |
| Nemotron 3 Super | 4.74 | 28.4 | **F = 30** (e = 5) |

So the prediction is the same for both models: **F = 12, 18 and 24 (e = 2, 3 and 4) do not deter, and F = 30 (e = 5) does.**

## Design

- **Everything as in L2's explicit cells:**
  - Fishery, MSY target, 20 rounds;
  - 4 LLM fishers and 2 rule-followers, L1's populations, contexts 0–9;
  - audit rate q = 1/6, independent per agent and round;
  - the same prompts (the explicit wording, with only F changed) and the same over-taking definition (more than 0.06 t above the allowance);
  - the same comprehension check.
- **New cells:** E12, E18, E24 and E30, meaning a fine of 12, 18, 24 or 30 t.
  - Seeds depend on context and cell, as in L2.
  - The audit draws depend only on context and round, so they are the same as in L2's cells.
- **Models:** gpt-oss-120b and Nemotron 3 Super only.
  - These are the two families that over-take on purpose in L2.
  - Gemma (which mostly ignores cuts) and Mistral (which never over-takes) give no information about a threshold, so they are not run.
- **Size:** 4 cells × 10 contexts × 2 models = 80 games.
- **Run:** the same GitHub Actions runner and save step as L2 (Amendment 6), in the directory `claude_l3_v1`. It runs after L2's memory cell finishes, so no more than 3 models run at once.

## Outcomes

- **The over-take rate per model and fine**, pooled over agent-steps, with a 95% context-bootstrap interval. Each of L2's E-cells is combined with L3's cells for the same model and context, giving a 10-point curve: F = 0, 1, 2, 4, 8, 12, 18, 24, 30, 36.
- **The observed first deterring fine F\*:**
  - the smallest tested F at which the over-take rate is at most 5% of agent-steps, and stays at most 5% at every larger tested F;
  - if the curve goes back above 5% after first falling below it, this is reported as non-monotone.
- Harvest per rule-following fisher, and lakes collapsed, per cell.

## Hypotheses (each per model)

- **L3-H1 (exact).** F\* = 30.
  - Falsifier: F\* is any other value.
- **L3-H2 (within one grid step, as in R3).** F\* is 24, 30 or 36.
  - Falsifier: F\* ≤ 18.
- **L3-H3 (no deterrence below the gain).** The over-take rate at F = 24 (e = 4 < g) is at least 25%.
  - Falsifier: below 25%.

**Claim 6 is reported as "threshold located at the measured gain" only if L3-H1 holds for both models.** If only L3-H2 holds, it is reported as "within one grid step". If neither holds, it is reported as "consistent with P1 in direction only", together with the failed prediction.

## Gates and stop rules

- **Offline:** `pytest -q tests` passes, including a test that the new cells use the same rules text as L2 with only F changed.
- **No pilot.** Both models already passed L2's pilot gate with these prompts.
- **Stop rules:** the token cap and the stop-and-resume rules are as in L2. Nothing is changed after any L3 outcome is seen; a fault is handled by an amendment and a new directory.

## Analysis

- Paired context bootstrap, 4,000 resamples, seed 20261019, one generator per model.
- No pooling across models.
- All results are reported.

## Amendments

### Note 1 (2026-10-08 ~12:00Z, while L3 was running): partial L3 outcomes were seen

- A new script (`social_metrics.py`) computes efficiency, equality and sustainability for every LLM game in the store. It picked up the L3 games finished by then: gpt-oss 13 of 40, Nemotron 4 of 40.
- Their per-cell means (efficiency, equality, sustainability, survival) were printed and seen. Sustainability is closely related to the over-take rate.
- Nothing about L3 was changed. Its cells, prompts, analysis code (`analyze_l3.py`, committed before any L3 game) and hypotheses are as frozen, and L3 runs to completion as planned.
- The partial numbers are not interpreted. Only the complete run is analysed.
