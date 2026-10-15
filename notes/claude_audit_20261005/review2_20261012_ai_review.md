# Evaluation of a second AI review of the revised draft (12 Oct 2026)

The draft reviewed is the one after commit `96dded1`. The checks below are post hoc and were verified on the stored games (branch `l2-results`). Script: `experiments/oversight/l2_review_checks.py`. Output: `runs/claude_l2_v1/l2_review_checks.json`.

## New checks

**Split-half calibration.**
- Method: estimate the observed excess on 5 populations without a fine, predict the stopping fine, then find the stopping fine on the other 5. This was done for all 252 splits.
- Prediction: 30 t in every split, for both models.
- gpt-oss-120b: the held-out stopping fine was 24 t in 252 of 252 splits (one step early).
- Nemotron 3 Super: the held-out stopping fine was 30 t in 90.5% of splits and 24 t in 9.5%.
- What it shows: reusing the same populations for calibration and testing does not drive the result.
- What it does not show: a test on new populations, or on a fresh run.

**Observed excess under the other wordings, with no fine.**

| Model | Explicit wording | Silent wording | Paraphrased wording |
|---|---:|---:|---:|
| gpt-oss-120b | 4.48 t (predicts 30 t) | 4.02 t (predicts 30 t) | 3.96 t (predicts 24 t) |
| Nemotron 3 Super | 4.74 t (predicts 30 t) | 3.74 t (predicts 24 t) | 4.81 t (predicts 30 t) |

- In 2 of 4 cases, the prediction moves by one grid step with the wording.
- Those wordings were tested only at 36 t, so whether the calibration transfers across wordings is untested.

**Audit budget in Table S3.** "Memory + extra checks" uses 3.78–4.07 checks per step, against about 1 for the other rules. The main text and the Figure 2c caption do not say so.

## Verdict by point

| Review point | Verdict |
|---|---|
| Three gain quantities; $\hat g_1$ is not Proposition 1's $g_1$ | Already done in the last revision (Key terms, §4.6, Table S6). |
| q is fixed for the LLMs, so the expected fine and the stated fine cannot be told apart | **Correct for the LLMs.** For the programmed agents, q (from 1/6 to 0.02) and s were varied (Fig. 2b, open points), so the expected fine is identified there. For the LLMs, the claim must be stated at q = 1/6. An experiment is needed to claim more. |
| Coarse grid; only 2 models; calibration and test on the same populations | Grid and model count are already stated in the Limitations. The same-population point is partly answered by the split-half check and by fines 12–30 being unseen with new sampling seeds. New populations need an experiment. |
| The realised gain is collective, and it is not $g^*$ | Already in §4.7. Add "not the marginal gain of one agent". |
| Fig. 2a compares across games | Defend: every game uses the same half-capacity line, and the paper compares only within a game. |
| Fig. 2c audit budgets unequal | **Correct, new.** Add to the caption. |
| Fig. 2e: "ever caught" is a weak metric | Partly defend: it is the mechanism of the failure, and harm is the main outcome. Detection delay is optional and needs only an analysis. |
| Annotate Fig. 3; add a prediction table for Fig. 4 | Cheap presentation fixes. Table S6 already holds the numbers. |
| The 36 t wording control does not test transfer of the calibration | **Correct, and the check above strengthens it.** Needs an experiment to claim transfer. |
| Narrowed conclusion | Close to the current one. Adopt "at a fixed audit probability" and "under one wording". |

## Experiments, if the claim is a calibration method (needs supervisor agreement; breaks the freeze)

The plan is one study (L4) with a frozen protocol, on 10 new populations per model, for gpt-oss-120b and Nemotron 3 Super.
- **Calibration:** E0, the explicit wording with no fine. Freeze the predictions from it.
- **Out-of-sample replication:** E24 and E30.
- **Matched expected fine:**
  - q = 1/3 at F = 12 and 15;
  - q = 1/12 at F = 48 and 60.
  - These match e = 4 and e = 5.
- **Wording transfer:** P0, P24 and P30.

Size: 10 cells × 10 populations × 2 models = 200 games, which is 2.5 times L3. Without the wording cells it is 140 games.
