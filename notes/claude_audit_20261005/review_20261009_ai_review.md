# Evaluation of an AI chat's review of the AAAI draft (9 Oct 2026)

The review was pasted by Ameer, and each point below was checked against the code, the data and the novelty notes. Labels:
- **[verified]**: checked against stored data or code in this session.
- **[post hoc]**: an analysis decided after the results were seen.

## Two checks run on the stored LLM games (post hoc, verified)

The data are the stored games on branch `l2-results`, from `claude_l2_v1` and `claude_l3_v1`.

**1. Measured one-round gain against the maximum available gain (review point 2).** For each model, the table compares three quantities, all in tonnes, from the games without a fine (E0):
- the gain as the paper measures it: the mean catch above the allowance, on rounds where the model over-took;
- the gain as Proposition 1(i) defines it: the most the model could have taken above its allowance (6 t minus the allowance), on the same rounds;
- that same maximum, on all rounds.

| Model | Measured gain (paper's g1, pre-registered) | Maximum available, over-take rounds | Maximum available, all rounds |
|---|---:|---:|---:|
| gpt-oss-120b | 4.48 | 5.17 | 5.07 |
| Nemotron 3 Super | 4.74 | 5.09 | 5.03 |

Using the proposition's own quantity (about 5.1 t), the first fine that would deter is 36 t, where the expected fine is 6 t. The models actually stopped at 24 t (gpt-oss-120b) and 30 t (Nemotron 3 Super). So both stopped one to two steps earlier than a risk-neutral agent that decides round by round.

The pre-registered quantity, the mean excess the model actually took, predicts better. But it is an empirical proxy, not the quantity in Proposition 1. This fits the failed H3: the LLMs are easier to deter than a risk-neutral agent.

**2. How stable the stopping fine is when the 10 populations are resampled (review point 6).** Paired bootstrap over populations, 4,000 resamples, seed 20261019, with the same "≤5% at this fine and every higher one" rule:
- gpt-oss-120b: the stopping fine is 24 t in 100% of resamples.
- Nemotron 3 Super: 30 t in 92% of resamples, 24 t in 8%.

## Point-by-point verdict

| # | Review point | Verdict | Action |
|---|---|---|---|
| 1 | Fine size versus expected-cost reasoning not separated | **Agree.** Already listed as future work in §5.2. | Narrow the claim to *prediction* rather than *mechanism*. Testing the mechanism needs a new experiment (see "Decision for Ameer"). |
| 2 | Measured g1 ≠ the proposition's g1 | **Agree, and it matters** (check 1) | Rename the measure "observed one-round excess"; report the 5.1 t maximum and the predictions it gives. |
| 3 | g* for the LLMs is realised, not maximised | **Agree, and it goes further.** The code takes the LLMs' net catch at F = 0 minus that at F = 36, per over-take round. That is the realised gain of all four LLMs over-taking together, against all four complying. Without the one-level restriction, over-taking only in the last round earns about g1, so max over all strategies ≈ g1. Part (ii) therefore holds only for agents that commit to one level for the whole game. | Rename it "realised whole-game gain" and drop "ignored the depleted stock". The supportable claim: "the over-taking they did lost fish over the game, yet they continued until the expected fine neared the round's gain." |
| 4 | "Behaved as in part (i)" is too strong | **Agree.** Over-taking at no fine was 55%, not 100%, the decline was gradual, and the models stopped early. | Change to "closer to part (i) than part (ii), but deterred earlier and more gradually than a risk-neutral agent." |
| 5 | The 5 of 5 programmed result is near built-in | **Mostly agree.** The agents optimise the payoff the proposition describes. It still tests the measurement procedure: out-of-sample populations, noise, and audits that miss. | Present it as validation of the measurement procedure, and lower its weight in the abstract. |
| 6 | Small effective sample of LLM games | **Agree; already in Limitations** | Add the bootstrap stability from check 2 (supplement plus one sentence). |
| 7 | "Keep audits random" contradicts Forest, where audits aimed at the plot furthest below the reviewer's forecast left 26% of the no-audit harm, against 65% at random | **Agree; this is a real inconsistency** | New recommendation: don't rank audits by report size; use random audits, or target by how far the outcome falls from the forecast. Also separate unpredictable *timing* (the known schedule failed) from random *targets*. |
| 8 | "Memory did not deter" | **Partly agree.** Over-taking fell from 55% to 20% for gpt-oss-120b, so memory deterred partially. "Did not stop" is accurate; "without deterring" in the Discussion is not. | Use three terms consistently: reduced over-taking, stopped over-taking, reduced harm. |
| 9 | No reviewer fallback when no cut meets the target | **Agree; easy fix.** The code chooses cut 0 when nothing qualifies (`calibrated_oversight.py`, lines 61 and 118). | Add "or 0 if none does" to the Methods. |
| 10 | Ye & Steinhardt and Okamoto are understated | **Agree on both.** Ye & Steinhardt's own two ingredients are reliability tracking and *escalating penalties for repeated misbehaviour*. Okamoto used 4 levels of audit likelihood × fine around break-even, not "one below and one above". | Rewrite both sentences in the Introduction and Related Work. Narrow the novelty claim: predicting the stopping fine for each model, from its own behaviour, in a repeated commons with stated audit odds. |
| 11 | Proposition 1 isn't new | **Agree; already cited to Becker** | Call it a benchmark that applies Becker's rule; state that ties go to compliance. |
| 12 | The prompt permits over-taking; no concealment | **Agree on scope.** Exact prompts are not in the supplement. | Add a scope sentence; add the full prompts, model identifiers and seed handling to the supplement. |
| 13 | "Predicted before deployment" overstates | **Agree** | "can be calibrated from the agent's behaviour without a fine, in our setting". |
| 14 | No true no-audit, no-fine control | **Agree; already a stated limitation.** It needs new games. | Keep as a limitation (or bundle into the optional experiment). |
| 15 | Wording table | Mostly duplicates the points above. "Deterrence began where the expected fine equalled the gain" refers to the programmed agents in the abstract, where the threshold *is* sharp, so that entry misreads the abstract. | Covered by the other fixes. |
| — | The AAAI-27 deadlines have passed | **Doesn't apply.** The plan is the AAAI-27 workshop (about 20 Nov), then ICML 2027. | Check the workshop's format and page limit. |

## Decision for Ameer (new experiment, against the freeze)

The review's main question is whether the paper shows we can *predict* the stopping fine, or *why* the models stop.

**Option A: no new experiments (recommended for the workshop).**
- Claim the prediction.
- Report the mechanism as consistent with the evidence but untested.
- Every fix above can be made in the text, or with the existing data.

**Option B: one LLM experiment, for the ICML version.**
- Hold the expected fine fixed and change the audit rate, e.g. q = 1/3 with F = 15 against q = 1/6 with F = 30.
- Add a true no-audit, no-fine control.
- Run 2 models × about 3 conditions × 10 populations = about 60 games, roughly 3/4 the size of L3 (80 games).
- It needs a frozen protocol first, and it needs the supervisor's agreement because it breaks the freeze.

## Done (9 Oct, option A chosen by Ameer)

All "fix in text" and "existing data" actions above are applied, with no new experiments.
- **New script:** `experiments/oversight/l2_review_checks.py`, with output in `runs/claude_l2_v1/l2_review_checks.json`.
- **Figure 4** adds the prediction from the largest available excess, drawn as tinted points.
- **Supplement:** Section I gives the exact prompts and settings. Section J gives the two post hoc checks as Table S6. Table S4 and the key terms are updated.
- **Build:** `build_main.py` now refreshes the paper's local figure copies on every build. The old Figure 4 had been stale there.
- **Main paper:** 8 pages, with the Conclusion on page 7 and the references running to page 8. That is within AAAI's 7 content pages plus 2 for references. The workshop's own limit still needs checking.
- **Option B** (equal expected fine at different audit rates) is not run. It is parked for the ICML version and needs supervisor agreement.
