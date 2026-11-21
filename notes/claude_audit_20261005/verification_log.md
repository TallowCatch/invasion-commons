# Verification log

All checks below were run on 4–5 October 2026 against commit `8812f4f`
(branch `codex/harvest-oversight-gap-stagea`). The repository was a fresh
clone from GitHub; your local repository is on the same commit. No new
simulation episodes were run. Everything is either a rerun of an existing
script or a re-analysis of saved data.

## 1. Integrity and reproduction

| Check | Command | Result |
| --- | --- | --- |
| Data hashes | `sha256sum` on `paper/.../data/analysis/*`, `data/sources/*` | All match `provenance.json` and `sources/manifest.json` |
| Core tests | `python -m pytest -q tests --ignore=tests/test_fishery_rl.py --ignore=tests/test_harvest_rl.py` | 236 passed, 11 skipped (the PyTorch RL tests were not run) |
| Coupled-local replay | `python -m experiments.replay_coupled_local --input-dir results/runs/budgeted_reviewer_confirmation_v1 --output-dir <tmp>` | 11,249 cases, 33,747 paired decisions, 0 disagreements |
| Figure 8 data | `python -m experiments.plot_reviewer_decisions --input-dir paper/.../data --output-dir <tmp>` | Data CSV identical to the committed one |
| Paper build | `pdflatex`/`bibtex`/`pdflatex` ×2 | No warnings; text identical to the committed `main.pdf` |
| Worked example | Recomputed Fishery context 0, step 14 | Demand 14.70, next stock 10.77, bound 20.65, chosen scale 0.5: all match |

## 2. Numbers verified against committed data

- **Confirmation results:** every count and rate in the decision table,
  the closed-loop outcomes, and the paired long-run differences
  (`data/analysis/*.csv`).
- **Stage A:** 72 rows; winners hybrid 13 / local 4 / global 1
  (`harvest_oversight_gap_stageA_ranking.csv`).
- **Threshold grid:** 9,000 rows, 25 threshold pairs, no duplicate keys.
- **September frozen-population checks** (`completed_checks/slow_regrowth/mechanism_episodes.csv`):
  - 19.70%, 5.33%, 0.125% and 8.30% unsafe time;
  - health 11.12 → 11.79 and return 980.48 → 1,028.95;
  - uniform caps 14.74 / 908.18;
  - 16 of 80 clean-prefix onsets for the fixed cutoff;
  - 8 own-model onsets across 6 episodes.
- **Messages vs none, paired over 10 populations:** −0.197 ± 0.213
  (95% t-interval); median −0.008.
- **Code facts used in the issues file:**
  - weather allowance (`budgeted_oversight.py:77`);
  - bounded-local formulas (`budgeted_oversight.py:121-122`, `oversight_protocol.py:81-91`);
  - regime-pack bug (`harvest_benchmarks.py:267-297`);
  - uniform collapse penalty (`evolution.py:587-590`);
  - 1.96 × SE intervals (`summarize_harvest_invasion.py:120`);
  - governor trigger 16 / cap 0.18 (`harvest_evolution.py:24-38`);
  - Fishery quota 0.07 (`run_governance_ablation.py:73`).

## 3. Numbers I could not verify (notes only)

Raw results for these live in git-ignored `results/` folders that are not
in the repository:

- all Fishery Phase 1 results;
- the March–April Harvest results (except one paper_v2 summary file);
- PPO;
- the September development pilots;
- the actor-pressure pilot;
- Clean Up.

They are quoted from your notes and labelled "(notes only)" in the project story.

## 4. New post hoc analyses (scripts in `scripts/`)

These were computed after seeing the confirmation results. They are
exploratory and need prospective confirmation.

| Script | What it computes | Key outputs |
| --- | --- | --- |
| `reanalysis.py` | Harvest decisions at k = 6 under three allowances (computed with the reviewers' own prediction functions); 4,000-draw risk per Harvest case; structure of the Fishery risky cases; where "safe" Fishery requests leave the stock; closed-loop stock and restriction frequency | `reanalysis.json` |
| `margin_sweep.py` | All three reviewers × budgets 0/3/6 under the original and the average-sized allowance, using the repository's own reviewer code | `margin_sweep_output.txt` (original-allowance rows reproduce the saved counts exactly) |

Run both from the repository root after unpacking the confirmation archive:

```sh
mkdir -p results/runs
tar -xzf paper/paper_v5_scalable_oversight_commons/data/sources/budgeted_reviewer_confirmation_v1.tar.gz -C results/runs
PYTHONPATH=. python3 notes/claude_audit_20261005/scripts/reanalysis.py
PYTHONPATH=. python3 notes/claude_audit_20261005/scripts/margin_sweep.py
```

`reanalysis.py` writes `reanalysis.json` next to itself. (It also contains
an unused helper function, `closed_states`; it is harmless.)

## 5. How the audit was done

- **I read:** `CLAUDE_HANDOFF.md`, `PROJECT_HISTORY.md`, the paper
  (`main.tex` and its appendix), `PAPER_CLOSEOUT_20260924.md`, the data
  README, and the confirmation protocol and closeout.
- **Two independent sub-audits** read the code and notes for the earlier
  experiments (February–June) and the September checks. They recomputed
  numbers wherever committed data existed. Their findings that I used were
  spot-checked against the code and data, as listed in section 2.
- **Your installed Codex skills** `scientific-experiment` and
  `experiment-planner` were read and used as the checklist:
  - same target and authority across arms;
  - independent units and no pseudoreplication;
  - post hoc work labelled as such;
  - prospective contracts before new runs;
  - no retrying until significant.
- **No suitable "science" plugin was available** in the Claude plugin
  catalogue for this kind of study.

## 6. Independent verification pass on these notes

After drafting, a separate checker re-derived the key claims from the data and
code without reusing my scripts. It confirmed:

- the margin re-analysis, with an independent re-implementation of the
  safety predicate;
- every table in the September reviewer write-up;
- the closed-loop statistics;
- the code citations;
- every "(verified)" number in the project story.

It also found errors and overstatements, which are now corrected:

- **Harvest allowance.** The original allowance is a valid union bound for
  both safety conditions. It is oversized only because the patch-failure
  condition never binds in these cases. The 4 risky approvals at 0.282 are
  borderline label noise, not a gap in the allowance. I added a sensitivity
  table across allowance sizes.
- **Fishery levels.** "Below 50" had mixed up the stock before and after
  regrowth. Corrected: all 694 safe requests leave less than 50 after
  harvest; 409 leave the regrown stock below 50.
- **Open vs closed loop.** In closed loop the joint reviewer cut only risky
  requests, so the 86%-vs-0% contrast was about case mix, not errors.
  Reframed.
- **Collapse timing.** Collapse timing in the unregulated Fishery runs is
  stock below 10 at a median of step 8. The median scored stock is 47.4,
  not 38.9.
- **Harvest weather streams.** The no-reviewer runs used 2 weather streams
  per context.
- **Stage A details.** Global caps had 3–14% unsafe time outside the
  strong-overseer setting. The actor ladder runs backwards in every
  no-overseer condition.
- **Collapse penalty.** It does influence Harvest's search-based entrant
  generator.
- **Hedging.** Wording on the "safety target" explanation now says it is
  untested, since the target was never varied.

## 7. Experiments R1 and S1 (5 October 2026)

| Check | Result |
| --- | --- |
| New unit tests (`tests/test_claude_calibrated_oversight.py`) | 4 passed. They confirm that R1/S1 rebuild the 23 September populations exactly, that Fishery joint review with full information equals the label, that bounded local review is never less cautious than joint, and that the reference labels behave sensibly at extremes. |
| Smoke runs | R1 and S1 each ran twice with byte-identical outputs (gzip written with a fixed timestamp) |
| R1 smoke gate | Joint review at k = 6 agreed with the reference on every resolved smoke case |
| S1 gate (H1) | With *d* = 0, every report-based protocol matched `full` exactly: 0 mismatches in the full run |
| Full runs | R1: 171 s, 3,648 episode records. S1: 235 s, 3,328 test plus 384 training episodes. Manifests with SHA-256 hashes are in each run folder. |

Reproduce from the repository root:

```sh
PYTHONPATH=. python -m pytest -q tests/test_claude_calibrated_oversight.py
PYTHONPATH=. python -m experiments.run_r1_repaired_reviewer --profile full --out results/runs/claude_r1_repaired_reviewer_v1
PYTHONPATH=. python -m experiments.analyze_r1_repaired_reviewer --run results/runs/claude_r1_repaired_reviewer_v1 --out results/runs/claude_r1_repaired_reviewer_v1/analysis
PYTHONPATH=. python -m experiments.run_s1_reporting_audit --profile full --out results/runs/claude_s1_reporting_audit_v1
PYTHONPATH=. python -m experiments.analyze_s1_reporting_audit --run results/runs/claude_s1_reporting_audit_v1 --out results/runs/claude_s1_reporting_audit_v1/analysis
```

## 8. Independent check of the R1 and S1 write-ups

**What the separate checker did.**

- Reran both analysis scripts. The outputs were byte-identical.
- Confirmed that the source hashes in both run manifests match the code.
- Checked the code logic: previous-request fill, the MSY residual, label
  vectors, no label leakage into reviewer decisions, independent reviewer
  and reference draws, and S1 report, audit, peer and exclusion handling.
- Matched every table cell against the data.

**What it found, and what has been corrected:**

- **Two typos:** a CI bound of −489, not −496, and a mean health of 11.28,
  not 11.29.
- **R1: the Fishery reversal depends on the fill rule** as well as the
  target. The interpretation has been rewritten, and both fills are now
  shown.
- **S1: safety in the audit and peer arms came from excluding cheaters.**
  The reviewer never needed to cut, so the earlier "audits correct beliefs"
  reading was wrong. Rewritten.
- **S1: lying paid in harvest terms in several audit arms.** The fine
  reversed it. Now disclosed.
- **S1, H4:** the Fishery `peer_collude` lie was profitable, so H4 is
  falsified and "lying never paid" was wrong. Corrected.
- **Terminology:**
  - "trusted monitoring with no audits" was self-contradictory; replaced;
  - the AI-control mapping is now stated as an analogy;
  - S1's safety measure is renamed "unsafe-action rate" to distinguish it
    from R1's "unsafe approval rate".
- **Ratio:** "7 times" corrected to 8.4 times.
- **Harvest collusion:** collusion was possible in 23 of 64 contexts; the
  earlier draft called it "rare". Corrected.
- **Disclosures added:**
  - the H4 check in R1 covered only k = 6;
  - Harvest used one weather stream per context;
  - S1's H1 check compared totals, not individual steps.

## 9. S1b and S2 (5 October 2026)

**Gates and checks**

| Check | Result |
| --- | --- |
| S1 refactor | Adding the belief and sanction factors reproduced the original S1 smoke run byte-for-byte |
| Smoke runs | S1b and S2 smoke runs were byte-identical on rerun |
| Gate A | The (corrected, exclusion+fine) cell matched S1 at *d* = 0.5 (0 mismatches) |
| Gate C | With everyone complying, `allow`, `rand2` and `peer` gave identical totals and 0 catches |
| Full test suite | Passed, excluding the PyTorch RL tests |

**Run sizes**

| Run | Episodes | Seconds |
| --- | ---: | ---: |
| S1b | 7,168 | 594 |
| S2 | 5,120 | 396 |

**Reproduce from the repository root:**

```sh
PYTHONPATH=. python -m experiments.run_s1b_ablation_msy --profile full --out results/runs/claude_s1b_ablation_msy_v1
PYTHONPATH=. python -m experiments.run_s2_compliance_deterrence --profile full --out results/runs/claude_s2_compliance_deterrence_v1
PYTHONPATH=. python -m experiments.analyze_s1b_s2 --s1b results/runs/claude_s1b_ablation_msy_v1 --s2 results/runs/claude_s2_compliance_deterrence_v1
PYTHONPATH=. python -m experiments.make_progress_figures
```

## 10. Independent check of the S1b and S2 write-up and figures

**Confirmed.** Every table value, interval and verdict, as well as the gate
results and run sizes.

**Corrected in the S1b/S2 results, figures 5 and 6, the figures guide and the the what-we-learned note update:**

- **A-H2.** It is true by construction, because under exclusion the belief
  switch does nothing.
- **Correction lasted one step.** The reviewer has no memory, so the result
  is limited to memoryless reviewers.
- **Exclusion pushed Harvest below the honest level.** It removes
  extraction as well as stopping lies.
- **Fishery under the MSY target.** Adaptive liars raised harvest slightly,
  and audits cost nothing against them, because they chose to be honest.
- **What S2 tests.** It is non-compliance (hidden over-extraction), not
  deception. Audits are perfect, the fine is flat, and audit rates are far
  above AI-control budgets.
- **Citations.**
  - Avenhaus is now cited only for the quote that was actually checked.
  - The unverified claim attributed to Greenblatt was removed.
  - The Becker wording is now explicitly unread.
- **Hypothesis wording.** C-H4 and C-H5 are now phrased according to their
  criteria and the design's limits.
- **Seeds and gate.** The seed-base wording and the narrower gate C check
  are disclosed.
- **Figures.**
  - Figure 5 now shows values on the bars and the honest reference line, and
    drops the duplicate "fine" column.
  - Figure 6 now plots only the tested points, shows per-agent values and
    total harvest, and states the design limits in its caption.

## 11. S3 (5 October 2026)

**Order of work.** The protocol (the S3 protocol) was committed as `087d392` before any S3 code existed. The runner, tests and Amendment 1 were committed as `7c6494d` before any full run.

**Engineering gates:**
- `pytest -q tests` passes: 254 passed, 11 skipped, including 6 new S3 tests.
- Smoke runs for parts A, C and D are byte-identical on rerun.
- **S2 equivalence.** The S3 Fishery episode reproduces 256 sampled saved S2 episodes. In 2 of them, the mean stock differs in the 10th significant digit. S2's own code gives the identical value on this machine.
- **S1/S1b equivalence on this machine.** The S3 `memoryless`, `trust` and `full` arms equal S1's episode functions in 112 of 112 checks.
- **Environment drift.** Saved S1 episodes are not exactly reproduced here, even by S1's own unchanged code. Example: Fishery `full`, context 48. Hence Amendment 1: every Part D contrast is computed inside the S3 run.

**Full runs.** One run each:

| Part | Seconds | Episodes |
| --- | ---: | ---: |
| A | 28 | 2,944 |
| B | 15 | 1,536 |
| C | 6 | 512 |
| D | 104 | 3,264 |

**Analysis.** `PYTHONPATH=. python -m experiments.oversight.analyze_s3` runs cleanly with `-W error::RuntimeWarning`. A first attempt had two analysis-code bugs, both fixed before any number was written up:
- a pandas index mismatch, which gave NaN for the no-check gain;
- the column name `mode` clashing with `DataFrame.mode`, which made the Part D selections empty.

**Re-checked by hand from the condition table:** the table values and the "two thirds of the gap" figure in the S3 results, and the usefulness-loss values in the Figure 13 caption.

## 12. S4 (5 October 2026)

**Order of work.** The protocol (the S4 protocol) was committed as `5da1d9e` before any code. The runner and tests were committed as `2836785` after the gates and before the full runs.

**Gates:**
- `pytest -q tests`: 258 passed, 11 skipped, including 4 new S4 tests.
- Smoke runs for A and B are byte-identical on rerun.

| Gate | Check | Result |
| --- | --- | ---: |
| 3 | S4 `fine` (q = 1/6, F = 6) equals the S3 `bern` episode | 16/16 |
| 4 | S4 `memory` with compliers equals `fine` with compliers | 16/16 |
| 5 | Part B equals the S3 Part D `memory` episodes (Harvest, Fishery one-step, Fishery MSY) | 24/24 |

**Full runs:**

| Part | Seconds | Episodes |
| --- | ---: | ---: |
| A | 25 | 2,432 |
| B | 7 | 256 |

**Analysis.** `analyze_s4` runs with `-W error::RuntimeWarning`.

**Floor-effect check [post hoc].** An instrumented rerun of `memory`, q = 1/3, d = 0.75, continue, over 64 contexts counted caught-cheater steps with the targeted allowance at 0. Result: 19,410 of 19,733 (98.4%). This instrumentation is diagnostic only. It changes no reported outcome.

## 13. R3 and T1 (7 October 2026)

**Order of work.** The protocols were committed as `49aef43` before any code. The code and tests were committed as `90ce85b` after the gates and before the full runs.

**R3 gates:**
- The smoke run is byte-identical on rerun.
- R3's episode (R2's `nc_episode`) equals `s3.fishery_episode` with S3 seeds in the pilot setting: 16 of 16.

**T1 gates:**
- The smoke run is byte-identical on rerun.
- **Forest:** T1's `random` and `signal` arms equal S3 Part D's `rand1` and `targ1` memory arms with S1 seeds (16 of 16). The `trust` arm equals S3's trust arm (8 of 8).
- **River:** T1 equals C1 Part B's `episode` with C1's own seeds (16 of 16; only the arm label differs).

**A bug caught before the run.** A unit test found that the T1 `trust` arm's protocol name (`report`) matched the report-aimed rule by prefix, so the trust arm would have been audited. It was fixed before any full run (`tests/test_r3_t1.py`).

**Full runs:**
- R3: about 1 minute.
- T1: 21 seconds, 640 episodes.
- Tests: 344 passed, 11 skipped.

**Post hoc diagnostic.** The share of liars ever caught came from a deterministic rerun of T1's arms with instrumentation (`runs/claude_t1_v1/t1_posthoc_distinct_liars.json`). The harm values matched the full run exactly.
