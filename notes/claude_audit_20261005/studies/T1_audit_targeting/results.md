# Results T1: aiming audits at the largest report fails, but not for the reason we thought

**Protocol:** `protocol.md` in this folder, frozen at 2026-10-07T11:31Z (commit `49aef43`) before any T1 code existed. The code and tests were committed as `90ce85b` after the gates and before the full run.

**A bug was found before the full run.** A unit test showed that the trust arm's protocol name also matched the report-aimed rule, so the trust arm would have been audited. It was fixed, and the gates rerun, before any full run. There are no protocol amendments.

**Run:** `results/runs/claude_t1_v1/`, one full run in 21 seconds. Tables are in `notes/claude_audit_20261005/runs/claude_t1_v1/`.

**Status of the numbers.** Verified. The intervals are 95% paired context-bootstrap intervals (4,000 resamples). The coverage numbers ("share of liars ever caught") come from a deterministic diagnostic rerun of the same arms after the results were seen, and are marked **[post hoc]**.

## Question

Does aiming audits at the agent whose (memory-corrected) report is largest catch fewer under-reporters than auditing at random? And does aiming at a physical signal do better?

## What was done

**Three games:**
- Fishery: 4 greedy agents, MSY target;
- Forest: 2 greedy agents;
- River: the compositional game.

The greedy agents under-report by half. The reviewer remembers each caught lie. There are no fines, and one audit per step.

**Arms:**
- trust (no audits);
- random;
- report-aimed;
- in Forest only, signal-aimed: the plot doing worst against prediction.

64 new populations per game.

## What came out

**Harm, as a percentage of steps:**

| Game | Trust | Random | Report-aimed | Signal-aimed |
| --- | ---: | ---: | ---: | ---: |
| Fishery (steps over the MSY limit) | 98.9 | **7.4** | 96.2 | — |
| Forest (unsafe steps) | 1.95 | 1.27 | 1.39 | **0.51** |
| River (unsafe steps) | 21.0 | **2.8** | 6.0 | — |

**Harm left, as a share of the harm with no audits:**

| Game | Random | Report-aimed | Signal-aimed |
| --- | --- | --- | --- |
| Fishery | 7.5% [6.5, 8.5] | 97.3% [96.1, 98.5] | — |
| Forest | 65% [51, 81] | 71% [58, 84] | 26% [16, 39] |
| River | 13% [10, 18] | 29% [22, 36] | — |

**Report-aimed minus random, harm in percentage points:**

| Game | Difference |
| --- | --- |
| Fishery | +88.8 [+87.4, +90.2] |
| Forest | +0.12 [−0.27, +0.61] |
| River | +3.2 [+1.7, +5.0] (replicates C1's +3.4 on new seeds) |

**Mechanism:**

| Game | Audits landing on a liar: random | Audits landing on a liar: report-aimed | Liars ever caught **[post hoc]**: random | Liars ever caught **[post hoc]**: report-aimed |
| --- | ---: | ---: | ---: | ---: |
| Fishery | 67% | **100%** | 100% | **25%** |
| Forest | 32% | 16% | 100% | 48% |
| River | 34% | **72%** | 100% | 76% |

**Hypotheses:**
- **T1-H1 (report-aimed is worse than random in at least 2 of 3 games): supported**, in Fishery and River. Forest shows no difference.
- **T1-H2 (report-aimed audits land on liars less often): falsified**, holding in 1 of 3 games only. In Fishery and River, report-aimed audits land on liars *more* often than random ones.
- **T1-H3 (signal-aimed is no worse than random in Forest): supported, and better.** Signal-aimed minus random harm is −0.76 points [−1.04, −0.47].

## What it means

**What the data show.** Aiming audits at the largest report does let more harm through in two of three games. But the failure is about **coverage**, not concealment.

- Report-aimed audits keep checking the same agent. In Fishery, the greedy liars' halved reports are still the largest requests, so the audit catches a liar at every step. But it is always the same liar: only 1 of the 4 is ever caught.
- Once memory has corrected that liar, the reviewer's belief about it is right. Auditing it again adds nothing, while the other three keep under-reporting unseen.
- Random audits, and in Forest signal-aimed audits, reach every liar sooner or later. Memory then corrects each one.

**This revises Conjecture C1** (`propositions.md`). The earlier explanation, that under-reporters hide by keeping their own reports small, is not what happens. A *deterministic* aim based on reports concentrates the audit budget on agents the reviewer has already corrected. That is the same lesson as the predictable schedule in S3: audits work when they are unpredictable and cover everyone.

**Signal-aimed audits worked best in Forest**, even though they landed on liars less often than random audits (12% against 32%). **[post hoc]** The likely reason is that they check where harm is actually building up, the plots doing worse than predicted, so the true information arrives where the decision matters most. This is an explanation, not a tested result.

**What it does not show:**
- adaptive liars, who could exploit a deterministic aim on purpose;
- aiming rules that mix randomness with a signal;
- per-agent signals in Fishery or River, where none exist because the resource is shared.

## What we now believe

**Claim 5, revised:** how audits are aimed matters because a predictable or deterministic aim leaves most agents unchecked. A known schedule (S3) and a fixed report-based aim (T1) both fail for this reason. Random audits, or audits aimed at a physical signal of harm, reach everyone.

We are confident about the coverage mechanism in Fishery and River (25% and 76% of liars ever caught). Forest's report-aimed result does not differ from random.
