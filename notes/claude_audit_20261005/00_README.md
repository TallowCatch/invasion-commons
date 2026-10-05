# Claude audit of the invasion-commons experiments (5 October 2026)

This folder is a plain-language record of what the project's experiments did,
what they found, what is wrong with their design, and what to do next. It was
written after reading the repository, re-checking the committed data, and
re-analysing the saved results of the most recent experiment.

**No new experiments were run.** Some analyses here are *post hoc*: they were
computed on already-saved data after the results were known. Such analyses are
labelled **[post hoc]**. They can suggest what is going on, but they are not
confirmation, because the decision to look was made after seeing the outcome.

Source state: branch `codex/harvest-oversight-gap-stagea`, commit `8812f4f`.
Your local copy was on the same commit. I could not open a shell on your
computer, so I could not check whether you have uncommitted local changes.

## Files

| File | What it contains |
| --- | --- |
| `01_PROJECT_STORY.md` | Each experiment from February to September, in order: question, what was done, result, what it means. |
| `02_REVIEWER_EXPERIMENT_EXPLAINED.md` | The most recent and most important experiment, with my re-analysis of its data. |
| `03_DESIGN_ISSUES.md` | Problems in how the experiments were designed (not how the paper is worded), ranked by impact, each with evidence and a fix. |
| `04_WHAT_WE_LEARNED_AND_NEXT.md` | What the evidence supports overall, and a concrete plan for what to do next. |
| `05_VERIFICATION_LOG.md` | What I checked, reran or could not check, and how to repeat it. |
| `06_LITERATURE_LEDGER.md` | Verified sources, the terms they use, and how our terms map onto them (scalable oversight, AI control, inspection games, deterrence, commons). |
| `07_PROTOCOL_R1_REPAIRED_REVIEWER.md` | Protocol for R1, frozen before the run. |
| `08_PROTOCOL_S1_REPORTING_AND_AUDITS.md` | Protocol for S1, frozen before the run, with one amendment made before the full run. |
| `09_RESULTS_R1_REPAIRED_REVIEWER.md` | R1 results: calibrated reviewers, limited checking, and the Fishery target test. |
| `10_RESULTS_S1_REPORTING_AND_AUDITS.md` | S1 results: misreporting, audits, peer reports and collusion. |
| `scripts/` | The two re-analysis scripts and their outputs, so every new number here can be reproduced. |

Experiment code and data in the repository:

- shared reviewer code: `fishery_sim/calibrated_oversight.py`;
- runners and analyses: `experiments/claude_oversight_common.py`,
  `experiments/run_r1_repaired_reviewer.py`,
  `experiments/run_s1_reporting_audit.py`,
  `experiments/analyze_r1_repaired_reviewer.py`,
  `experiments/analyze_s1_reporting_audit.py`;
- tests: `tests/test_claude_calibrated_oversight.py`;
- raw runs and analysis tables: `results/runs/claude_r1_repaired_reviewer_v1/`
  and `results/runs/claude_s1_reporting_audit_v1/`.

Suggested reading order: the update at the top of 04, then 09 and 10 (the
new experiments), then 02, 03 and 01 for background.

## The bottom line in five sentences

1. The project asks what a reviewer must know to block harmful *combined*
   resource requests without blocking harmless ones.
2. In the latest experiment, the reviewers know the exact rules of the game
   and the agents never lie or adapt. So who wins is largely decided by
   arithmetic: a reviewer that adds up everyone's requests is exact, and the
   two "local" reviewers are deliberately loose approximations of it.
3. In Harvest, the headline numbers were mostly caused by the size of an
   uncertainty allowance for weather. It is about 3.6 times larger than the
   only safety condition that actually binds in these cases requires. I
   recomputed the decisions with an allowance sized for that condition
   **[post hoc]**. Then the joint reviewer restricts 0 of 3,687 safe
   requests instead of 2,521, and the "optimistic local" reviewer starts
   approving 95 risky ones. Harvest then looks like Fishery.
4. The most interesting finding is probably about the safety *target*, not
   information. In Fishery, "safe" means the stock stays above 10. But the
   stock regrows fastest when 50 is left after harvest. So a reviewer that
   approves everything that is safe for one step pushes the stock well
   below its most productive level. Better information produced *less*
   long-run harvest: 1,084 → 835 → 551.5 as inspection went 0 → 3 → 6. The
   target was never varied, so this explanation still needs testing.
5. The next useful work: first fix the target and the uncertainty allowance
   and rerun the existing design cheaply. Then test the part that is
   actually open: agents who can misreport, reviewers who must pay to check,
   and actors who adapt to the reviewer.

## Glossary (used throughout)

- **Commons / shared resource.** A stock (fish, or plant health on patches)
  that everyone draws from and that regrows over time.
- **Agent / actor.** One participant that asks to take some amount each step.
- **Policy / strategy.** The fixed rule an agent uses to decide its request.
  Here these are small sets of numbers (thresholds and fractions), not
  learning agents and not live language models.
- **Request (proposal).** What an agent asks to take this step, as a fraction
  of its maximum. The six requests together are the *joint request*.
- **Reviewer (overseer, monitor).** A rule that looks at some of the requests
  and may scale *all* of them down by the same factor (1, 0.75, 0.5, 0.25 or 0).
- **Inspection budget (k).** How many of the six requests the reviewer
  actually sees (0, 3 or 6). Unseen requests are assumed to be the maximum.
- **Joint reviewer.** Adds up the requests it sees (and assumes the maximum
  for unseen ones), then predicts the next state exactly.
- **Bounded local reviewer.** Builds its prediction from per-agent reports
  while assuming the worst about other agents (Fishery: everyone takes as
  much as the largest request; Harvest: every neighbour takes the maximum).
- **Optimistic local reviewer.** Ignores what other agents do to the
  resource (Fishery: counts only the largest request; Harvest: ignores spillover
  from neighbours).
- **Coupled local reviewer.** Local reports that also include neighbours'
  inspected contributions. Mathematically identical to the joint reviewer.
- **One-step label ("reference").** Whether the original joint request would
  leave the system safe *one step later*. Exact in Fishery. In Harvest it is
  estimated from 128 random weather draws: "safe" if the risk is clearly
  below 5%, "risky" if clearly above, otherwise "unresolved".
- **Safe restricted.** A one-step-safe request that the reviewer scaled down
  (even partially). This is the "unnecessary restriction" error.
- **Risky approved.** A one-step-risky request the reviewer let through
  unchanged. This is the "missed harm" error.
- **Context.** One freshly generated population of six agents plus its
  random seeds. This is the *independent unit*: things measured within the
  same context are not independent of each other.
- **Open loop vs closed loop.** *Open loop*: judging requests recorded from a
  run in which nobody intervened. *Closed loop*: actually running the game
  with the reviewer in charge, so its decisions change the future states.
- **Uncertainty allowance (margin).** In Harvest, weather adds noise. The
  reviewers subtract a safety buffer from their predictions so that they
  stay safe despite noise.
- **MSY (maximum sustainable yield).** The largest harvest that can be taken
  every step forever. For this kind of regrowth it happens when the stock
  after harvest is half its maximum.
- **Confound.** Two things change together, so you cannot tell which one
  caused the effect.
- **Pseudoreplication.** Treating repeated measurements of the same unit
  (time steps, seeds, agents) as if they were independent samples. This
  makes results look more certain than they are.
- **Post hoc.** Decided after seeing the data. It is fine for generating
  ideas, but not for confirming them.
