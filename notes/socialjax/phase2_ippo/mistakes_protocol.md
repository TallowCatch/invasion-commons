# Mistakes test: an overseer that misses and wrongly flags (written 21 Nov 2026, before any run)

**What it tests.** Two of the general claims in `notes/phd_plan/2026-11-21_general_formulation.md`:
- **C1 (deterrence):** over-harvesting stops paying when the chance of being checked × (1 − miss rate) × penalty is at least the gain.
- **C2 (cost to honest agents):** at a fixed average penalty, false alarms cost honest agents the same on average whatever the number of checks.

It is also the first "less capable overseer" result for the bounding question.

**Setting.** Same as stage 1 (`protocol.md`):
- Harvest Open, 7 IPPO agents, individual rewards;
- rule m = 3, random checks, window 1;
- 1e8 steps from scratch, seeds 1–3, last 10 updates reported.

**One change to the wrapper (21 Nov), made before any run with false alarms.** A false alarm now falls only on an honest harvest: an apple taken within the rule, judged wrongly. Before, it could fall on any checked agent, even one standing still. That version is noise an agent cannot act on, and at our settings it would have cost an honest agent about 250 apples a game. No earlier result is affected, because every earlier run had both mistake rates at 0.

## Runs (24)

| Arm | Mistake | Checks k | Average penalty c (apples, perfect inspector) | Runs |
|---|---|---|---|---|
| M | misses α = 0.5 | 1, 7 | 1, 2 | 12 |
| F | false alarms on honest harvests β = 0.25 | 1, 7 | 1, 2 | 12 |

The penalty per catch is P = 7c/k, as before.

## Predictions

**What changes for the agent.** Taking an illegal apple rather than a legal one costs on average (k/7) × (1 − α − β) × P = (1 − α − β) × c apples. In the M arm that is 0.5c, and in the F arm it is 0.75c.

| Arm | Effective average penalty | Prediction, from stage 1 (dies at 0.5, survives at 1) |
|---|---|---|
| M, c = 1 | 0.5 | **orchard dies**, about 68 apples, at both k |
| M, c = 2 | 1.0 | **orchard survives**, close to the perfect inspector at c = 1 (441–477 apples) |
| F, c = 1 | 0.75 | **open**: this is between the tested 0.5 and 1, so it places the threshold |
| F, c = 2 | 1.5 | **orchard survives** |

- **The cost to honest harvesting in the F arm.** Each legal apple costs on average β × c = 0.25 or 0.5 apples, which is less than the apple itself. So agents should keep harvesting legally. Apples collected should be at or a little below the perfect-inspector numbers at the same effective penalty.
- **k at a fixed average penalty (C2).** On average the cost is the same at k = 1 and k = 7. But at k = 1 each wrongful fine is 7 times larger and 7 times rarer.
  - If k = 1 does clearly worse than k = 7 in the F arm (seed ranges not overlapping), learners care about how uneven the losses are, not just the average. That is a result in itself.
  - If they match, C2 holds.

## What would change the plan

- If M at c = 2 fails, the (1 − α) scaling in C1 is wrong for learners. Then misses hurt more than their average effect, and the general claim needs changing.
- If F at c = 2 fails, false alarms do more than shrink the deterrence gap, for example by putting agents off harvesting altogether. Check the legal-harvest counts.
