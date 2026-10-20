# Results L6: language-model foresters next to a programmed over-cutter (gpt-oss-120b)

- **Protocol:** `protocol.md` in this folder (Amendment 1: gpt-oss only; F = 0, set by the pilot gate).
- **Data:** 40 games (4 cells × 10 populations), stored in `claude_l6_v1/` on the `l2-results` branch.
- **Analysis:** `experiments/oversight/analyze_l6.py`, written before any game, run on 13 Oct after all 40 games were complete. Summary: `runs/claude_l6_v1/l6_summary_gpt-oss_120b-cloud.json`.

## Pre-registered outcomes

Forester over-take rate: the share of a forester's rounds with a cut more than 0.06 units above its allowance, by ring distance d from position 0.

| Cell | d = 1 | d = 2 | d = 3 | Rounds breaking the 5% constraint | Collapsed |
|---|---:|---:|---:|---:|---:|
| Control, hidden | 25.7% | 27.7% | 27.5% | 31% | 0 |
| Control, visible | 31.8% | 29.5% | 29.5% | 37% | 0 |
| Defector, hidden | 49.8% | 48.8% | 48.0% | 74% | 0 |
| Defector, visible | 58.0% | 56.8% | 50.5% | 86% | 0 |

| Hypothesis | Estimate [95% interval] | Holm p | Verdict |
|---|---|---|---|
| H1: neighbours' plots less healthy under D | −1.09 health [−1.94, −0.22] | 0.016 | **holds** (the manipulation reached the neighbours) |
| **H2: over-taking at d = 1 higher under D (primary)** | **+25.1 points [+20.2, +30.6]** | < 0.001 | **holds** |
| H3: the D − C difference is larger when cuts are visible | +2.2 points [−4.8, +9.5] | 0.29 | **does not hold** |
| H4: the effect falls with distance (d = 1 against d = 3), descriptive | +4.4 points [−0.4, +8.8] | — | no clear gradient |

**Minimum detectable effects** (80% power): 8 points for H2, 11 points for H3.

## Post hoc: how the spillover happened

This analysis was decided after seeing the results.

| Cell | Reviewer's cut factor | Forester request | Forester allowance | Forester actual cut (units) |
|---|---:|---:|---:|---:|
| Control, hidden | 0.76 | 2.24 | 1.73 | 2.37 |
| Control, visible | 0.73 | 2.25 | 1.65 | 2.31 |
| Defector, hidden | 0.36 | 2.00 | 0.82 | 2.34 |
| Defector, visible | 0.22 | 1.83 | 0.53 | 2.24 |

With the over-cutter present, the reviewer cut everyone's requests much harder (factor 0.76 → 0.22–0.36), so forester allowances fell by half or more. The language-model foresters kept cutting about the same amount (2.2–2.4 units) and even asked for slightly less. So the rise in over-taking comes mainly from **allowances shrinking under a shared cut**, not from foresters cutting more or copying their neighbours. This also explains why:
- the rise is almost the same at every distance (H4), because the cut is shared by all six agents;
- seeing neighbours' cuts adds nothing (H3).

## What it means

- **What it shows:** one over-cutter made the other agents break their allowances far more often, across the whole ring, and pushed the forest past its safety constraint in 74–86% of rounds against 31–37%.
- **How:** the cheating spread through the oversight mechanism itself. The reviewer's shared cut passed the cost of one agent's damage onto everyone's allowance, and the language-model agents did not accept the tighter limits.
- **What it does not show:** there is no evidence of social contagion (imitating neighbours). Whether stricter allowances caused over-taking, or the models simply cut what they felt was reasonable whatever the allowance, is not separated by this design.
- **Link to earlier results:** this matches the L2 post hoc spillover check, where about half the peer effect ran through the shared cut. It also matches the programmed-agent finding that lowering everyone's allowance to absorb cheaters backfires.
- **Scope:** one model (gpt-oss-120b), no fine (F = 0, set by the pilot gate), 10 populations.
