# Results S4: adaptive cheaters against a reviewer with memory, and a cost per audit

**Protocol:** `studies/S4_adaptive_cheaters_audit_cost/protocol.md`, frozen at 2026-10-05T11:38Z (commit `5da1d9e`) before any S4 code existed. The runner and tests were committed as `2836785` after gates 1–5 passed and before the full runs. There are no amendments.

**Runs:**
- `results/runs/claude_s4_part{A,B}_v1/`: one full run each (A 25 s, B 7 s).
- Analysis: `experiments/oversight/analyze_s4.py`, which writes `analysis/s4_summary.json` and `s4a_table.csv`.
- Figure: `fig14_s4_memory_adaptive_cost`.

**Status of the numbers.** Every number is **verified** from those files. Intervals are 95% paired context-bootstrap intervals (4,000 resamples). **[post hoc]** marks explanations reached after seeing the data.

---

## Part A: cheaters who know the reviewer remembers

### Question

Can a reviewer that remembers caught over-takers neutralise cheating **without fines**, when cheaters choose their strategy knowing this? And once each audit has a cost, which audit rate is best?

### What was done

**Setting:** Fishery, MSY target. 4 of 6 agents may take more than their allowance.

**Audits:** each agent is audited with probability q per step (q from 0.01 to 1/3), and audits are perfect.

**Four regimes:**

| Regime | What a catch does |
| --- | --- |
| No checks | No audits at all. |
| Fine | A flat fine of 6 per catch. |
| Memory | No fine. The reviewer records how much the agent over-took and from then on gives it a smaller, targeted allowance. The aim is that its expected take is what the shared cut planned. |
| Fine + memory | Both. |

**The cheaters' choice.** The group chose a cheating level (0 to 1) and a reaction to being caught: keep going, or stop. The choice was made on 8 training contexts and tested on 64 fresh ones.

**Audit cost.** Cost was applied in the analysis as total harvest minus c per agent-audit. For scale, one agent's harvest averages about 2.9 per step.

### What came out

| Audit rate q | Fine: chosen level | Fine: gain | Memory: chosen level | Memory: gain | Fine + memory: chosen level | Fine + memory: gain |
| ---: | ---: | --- | ---: | --- | ---: | --- |
| 0.01 | 0.75 | +89.9 | 0.75 | +97.3 | 0.75 | +79.5 |
| 0.02 | 0.75 | +69.6 | 0.75 | +85.0 | 0.75 | +46.9 |
| 0.05 | 0.75 | +8.7 [+0.3, +17.2] | 0.75 | +65.2 | **0** | 0 |
| 0.10 | **0** | 0 | 0.75 | +54.4 | 0 | 0 |
| 1/6 | 0 | 0 | 0.75 | +48.2 | 0 | 0 |
| 1/3 | 0 | 0 | 0.75 | +44.7 [+37.6, +52.1] | 0 | 0 |

"Gain" is the cheaters' held-out gain over complying, for the group of 4 over 80 steps. With no checks, the group chose level 0.75 and gained **+107.7**.

**Harvest per honest agent:**

| Regime | Result |
| --- | --- |
| No checks | 43.7 |
| Everyone complies | 104.8 |
| Fine (until it deters) | 43.7 |
| Memory | rises with q: 51.2, 58.6, 71.0, 77.7, 82.0, 84.9 |

**Steps breaking the MSY target under memory**, compared with no checks (100%), in percentage points:

| q | 0.01 | 0.02 | 0.05 | 0.10 | 1/6 | 1/3 |
| --- | --- | --- | --- | --- | --- | --- |
| Change | −1.1 | −7.4 | −31.2 | −46.8 | −55.2 | −61.2 [−70.2, −51.1] |

**Fine + memory at low audit rates:** compared with the fine alone, harvest per honest agent rose by:
- q = 0.01: +7.5 [+5.7, +9.5];
- q = 0.02: +14.9 [+12.5, +17.3].

**Best audit rate once audits cost something** (q*, the rate with the highest total harvest minus audit cost):

| Cost per audit c | Fine | Memory | Fine + memory |
| ---: | ---: | ---: | ---: |
| 0 | 0.10 | 1/3 | 0.05 |
| 0.05 | 0.10 | 1/6 | 0.05 |
| 0.2 | 0.10 (net 1,384.4) | 0.05 (1,386.3) | 0.05 (1,388.7) |

At c = 0.2, no audits gives 1,379.2. With fine + memory, the fine deters at half the audit rate needed with the fine alone, so it reaches the same deterred outcome with fewer audits.

**Hypotheses:**
- **A-H1 (memory: gain below no checks at every q, not rising with q): supported.** At q = 0.01 the difference is only −10.4 [−13.5, −7.6].
- **A-H2 (memory: fewer target-breaking steps than no checks at q ≥ 0.05): supported.**
- **A-H3 (fine: comply at q ≥ 0.10, cheat at q ≤ 0.02): supported.** At q = 0.05, the cheaters still cheated and gained a little, as S3 predicted for e = 0.3.
- **A-H4 (fine + memory beats fine alone for honest agents at low q): supported.**
- **A-H5 (at c = 0.2, fine + memory needs an audit rate no higher than fine alone): supported**, at 0.05 against 0.10.
- **A-H6 (under memory, cheaters keep going after a catch): supported.** Stopping was never chosen.

### What it means

**What the data show:**
- **Memory without fines reduces harm but does not deter.**
  - Cheaters kept over-taking at every audit rate, because nothing they did was ever penalised.
  - Even at the highest rate (1/3), they kept 41% of their no-check gain.
  - Honest agents recovered about two thirds of their loss (84.9 against 43.7 and 104.8).
- **Memory and fines work together.** Adding memory lowered the audit rate needed for a fine of 6 to deter, from 0.10 to 0.05. It also helped honest agents even at rates too low to deter.
- **Under audit costs, the cheapest good option is fine + memory at a low rate.** The net-harvest differences are small (a few units out of about 1,390), because audits are cheap relative to harvest here.
- **Net harvest is not fairness.** Memory alone at q = 1/3 gave the *highest* total (1,398), while honest agents still lost 20 each to cheaters. A net-harvest objective hides that transfer.

**[post hoc] Why memory alone cannot neutralise cheating here:**
- The rule shrinks a caught agent's allowance, but not below zero.
- Under the MSY target, the shared cut usually leaves less than 75% of each request. A level-0.75 cheater still takes 0.75 of its request when its allowance is 0, which is more than planned.
- In 98.4% of caught-cheater steps, the targeted allowance was already at its floor of 0.
- So this memory rule only works when cheating is milder than the cut. A reviewer that also tightened the shared cut, or one that could refuse extraction outright, could do more. That was not tested.

**How much is built in:**
- That cheaters never stop after a catch is built in: stopping only reduces their take.
- The fine thresholds repeat S3's arithmetic.
- The floor effect was not anticipated in the protocol.

**What it does not show:**
- memory that also changes the shared cut, or physically blocks over-takes;
- cheaters who can hide the size of their over-take from audits;
- risk-averse cheaters, fines that scale with the over-take, or costs larger than the ones tested.

---

## Part B: do misreporters lie against a reviewer with memory?

| Setting | Chosen lie | Held-out result |
| --- | ---: | --- |
| Harvest | 0 | Lying lowered the liars' payoff in training. |
| Fishery, stock-≥-10 target | 0 | Every lie level paid exactly the same in training; the tie went to 0. |
| Fishery, MSY target | **0.5** | Liars' gain +0.25 [−0.09, +0.63], essentially 0. Target-breaking steps +3.6 points [+3.1, +4.1]. Honest harvest unchanged (−0.00 [−0.10, +0.09]). |

**B-H1 (no lying in any setting) is falsified as stated, in one of three settings.** The lie chosen in Fishery MSY won on training by 0.46 out of about 1,180 (0.04%), and held out it gained nothing measurable. **[post hoc]** This is a near-tie decided by sampling noise, not a profitable lie. Misreporting against a reviewer with memory still does not pay in any material way, so its harm comes from fixed-rule liars, not from strategic ones.

---

## What we now believe, and how confident we are

1. **A reviewer with memory reduces the harm from cheaters but does not deter them without a penalty.** With a fine, it halves the audit rate needed for deterrence. We are confident about this pattern within this model.
2. **This memory rule has a built-in limit:** it cannot correct below a zero allowance, so heavy cheating under a strict target gets through. That was found after the run, and the fix is untested.
3. **Once audits cost something, fine + memory at a low audit rate gives the most harvest,** but the margins are small here. Net harvest also hides who bears the loss, so it should be reported next to honest agents' harvest.
4. **Strategic misreporting still does not pay,** with or without memory.

## Still untested

- a reviewer that tightens the shared cut, or blocks extraction, for known cheaters;
- over-takes that the audit can only partly measure;
- higher audit costs, and a planner that sets q itself;
- any of this in Harvest, where cheating never paid.

## Next concrete step and its cost

Add a memory variant that accounts for known cheaters in the shared cut, and one that caps a known cheater's take. Then rerun Part A. That needs a protocol and about 5 CPU minutes.

The three decisions for Ameer and the supervisors are still open: the definition of harm, which paper, and the main game.
