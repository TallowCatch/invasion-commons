# Results C1: a third game where harm comes from a combination of actions

**Protocol:** `studies/C1_compositional_harm/protocol.md`, frozen at 12:10Z
before any code. Amendment 1 (calibration and implementation choices, 12:15Z)
was written before any test context was run.

**Run:**
- `results/runs/claude_c1_v1/`: one full run, no reruns.
- Calibration is in `results/runs/claude_c1_calibration_v1/`.

**Status of the numbers:** verified, read from `analysis/c1_*.csv`.
Intervals are 95% paired context-bootstrap intervals (4,000 resamples).
Holm correction is applied across C1-H1 to C1-H4.

## Question

Do the reviewer findings hold in a game where harm comes from **combining**
actions, not from their sum?

## What was done

**The game.** Six agents discharge two reagents, three agents for each.
- River damage per step is κ · X · Y + λ (X + Y), with κ = 32 and λ = 1.6.
  The reaction term is 75–94% of damage.
- An additive control has damage λ′ (X + Y), with λ′ matched so both games
  have the same safe total at the starting river quality.

**Comparisons:**
- four reviewers: `joint`, a fair-share per-agent `quota`,
  `local_optimistic` and `local_bounded`;
- Part B: two under-reporters, and 1 random or 1 targeted audit per step with
  memory.

64 test contexts.

## What came out

**Open loop** (risky requests approved / safe requests cut):

| Reviewer | Compositional game | Additive control |
| --- | --- | --- |
| joint | **0.3%** / 0.03% | 1.7% / 0.3% |
| quota | 0% / **82%** | 0% / **89%** |
| local_optimistic | **100%** / 0% | 100% / 0% |
| local_bounded | 0% / 89% | 0% / 100% |

**Closed loop, compositional game,** compared with joint:
- local_optimistic: +20.0 points of steps unsafe [16.2, 24.1];
- no reviewer: +31.5 points [24.1, 39.6];
- quota and local_bounded: about as safe as joint, but 4.8 and 11.5 less
  payoff.

**Part B, compositional game.** Targeted audits gave **+3.4 points** more
unsafe steps than random audits [1.8, 5.3].

**Hypotheses:**
- **C1-H1** (the quota matches joint in the additive game): **not
  supported.** The quota cut 89% of safe requests there too.
- **C1-H2** (the quota fails in the compositional game): **supported.**
- **C1-H3** (optimistic local > 20% risky approved, joint < 2%):
  **supported.** This is close to true by construction.
- **C1-H4** (targeted audits beat random): **not supported.** The effect ran
  the other way.

## What it means

**What it shows:**
- **The reviewer ranking carries over to a third, structurally different
  game.** Joint review stays near-exact with a known model, and per-agent
  rules err in their usual directions.
- **A fair-share quota fails whenever demand is unequal**, whether or not
  harm is compositional. **[post hoc]** The experiment therefore did not
  isolate a quota failure caused by compositional harm. Its premise, that a
  quota works in an additive commons with unequal agents, was wrong.
- **Audits targeted by report size can backfire against under-reporters.**
  **[post hoc]** Under-reporters make their own reports small, so a rule that
  audits the most influential report looks away from them. This contrasts
  with S1 Harvest, where targeting used a physical signal and helped. See
  `propositions.md`, Conjecture C1.

**What it does not show:**
- whether the compositional structure changes anything when the model is
  unknown (it was known here);
- targeting that combines report size with a physical signal.

## Design caveats

- The game parameters were calibrated on 8 calibration contexts, which is a
  reasonable but arbitrary target region.
- The quota rule counts a request as "restricted" if any one agent is cut.
  That makes its usefulness loss look large. On average it kept 81% of the
  requested total.
