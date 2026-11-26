# Protocol C1: a third game where harm comes from a combination of actions

**Frozen:** 2026-10-05T12:10Z, before any C1 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It addresses **caveat 1** (two
games only) in `background/caveats_assessment.md`.

Fishery and Harvest share three features:
- one number per agent;
- harm that grows with the total taken;
- one shared cut.

GovSim's three scenarios do not add a new structure either: its paper states
that their "underlying resource regeneration dynamics ... are mathematically
equivalent" (Piatti et al. 2024, arXiv:2404.16698v4).

C1 breaks the second feature. Harm comes from a **combination** of actions,
the case Hu & Wang (2026) call "compositional harm" (the literature ledger).

## The game ("two-reagent river")

- **Agents.** 6 agents, 80 steps. Three are *type X* (they discharge reagent
  X), three are *type Y*. Each agent requests a discharge pᵢ ∈ [0, 1] per
  step from a threshold policy, as in Fishery: a low request when the river is
  dirty, a high one when it is clean. Two agents (one of each type) are
  stress agents with high requests.
- **Payoff.** Each agent earns its discharge (production).
- **River.** A river quality Q ∈ [0, 100] starts at 80.
  - Damage this step: D = κ · X_tot · Y_tot + λ · (X_tot + Y_tot), where
    X_tot and Y_tot are the summed discharges of each type.
  - Q' = Q − D + r Q (1 − Q / 100) + ε, with weather ε ~ N(0, σ²).
  - The reaction term κ is large and the linear term λ is small, so the
    harm is mostly compositional.
- **Unsafe.** A step is unsafe if Q' < 30.
- **Additive control (C1-add).** The same game with D = λ′ (X_tot + Y_tot).
  λ′ is chosen so that the total discharge that is safe at Q = 80 equals
  that of the compositional game at the *typical* mix (equal X and Y totals).
- **Parameters** (fixed by a calibration on 8 calibration contexts, base
  1,490,000,000, before any test context is run, then frozen in an
  amendment): κ, λ, λ′, r, σ. They are chosen so that, with no reviewer:
  - about 30–60% of steps are risky in the compositional game;
  - the additive control has a similar share.

## Reviewers (all choose one shared cut from {1, 0.75, 0.5, 0.25, 0} with a 5% chance constraint, as in R1)

| Reviewer | What it computes |
| --- | --- |
| `joint` | damage from all requests together (exact model) |
| `quota` | **per-agent quota.** Each agent may discharge at most S / 6, where S is the total discharge that is safe at the typical mix. Agents over quota are cut to it. This is the natural local rule for an additive commons. |
| `local_optimistic` | each agent checked alone: its own damage with the other type absent |
| `local_bounded` | each agent checked assuming the other type discharges at its maximum |

## Misreporting and audits (Part B)

- The stress agents under-report by half.
- One audit per step:
  - `random`: one agent at random;
  - `targeted`: the agent whose report most changes the predicted damage. In
    the compositional game that is the largest report of the type whose
    total is *larger*, because the product is most sensitive there.
- A caught liar is corrected using memory, as in S3 Part D (reports rescaled
  by its measured lie). No fines or exclusion.

## Measures

- Open loop (requests from no-reviewer runs): usefulness loss and unsafe
  approvals per reviewer.
- Closed loop: total payoff, share of steps unsafe, final Q.
- Part B: share of steps unsafe under trust, random audit and targeted
  audit.

## Hypotheses (confirmatory family C1)

- **C1-H1.** In the additive control, `quota` is within 3 points of `joint`
  on both error rates.
- **C1-H2.** In the compositional game, `quota` either approves more than 5%
  of risky requests or cuts more than 20% of safe ones. (A per-agent quota
  cannot be calibrated when harm depends on the mix.)
- **C1-H3.** In the compositional game, `local_optimistic` approves more than
  20% of risky requests, against under 2% for `joint`. This is close to true
  by construction, and is included to quantify it.
- **C1-H4.** In Part B, targeted audits give fewer unsafe steps than random
  audits in the compositional game, with a paired interval excluding 0.

## Design

- **Independent unit:** the context; 64 test contexts (population base
  1,400,000,000, weather 1,410,000,000).
- **Analysis:** paired context bootstrap, 4,000 resamples, seed 20261014, 95%
  intervals; Holm correction across C1-H1 to H4.

## Engineering gates

1. Unit tests:
   - with κ = 0 the two damage functions agree when λ = λ′;
   - `joint` is exact with σ = 0;
   - the targeted audit picks the analytically most sensitive agent.
2. The calibration amendment is written before any test context is run.
3. A smoke run is byte-identical on rerun.

## Run budget and stop rules

One calibration, one full run, no reruns after seeing results. A crash may be
repaired once.

## Amendments

### Amendment 1 (calibration)

**Written:** 2026-10-05T12:15Z, after the calibration stage and before any
test context (population base 1,400,000,000) was run.

**Frozen values** (in `fishery_sim/two_reagent.py`, `FROZEN`):

| Parameter | Value |
| --- | --- |
| κ | 32 |
| λ | 1.6 (λ = 0.05 κ, ratio fixed before the search) |
| λ′ (additive control) | 22.917 (matching rule below) |
| r | 1.0 |
| σ | 3 |
| Request ranges (`wide_low`) | normal agents: low U(0.00, 0.10), high U(0.25, 0.50); stress agents: low U(0.10, 0.30), high U(0.70, 0.95) |
| Policy thresholds | U(35, 75): request low when Q < threshold |
| Stress agents | one per type, chosen uniformly within the type, per context |

- **Matching rule for λ′.** Headroom at Q = 80: H = 80 + r·80·0.2 − 30 −
  1.645σ (analytic 5% normal quantile). The compositional safe total S solves
  κS²/4 + λS = H at the equal mix, giving S = 2.665. Then λ′ = H/S, so both
  games have the same safe total, 2.665, at Q = 80.
- **Calibration outcome at the chosen point** (8 calibration contexts, no
  reviewer, 80 steps):

  | Game | Steps risky (exact one-step risk > 5%) | Steps starting from Q ≥ 30 | Risky among those |
  | --- | ---: | ---: | ---: |
  | Compositional | 45.6% | 56.6% | 9.7% |
  | Additive | 51.6% | 54.4% | 12.4% |

- **How compositional the harm is.** The κXY term is 94% of the damage at
  the mean high requests and 75% at the mean low requests.

**How the values were chosen.** The search is recorded in
`results/runs/claude_c1_calibration_v1/calibration.json`, produced by
`run_c1_compositional_harm --stage calibrate`.

- **Grid:** 3 request-range sets × κ ∈ {16, 20, 24, 28, 32} × r ∈ {0.7, 0.85,
  1.0, 1.2} × σ ∈ {3, 5, 7}, which is 180 points.
- **Feasibility:**
  - the no-reviewer risky share over all steps is in [30%, 60%] in both
    games;
  - at least 50% of steps start from Q ≥ 30 in both games.
- **Choice:** the feasible point minimising |risky_comp − 0.45| +
  |risky_add − risky_comp|. Ties go to the smaller κ, then r, then σ.
- **Result:** 9 of 180 points were feasible, and the rule picked the point
  above.
- **Coarse look before the grid** (same calibration contexts, not saved):
  - With the Fishery request ranges, the additive control was risky at about
    90% or more of steps at every κ, r and σ tried.
  - The reason is the matching rule. It fixes λ′ at Q = 80, and linear damage
    is larger than the quadratic compositional damage at every total below
    S(80).
  - This is why two request-range sets with a wider low/high gap were added.
  - The three sets are named in `two_reagent.RANGE_SETS`.

**Monte Carlo draws.** These are as planned and were not reduced:
- reviewer chance constraint: 400 draws;
- closed-loop labels: 2,000 draws;
- open-loop labels: 4,000 draws.

**Seeds** (the protocol fixed only the population and weather bases):

| Stream | Seed | Shared across |
| --- | --- | --- |
| Population | 1,400,000,000 + c | – |
| Weather | 1,410,000,000 + c: one standard-normal shock per step, scaled by σ | all arms of a context |
| Reviewer draws | `stable_seed`(1,420,000,000, "closed" or "open", game, c, t) | all arms at the same step (common random numbers) |
| Reference labels | `stable_seed`(1,430,000,000, game, Q, requests); +1 for open loop | – |
| Random audits | `stable_seed`(1,440,000,000, game, c, t) | – |
| Calibration | population 1,490,000,000 + c, weather 1,495,000,000 + c | – |

**Implementation choices the protocol did not pin down.** These were fixed
here, before any test context was run.

1. **Labels.**
   - A step's risk is P(Q′ < 30).
   - Labels follow R1: Monte Carlo with a Wilson 95% interval. A request is
     safe if the upper bound < 5%, risky if the lower bound > 5%, and
     unresolved otherwise.
   - Open- and closed-loop rates are scored over steps that start from
     Q ≥ 30, as in R1's "currently safe state".
2. **No episode end.**
   - Q is clipped to [0, 100] and all 80 steps are run.
   - "Share of steps unsafe" is the share of the 80 steps whose next Q is
     below 30.
3. **`quota`.**
   - S is the largest equal-mix total whose *estimated* risk is ≤ 5%. It uses
     the reviewer's own 400 draws (the same draws as `joint`) and is capped
     at 6.
   - Each agent whose believed request exceeds S/6 is cut to S/6. Others are
     untouched.
   - It does not use the shared-cut menu.
   - A request counts as "approved" if no agent is cut.
   - Retained share = executed total ÷ requested total.
4. **Local reviewers.**
   - `local_optimistic` checks agent i with every other agent absent.
   - `local_bounded` checks agent i alone of its type, with the other type at
     its maximum (3 units). Both are scaled by the candidate cut.
   - The shared cut must pass all six checks.
5. **Targeted audit.**
   - The audited agent is argmaxᵢ ∂D/∂pᵢ · pᵢ, evaluated at the reviewer's
     memory-corrected beliefs before the audit. Ties go to the lowest index.
   - In the compositional game, agent i's score is
     pᵢ · (κ · [other type's total] + λ). In the additive game it is the
     largest belief.
   - **Note:** the protocol's gloss ("the largest report of the type whose
     total is larger") does not follow from this formula in general. With
     equal shares within each type, the two types tie. The formula is
     implemented.
6. **Part B.**
   - All arms use the `joint` reviewer.
   - The liars are the two stress agents and report 0.5 pᵢ.
   - Memory is as in S3 Part D (`memory`): an audit reveals the audited
     agent's true request for that step. There are no fines.
   - Part B runs in both games. C1-H4 uses the compositional game; the
     additive game is descriptive.
   - Part A `joint` (same reviewer draws) is the full-information reference.
7. **Closed-loop error rates** (unsafe approvals and usefulness loss on the
   steps actually faced) are reported as secondary outcomes, as in R1.
8. **Hypothesis tests.**
   - Rates are pooled over contexts (sum of numerators ÷ sum of
     denominators).
   - The bootstrap resamples whole contexts: 4,000 resamples, seed 20261014.
     All hypotheses use the same resample indices.
   - Each hypothesis gets a bootstrap p = (1 + number of resamples in which
     its claim fails) / 4,001.
   - Holm step-down at α = 0.05 is applied across C1-H1 to C1-H4.
   - **Verdicts:**
     - "supported": the claim holds at the point estimate and the Holm
       test rejects;
     - "not supported": the claim fails at the point estimate;
     - "inconclusive": otherwise.
   - **Claims:**
     - **H1:** |quota − joint| < 3 points on *both* open-loop rates
       (additive game).
     - **H2:** quota's unsafe approval rate > 5% *or* its usefulness loss
       > 20% (compositional game, open loop).
     - **H3:** `local_optimistic` unsafe approval rate > 20% *and* `joint`
       < 2% (compositional game, open loop).
     - **H4:** targeted − random share of steps unsafe < 0, with the 95%
       interval excluding 0 (compositional game, Part B).
9. **Smoke profile:** 2 contexts, 20 steps.
