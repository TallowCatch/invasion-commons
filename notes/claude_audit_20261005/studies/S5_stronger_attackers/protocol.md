# Protocol S5: stronger attackers, and memory rules that close S4's gap

**Frozen:** 2026-10-05T12:05Z, before any S5 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It was written knowing S1–S4
(the R1 results–18). It addresses **caveat 2** (fixed-rule agents) in
`background/caveats_assessment.md`, and S4's open question (a memory rule that
cannot cut below a zero allowance).

Terms follow `literature_ledger.md`. The attackers here are a red team in
the sense of Greenblatt et al. (2023/24): attack policies built against a
fixed protocol. Their results are a lower bound on what an adversary could do.

## Question

1. Do the S2–S4 conclusions survive attackers that are stronger than one
   shared cheating level?
2. Do two new memory rules remove the S4 floor problem?

## Fixed for every cell

- Fishery, MSY target, 6 agents, 80 steps. Stress agents may over-take
  (S2–S4 model: a cheater with level d takes a + d (p − a)).
- Bernoulli audits with probability q per agent per step; perfect detection.
- 32 training contexts (population base 1,350,000,000) and 64 test contexts
  (population base 1,300,000,000). Audit seed base 1,345,000,000.

## Reviewer regimes

| Regime | What a catch does |
| --- | --- |
| `none` | no audits |
| `fine` | flat fine F = 6 per catch (S2–S4) |
| `memory` | S4's targeted allowance (the S4 protocol) |
| `memory_cut` | **new.** The reviewer keeps each caught agent's mean observed over-take d̄ᵢ and chooses the shared cut so that the *expected* total take, a + d̄ (p − a) per agent, meets the target. |
| `memory_cap` | **new.** After its first catch, an agent is checked every step from then on (its take is enforced at its allowance). These extra checks are counted as audits. |
| `fine+memory_cut` | both |

Audit rates: q ∈ {0.02, 0.05, 0.10, 1/6}.

## Attackers (three tiers)

- **T0, fixed group level.** One level d ∈ {0, 0.25, 0.5, 0.75, 1} shared by
  all cheaters, plus S4's reaction to a catch (continue or stop). Same as S4.
- **T1, per-cheater, state-dependent.** Each cheater i has 4 levels from
  {0, 0.25, 0.5, 0.75, 1}, one for each combination of:
  - stock above or below 60;
  - never caught or caught before.

  It also has one "lie low" length L ∈ {0, 2, 5} steps after any audit of
  itself. The search is iterated best response: each cheater in turn
  improves its own parameters by coordinate ascent on the 32 training
  contexts, holding the others fixed. It stops after 3 full rounds, or
  earlier if nothing changes. The search starts from the T0 solution.
- **T2, trained neural policy.** One shared policy network for all cheaters,
  trained with PPO (Schulman et al. 2017; not yet in the ledger).
  - **Observations:** stock / K, own request, own allowance, steps since own
    last audit (capped at 10, scaled), caught-before flag, t / T.
  - **Action:** a level in {0, 0.25, 0.5, 0.75, 1}.
  - **Reward:** the agent's own payoff including fines.
  - **Training:** fixed budget of 6,000 episodes on the training contexts per
    cell, in only 4 cells: `fine` q = 0.05, `memory` q = 1/6, `memory_cap`
    q = 1/6, `fine+memory_cut` q = 0.05.
  - The best checkpoint is chosen by training payoff only.

**Validity check (manipulation check).** Tier n counts as stronger than tier
n − 1 in a cell only if its held-out cheater gain is higher and the paired 95%
interval excludes 0. If T1 or T2 is never stronger than T0 in any cell, that
is reported as a failed manipulation. As in phases 6b and 6e (the project story), more
search does not automatically mean a stronger actor.

## Outcomes

- **Cheater gain:** cheaters' held-out net payoff minus their payoff when
  complying, summed over the group, over 80 steps.
- Harvest per honest agent.
- Share of steps breaking the MSY target.
- Audits used per step (including `memory_cap`'s enforced checks).

## Hypotheses (confirmatory family S5)

- **S5-H1 (deterrence is robust).** Under `fine` at q ≥ 0.10 (e ≥ 0.6, above
  the 0.356 break-even), no attacker tier gains more than 5 held out.
  - Falsifier: any tier with gain > 5 and an interval excluding 0.
- **S5-H2 (the new memory rules fix S4's floor).** Against T0 at q = 1/6, the
  cheater gain under `memory_cut` and under `memory_cap` is each below 25%
  of the no-check gain. (S4's `memory` kept 45%.)
- **S5-H3.** Against `memory_cap` at q = 1/6, T1 and T2 do not raise the gain
  by more than 10 over T0.
- **S5-H4 (a richer attacker finds a gap).** Against `fine+memory_cut` at
  q = 0.05 (e = 0.3, just under break-even), T1 finds a strategy with
  held-out gain > 5.
  - This is a prediction that richer attackers matter. Its failure is
    informative.

Everything else is exploratory.

## Analysis

- Paired context-bootstrap contrasts, 4,000 resamples, seed 20261013, 95%
  percentile intervals.
- Holm correction across the S5 family, as in R2.
- The training-vs-held-out gap is reported for every chosen strategy.

## Engineering gates

1. `pytest -q tests` passes, with new tests for:
   - `memory_cut`'s expected take meets the target for known d̄;
   - `memory_cap` enforces the allowance after a catch;
   - T1 parameters that match a T0 level reproduce T0 episodes exactly.
2. **Replication gate.** `memory` with T0 at q = 1/6 reproduces S4's chosen
   d* and reaction when given S4's seeds.
3. A smoke run is byte-identical on rerun (PPO with a fixed torch seed and one
   thread).

## Run budget and stop rules

- T0 and T1: one full run.
- T2: one training run per cell with the fixed budget, and no tuning of PPO
  hyperparameters after seeing held-out results. Hyperparameters may be
  tuned on training payoff during the smoke stage only.
- A crash may be repaired once, keeping the failed attempt.

## Amendments

### Amendment 1 (2026-10-05T14:46Z, before any full run)

Implementation choices the protocol left open, fixed after the smoke run and
before any full run. No cell, tier or hypothesis is dropped.

- **Cheaters are indexed by rank.** "Cheater i" in T1 means the k-th stress
  agent (sorted by index) in each context, since the population changes from
  context to context.
- **Training contexts get their own audit stream** (seed from the audit base
  plus a "train" tag); test contexts use the S3/S4 derivation. Audit draws are
  shared by all regimes, tiers and levels at the same (context, step, q).
- **Rules of the T1 search.** T0 picks the level and reaction that maximise the
  group's mean payoff (as S4), ties to the smaller level, then "continue". T1
  then maximises each cheater's *own* payoff (iterated best response, as
  written), not the group's. A coordinate moves only on a strict improvement
  (> 1e-9), ties go to the smaller value, and a cheater makes at most 3
  coordinate sweeps per turn. "Lie low" means level 0 for the L steps after any
  audit of that cheater, caught or not.
- **Details of the memory rules.** Under `memory_cut`, caught agents keep the
  plain allowance; only the shared cut uses d̄. Under `memory_cap`, enforcement
  starts the step after the first catch, and each enforced agent-step counts as
  one audit (its random draw is not counted again).
- **T2 is evaluated as a stochastic policy.** Levels are sampled from the
  trained policy, with one fixed random stream per (split, context) (new seed
  base 1,348,000,000). It is not evaluated by taking the most likely action.
  Reason: in the smoke run the most-likely-action version of a still-uncertain
  policy chose level 1 everywhere and collapsed the stock, which is not the
  policy PPO trained. Checkpoints are scored on all 32 training contexts every
  5 updates (40 episodes per update), at the start and at the end. The best is
  the highest mean group cheater payoff, which for a shared policy ranks the
  same as mean own payoff.
- **PPO hyperparameters.** Four settings were compared on training payoff only
  (2,000 episodes; cells `fine` 0.05, `memory_cap` 1/6, `fine+memory_cut`
  0.05). None was clearly better, so the defaults are kept: 2×64 tanh MLP,
  lr 3e-4, γ = 1, GAE λ = 0.95, clip 0.2, 4 epochs, minibatch 512, entropy
  0.01, reward scale 0.1. One PPO seed per cell (base 1,347,000,000).
- **How the hypotheses are tested.** Bootstrap p-values are twice the share of
  resamples on the wrong side of the threshold, capped at 1. Holm correction
  covers all 9 components of H1–H4 (H1: 4, H2: 2, H3: 2, H4: 1).
  - H1 is falsified if a tier has a gain above 5 and a Holm-adjusted p < 0.05
    for gain > 0. The uncorrected 95% interval version is also reported.
  - H4 uses the same rule in the other direction: it is supported if the gain
    is above 5 and the Holm-adjusted p for gain > 0 is below 0.05. Whether the
    interval lies wholly above 5 is also reported.
  - H2 compares a ratio with 0.25, and H3 compares a difference with 10. Each
    component is supported if its estimate is below the threshold and its
    Holm-adjusted p is below 0.05. It is falsified if the interval lies above
    the threshold, and inconclusive otherwise.
  - The 4 named T2 cells do not include a `fine` cell with q ≥ 0.10, so H1 is
    tested on T0 and T1 only.
- **Training-vs-held-out gap.** It uses an unpaired bootstrap, because
  training and test contexts are different populations.
- **Added exploratory outcome.** Each cheater rank's own gain, and its maximum
  across ranks, is reported because T1 and T2 maximise each cheater's own
  payoff while the primary outcome is the group sum.
- **Runtime.** No reduction is needed. Smoke timing gives about 25–45 minutes
  for T0 + T1 and about 15 minutes for T2 on one core, within budget.
