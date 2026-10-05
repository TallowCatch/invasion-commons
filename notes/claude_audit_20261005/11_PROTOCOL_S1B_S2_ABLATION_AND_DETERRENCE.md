# Protocol S1b and S2: what audits actually do, and whether checking deters a cheater who gains privately

**Frozen:** 2026-10-05T00:35Z, before any outcome of these runs existed.

This is a local protocol, not a public preregistration. It is not edited
after results exist; changes go under "Amendments". It builds on the results
of R1 and S1 (files 09 and 10), which were already known when it was
written.

Terms follow `06_LITERATURE_LEDGER.md`. The comparison to AI control
(Greenblatt et al. 2023/24) and to inspection games (Avenhaus et al. 2002) is
an analogy. Our actors are fixed rules plus a one-parameter search, and our
audit rates are far above the 2% used in AI control.

---

## Part A: S1 ablation. Does an audit help by correcting the reviewer, or by removing the cheater?

### A1. Question

In S1, audits made the system safe, but the reviewer never needed to cut
anything. Safety came from excluding caught agents. Part A separates the two
things an audit can do:

- **Belief correction.** The reviewer uses the audited agent's true request
  in this step's decision.
- **Sanction.**
  - `none`: no penalty;
  - `fine`: the caught agent pays 12 and keeps harvesting;
  - `excl+fine`: the caught agent is excluded for 6 steps and pays 12, which
    is S1's rule.

### A2. Design

Everything else is identical to S1: same code, games, reviewer and seeds.
The factors are:

- **Belief correction:** {on, off} × **sanction:** {none, fine, excl+fine}.
- **Protocols:**
  - Harvest: `rand2`, `targ1` and `peer`;
  - Fishery (one-step target, as in S1): `rand2` and `peer`.
- **Misreporting:** fixed *d* = 0.5, plus the S1 adaptive search over
  *d* ∈ {0, 0.25, 0.5, 0.75} on the same 8 training contexts.

The cell (on, excl+fine) reproduces S1 and serves as a replication check.

### A3. Hypotheses

- **A-H1.** With belief correction on and no sanction, unsafe actions are
  lower than in S1's `report` arm at *d* = 0.5. Prediction: Harvest `rand2`
  at least 1 point lower.
  - Falsifier: no reduction.
- **A-H2.** With belief correction off and `excl+fine`, unsafe actions stay
  close to S1's ~0. Prediction: Harvest `targ1` and `rand2` within
  0.5 points of the (on, excl+fine) cell.
  - Falsifier: more than 0.5 points higher.
- **A-H3.** With no sanction, the adaptive search finds *d** > 0 under
  `rand2`.
  - Falsifier: *d** = 0.

---

## Part B: S1 in Fishery with the MSY target

### B1. Question

Do S1's Fishery results hold when the reviewer's target is the productive
level (at least 50 left after harvest), which R1 supports?

### B2. Design

Identical to S1 Fishery, except the target is `msy`. Same protocols, same
*d* levels, same adaptive search.

### B3. Hypotheses

- **B-H1.** Trusting reports at *d* = 0.5 raises unsafe actions above `full`.
  - Falsifier: no difference.

---

## Part C: S2. Cheaters who take more than they are allowed

### C1. Question

In S1, lying rarely paid, probably because the gain from a lie was shared.
S2 gives cheating a **private** gain, so that deterrence can be tested.

Each agent reports its request truthfully. The reviewer sets each agent's
**allowance**: a_i = λ × r_i, using the chance-constrained joint rule.
Honest agents take exactly their allowance. A **non-complying** agent with
level *d* ∈ [0, 1] takes a_i + *d* × (p_i − a_i). It keeps a share *d* of
the cut it was supposed to accept.

The reviewer's prediction assumes compliance. So non-compliance is hidden
extra extraction: a private gain to the cheater, with shared costs in
Fishery and mostly own-patch costs in Harvest.

This is the classic compliance-and-enforcement problem (inspection games;
Becker 1968; Allingham & Sandmo 1972, the latter two at metadata depth only).
Real extraction is only seen when the agent is **audited**, which checks what
it actually took this step.

### C2. Design

**Games and target.**

- **Harvest:** `mix2`, slow renewal, one-step chance constraint. The 2
  stress agents can be non-compliant.
- **Fishery:** `mix4`, **MSY target**, the default from now on because of
  R1. The 4 stress agents can be non-compliant.

**Audits happen after extraction.** An audited agent that took more than
its allowance is caught and pays a fine *F*. There is no exclusion, which
avoids the incapacitation effect seen in S1. Audits never change the current
step's decision.

**Protocols** (each with fine *F* ∈ {0, 6, 12, 24}):

| Protocol | What happens |
| --- | --- |
| `allow` | Allowances only; nobody checks. Independent of *F*. |
| `rand1`, `rand2` | Audit 1 or 2 random agents per step. |
| `targ1` (Harvest only) | Audit the agent whose patch came in furthest below the reviewer's prediction, using a decaying sum of signed shortfalls, factor 0.8. |
| `peer` | The clockwise neighbour observes and reports the agent's extraction. Every non-compliance is caught. |
| `peer_collude` | As `peer`, but a non-complying monitor stays silent about a non-complying neighbour. |

**Adaptive search.** For each protocol × *F* cell, choose a common
*d* ∈ {0, 0.25, 0.5, 0.75, 1}. The choice maximises the cheaters' mean net
payoff (harvest minus fines) on 8 training contexts, with ties going to the
smaller *d*. Training seeds are the same separate bases as S1.

**Test.** 64 test contexts at *d* = 0 (all comply), *d* = 1 (full
non-compliance) and *d* = *d** (adaptive). New seed bases:

| Seed | Base |
| --- | --- |
| Fishery population | 800,000,000 |
| Harvest population | 810,000,000 |
| Weather | 820,000,000 |
| Reviewer | 830,000,000 |
| Reference | 840,000,000 |
| Audit | 845,000,000 |
| Training: Fishery population / Harvest population / weather | 850M / 860M / 870M |

**Outcomes**, never combined:

- unsafe-action rate on executed actions, as in S1;
- total harvest, and honest agents' harvest;
- cheaters' net payoff;
- audits per step;
- catches;
- *d**.

### C3. Hypotheses

- **C-H1 (manipulation check).** Under `allow`, *d** > 0 in Fishery and the
  held-out gain over *d* = 0 is positive.
  - Falsifier: *d** = 0, or the held-out gain interval includes or is below 0.
  - If falsified, deterrence results are uninformative.
- **C-H2.** Under `allow` with *d* = 1, the unsafe-action rate is higher than
  with *d* = 0, in both games.
  - Falsifier: the interval includes 0.
- **C-H3 (deterrence).** Under random audits, *d** is non-increasing in
  *F*, and *d** = 0 at *F* = 24 for `rand2`.
  - Falsifier: *d** increases with *F* for some protocol, or `rand2` at
    *F* = 24 has *d** > 0.
- **C-H4.** In Harvest, `targ1` reaches *d** = 0 at an *F* no higher than
  `rand1` does.
  - Falsifier: `targ1` needs a higher *F*.
- **C-H5.** `peer` gives *d** = 0 for every *F* > 0. `peer_collude` gives
  *d** > 0 for at least one *F* > 0 in Fishery, where adjacent cheaters are
  common.
  - Falsifier: either fails.

---

## Analysis (all parts)

- Paired context-bootstrap contrasts: 4,000 resamples, seed 20261007, 95%
  percentile intervals.
- No p-values, no pooling across games.
- Exploratory figures are made after the analysis and do not change any
  verdict.

## Execution and stop rules

- **Engineering gates:**
  - unit tests pass;
  - a smoke run of 2 contexts × 20 steps is byte-identical on rerun;
  - Part A, the (on, excl+fine) cell, reproduces S1's d = 0.5 results
    exactly for the same seeds;
  - Part C, `allow` with *d* = 0, gives the same decisions as an honest
    reviewer run.
- **Run budget:** one full run per part, 45-minute cap each.
- **Changes:** no changes after seeing the full results. A crash may be
  repaired once, in a new directory, keeping the failed attempt.

## Amendments

(none yet)

### Notes added after the results (2026-10-05; these change no design element)

- **Training seeds.** C2's sentence "Training seeds are the same separate
  bases as S1" contradicts the seed table in this same file, which the code
  follows (850M/860M/870M). The seed table is what was run.
- **Hypothesis A-H2.** It turned out to be true by construction. Under
  exclusion, the belief switch has no effect (see file 12, Part A).
