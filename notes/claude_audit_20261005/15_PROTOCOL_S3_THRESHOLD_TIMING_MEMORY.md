# Protocol S3: the deterrence threshold, imperfect audits, cheaters who time their cheating, and a reviewer with memory

**Frozen:** 2026-10-05T11:20Z, before any S3 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It builds on the results of S1,
S1b and S2 (files 10 and 12), which were already known when it was written.

Terms follow `06_LITERATURE_LEDGER.md`:
- "audit rate" corresponds to an *audit budget* in AI control (Greenblatt et al.
  2023/24, arXiv:2312.06942);
- "expected fine" is the expected-penalty idea usually attributed to Becker
  (1968, doi:10.1086/259394, checked at metadata depth only);
- the audit schedules relate to *inspection games* (Avenhaus, von Stengel &
  Zamir 2002, concept only).

These are analogies. Our actors are fixed rules plus a small strategy search.

---

## Parts A and B: where is the deterrence threshold, and do imperfect audits move it?

### Question

In S2, any tested fine (6, 12 or 24) with one random audit per step stopped
fixed-level cheaters in Fishery. Arithmetic done after that run put the
threshold near a fine of 2, below the grid. Parts A and B ask:
- where the switch from cheating to complying actually happens;
- whether it depends only on the **expected fine per agent-step**,
  e = q × s × F;
- whether lower audit rates and imperfect audits shift it in that one
  combination.

Here:
- q is the chance that an agent is audited in a step;
- s is the chance that an audit catches an over-take;
- F is the flat fine per catch.

### Design

**Kept from S2 Part C:** the setting, code path and cheating model.
- Fishery, MSY target, `mix4`.
- The 4 stress agents may take a_i + d × (p_i − a_i), where a_i is their
  allowance and p_i their request.
- Audits check the step's actual extraction afterwards and charge a flat fine
  F per catch. There is no exclusion, and audits never change the current
  decision.

**New audit protocol `bern`.** Each agent is audited independently with
probability q each step. An audited agent that over-took is caught with
probability s.

**Grid.** Fines are set from a grid of expected fines per agent-step:
e ∈ {0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.8, 1.0}, with F = e / (q × s).

- **Part A** (perfect audits, s = 1): q ∈ {1/6, 0.10, 0.05, 0.02}.
- **Part B** (imperfect audits): q = 1/6 with s ∈ {0.5, 0.25}.

The cell e = 0 is "audit but no fine". It is the same for every q and s.

**Adaptive search, as in S2.** For each cell, one common d is chosen from
{0, 0.25, 0.5, 0.75, 1}. It maximises the cheaters' mean net payoff on 8
training contexts, with ties going to the smaller d.

**Test.** 64 fresh test contexts at d = d*, plus d = 0 (all comply) and the
`allow` arm (no checks) at d*_allow.

**Seed bases (new):**

| Seed | Base |
| --- | --- |
| Fishery population | 900,000,000 |
| Reviewer | 930,000,000 |
| Audit | 945,000,000 |
| Detection (s < 1) | 946,000,000 |
| Training Fishery population | 950,000,000 |

Fishery has no weather, so the other S2 bases are unused.

**Also recorded:** for each cheater, the number of steps on which it
over-took. This allows the threshold to be predicted from the data.

### Prediction made now, from S2

- Held out, unchecked cheating gained about 109 per group of 4 cheaters
  (file 12), which is about 27 per cheater.
- The cheater over-takes on at most 80 steps, so the expected fine is
  e × (number of over-take steps).
- A risk-neutral cheater should therefore switch to complying once
  e × n_over > about 27, that is, near e ≈ 0.34 if it over-takes on nearly
  every step.

### Hypotheses

- **A-H1 (bracket).** For every q in Part A:
  - the search picks d* > 0 at e ≤ 0.2;
  - it picks d* = 0 at e ≥ 0.5.
  - Falsifier: any Part A cell outside this bracket.
- **A-H2 (only the expected fine matters).** The smallest grid value of e
  with d* = 0 (call it e*) is the same for all four q, within one grid step.
  - Falsifier: e* differs by more than one grid step between any two q.
- **B-H1 (imperfect audits work through the same product).** With
  q = 1/6, e* at s = 0.5 and at s = 0.25 is within one grid step of e* at
  s = 1.
  - Falsifier: a larger difference.
- **A-H3 (held-out check).** At e*, the held-out cheater payoff at d* is
  not above complying.
  - Falsifier: the context-bootstrap interval for (d* payoff − comply
    payoff) lies entirely above 0.

A-H2 and B-H1 could fail because a single training sample of 8 contexts
gives noisy catch counts at low q. If so, that failure is the result: at low
audit rates, the threshold a cheater infers from limited experience is
unreliable.

---

## Part C: cheaters who time their cheating

### Question

Do audits still deter when cheaters can tell when an audit is likely?

### Design

Fishery, MSY target, the same cheating model, and the 64 test and 8
training contexts from Parts A and B. Two schedules have the **same average
audit rate per agent-step, 1/6**:

- **Random (`bern`, q = 1/6):** unpredictable.
- **Periodic (`periodic6`):** every agent is audited on steps
  t ≡ 5 (mod 6) and on no other step. The schedule is known to the agents.

The strategy space has two dimensions:
- level d ∈ {0, 0.25, 0.5, 0.75, 1};
- timing ∈ {`always`, `avoid`}.

An `avoid` cheater over-takes only on steps where its known chance of being
audited is below 1/2. Under the random schedule that chance is always 1/6,
so `avoid` behaves exactly like `always`.

**Fines:** F ∈ {0, 6, 24}. The search covers all 10 strategies, ties go to
(smaller d, `always`), and the test is on 64 contexts.

### Hypotheses

- **C-H1.** Under `periodic6` with F ∈ {6, 24}, the search picks `avoid`
  with d* > 0. The held-out gain over complying is positive (interval
  above 0).
  - This is **largely true by construction**: an `avoid` cheater is never
    audited while it over-takes. The informative part is its size (C-H3).
  - Falsifier: d* = 0, or the gain interval includes 0.
- **C-H2.** Under random audits at F ∈ {6, 24}, d* = 0.
  - Falsifier: d* > 0.
- **C-H3 (size).** Under `periodic6` at F = 24, harvest per honest agent
  falls by at least half of the drop seen with no checks (`allow` vs
  comply, same seeds).
  - Falsifier: a smaller drop.

---

## Part D: a reviewer that remembers caught lies

### Question

Can audits make the system safer by improving the reviewer's decisions,
without excluding or fining anyone? In S1b, a reviewer with no memory
barely improved. Here the reviewer remembers what each audit revealed.

### Design

This is the S1 misreporting setting. Misreporters report
r_i = (1 − d) × p_i. Audits reveal p_i. The reviewer uses the calibrated
joint chance constraint, and the sanction is `none`: nobody is excluded or
fined.

**Belief modes:**

| Mode | What the reviewer believes |
| --- | --- |
| `trust` | b_i = r_i. Identical to S1 `report`. |
| `memoryless` | The audited agent's true request is used this step only. Identical to S1b, belief on, sanction none. |
| `memory` | Each audit that reveals r_i < p_i records d̂ = 1 − r_i / p_i for that agent. On later steps, an unaudited agent with any record gets b_i = min(1, r_i / (1 − mean d̂_i)). Audited agents get their true request. |
| `memory_max` | An agent caught even once is from then on believed to request the maximum, b_i = 1, unless audited that step. This is a conservative memory. |

**Liars:**
- `fixed` d = 0.5, as in S1 and S1b;
- `noisy`: each misreporter draws d_t independently from Uniform(0, 1) every
  step (mean 0.5), so memory can only estimate the lie.

**Protocols:**
- Harvest (one-step chance constraint): `rand1`, `rand2`, `targ1`;
- Fishery: `rand1`, `rand2`, under both the one-step target (paired with
  S1b) and the MSY target (the default from S1b on).

`peer` is left out because neighbour reports already reveal every request
each step, so memory cannot add anything.

**Seeds.** The S1 bases (700M–745M), so that every arm is paired with S1
and S1b by context. A new base, 947,000,000, is used for the noisy-liar
draws. This part reuses S1's contexts by design, as S1b did, so it is not a
fresh-seed test.

**Outcomes**, never combined:
- share of the 80 steps in an unsafe state (`unsafe_fixed`): primary for
  Harvest and the Fishery one-step target;
- share of steps breaking the MSY target: Fishery MSY;
- unsafe executed actions;
- usefulness loss (requests labelled safe that were cut);
- honest and total harvest.

### Hypotheses (fixed liars, d = 0.5)

- **D-H1.** Harvest `rand2`: `memory` lowers the unsafe share below
  `memoryless` by at least 0.5 points.
  - Falsifier: a smaller reduction, or an interval including 0.
- **D-H2.** Fishery one-step `rand2`: `memory` brings the unsafe share
  below 10%. S1b had memoryless at 68.7% and trust at 80.4%.
  - Falsifier: 10% or more.
- **D-H3.** `memory_max` is at least as safe as `memory`, but cuts more
  safe requests (higher usefulness loss) in Harvest `rand2`.
  - Falsifier: usefulness loss is not higher.
- **D-H4.** Against noisy liars, `memory` is less safe than against fixed
  liars (higher unsafe share in Harvest `rand2`).
  - Falsifier: not higher.

**Known in advance.** Against a fixed-fraction liar, one catch reveals d
exactly, so `memory` is close to full information after the first catch.
That makes the fixed-liar result easy by design. The noisy-liar arm is the
harder test.

---

## Analysis (all parts)

- Paired context-bootstrap contrasts: 4,000 resamples, seed 20261008, 95%
  percentile intervals.
- No p-values, no pooling across games.
- d* and e* are reported as found; they are not tested statistically.
- Figures are made after the analysis and do not change any verdict.

## Engineering gates (before the full run)

1. `pytest -q tests` passes.
2. A smoke run (2 contexts × 20 steps) is byte-identical on rerun.
3. **S2 replication.** With S2's seeds and protocol `rand1`, the S3 code
   reproduces S2's Fishery `allow` and `rand1` F = 6 test episodes exactly
   (64 contexts, at d = 1 and d = d*).
4. **S1b replication.** `memoryless` at d = 0.5 with S1 seeds reproduces S1b
   Part A (belief on, sanction none) exactly, and `trust` reproduces S1
   `report` exactly.

## Run budget and stop rules

- One full run per part, with a 45-minute cap each.
- No changes after seeing the full results. A crash may be repaired once, in
  a new directory, keeping the failed attempt.
- If a gate fails, fix the code and rerun the gate. This is allowed because
  no full-run outcome exists yet.

## Amendments

(none yet)
