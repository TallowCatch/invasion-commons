# Protocol S4: adaptive cheaters against a reviewer with memory, and a cost per audit

**Frozen:** 2026-10-05T11:38Z, before any S4 code was written or run.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". It builds on the results of S1b,
S2 and S3 (the S1b/S2 results and S3 results), which were already known when it was written.

Terms follow `literature_ledger.md`:
- "audit rate" corresponds to an *audit budget* (Greenblatt et al. 2023/24);
- "audit cost" corresponds to *costly state verification* (Townsend 1979,
  doi:10.1016/0022-0531(79)90031-0, metadata depth only).

These are analogies.

## Why the main test uses cheating, not misreporting

In S1b, misreporting never paid, even with no sanction at all (A-H3
falsified, the S1b/S2 results). A search for the best lie against a reviewer with
memory would therefore almost certainly return "do not lie" and teach
little.

In the S2/S3 non-compliance model, hidden over-extraction **does** pay
privately in Fishery: +28.5 per cheater with no checks (the S3 results). So the
main test (Part A) asks whether memory still helps against cheaters who
choose their strategy knowing the reviewer remembers. Part B is a small
check that misreporting still does not pay once the reviewer has memory.

Harvest is left out of Part A because cheating never paid there (S2).

---

## Part A: cheaters who adapt to a reviewer with memory, and the price of auditing

### Question

1. Can a reviewer that remembers caught over-takers neutralise cheating
   **without fines**, when cheaters choose their strategy knowing this?
2. With a cost per audit, which audit rate gives the most harvest net of
   audit costs, under fines, memory, or both?

### Design

**Kept from S2/S3:** Fishery, MSY target, 4 stress agents who may
over-take. Reports of requests p_i are truthful. The reviewer picks one
shared cut λ, giving allowances a_i = λ × p_i. A cheater with level d takes
a_i + d × (p_i − a_i).

**Audits:** Bernoulli. Each agent is audited with probability q each step,
with perfect detection, checking the step's actual take after extraction.

**Regimes:**

| Regime | What a catch does |
| --- | --- |
| `none` | No audits. |
| `fine` | Flat fine F = 6 per catch, as in S2. Nothing else. |
| `memory` | No fine. The reviewer records the agent's observed over-take fraction, d̂ = (taken − a) / (p − a). From then on it gives that agent a **targeted allowance** a′ = max(0, (λ − d̄) / (1 − d̄)) × p, where d̄ is the mean of the recorded fractions (a′ = 0 if d̄ = 1). The agent's expected take is then λ × p, as planned. Other agents are unaffected. |
| `fine+memory` | Both. |

**Audit rates:** q ∈ {0.01, 0.02, 0.05, 0.10, 1/6, 1/3}.

**Cheater strategy (adaptive).** The group chooses a level
d ∈ {0, 0.25, 0.5, 0.75, 1} and a reaction to being caught: `continue`
(keep the same level) or `stop` (comply for the rest of the episode). The
choice maximises the cheaters' mean net payoff on 8 training contexts.
Ties go to the smaller d, then `continue`. The search covers all 10
strategies in every regime × q cell.

**Test.** 64 fresh test contexts at the chosen strategy, plus `comply`
(d = 0) and `none` at its own d*.

**Audit cost (accounting only).** Net harvest = total harvest − c × number
of audits, for c ∈ {0, 0.02, 0.05, 0.1, 0.2} per agent-audit. Fines are
transfers between agents and are excluded. Cost changes nobody's
behaviour, so it is applied in the analysis to the same runs. For each
regime and c, report the audit rate with the highest held-out mean net
harvest (q*). The `none` regime has zero audits.

**Seed bases (new):**

| Seed | Base |
| --- | --- |
| Fishery population | 1,000,000,000 |
| Audit | 1,045,000,000 |
| Detection (unused, s = 1) | 1,046,000,000 |
| Training population | 1,050,000,000 |

### Prediction made now

- **Memory:**
  - Before its first catch, a cheater gains as in S2.
  - After it, its expected take falls back to λ × p, so cheating gains
    nothing more.
  - The expected number of steps before the first catch is about 1/q.
  - Cheaters' gain should therefore fall as q rises, but not to zero,
    because nobody is fined.
- **Fines at F = 6:** from S3, deterrence starts at e = q × 6 ≈ 0.36, so
  cheaters should comply at q ≥ 0.10 and cheat at q ≤ 0.05.

### Hypotheses

- **A-H1.** Under `memory`, the held-out cheater gain over complying is
  below the `none` gain at every q, and it does not increase with q.
  - Falsifier: at some q the gain is not below `none` (the interval for
    memory − none includes or exceeds 0), or it rises from one q to the
    next by more than the overlap of the two intervals.
- **A-H2.** Under `memory`, steps breaking the MSY target are lower than
  under `none` at every q ≥ 0.05.
  - Falsifier: the interval for (memory − none) includes or exceeds 0 at
    some q ≥ 0.05.
- **A-H3.** Under `fine`, d* = 0 at q ∈ {0.10, 1/6, 1/3} and d* > 0 at
  q ∈ {0.01, 0.02}. q = 0.05 (e = 0.30) is left unpredicted, because S3
  showed it lies at the edge.
  - Falsifier: any of the five predicted cells is wrong.
- **A-H4.** At q ∈ {0.01, 0.02}, `fine+memory` gives higher honest-agent
  harvest than `fine`.
  - Falsifier: the interval for the difference includes or is below 0 at
    either q.
- **A-H5 (cost).** At c = 0.2, the best audit rate under `fine+memory` is no
  higher than under `fine`.
  - Falsifier: q*(fine+memory) > q*(fine).
- **A-H6 (does adapting to memory matter?).** Under `memory`, the chosen
  reaction to a catch is `continue` whenever d* > 0. Stopping after a catch
  only lowers the cheater's take below λ × p.
  - Falsifier: `stop` is chosen in some cell with d* > 0.

---

## Part B: do misreporters lie against a reviewer with memory?

### Design

The S3 Part D setting: S1 seeds for training and test, sanction none,
reviewer mode `memory`, protocol `rand2`. Cells:
- Harvest, one-step chance constraint;
- Fishery, one-step target;
- Fishery, MSY target.

Adaptive search over d ∈ {0, 0.25, 0.5, 0.75} on S1's 8 training
contexts. The test is on 64 contexts at d*.

### Hypothesis

- **B-H1.** d* = 0 in all three cells, as S1b found without memory.
  - Falsifier: d* > 0 in any cell.

---

## Analysis

- Paired context-bootstrap contrasts: 4,000 resamples, seed 20261009, 95%
  percentile intervals.
- No p-values, no pooling across games.
- d* and q* are reported as found.
- Figures come after the analysis and change no verdict.

## Engineering gates (before the full run)

1. `pytest -q tests` passes, with new unit tests for:
   - the targeted allowance;
   - the over-take estimate;
   - the `stop` reaction.
2. A smoke run is byte-identical on rerun.
3. **Regime `fine` equals S3.** With S3's seeds, `fine` at q = 1/6 and
   F = 6 gives the same test episodes as the S3 `bern` cell with the same
   q and F, for 8 sampled contexts.
4. **Memory changes nothing for compliers.** With d = 0, `memory` equals
   `none` with audits, for 8 sampled contexts.
5. **Part B code path.** With mode `memory` at d = 0.5, Part B
   reproduces the S3 Part D `memory` / `fixed` / `rand2` episodes on this
   machine, for 8 sampled contexts.

## Run budget and stop rules

- One full run per part, with a 45-minute cap each.
- No changes after seeing the full results. A crash may be repaired once,
  in a new directory, keeping the failed attempt.
- Gate failures may be fixed before the full run.

## Amendments

(none yet)
