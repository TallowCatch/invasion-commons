# Results S1b and S2: what audits actually do, and how fixed-level cheaters respond to audits plus fines

**Protocol.** `studies/S1b_S2_ablation_and_deterrence/protocol.md`, frozen at
2026-10-05T00:35Z, before these runs. The deviations from it are listed
under "Departures from the protocol" below.

**Runs.**

| Run | Folder | Episodes | Seconds |
| --- | --- | ---: | ---: |
| S1b | `results/runs/claude_s1b_ablation_msy_v1` | 7,168 | 594 |
| S2 | `results/runs/claude_s2_compliance_deterrence_v1` | 5,120 | 396 |

The analysis is in `experiments/analyze_s1b_s2.py`. Figures come from
`experiments/make_progress_figures.py` (`figures/fig5`, `fig6`).

An independent check corrected several statements in an earlier draft of
this file (see the verification log §10). Every number below is **verified** from the run
data. Explanations marked **[post hoc]** were not tested.

**Two measures of "unsafe"** appear below. Each table says which one it uses.

- **Share of the horizon unsafe:** the fraction of the 80 steps spent below
  the safety line, with collapse counted for the rest of the run.
- **Per-step unsafe-action rate:** among steps that start safe, the share
  whose executed action had more than 5% one-step risk.
  - In Fishery S2, "risk" here means breaking the MSY target, which is a
    soft failure, not collapse.

---

## Part A: what part of an audit makes the system safe?

### What was done

S1 was rerun on the same seeds. Misreporters under-reported by half
(*d* = 0.5). Two things were switched separately:

- **Correction:** whether the audited agent's true request is used in that
  step's decision.
- **Sanction:** none, a fine, or exclusion for 6 steps plus a fine.

Fishery used the one-step target, as in S1.

Two design facts, found by the independent check, limit what Part A can
show:

1. **The reviewer has no memory.** A caught liar's true request is used
   only in the step it was audited. The next step, the reviewer trusts that
   agent's under-report again. A reviewer that remembered the size of the
   lie would do much better. That was not tested.
2. **Under exclusion, the correction switch does nothing.** A caught agent
   is set to 0 both in its true request and in the reviewer's belief, and an
   uncaught agent's report equals its request. So "exclusion with
   correction" and "exclusion without correction" are mechanically
   identical.

   Exclusion also gives the reviewer exact information about the excluded
   agent for 6 steps, and it removes that agent's extraction.

### What came out

Share of the horizon unsafe:

| Setting | Trust reports | Correction, no sanction | Exclusion + fine (S1 rule) | Everyone honest (full verification) |
| --- | ---: | ---: | ---: | ---: |
| Harvest, 2 random audits | 1.9% | 1.7% | 0.2% | 0.6% |
| Harvest, 1 targeted audit | 1.9% | 1.3% | 0% | 0.6% |
| Harvest, neighbour reports | 1.9% | 0.6% | 0% | 0.6% |
| Fishery, 2 random audits | 80% | 69% | 0% | 0% |
| Fishery, neighbour reports | 80% | 0% | 0% | 0% |

- **The fine did nothing on its own.** "Fine only" gave the same results as
  "no sanction" in every cell. That is expected, because these misreporters
  lie at a fixed level whatever the penalty.
- **How often liars were out.** Under exclusion they were excluded most of
  the time: about 81% of misreporter-steps in Harvest with 2 random audits.
  In Fishery, that removes 4 of the 6 agents for most of the run.

Per-step unsafe-action rate, "correction, no sanction" minus "trust reports",
with 95% intervals over contexts:

| Setting | Difference |
| --- | --- |
| Harvest, 2 random audits | −0.6 points [−0.8, −0.3] |
| Harvest, 1 targeted audit | −1.0 [−1.4, −0.6] |
| Harvest, neighbour reports | −2.6 [−3.1, −2.2] |

**Hypotheses:**

- **A-H1: falsified.** The prediction was a drop of at least 1 point; the
  observed drop was 0.6.
- **A-H2: true by construction.** Its "support" is not evidence, for the
  reason given in design fact 2.
- **A-H3: falsified.** The best lie was 0 in every cell, even with no
  sanction.

### What it means

**What the data show.** With a reviewer that has no memory, using one or two
audit results per step barely improved decisions. Only complete information
from neighbour reports brought correction-only down to the honest level.

The safety in S1's audit arms came from **removing the caught cheaters'
extraction for many steps**. Exclusion even pushed Harvest *below* the honest
level (0–0.2% vs 0.6%), because it takes the aggressive agents out of the
game.

**What they do not show:**

- that partial checking is of little value in general. A reviewer that
  remembers and corrects for a caught agent's lie was not tested;
- a clean separation of "sanction" from "information". Exclusion supplies
  both.

---

## Part B: S1 in Fishery with the MSY target

| Condition | Steps breaking the MSY target | Total harvest | Honest agents' harvest | Mean stock |
| --- | ---: | ---: | ---: | ---: |
| Full verification | 0% | 1,393 | 206.2 | 70 |
| Trust reports, liars at fixed *d* = 0.5 | 98.8% | 1,388 | 204.6 | 62 |
| Trust reports, adaptive liars (chose *d** = 0.25) | 75.2% | **1,401** | **207.0** | 66 |

- Under trusted reports at *d* = 0.5, the rise in target-breaking steps was
  +98.8 points [+98.0, +99.4].
- No condition ever fell below the old collapse line of 10.
- **B-H1: supported**, but the "harm" here is soft. Breaking the MSY target
  this way left the stock near 62–66. It even slightly *raised* total and
  honest harvest at the adaptive level.

**Liars at a fixed level vs liars who respond to the audits:**

- **Fixed *d* = 0.25 or 0.5.** Audits excluded them, and that cost harvest:
  - rand1: 854;
  - rand2: 476;
  - peer: 284;
  - against 1,393 with full verification.

  **[post hoc]** The likely reason is that the reviewer was already holding
  the stock at the productive level, so removing agents only lost harvest.
- **Adaptive liars.** They chose *d** = 0 under every audit protocol, so
  against them audits cost nothing (1,393).
- **Colluding neighbours, fixed *d* = 0.5:** 35% of steps broke the target,
  with harvest of 1,218.

---

## Part C (S2): hidden over-extraction that pays privately

### What was done

- Agents report their requests **truthfully**, so this is not misreporting.
- The reviewer sets each agent's allowance from those reports.
- Would-be cheaters then take more than their allowance. They keep a share
  *d* of the cut, from *d* = 0 (comply) to *d* = 1 (take the full request).
- This is **non-compliance**: hidden over-extraction, not a false report.
- Audits check what each agent actually took, after extraction. They are
  perfect (any over-take of an audited agent is caught) and charge a **flat
  fine per catch**, whatever the size of the over-take. There is no
  exclusion.
- **Cheating level.** One level *d* is chosen for the whole group of
  cheaters, on 8 training contexts. It is then tested on 64 new contexts.
- **Settings:**
  - Fishery: MSY target, 4 would-be cheaters;
  - Harvest: 2 would-be cheaters.
- **Audit coverage.** One audit per step covers 17% of agents. That is far
  above the 2% audit budgets in AI-control work (Greenblatt et al.
  2023/24). The numbers are not comparable.

### What came out: Fishery

**With no checks, cheating paid.**

| Outcome | Result |
| --- | --- |
| Cheating level chosen | *d** = 0.75 |
| Cheaters' gain over complying (held out) | +109 [+102, +115] |
| Harvest per honest agent | 106 → 45 |
| Net payoff per cheater | 295 → 322 |
| Total harvest | barely changed: 1,393 → 1,378 |
| Steps breaking the MSY target | every step |
| Stock | held at about 60 |

With *d* = 1, the stock spent 52% of the horizon below the collapse line.

**With checks.** Under all four checking rules (1 or 2 random audits per
step, or neighbour reports with or without collusion):

- the chosen level dropped to *d** = 0 **at every tested fine (6, 12, 24)**;
- the outcome returned exactly to the all-comply result;
- at a fine of 0, checking changed nothing (*d** = 0.75).

Hypotheses C-H1, C-H2 and C-H3 are **supported**.

**Where the threshold probably is** (arithmetic done after the run):

- Average gain per cheater per step: 109 ÷ 4 ÷ 80 ≈ 0.34.
- Expected fine per cheating step with one random audit:
  1/6 × fine. This exceeds 0.34 once the fine is above about 2.
- So the threshold is probably near 2, well below the smallest tested fine
  of 6. The grid is too coarse to show it.

This is the textbook expected-penalty reasoning usually attributed to
Becker (1968). We have verified only the paper's identifier, not its text.

### What came out: Harvest

The chosen cheating level was 0 at every fine, including with no checks at
all.

With full cheating (*d* = 1) and no checks:

- the per-step unsafe rate rose by 1.4 points [+1.0, +1.8];
- cheaters' payoff fell by 6.5;
- honest agents' harvest also fell, by 2.4 [−4.5, −1.0].

**[post hoc]** The likely reason is that over-taking mostly damages the
cheater's own patch.

**Hypotheses:**

- **C-H4:** not falsified under its stated criterion, but uninformative.
  Both checking rules reach *d** = 0 at a fine of 0, because cheating never
  paid in Harvest.
- **C-H5:** "neighbour reports deter cheating" is supported in Fishery.
  "Collusion sustains cheating" is falsified as stated, but this is largely
  a design artefact: one cheating level is chosen for the whole group, so
  shielded cheaters could not cheat while exposed ones complied. Treat it as
  a limit of the test, not a finding.

### What it means

**What the data show.** In Fishery, hidden over-extraction moved harvest from
honest agents to cheaters. Against these fixed-level, group-chosen cheaters,
any tested combination of at least one perfect random audit per step and a
flat fine of 6 or more removed the incentive completely.

This matches the inspection-game idea that "inspector leadership serves as
deterrence": the inspector commits to a checking rule in advance (Avenhaus
et al. 2002, full text checked).

**What it does not show:**

- the minimum audit rate or fine that works;
- deterrence against cheaters who time their cheating or adapt within an
  episode;
- deterrence with imperfect audits, or fines scaled to the over-take;
- any result in Harvest.

---

## Departures from the protocol, and further caveats

- **Seeds.** File 11 says S2 training reuses "the same separate bases as
  S1". The code and the S1b/S2 protocol's own seed table use new bases
  (850M/860M/870M). This is harmless and noted here.
- **Gate C.** It was checked by comparing totals for `allow`, `rand2` and
  `peer` with everyone complying. The frozen wording asks for "the same
  decisions as an honest reviewer run". The adaptive cells with *d** = 0
  also match the all-comply outcome exactly.
- **Audit schedules.** The audit random-number seed does not include
  "training vs test". So training contexts 0–7 share audit schedules with
  test contexts 0–7. The populations differ, so the effect should be small.
- **Search grid.** The search is coarse: five levels of *d*, and fines of
  0, 6, 12 and 24.
- **Conditions chosen after S1.** The MSY target in Fishery, and dropping
  exclusion from S2, were both decided after seeing S1. The runs used fresh
  seeds.

## Where this leaves us

**What we now believe, in these two simulated games:**

- With honest agents, a reviewer's errors depend on how it combines
  information.
- Acting on unverified reports is unsafe.
- A memoryless reviewer gains little from one or two audits per step. In
  our runs, the safety came from removing cheaters or from complete
  information.
- When over-extraction pays privately (Fishery), perfect random audits plus
  a flat fine removed the incentive for fixed-level cheaters.

**Still untested:**

- a reviewer that remembers caught lies;
- cheaters who time their cheating;
- imperfect audits and fines scaled to the over-take;
- the actual threshold.

**Next, each a few CPU minutes:**

1. **Locate the threshold.** Fines from 0 to 6 in steps of 1, with audit
   rates below one per step. Compare with the about-2 prediction.
2. **Imperfect audits.** Catch probability below 1.
3. **Timing cheaters.** Cheaters who cheat right after being audited. This
   is the first truly adaptive red team.
4. **A remembering reviewer.** After a catch, it corrects that agent's
   later reports.
