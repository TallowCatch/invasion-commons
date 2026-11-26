# H5 results: how much commons harm passes the individual rule? (Harvest, 23 Nov 2026)

Protocol, definitions and predictions (fixed before measuring): `protocol.md`. Code:
`experiments/socialjax/measure_lpgf.py`; tests: `tests/test_lpgf.py` (9). Summaries: `lpgf_summary.json`; raw
per-episode numbers (git-ignored): `results/h5_lpgf/lpgf.json`.

**Setup.** SocialJax Harvest Open, 7 agents, 999 steps per episode. 5 trained IPPO runs (stage 1, shared network,
1e8 steps, seed 1 each), each played under its own overseer, and 3 scripted populations with no overseer.
**32 episodes per policy, 256 in all.** Brackets are 95% bootstrap intervals over episodes.

**Short answer.** In Harvest, no permanent harm passed the rule. All 580 patch extinctions we saw (8 policies,
256 episodes) involved at least one eat that broke the rule. Local damage that heals is a different story: in the
overseen runs, 60–93% of newly dead cells came from legal eats, but 92–95% of those cells grew back within the
episode. H5's Harvest prediction holds. Harvest gives no reason to add the commons view.

---

## 1. Permanent loss: patches that lose their last apple

A patch with no apples can never regrow, so this is the only loss that is permanent under the game's rules.

| Policy | Orchard alive at end | Patch extinctions (episodes with any) | Cells lost per episode | Share from legal eats only (c) | Share in all-pass steps (b) |
|---|---|---|---|---|---|
| A: no overseer | 0 / 32 | 192 (32) | 64.0 | **0 of 192** | 0% |
| B: random, k = 1 | 32 / 32 | 3 (3) | 1.2 [0.0, 2.4] | **0 of 3** | 0% |
| B: random, k = 2 | 32 / 32 | 0 | 0 | n/a | n/a |
| B: random, k = 7, penalty 2 (perfect full inspection) | 32 / 32 | 0 | 0 | n/a | n/a |
| D: history, k = 2 | 32 / 32 | 1 (1) | 0.2 [0.0, 0.6] | **0 of 1** | 0% |
| Bots: 7 honest | 32 / 32 | 0 | 0 | n/a | n/a |
| Bots: 7 greedy | 0 / 32 | 192 (32) | 64.0 | **0 of 192** | 0% |
| Bots: 1 greedy + 6 honest | 0 / 32 | 192 (32) | 64.0 | **0 of 192** | 0% |

(k = agents checked per step out of 7; penalty in apples per catch, 1 unless stated.)

**Why zero is structural, not luck.** The step that ends a patch must eat all of its remaining apples. If 3 or
fewer remain, each has at most 2 apples near it, so every one of those eats breaks the rule. A fully legal ending
needs 4 or more agents to eat 4 or more mutually close apples in the same step, each seeing 3 or more apples near
it. That never happened in 256 episodes. So in Harvest, with this rule, a perfect check of the right agent at the
right step could stop every permanent loss.

Two side observations:
- The overseen runs keep the orchard alive in every episode, as in training, but k = 1 and history k = 2 still lost
  a whole patch in 3 and 1 of 32 episodes. Every one of those was a violation that went unchecked: a budget
  failure, not a failure of what the rule can see.
- One greedy agent among six honest ones empties the orchard in all 32 episodes. All 192 patch losses trace to
  violations, which here means mostly that one agent's eats.

## 2. Local damage that heals: newly dead cells

A cell is locally dead when it has no apple on it or within two squares, so it cannot regrow next step. A
neighbour can regrow and revive it, so this is temporary damage.

| Policy | New dead cells per episode | From legal eats (c) | of which joint | In all-pass steps (b) | Revived later | Still dead at end, per episode | of which from legal eats | Legal eats that kill a cell | Violating eats that kill a cell |
|---|---|---|---|---|---|---|---|---|---|
| A: no overseer | 64.8 [64.3, 65.3] | 3.9% [2.9, 4.9] | 0% | 1.5% | 1% | 64.0 | 4% | 4.9% | 64% |
| B: random, k = 1 | 51.8 [45.7, 58.3] | 60% [56, 64] | 0.9% | 58% | 92% | 4.0 [2.6, 5.6] | 38% [26, 54] | 5.9% | 24% |
| B: random, k = 2 | 86.7 [79.3, 93.6] | 75% [73, 78] | 0.7% | 74% | 94% | 4.8 [4.0, 5.8] | 63% [49, 78] | 10.6% | 36% |
| B: random, k = 7, penalty 2 | 5.5 [3.6, 7.6] | 93% [89, 97] | 1.7% | 93% | 95% | 0.3 [0.1, 0.5] | 100% (9 cells) | 0.9% | 6% |
| D: history, k = 2 | 50.9 [44.3, 57.7] | 72% [69, 76] | 1.4% | 71% | 94% | 2.9 [2.0, 4.0] | 55% [41, 75] | 6.5% | 31% |
| Bots: 7 honest | 41.9 [36.5, 47.5] | 99.9% | 8.0% | 99.9% | 96% | 1.5 [1.1, 2.0] | 100% | 7.3% | (2.2 violations per episode) |
| Bots: 7 greedy | 64.8 [64.3, 65.5] | 10.5% [9.0, 12.1] | 0.9% | 3.3% | 1% | 64.0 | 10% | 13.9% | 64% |
| Bots: 1 greedy + 6 honest | 66.9 [65.6, 68.3] | 6.6% [5.3, 7.9] | 0.5% | 5.7% | 4% | 64.0 | 5% | 6.1% | 77% |

What this shows:
- **When the orchard collapses, violations do the damage.** In the three collapsing populations, only 4–11% of
  local deaths come from legal eats, and a violating eat is about 5 to 13 times as likely to kill a cell as a legal one.
- **When the orchard is kept, most local deaths are legal, and they heal.** The rule protects the eater's own
  neighbourhood, not the cells beyond it, so a legal eat at the edge of a patch can strip a cell on the far side.
  This is the everyday churn of harvesting near an edge: 92–96% of those cells revive within the episode, and no
  patch was lost to it.
- **Joint legal harm is rare.** Several legal eats in one neighbourhood in one step account for at most 1.7% of
  local deaths in the trained runs, and 8% for the honest bots, which move as a pack.
- Under perfect full inspection (k = 7), local death is rare (5.5 cells per episode) and almost all of it is legal,
  because almost nothing illegal is left (5.2 violations in 536 eats).

The cells still dead at the end are censored: the episode stops at step 999, and their patches still have apples,
so many would likely revive later. The "still dead at end" split was added after a 2-episode smoke run
(`protocol.md` §7), so treat it as secondary.

## 3. The June predicate: apples below τ × initial apples

Initial apples: 64. Each cell: downward crossings per episode, then the share of crossings at steps where every
agent passed the rule (LPGF (a)).

| Policy | τ = 0.1 | τ = 0.25 | τ = 0.5 |
|---|---|---|---|
| A: no overseer | 1.0, 6% [0, 16] | 1.0, 3% [0, 9] | 1.0, 18% [6, 31] |
| B: random, k = 1 | 0 | 0 | 8.2, 84% [81, 87] |
| B: random, k = 2 | 0 | 0 | 3.1, 87% [80, 92] |
| B: random, k = 7, penalty 2 | 0 | 0 | 43.6, 98% [98, 99] |
| D: history, k = 2 | 0 | 0 | 7.9, 91% [87, 95] |
| Bots: 7 honest | 0 | 0 | 19.8, 100% |
| Bots: 7 greedy | 1.0, 12% [3, 25] | 1.0, 16% [3, 28] | 1.0, 6% [0, 16] |
| Bots: 1 greedy + 6 honest | 1.1, 17% [6, 30] | 1.7, 39% [27, 52] | 1.1, 53% [36, 69] |

- At τ = 0.5 the June measure says most crossings in the overseen runs are "local pass, global fail". This is an
  artefact of the threshold: sustainable play settles at 23–29 apples, just under 0.5 × 64 = 32, so ordinary legal
  eating crosses the line again and again. The honest bots score 100% on it while keeping every patch alive.
- At τ = 0.1 and 0.25 only the collapsing populations cross, and mostly at steps with a violation (3–39%).
- June's raw form (all pass and unsafe, over all steps) gives 81–97% at τ = 0.5 for every policy, and 90–97% at
  every τ for the collapsing ones, because after a collapse every step is "all pass, unsafe". It measures how long
  the orchard has been dead, not whether the rule missed anything. We do not use it.

## 4. Predictions

| Prediction (protocol §5) | Result |
|---|---|
| P1 (main): legal-only share of permanent loss < 10% | **Holds.** 0 of 580 patch extinctions; 0% in each of the 5 policies that lost a patch |
| P2: legal share of local death < 50% for learned policies and greedy bots | **Holds for collapsing populations (4–11%); fails for overseen learners (60–93%).** Those deaths heal (92–95%) |
| P3: June rate high at τ = 0.5, lower at τ = 0.1 | **Holds** (84–98% vs no crossings for the overseen runs) |
| P4: orchard survival as in training | **Holds** (A: 0/32; all overseen runs 32/32) |
| P5: honest bots' harm is legal but loses no patch | **Holds** |
| Refuted if legal-only permanent loss ≥ 25%, or k = 7 loses patches to legal eats | Not refuted |

## 5. Answers

**Does H5's Harvest prediction hold?** Yes. With perfect full inspection the orchard stays alive (32 of 32
episodes, no patch lost), and the measured LPGF on permanent loss is 0%. Low LPGF and full protection go
together, as H5 says. The miss on P2 is about temporary local damage, which the rule was never meant to stop and
which heals.

**What share of harm escapes individual inspection, and for which policies?**
- Permanent loss: none, for every policy.
- Temporary local death: most of it in runs where an overseer keeps the orchard alive (60–93%), almost all of it for
  rule-keeping bots, and little of it (4–11%) when the orchard collapses.
- The June threshold predicate at τ = 0.5 reports high LPGF (84–100%) for sustainable play. That says more about the
  threshold than about the rule.

**Does this justify adding the commons view as the next step?** Not on Harvest evidence. Every permanent loss
passed through a violation an individual check could have caught. Where patches were lost under an overseer, the
cause was the budget (unchecked violators), not something only a commons view could see. The commons view has to be
justified elsewhere. Clean Up is the natural test: its harm (pollution) builds up from inaction, and the rule only
fires on eats. The script is ready for its runs (`protocol.md` §6): pollution crossings of 0.4, step-level and
W-window LPGF.

## 6. Caveats

- One training seed per overseer setting; 32 episodes each. The bootstrap covers episode-to-episode variation, not
  seed-to-seed variation.
- Shared-network learners at 1e8 steps (stage 1). Separate-network or worst-case learners were not measured.
- The attribution looks one step ahead: a violating eat is counted as preventable because stopping it would have
  left the cell alive that step. For patch loss, §1's argument shows the conclusion does not depend on this: a
  patch's last apples can only go legally in a joint eat of 4 or more.
- "Dead at end" is censored at step 999.
- Honest bots break the rule 2.2 times per episode, presumably through their 10% random moves. This does not affect
  the conclusions.
- For an empty patch-loss class, the 0% has no bootstrap interval. With 192 patch losses over 32 episodes per
  collapsing policy, a legal share of more than a few percent would very likely have shown up.

## 7. Related work

- **Makins, Angelini, Shams and Phuong (2026)**, *Multi-Agent AI Control: Distributed Attacks Hamper Per-Instance
  Monitors* (arXiv:2607.07368). Attackers split a harmful side task across coordinating agents, so each commit looks
  harmless to a per-commit monitor. This is local pass, global fail with deliberate splitting. Our Harvest result is
  the opposite case: learners that are not coordinating an attack, plus a rule whose permanent harm cannot be
  split. The joint legal kill is the Harvest version of a split attack, and it was rare (at most 1.7% of local
  deaths in trained runs, never a patch).
- **Perolat, Leibo, Zambaldi, Beattie, Tuyls and Graepel (2017)**, *A Multi-Agent Reinforcement Learning Model of
  Common-Pool Resource Appropriation* (NIPS 30; arXiv:1707.06600). This is the commons game Harvest descends from:
  apples regrow faster with more apples nearby, and never once a local patch is empty. Its sustainability metric
  (the average time at which rewards are collected) is an episode-level summary of the same collapse that our patch
  extinction counts event by event.
- **Our June–July 2026 draft** (`notes/private/yali_formulations/scalable_oversight_formulation_yali.tex`,
  17 Jun; method note, 24 Jul; summary in `../lit/C_our_prior_formulations.md`). It defined LPGF as "every agent
  passes locally and the next state is globally unsafe", reported as a rate over the episode, with a swept global
  threshold. §3 shows why we count transitions instead: counted over all steps, that rate mostly measures time
  spent after a collapse.

Both papers were checked to exist by web search on 23 Nov 2026 (arXiv abstract pages and NeurIPS proceedings).

## 8. Clean Up (24 Nov 2026)

Measured as set out in protocol §6, before looking at any Clean Up trace. 21 trained runs (7 arms × 3 seeds) plus 3 scripted populations, 32 episodes each, CPU only. The `presume` arm (3 runs) could not be replayed, because the step-for-step overseer used for replay does not support presumed guilt; it is left out. Summary: `lpgf_cleanup_summary.json`.

**Harm event:** pollution rising to 0.4, where apple regrowth stops. **LPGF (a):** share of those crossings at steps where every agent passed the rule. **(a_W):** share with no violation by anyone in the rule's own 100-step window.

| population | violations per game | share of steps polluted (≥ 0.4) | crossings per game | LPGF (a) | LPGF (a_W) | orchard alive at end |
|---|---|---|---|---|---|---|
| 7 honest bots (never violate) | 0 | 0.30 | 6.1 | 1.00 | **1.00** | **0.38** |
| 1 greedy + 6 honest bots | 65 | 0.72 | 4.5 | 0.99 | 0.06 | 0.00 |
| learners, no overseer | 14.6 | 0.96 | 0.7 | 0.89 | 0.00 | 0.00 |
| learners, random, w = 50 | 7.2 | 0.98 | 0.3 | 0.97 | 0.00 | 0.00 |
| learners, random, w = 150, c = 1 | 11.5 | 0.82 | 5.4 | 0.98 | 0.57 | 0.46 |
| learners, random, w = 150, c = 2 | 13.1 | 0.65 | 12.9 | 0.98 | 0.72 | 0.85 |
| learners, repeat offenders first, w = 150 | 21.6 | 0.66 | 9.9 | 0.97 | 0.61 | 0.71 |
| learners, 2 overseers random, w = 150 | 15.8 | 0.81 | 3.3 | 0.98 | 0.69 | 0.74 |
| learners, 2 overseers sharing, w = 150 | 15.6 | 0.71 | 7.5 | 0.98 | 0.54 | 0.41 |

(Learner rows: mean of 3 seeds.)

**Answers.**
- **LPGF is high in Clean Up.** In the overseen learner runs, 54–72% of the times pollution reaches the no-regrowth level, nobody broke the rule in the previous 100 steps. In Harvest the comparable figure was 0 of 580.
- **Full compliance does not protect the commons.** Seven agents who never break the rule still let pollution cross the threshold 6 times a game, and the orchard is empty at the end of 62% of games. So even perfect inspection of every agent cannot keep Clean Up safe under this rule. The prediction of H5 holds, and its falsifier ("high LPGF still fully protected by perfect full inspection") is not met.
- Where pollution stays above 0.4 throughout (no overseer, w = 50), there are few crossings to count, so a_W = 0 there means "already lost", not "the rule worked".
- **Why.** Dirt spawns whatever anyone eats; the rule only forbids eating without having cleaned, and an agent who cleans once per 100 steps passes it while cleaning far less than the commons needs.

**Caveats.** The "never violate" population is scripted, not learned. Clean Up learners were unstable at 1e8 steps (several runs peak mid-training and decline), so the learner rows describe end-of-training policies, not converged ones.

**Consequence.** Clean Up is the case for the commons view (the overseer also observing the state of the commons, or a group predicate in the rule). Per Decision 1 this was planned as a next step, to be justified if LPGF turned out large. It has.
