# The project story, experiment by experiment

Each entry gives:

- the question, in plain words;
- what was done;
- what came out;
- what it means.

Design problems are summarised here, with details in `background/design_issues.md`.

**Verification status.** Most raw results before September live in
git-ignored `results/` folders and are not in the repository.
- Numbers marked **(verified)** were recomputed from committed data.
- Numbers marked **(notes only)** are as stated in your notes or older
  papers, and could not be checked against data.

---

## Phase 1: Fishery (February–March 2026)

**Question.** Strategies keep getting replaced by new, greedier ones. Do
monitoring, quotas, fines, adaptive quotas or temporary closures keep a single
shared fish stock from collapsing?

**What was done**

- **The game.** A population of threshold strategies (default 12) fished
  one stock for 200 steps.
- **Turnover.** After each "generation", the worst 20–30% of strategies by
  their own payoff were replaced by mutated copies of better ones. Some
  replacements were instead written by a small live language model (Qwen 2.5 3B).
- **Governance options** were compared across difficulty tiers (how fast the
  stock regrows), with 5 independent evolutionary runs per condition.
- **A learned-agent (PPO) check** used 5 seeds.

**What came out (notes only)**

- In the medium tier, monitoring plus sanctions cut the collapse rate by
  about 11 percentage points with mutation-made strategies and about 8 with
  language-model-made ones.
- Adaptive quotas and closures cut collapse by 92%, identically for every
  strategy source.
- With PPO agents, an adaptive quota took collapse from 81% to 0%.

**What it means**

- Enforced limits protect the stock in this simulator. That part is solid
  and unsurprising.
- Three things weaken the stronger claims:
  - **The static quota could not work.** It allows each agent 7% of the
    stock per step. With 12 agents that is 84% per step, which is
    unsustainable even if everyone obeys. "Adaptive beats static" therefore
    mostly means "a sustainable limit beats an unsustainable one".
  - **The confidence intervals are too narrow.** With only 5 runs, the
    method used makes intervals too tight. Rebuilt properly, the headline
    11- and 8-point effects probably include zero.
  - **The language-model arm was mostly not the model.** Its outputs were
    randomly perturbed and pushed towards greed after generation, and
    failures silently fell back to ordinary mutation.

---

## Phase 2: Harvest (March–April 2026)

**Question.** Each agent has its own patch, and over-harvesting damages its
neighbours. Is the resource better protected by central caps, by peer
communication with credit transfers, or by both?

**What was done**

- **The game.** Six agents on a ring for 80 steps.
- **Four packages were compared:**
  - none;
  - "bottom-up" (messages and credits, no central governor);
  - "top-down" (a central governor sets caps);
  - "hybrid" (messages plus caps targeted at neighbourhoods).
- **Additions over the phase:** named scenario presets ("community
  irrigation", "forest co-management"), institutional frictions (missed
  detections, delays, limited targeting, costs), and a PPO check.

**What came out**

- **Hybrid usually ranked first**, but by small margins. The average patch
  health gain was +0.044 on a 0–20 scale (verified for one summary file),
  while harvest fell slightly.
- **There were no garden collapses in any compared cell**, so the ranking
  rested on tiny health differences.

**What it means**

- Central caps raise resource health at some cost to harvest. Communication
  alone raises harvest but protects less.
- **Hybrid changes three things at once** (messages, targeting, neighbour
  expansion), so we cannot say which part helps.
- **The scenario names are parameter bundles,** not models of real
  irrigation or forestry systems.
- **The PPO reward was tuned on the evaluation metrics.** Its training
  reward included bonuses for exactly the behaviour later measured.

---

## Phase 3: Stage A, "local pass / global fail" (June 2026)

**Question.** Can a simple per-agent check ("is my request below 40% of the
maximum?") pass while the whole system becomes unsafe? And does this get worse
as actors become "stronger" relative to the overseer?

**What was done**

- **Grid:** 2 scenarios × 3 actor-search settings × 3 overseer settings ×
  4 governance packages = 72 condition rows, each with 5 runs × 15
  generations.
- **"Local-pass/global-fail" (LPGF)** counted steps where every *request*
  passed the 40% check but the next state was globally unsafe.
- **A "capability gap" score** was computed as actor rank minus overseer rank.

**What came out (verified)**

- **Hybrid ranked first in 13 of 18 cells,** local in 4 and global in 1.
- **Average unsafe time:** none 19.4%, local 13.9%, global caps 6.1%,
  hybrid 4.1%.
- **The illustrative 29-step LPGF case** happened entirely *after* the
  system was already unsafe.

**What it means**

- **Caps keep patches healthier.** That part holds.
- **The "local" package has no overseer at all.** It is messages and credits
  only. So it cannot show that "local oversight fails".
- **The capability-gap score is not a measurement.**
  - It adds up unrelated things (search size, detection rate, delay, cost).
  - The actor manipulation ran backwards. In every condition without an
    overseer (none and bottom-up, in both scenarios), the *weakest* actor
    setting caused the most unsafe time. For example, forest with no
    governance gave 0.381 / 0.290 / 0.244 for low / medium / high.
  - The apparent "gap" effect came entirely from the overseer setting.
- **The central governor almost never switches off.** It triggers when
  average health falls below 16, well above the safety line of 10. So "global"
  is effectively a near-permanent cap. With the strong-overseer setting, its
  zero unsafe time follows from how it was tuned. With limited or weak
  overseers it still had 3–14% unsafe time (6.1% on average).
- **About half of hybrid's 13 wins are ties.** Both hybrid and top-down
  were at zero unsafe time, and hybrid won on health differences of
  0.01–0.06.
- **The "stress" test regimes were weaker than labelled** because of a code
  bug. They were built from the base tier's numbers, not the scenario's.
  For example, "slow regrowth" was only a 7% reduction in community
  irrigation instead of 15%.

---

## Phase 4: Threshold replay (June 2026)

**Question.** Do Stage A's conclusions depend on exactly where the safety
lines were drawn?

**What was done.** The same trajectories were relabelled under 25
combinations of the local cutoff and the global health threshold, giving
9,000 rows.

**What came out (verified)**

- Labels move smoothly as the lines move. The default lines were not
  cherry-picked.
- Hybrid still wins most cells.

**What it means**

- It is a sensitivity check, not new evidence: the 9,000 rows are
  relabellings of the same runs.
- One thing to correct in the notes: the script re-simulated the runs rather
  than reading saved trajectories. The results are identical either way,
  because the thresholds do not affect the dynamics.

---

## Phase 5: Small language-model strategy banks (summer 2026)

**Question.** Can small open models write valid Harvest strategies, and do
the earlier patterns still hold with them?

**What was done**

- Qwen 2.5 3B and Llama 3.2 3B each wrote 32 "cooperative" and 32
  "exploitative" strategies offline, as JSON.
- These were evaluated in fixed populations with different mixes.

**What came out**

- **(verified)** All-exploitative populations collapse without governance,
  and none collapse under caps.
- **A later control** built strategies directly from the numbers given in the
  prompt, with no language model. It produced the same broad pattern.

**What it means**

- The interface works.
- The prompt's numbers, not the model's reasoning, drive the behaviour.
  Qwen copied all 14 prompted numbers in 35 of its 64 strategies.
- Because the cap rule overrides strategy content, this mostly re-tests
  "caps work".

---

## Phase 6: September validation checks

### 6a. Frozen-population mechanism checks (verified)

**Question.** Take the same evolved populations and apply different
interventions. Which ones actually prevent unsafe states?

**What was done**

- **Populations:** 20 final-generation populations, plus a later
  slow-regrowth follow-up chosen after the first screen showed few failures.
- **Arms:** no intervention, messages only, announcement only, caps, a fixed
  40% cutoff, a "local state" filter (predicts your own patch next step), and
  a joint reference.

**What came out (slow regrowth, high stress)**

| Arm | Unsafe time | Return |
| --- | ---: | ---: |
| No intervention | 19.70% | 980.48 |
| Fixed cutoff | 5.33% (16/80 episodes had failure with every action passing) | 1,046 (highest) |
| Local state filter | 0.125% | 1,028.95 |
| Joint reference | 8.30% | — |
| Messages only | 0% | — |
| Announcement only | 0% | — |

**What it means**

- **The 19.70% comes from 4 of the 10 populations,** with unsafe rates of
  59%, 23%, 30% and 84%; the rest are near 0. Paired over the 10
  populations, the improvement from messages, announcements or the local
  state filter is −19.7 ± 21.3 points (95% t-interval), so it includes zero.
  With so few populations, this is descriptive.
- **Messages and announcements "work" because the strategy code obeys them
  automatically.**
  - Every agent lowers its request when neighbours announce restraint.
  - Every agent caps itself at an announced cap plus at most 0.25.
  - Nobody chose to comply, and these traits were never under selection.
  - So these results say nothing about agents who could ignore a message.
- **The fixed cutoff's failures happen because the cutoff (2.4 per patch)
  is slightly above the most this slow-regrowing patch can sustain
  (2.38).** That is a real example of "every action passed, the system still
  failed", but it is built into this setting rather than discovered.
- **The joint reference had a different safety goal** and a different
  weather assumption from the local filter. So comparing the two does not
  isolate information. This is the motivation for the matched reviewer
  experiment.

### 6b. Does more strategy search make stronger actors? (verified)

**Question.** If you pick the best of more mutated candidates (1, 6 or 12),
does the chosen strategy really earn more in new episodes?

**What came out**

- **Moderate stress:** picking from 6 instead of 1 raised held-out score by
  about 9 (interval 2.5–15.4).
- **12 candidates:** the gain is driven by one context out of 12.
- **High stress:** no reliable gain.
- **Horizon:** the two selection horizons usually picked the same
  candidate, so the "horizon" factor was barely tested.
- **"Held-out" meant new weather with the same partner agents.**

**What it means.** More search sometimes helps an agent's own payoff. There
is no validated ladder of actor strength.

### 6c. Language-model strategies vs prompt-number control (verified)

**What it means.** The broad collapse/protection pattern does not need a
language model.

### 6d. Development pilots (notes only)

- **Matched-oversight pilot, decision-case coverage, held-out policy pilot,
  budgeted-reviewer pilot (400 episodes).** Together these set up the final
  confirmation: same safety target, same action menu, and inspection
  budgets 0, 3 and 6.
- **The pilot already showed the long-run reversal.** In Fishery, the
  least-informed reviewer earned 1,109 against 527 for the fully informed one.
- **The pilot settings were then picked for the confirmation** because they
  produced both safe and risky requests.

### 6e. Actor-pressure pilot (notes only)

**Question.** Does searching over 1, 8 or 32 candidates produce stronger
actors that make reviewers err differently?

**What came out**

- The entrant's payoff rose: 139.9 → 177.9 → 181.5.
- **The 8-candidate search picked the same strategy as the 1-candidate
  search in 3 of 4 contexts.**
- **The search never saw the reviewer.**
- **Every reviewer still rejected all risky requests.**

**What it means.** The "stronger actor" manipulation did not work. Your notes
already call it a negative design check, correctly.

### 6f. Clean Up as a third game (notes only)

**What came out**

- The first attempt crashed on an observation-parsing error.
- The second ran, but:
  - every episode started already polluted;
  - no policy reached the clean target;
  - no apples were harvested by either variant, so a productive vs
    free-rider contrast was impossible.

**What it means**

- The admission gate effectively required *restoring* a polluted river, and
  the adapter notes had already shown that the starting map is polluted.
- Clean Up is not a result. A redesign would need clean starting states and
  competent policies first.

---

## Phase 7: The fresh-seed reviewer confirmation (23 September 2026)

This is the main current evidence; see `background/september_reviewer_experiment.md`.

- **Reported result.** Joint review cuts fewer safe requests than a deliberately
  conservative local rule, without approving more risky ones. A local rule
  that shares the same inspected numbers is identical to joint review. Good
  one-step decisions did not give the best long-run harvest.
- **My re-analysis [post hoc]:**
  - The Fishery decision results follow largely from arithmetic.
  - The Harvest results were dominated by an uncertainty allowance about 3.6
    times larger than the binding safety condition needs.
  - The long-run reversal is most likely caused by a safety line far below
    the resource's most productive level. This is untested, because the
    line was never varied.
