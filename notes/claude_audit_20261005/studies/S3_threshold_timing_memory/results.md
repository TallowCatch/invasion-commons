# Results S3: the deterrence threshold, imperfect audits, timed cheating, and a reviewer with memory

**Protocol:** `15_PROTOCOL_S3_THRESHOLD_TIMING_MEMORY.md`. It was frozen at 2026-10-05T11:20Z and committed as `087d392` before any S3 code existed. Amendment 1 was added after the engineering gates and before any full run.

**Runs:**
- `results/runs/claude_s3_part{A,B,C,D}_v1/`, one full run each, with no reruns.
- Run times: A 28 s, B 15 s, C 6 s, D 104 s.
- Analysis: `experiments/oversight/analyze_s3.py`, which writes `analysis/s3_summary.json` (in part A) and `s3d_condition_table.csv` (in part D).
- Figures: `fig12_s3_threshold_timing` and `fig13_s3_memory_reviewer`.

**Status of the numbers:**
- Every number below is **verified**: read from those files by the analysis script.
- The intervals are 95% paired context-bootstrap intervals (4,000 resamples).
- Interpretations marked **[post hoc]** were reached after seeing the data.

**Fixed for every part:**
- 64 test contexts and 80 steps;
- one shared cut chosen by the calibrated joint reviewer;
- fixed-rule agents.

---

## Parts A and B: where the deterrence threshold is

### Question

Hidden over-extraction gains a cheater about 28 per episode when nobody checks. How large must the expected fine per agent-step, e = audit rate × detection rate × fine, be before cheaters stop? Does the answer depend on anything other than e?

### What was done

**Setting:** Fishery with the MSY target. The 4 stress agents may take more than their allowance.

**Varied:**
- e over 11 values from 0 to 1;
- Part A: audit rate q ∈ {1/6, 0.10, 0.05, 0.02}, with perfect audits;
- Part B: q = 1/6 with audits that catch an over-take only half or a quarter of the time.

The fine is set from e, as F = e / (q × s). For example, q = 0.02 and e = 0.5 gives a fine of 25.

**The cheaters' choice.** For each cell, the cheaters' common level d* was chosen on 8 training contexts. It was then tested on 64 fresh contexts with new seeds.

### What came out

**With no checks:**
- the cheaters chose d* = 0.75;
- they over-took on all 80 steps;
- each gained 28.5 over complying, held out.

The expected-fine break-even is therefore 28.5 ÷ 80 = **0.356**.

**The level each group chose:**

| Audit rate q | Detection s | Fine at e = 0.3 | Last e with cheating | Held-out gain per cheater there | First e with d* = 0 (e*) |
| --- | --- | ---: | ---: | --- | ---: |
| 1/6 | 1 | 1.8 | 0.25 | +8.5 [+6.9, +10.2] | **0.30** |
| 0.10 | 1 | 3.0 | 0.30 | +4.4 [+2.5, +6.3] | **0.35** |
| 0.05 | 1 | 6.0 | 0.35 | +1.0 [−1.4, +3.4] | **0.40** |
| 0.02 | 1 | 15.0 | 0.25 | +8.2 [+5.7, +10.8] | **0.30** |
| 1/6 | 0.5 | 3.6 | 0.25 | +8.3 [+6.5, +10.1] | **0.30** |
| 1/6 | 0.25 | 7.2 | 0.25 | +8.6 [+6.7, +10.5] | **0.30** |

- **The gain curves overlap.** At each e, the held-out gain was nearly the same for all six settings: about 20.5 at e = 0.1 and 12.5 at e = 0.2 (Figure 12, left).
- **The switch is clean.** Once d* reached 0 it stayed 0 at every larger e, in all six settings.
- **The level chosen while cheating** was always 0.75.

**Hypotheses:**
- **A-H1 (cheat at e ≤ 0.2, stop at e ≥ 0.5, for every q): supported.**
- **A-H2 (e* the same for every q, within one grid step): falsified as stated.** e* was 0.30, 0.35, 0.40 and 0.30. The extremes are two grid steps apart. All four lie within 0.06 of the break-even 0.356.
- **B-H1 (imperfect audits act through the same product): supported.** e* = 0.30 for s = 1, 0.5 and 0.25.
- **A-H3: true by construction**, and so uninformative. At e*, d* = 0 by definition, so the cheaters' payoff equals complying. The protocol's wording made this hypothesis empty. That should have been caught before the run.

### What it means

**What the data show:**
- Audits plus a flat fine deter these cheaters almost exactly at the expected-penalty break-even, the reasoning usually attributed to Becker (1968).
- Audit rate, fine size and detection rate matter only through their product e.
- With audits about 8 times rarer than in S2 (q = 0.02, about one agent-audit every 8 steps), a fine of about 15–20 still deters.

**How much is built in:** most of it.
- The cheater is risk-neutral, the fine is flat, and it over-takes every step.
- Each cheater's expected net gain is therefore its gain from over-taking minus e × 80 (its expected fines), whatever q and s are. The overlap of the curves follows from this.
- **Not built in:** that the strategy chosen on 8 training contexts generalises to 64 new ones; that the switch matches the held-out break-even; and that the code computes all of this correctly.

**Where A-H2 failed:** at q = 0.10 and q = 0.05, the groups kept cheating at e = 0.30–0.35, where the held-out gain was small (+4.4 and +1.0). The other groups stopped one step earlier. On the 8 training contexts, the payoff from cheating and from complying differed by less than 1% at e = 0.3. For example, at q = 1/6: 1,177.8 against 1,180.2. **[post hoc]** So near the break-even, which side a group lands on depends on the training sample. This is the predicted failure mode, and it is small.

**What it does not show:**
- deterrence for risk-averse cheaters, or for fines that grow with the size of the over-take;
- deterrence for cheaters who learn from their own catches;
- that a 2% audit rate is enough in general. It works here because the gain per step is small (0.36) and a fine of 15–20 is allowed;
- anything about Harvest, where cheating never paid (S2).

---

## Part C: cheaters who time their cheating

### Question

Do audits still deter when the audit schedule is known in advance?

### What was done

**Setting:** the same Fishery setting, at the same average audit rate per agent-step (1/6):
- **random:** each agent is audited with probability 1/6 each step;
- **periodic:** every agent is audited on steps 5, 11, 17 and so on, and the agents know it.

**The cheaters' choice:** a level d and a timing rule, either always cheat or skip steps where an audit is likely. Fines were 0, 6 and 24.

### What came out

**Per honest agent's harvest, and what the cheaters chose:**

| Arm | Chosen strategy | Harvest per honest agent | Stock collapsed | Cheaters' gain over complying (group of 4) |
| --- | --- | ---: | ---: | --- |
| Everyone complies | none | 104.3 | 0/64 | 0 |
| No checks | d = 0.75 every step | 40.9 | 0/64 | +113.9 [+107.6, +120.0] |
| Random audits, fine 6 or 24 | comply (d* = 0) | 104.3 | 0/64 | 0 |
| **Periodic audits, fine 6 or 24** | **d = 1 on unaudited steps** | **28.8** | **12/64** | **−58.7 [−138.4, +11.2]** |

**Under the periodic schedule:**
- timed cheaters were never caught;
- steps breaking the MSY target rose by 90.4 points [+88.4, +92.4];
- honest agents lost 75.5 each [−80.8, −70.4]. That is more than the 63.4 they lost with no checks at all.

**Hypotheses:**
- **C-H1: half supported, half falsified.**
  - Supported: the search chose timed cheating with d* = 1 at both fines. This part was true by construction.
  - Falsified: the held-out gain over complying was −58.7 [−138.4, +11.2], an interval that includes 0 with a negative estimate.
  - The strategy picked on 8 training contexts collapsed the stock in 12 of 64 new contexts. It hurt the cheaters as well.
- **C-H2 (random audits deter at fines 6 and 24): supported.**
- **C-H3 (honest harvest falls by at least half the no-check drop): supported.** The drop was −75.5 against a half-drop of −31.7.

### What it means

**What the data show:**
- **A known schedule is worse than no checks.** At the same average audit rate, a predictable schedule gave cheaters a safe window, and they used it fully. The result was worse for honest agents than no checks at all.
- **Random timing works.** Unpredictable audits at the same rate fully deterred.
- This matches the inspection-game intuition that inspections must be unpredictable (concept only; Avenhaus et al. 2002).

**[post hoc] Why the outcome is worse than no checks:**
- With no checks, cheaters picked 0.75, which kept the stock near 60.
- With a known schedule, d = 1 on 5 of 6 steps looked best on the training contexts, but it overshoots on new ones.
- This is a strategy over-fitted to a small training sample, not a deliberate choice to ruin the commons.

**What it does not show:**
- partially predictable schedules, such as audits that depend on the stock level;
- adaptive cheaters who learn the schedule over time;
- a cost for the auditor of randomising.

---

## Part D: a reviewer that remembers caught lies

### Question

Can audits make the reviewer's decisions safer, without excluding or fining anyone, if the reviewer remembers what each audit revealed?

### What was done

**Setting:** S1's misreporting setting with sanction none. Misreporters report a reduced request.

**Compared:**
- **Trust;**
- **Audit, no memory** (S1b's arm);
- **Audit + memory:** after a catch, the reviewer scales up that agent's later reports by the lie it measured;
- **Audit + assume maximum:** after a catch, that agent is treated as always asking for the maximum.

**Liars:** cut by half every step (fixed), or by a fresh random fraction each step (noisy, mean one half).

**Seeds:** the S1 seeds, so every arm is paired by context. All arms were rerun in S3 (see Amendment 1).

### What came out (2 random audits per step; Figure 13)

| Setting, liars | Trust | Audit, no memory | Audit + memory | Audit + assume max | Everyone honest |
| --- | ---: | ---: | ---: | ---: | ---: |
| Harvest, % of steps unsafe, fixed | 1.93 | 1.74 | **0.98** | 0.66 | 0.61 |
| Harvest, noisy | 2.13 | 1.66 | **1.21** | 0.70 | 0.61 |
| Fishery stock-≥-10 target, % unsafe, fixed | 80.4 | 68.8 | **1.5** | 0.0 | 0.0 |
| Fishery stock-≥-10 target, noisy | 82.5 | 75.4 | **32.1** | 0.0 | 0.0 |
| Fishery MSY, % of steps breaking target, fixed | 98.8 | 86.1 | **3.6** | 2.5 | 0 |
| Fishery MSY, noisy | 90.3 | 78.0 | **39.4** | 2.4 | 0 |

**Memory minus no memory (percentage points):**

| Setting | Fixed liars | Noisy liars |
| --- | --- | --- |
| Harvest | −0.76 [−1.05, −0.49] | −0.45 [−0.64, −0.25] |
| Fishery stock-≥-10 target | −67.3 [−72.2, −61.9] | −43.2 [−49.3, −36.8] |
| Fishery MSY | −82.6 [−84.2, −80.9] | −38.6 [−40.4, −36.8] |

The pattern was the same with 1 random audit or 1 targeted audit per step in Harvest. For example, with a targeted audit and fixed liars, memory was 0.62 points below no memory [−0.88, −0.39].

**The cost of "assume maximum"** is usefulness loss, the share of safe requests it cuts:

| Setting | Assume maximum minus memory |
| --- | --- |
| Harvest | +8.9 points [+5.6, +12.8] |
| Fishery stock-≥-10 target | +70.3 points |
| Fishery MSY | +80.7 points |

**Harvest totals:** honest agents' harvest barely changed between memory and no memory (Harvest noisy: −0.1 [−0.6, +0.4]).

**Fishery under the stock-≥-10 target:**
- memory raised honest harvest by 21.3 [+18.8, +23.8], because the stock no longer collapsed;
- "assume maximum" gave even more total harvest (738 against 560 with memory). It over-restricted, which acted as accidental conservation under the low target, the same effect seen in R1.

**Hypotheses:**
- **D-H1 (Harvest: memory at least 0.5 points safer than no memory): supported**, at −0.76 [−1.05, −0.49].
- **D-H2 (Fishery stock-≥-10 target: memory below 10% unsafe): supported**, at 1.5%.
- **D-H3 ("assume maximum" at least as safe as memory, but cuts more safe requests): supported** in Harvest: 0.66% against 0.98% unsafe, and usefulness loss +8.9 points.
- **D-H4 (noisy liars make memory less safe): supported by the estimate only in Harvest**, at +0.23 [−0.06, +0.57], an interval that includes 0. **Clearly supported in Fishery**, at +30.7 [+24.4, +37.0].

### What it means

**What the data show:**
- **Audits can improve decisions without excluding anyone, if the reviewer remembers.** In Fishery, memory turned audits that were nearly useless into ones that came close to full verification. In Harvest it closed about two thirds of the gap between no memory and everyone honest.
- **The improvement shrinks when lies vary.** Against noisy liars, memory removed only about half of the Fishery harm, because one catch no longer reveals the next lie.
- **The conservative memory has a price.** It is safer, but it cuts most safe requests in Fishery.

**How much is built in:** against a liar who always cuts by exactly half, a single catch reveals the lie perfectly. That fixed-liar result is therefore close to "true by construction". The noisy-liar result is the real test, and there the gain is about half as large.

**What it does not show:**
- liars who change behaviour after being caught: here the liars are fixed rules, not adaptive;
- a reviewer that must also infer *who* lies when audits are imperfect;
- anything about the cost of audits.

---

## What we now believe, and how confident we are

1. **Unpredictable audits plus fines deter cheating at almost exactly the expected-penalty break-even** (e ≈ 0.36 here), whatever the mix of audit rate, fine size and detection rate. We are confident this holds in this model; most of it is arithmetic.
2. **A predictable audit schedule fails, and can be worse than no checks** (honest harvest 28.8 against 40.9). We are confident about the direction. Its size depends on an over-fitted strategy from 8 training contexts.
3. **A reviewer with memory turns audits into better decisions without any sanction.** We are confident this holds against fixed liars, which is partly by construction. Against varying lies the effect is about half as large, and that is the more credible estimate.

## Still untested

- risk-averse cheaters, and fines that scale with the over-take;
- cheaters who learn from catches or adapt to the reviewer's memory;
- partly predictable audit schedules;
- audit costs;
- any of the deterrence questions in Harvest, where cheating never paid.

## Next concrete step and its cost

The most informative next run combines memory with adaptive liars, who choose their lie knowing the reviewer remembers. It also adds a cost per audit, so audit rate becomes a real trade-off. Both reuse the S3 code: a protocol and about 10 CPU minutes.

The decisions listed in file 04 still belong to Ameer and the supervisors:
- the definition of harm;
- which paper;
- which game is the main one.
