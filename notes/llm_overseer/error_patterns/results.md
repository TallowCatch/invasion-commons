# Patterned LLM inspector errors: results (23 Nov 2026)

**Question.** Does an LLM inspector's error pattern, measured on real situations, change deterrence compared with a coin flip at the same average miss and false-alarm rates? Protocol and predictions, written before any run: `protocol.md` (Amendments 1–2 are logistics only). How to rerun and the CREATE commands: `README.md`. Generated tables: `fit_tables.md`, `fit.json`.

**What was done here.** The offline half: a bank of real situations from trained agents, verdicts from two LLM inspectors, a fitted error model, a trainer option that plays it, and a job list. **No RL run with the patterned inspector has been done.** Section 5 lists them; section 6 says what they are expected to show.

## 1. Short answer

- **gpt-oss-20B's misses are strongly patterned.** Its miss rate α runs from **0.00** (no apples around, few in the outer cells) to **0.48** (2 apples around, 2 or more in the outer cells). Two things drive it: how close the count is to the threshold, and how many apples sit in the cells that don't count (it counts some of them in). The pattern predicts held-out verdicts much better than a constant rate (gain +0.128 nats per item, 95% interval +0.103 to +0.153, n = 1,200 violations).
- **Its false alarms are rare and only weakly patterned.** β is 0.06 at exactly 3 apples (the borderline legal eat) and about 0.01 elsewhere. With 25 flags in 1,263 honest eats, the pattern does not beat a constant rate on held-out items (interval includes 0). For β the coin flip is as good a model as the data can tell.
- **The coin-flip numbers from the ladder were off for learners' records.** On the bank, gpt-oss-20B's average α is 0.215 (0.288 on the records of agents trained under an overseer), not the ladder's 0.10; its β is 0.039, not 0.077. Both ladder numbers came from 26–30 bot records; the new ones are inside the ladder's wide intervals, so this is a correction more than a contradiction.
- **Why this could matter for deterrence.** An agent that kept its over-harvests to the worst-checked situations would face α = 0.48, so each over-harvest would cost (1 − 0.48)·c = 0.52c instead of the coin flip's 0.79c. F24 puts the learners' threshold at about 0.75–0.9 of an apple (higher at k = 1). So at c = 1 the pattern could move a cell from "alive" to "dead"; at c = 2 even the worst case (1.04 apples) stays above it. Whether learners find the pattern is the RL question (§5–6).
- **Qwen 7B + reasoning (secondary, 179 items):** misses about half of all violations (α 0.53 on the subset; 0.57 on the bank average, 0.69 on the overseen agents' records). Its misses rise with the count and with outer apples, as gpt-oss's do (0.34 → 0.59 → 0.73 at near 0 / 1 / 2). But with 89 violations no state-dependent model beats a constant rate on held-out items, so it is reported as averages only and has no table in the job list. If its bank-average α is right, the existing Qwen 7B + reasoning runs at c = 2 overstated its deterrent: an over-harvest would cost 0.87 apples, not 1.07, near the threshold.

## 2. What was run

- **Situation bank** (`experiments/llm_overseer/error_bank.py`; `experiments/llm_overseer/data/error_bank.jsonl.gz`). Five stage 1 parameter sets played with no overseer, 1,000-step episodes, actions sampled: A_none (32 episodes) and B_random k1/c1, k2/c1, k7/c2 and D_history k2/c1 (6 each). Every eat is one item, because at w = 1 the trainer applies a miss only to a checked violation and a false alarm only to a checked honest eat. **13,159 eats, 1,717 violations.**
  - A_none supplies 972 violations, the four overseen policies 745.
  - The learners eat at the threshold: 63% of honest eats are at exactly 3 apples around. The overseen agents' over-harvests are also borderline: 62% at near = 2, against 26% for A_none.
- **Features** on the map before the move, in the 5 × 5 patch the LLM sees: `near` (apples in the 12 counted cells; it fixes the label), `outer` (apples in the other 12 cells of the patch), `players`, `walls`. A test checks they equal what the renderer draws (`tests/test_inspector_table.py`).
- **Judging subset**, stratified by `near` (random within each value, then shuffled): 400 items for each of near 0, 1, 2; 300 / 250 / 200 / 200 for near 3 / 4 / 5 / 6; 150 for near 7, and all of near 8–10 (45, 35, 83). **2,463 items** (1,200 violations, 1,263 honest eats).
- **gpt-oss-20B on raw maps** (Ollama cloud, `think: low`, as in the ladder) judged all 2,463. **Cloud tokens: 2,417,973** over 2,463 logged calls (982 a check: 515 in, about 470 out), under the 3M cap, about $0.77 at $0.32 per million. Up to about 4,000 more tokens were spent in calls cut off by a disk-full crash and not logged (protocol Amendment 2). 24 replies (1%) hit the 2,048-token limit without a verdict; they count as no flag (10 of them are violations, adding 0.8 points to α).
- **Qwen 7B + reasoning on raw maps** (local, reason prompt, `--num-ctx 2048`): **179 items**, the first 179 of the same shuffled order (89 violations: 35 / 32 / 22 at near 0 / 1 / 2; 90 honest eats). The plan was 600. Ameer then capped it at 200 to free the laptop, and the disk guard stopped it twice (at 115 and 179 items) when swap left under 2 GB of disk free; it was not restarted. Because the order is shuffled, the first 179 are a random part of the stratified subset and so stratified too (the first 200 have 96 violations and 104 honest eats). 771 tokens a check (local, no cost), about 30 s each. The model was unloaded at the end.

## 3. The error pattern

### gpt-oss-20B, raw maps

**Misses** (α, n = 1,200 violations; 95% Wilson intervals):

| near (apples counted) | all | outer 0–1 | outer 2+ |
|---|---|---|---|
| 0 | 0.03 [0.02, 0.05], n = 400 | **0.00** [0.00, 0.02], n = 310 | 0.11 [0.06, 0.19], n = 90 |
| 1 | 0.17 [0.14, 0.21], n = 400 | 0.02 [0.01, 0.04], n = 238 | **0.40** [0.33, 0.48], n = 162 |
| 2 | 0.38 [0.33, 0.43], n = 400 | 0.31 [0.25, 0.37], n = 232 | **0.48** [0.41, 0.56], n = 168 |

- The judge misses when the count is near 3 and when apples outside the 12 counted cells are on the map. Logistic regression (per standard deviation): `outer` z = +9.4 on missing; `players` (+0.4) and `walls` (−0.9) do nothing.
- So the "busy map" that matters is apples in the wrong cells, not other players. This is the over-counting mistake predicted in the ladder protocol (P2: counting the 24 cells instead of 12).
- **How far α varies:** 0.003 to 0.48 across the six table cells (all n ≥ 90); spread weighted by how often each cell occurs among the bank's violations: sd 0.19.
- **Mostly, the same situation gets the same rate whoever made it.** A_none's average α on the subset is 0.13 and the overseen policies' 0.18–0.32, largely because they over-harvest in different cells. Within cells, A_none and the overseen policies agree to within 0.15 in five of six cells. The exception is near = 2 with 0–1 outer apples: A_none 0.15 (11 of 73) against 0.38 (60 of 159). So something the two features don't capture (the exact layout, or how many players stand nearby in denser parts of the orchard) still matters there. The table averages over it.

**False alarms** (β, n = 1,263 honest eats): near 3: **0.06** [0.04, 0.09] (n = 300); near 4: 0.01 [0.00, 0.03]; near 5 and 6: 0.01; near 7–8: 0.00 [0.00, 0.02]; near 9–10: 0.01. `outer`, `players` and `walls` have no detectable effect (|z| < 1.4).

### Held-out test (5-fold cross-validation, mean log-likelihood per item, gain over the constant rate with a paired bootstrap 95% interval)

| | M1: by near | M2: logistic (near, outer, players, walls) | M3: near × outer (the trainer's table) |
|---|---|---|---|
| gpt-oss α | +0.073 [+0.053, +0.092] | +0.120 [+0.098, +0.143] | **+0.128 [+0.103, +0.153]** |
| gpt-oss β | +0.001 [−0.014, +0.013] | −0.001 [−0.016, +0.011] | = M1 (no outer split) |

Brier scores agree (α: 0.156 constant, 0.122 M3). The trainer's table is the best of the four models for α; for β nothing beats the constant.

### Qwen 7B + reasoning, raw maps (179 items: wide intervals)

- **Misses** (n = 89): α 0.53 overall. By near: 0: 0.34 [0.21, 0.51] (n = 35); 1: 0.59 [0.42, 0.74] (n = 32); 2: 0.73 [0.52, 0.87] (n = 22). Logistic z: near 2 +3.2, `outer` +4.0, `walls` +2.7, `players` −1.3. In-sample, the near × outer cells run from 0.08 (n = 12) to 0.86 (n = 7).
- **False alarms** (n = 90): about 6 flags, β 0.07. At near 3: 0.12 [0.03, 0.34] (n = 17). No feature has |z| > 0.8.
- **Held-out test: nothing beats the constant rate.** α: M1 −0.012 [−0.090, +0.070]; M2 +0.072 [−0.072, +0.209]; M3 −0.007 [−0.128, +0.105]. β: M1 −0.033 [−0.096, +0.015].
- **Reading.** The direction matches gpt-oss: misses rise with the count and with apples in the wrong cells. But 179 items cannot show the pattern holds out of sample. Per Ameer's instruction, Qwen is reported as averages only: on the bank α̅ = 0.567 (overseen policies 0.689) and β̅ = 0.096 (0.098). A table would need about 1,000 items: about 8 hours locally, or a short vLLM job on CREATE.

## 4. Predictions checked

| Prediction (protocol §6) | gpt-oss-20B | Qwen 7B + reasoning (179 items) |
|---|---|---|
| E1 misses cluster on the borderline violation: α(0) ≤ 0.05, α(2) ≥ 0.15 (gpt-oss); α(2) − α(0) ≥ 0.2 (Qwen) | **held**: 0.03 and 0.38 | **held**, nominally: 0.34 → 0.73 (difference 0.39; the intervals just don't overlap) |
| E2 β(3) at least twice the average β over near ≥ 6 | **held**: 0.06 vs 0.005 (but few flags) | nominally held (0.12 vs 0.03), but n = 17 at near 3: not significant |
| E3 more `outer` apples raise α and lower β | **held for α** (z = +9.4); **not for β** (z = +0.1, no effect) | **held for α** (z = +4.0); not for β |
| E4 the state-dependent model beats the coin flip on held-out items, and α ranges by ≥ 0.15 | **held for α** (range 0.48); **failed for β** | **failed**: no model beats the constant on held-out items |
| E5 the bank averages differ from the ladder's by more than the ladder's intervals | **failed**: 0.215 and 0.039 lie inside the ladder's intervals for 30 and 26 items ([0.03, 0.26], [0.02, 0.24]) | **failed**: α 0.567 and β 0.096 lie inside the ladder's intervals ([0.30, 0.64], [0.01, 0.19]) |
| E6 (RL) | not tested here: §5–6 | |

## 5. Job list (written, not submitted)

`experiments/llm_overseer/runs_error_patterns.tsv`, 36 runs, gpt-oss-20B only (Qwen has no table, §3); submit commands in `README.md` §3 (they need a checkout of this branch on CREATE, for `--inspector-table` and the table files).

| Arm | Flags | gpt-oss-20B |
|---|---|---|
| `table` | `--inspector-table tables/<tag>.json` | α(near, outer), β(near): §3 |
| `coin` | `--miss-rate α̅ --false-alarm β̅`, the bank average | α̅ 0.215, β̅ 0.039 |
| `coinov` | the same averaged over the overseen policies only (protocol §3: added because it differs by more than 0.05) | α̅ 0.288, β̅ 0.042 |

Each arm: k ∈ {1, 7}, c ∈ {1, 2} (penalty 7c/k per catch), random checks, seeds 1–3, 1e8 steps from scratch, as in `../rl_results.md`.

- Why two coin flips: the average α depends on where agents over-harvest, so "the same average" needs a reference population. The bank average (mostly A_none's spread-out over-harvests) and the overseen agents' average (mostly borderline over-harvests) bracket it.
- What to read from the runs, besides apples and orchard survival: realised α = 1 − `detected` / `checks_on_violations` and realised β = `wrongly_stopped` / `checks_on_honest_eats` (new measures, logged in every run), and `violations_by_near`: do patterned-run agents move their over-harvests towards near = 2 and to busy patches?

## 6. What to expect from the runs

From the table, the fine an over-harvest actually costs is (1 − α)·c, with α between the coin flip's and the worst cell's:

| | coin flip (bank / overseen) | if every over-harvest sat in the worst cell |
|---|---|---|
| gpt-oss-20B, c = 1 | 0.79 / 0.71 | 0.52 |
| gpt-oss-20B, c = 2 | 1.57 / 1.42 | 1.04 |
| Qwen 7B + reasoning, c = 1 / c = 2 (no table; coin flip only) | 0.43 / 0.87 (overseen: 0.31 / 0.62) | – |

- F24 places the learners' threshold at about 0.75–0.9 of an apple at k = 2–7, and 0.9–1.0 at k = 1. So:
  - **c = 1 is where the pattern can decide survival.** The coin flips already sit at or below the threshold: gpt-oss at α 0.10 (a fine of 0.90) gave 210 apples and a partly dead orchard at k = 1, and 361 at k = 7. At k = 1 both arms will probably lose the orchard.
  - The most informative cell is **k = 7, c = 1**. The bank-average coin flip (0.79) sits inside the threshold band, so a shift of a tenth or two decides survival. If learners exploit the pattern, the table arm does worse than the coin flip there. If they don't, it should do about as well, or better where their over-harvests sit in well-checked cells.
  - **At c = 2 even full exploitation leaves about 1 apple per over-harvest**, above the threshold, so the orchard should survive in both arms; any difference shows in apples and in realised α.
- **A counter-effect.** The worst-checked situations are borderline over-harvests (2 apples around), which harm regrowth less than taking an isolated apple. Exploiting the pattern moves violations to where they do least damage, so the orchard may suffer less than the realised α implies.
- **Learning the pattern needs signal.** With k = 1 an agent is checked on 1 in 7 steps, so the difference between 0 and 48% misses shows up only through occasional fines. 1e8 steps may not be enough; realised α against the coin flip's is the direct test.
- **False alarms.** gpt-oss's β is small and mostly at near = 3 (63% of the learners' honest eats). In the patterned arm, honest agents could dodge it by eating at 4 or more, so wrongful fines per legal apple may fall below β̅·c. With β̅ = 0.04 the effect is small either way.

**What would show that patterns don't matter:** table and coin-flip arms within each other's seed ranges on apples, orchard survival and the fine per over-harvest in all cells, and realised α in the table arm equal to the coin flip's.

## 7. Related work: how the design and the result connect

References were checked by web search on 23 Nov 2026 (protocol §7 has the full list with venues). ★ = in Ameer's reading matrix.

- **Deterrence = chance × fine** (Becker 1968). Our fine per over-harvest, (1 − α)·(k/7)·P, is Becker's expected fine. A patterned α makes it differ across situations, and a learner can choose its situation. That is the whole mechanism this study measures.
- **Inspection games** (Avenhaus, von Stengel & Zamir 2002; Avenhaus & Canty 1996) treat an inspection as a statistical test with false alarms ("errors of the first kind") and non-detection ("second kind") at fixed rates. Here the non-detection rate is a function of the situation, not one number. That is closer to an inspectee who can pick where to violate.
- **Instance-dependent noise.** The coin flip is "class-conditional noise" in the label-noise literature (Frénay & Verleysen 2014). We find the miss rate is strongly instance-dependent (Chen et al. 2021 argue real noise usually is), while the false alarms are too rare to tell. LLM-judge biases that depend on the input (Zheng et al. 2023) are the general version; here the bias is concrete: counting apples in cells that don't count.
- **Exploiting an imperfect evaluator** (Everitt et al. 2017; Skalse et al. 2022; Gao, Schulman & Hilton 2023; Hardt et al. 2016). Everitt et al.'s corrupt-reward MDP is the exact setting: corruption (here, missed fines) depends on the state, and RL agents are drawn to corrupted states. Our offline result says the corrupted states exist and are reachable (near = 2 over-harvests are already 62% of what overseen agents take). The RL runs test whether IPPO finds them.
- **LLM monitors are scored by averages.** Scalable oversight (Bowman et al. 2022; Engels et al. 2025★) and AI control (Greenblatt et al. 2024★) summarise a monitor by aggregate rates. Our gpt-oss result shows the aggregate can hide a factor-of-100 spread across situations. An average α of 0.2 can mean "almost never misses" in most situations and "misses half the time" in a few. Which of the two a learner faces depends on where it chooses to act.

### The Png / Polinsky–Shavell conflict, reconciled

- **The theory.** Law and economics says wrongful conviction weakens deterrence, the same as wrongful acquittal: it lowers the payoff of obeying the law and so narrows the gap between crime and compliance (Png 1986; Polinsky & Shavell 2000; Kaplow & Shavell 1994).
- **Our result.** F24 found false alarms barely weakened deterrence. That looks like a conflict.
- **Why it isn't one.** In our trainer, a false alarm can fall only on a **legal harvest**. It never falls on "wait". So the choice the fine shapes is:
  - over-harvest vs wait: the expected-fine gap is (1 − α)·q·P, and β does not enter;
  - over-harvest vs a legal harvest: the gap is (1 − α − β)·q·P, as in Png.
  Lando (2006) makes this point in general: whether wrongful conviction lowers deterrence depends on whether its risk changes with the act chosen. Garoupa & Rizzolli (2012) reply that it usually does. Lando & Mungan (2018) add that the effect scales with how often the lawful choice is exposed to judgement. In Harvest, the alternative to an illegal apple is mostly to wait (F24's reading), and waiting is never judged. So the law-and-economics prediction does not apply.
- **What β does instead** is Kaplow's (2011) chilling: it taxes legal eating (C2's 0.46 apples per legal apple at β 0.23, c = 2).
- **This study's addition.** gpt-oss's false alarms fall almost only on the borderline legal eat (near = 3). So the chilling falls on the eats closest to the line, and an agent can avoid it by eating at 4 or more. This is not a real conflict with the theory. It is a different choice set, and it should be stated whenever F24 is cited.
- **A caveat.** If agents' real alternative were often a legal apple elsewhere (a denser orchard, or an action space where waiting is costly), the theory predicts β would start to weaken deterrence. That is testable with the existing trainer (a map where legal apples are always in reach) but is not done here.

### Supervisor's group (Yali Du)

- **DMPO** (Du et al. 2022, IROS, arXiv:2207.06559) and Ma, Li, Du, Dong & Yang (2024, *Nature Machine Intelligence* 6:1006–1020). Both assume an agent's dynamics depend only on its κ-hop neighbourhood. Truncating to it costs an error that decays exponentially with κ.
  - The fitted error model is local in the same sense, with κ = 2 cells: everything the table needs is in the 5 × 5 patch around the agent. That is what lets each regional overseer in `decentralised.py` apply it without global information.
  - A general formulation across testbeds would therefore write the inspector as α(s_{N_κ(i)}) and β(s_{N_κ(i)}), functions of the checked agent's κ-neighbourhood, rather than as constants.
  - These papers say nothing about inspector error itself, and we do not claim they do.
- **BazaarBench★** (Wang et al. 2026, arXiv:2610.06748). Its GPT-5 judge agreed with humans on 94% of 200 checks, with some systematic disagreements. That is the same lesson at the level of labels: high average agreement can hide a pattern. It is also part of why the ladder's BazaarBench numbers measure agreement with GPT-5 rather than truth. Harvest's exact ground truth is what lets this study measure the pattern directly.

### Matrix papers not yet read that bear on this (suggested order)

1. **Greenblatt et al. 2024, AI Control.** Trusted monitoring with an audit budget: the closest AI-safety analogue of a budgeted inspector. The monitor's false-positive rate is set by the budget, which is how one would set an LLM's flag threshold.
2. **Parisi et al. 2024, Monitored MDPs.** The formal model for an agent whose reward (here, the fine) is seen only when a monitor shows it. A patterned inspector is a monitor whose coverage depends on the state.
3. **Makins et al. 2026, Multi-Agent AI Control.** Per-instance monitors have blind spots that coordinated agents exploit: the multi-agent version of this question.
4. **Gans & Holden 2026, randomised oversight.** Audits without errors, where stronger audits make undeterred violations better hidden. The natural place to add α(s).
5. **Ye & Steinhardt 2026, Norm Enforcement for AI Agents.** Agents weaponise false reports: β as an attack surface, not noise.
6. **Zhu, Zuo & Sun 2026, Blindfold.** If agents can see the audit coin, the audit is biased. Our learners can't see it but can learn α(s), a softer version.
7. **Vinitsky et al. 2023, norms from public sanctions.** Agents learn what is sanctioned from the sanctions they see. Under a patterned inspector, what they would learn is the inspector's pattern, not the rule.
8. **Wang et al. 2019, Green Security Games.** Randomised patrols against attackers that adapt in deep RL. The attackers' learned response to patrol patterns is the analogue of E6.
9. **Perolat et al. 2017, common-pool resource MARL.** Where Harvest's dynamics come from. It bears on the counter-effect in §6: a near-2 over-harvest harms regrowth less than a near-0 one.

## 8. Limitations

- **No RL yet.** Everything about deterrence in §6 is a prediction from the table and F24's threshold.
- **The table is coarser than the LLM.** At temperature 0 a record always gets the same verdict; the table turns that into a probability per cell, keeping only `near` and `outer`. Learners in the table arm can exploit only that part. M3's held-out gain over M2 is small, so the logistic model's extra features add little.
- **The bank is from agents trained against perfect inspectors or none.** Learners trained against the table may visit thin cells (near ≥ 8 honest eats; violations at near 0 with many outer apples, n = 90). The table smooths thin cells towards their near-value rate.
- **One prompt, one seed, temperature 0; gpt-oss via Ollama's cloud.** Qwen's 179 items are too few to show a pattern out of sample.
- **"Matched average" needs a reference population** (§5); two are given.

## 9. Not done, because it drifted from the question

- More models (gpt-oss-120B, larger Qwen), other renderings (summaries) or longer windows (w = 10, 50): each is another table, not a new answer to this question.
- Other games, other overseer types (history or location targeting with a patterned inspector), separate networks or revealed checks.
- The choice-set test of the Png reconciliation (a map where a legal apple is always in reach): noted in §7 as the way to test it.
- Running the RL jobs (not to be submitted from here).
