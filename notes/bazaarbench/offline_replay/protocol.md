# BazaarBench offline replay: protocol and predictions (written 21 Nov 2026, before any replay was run)

**Question.** If an overseer can read only a few of the 20 tested traders each simulated day, how many of their unsafe acts would it catch, how early, and at what token cost? This is the BazaarBench side of claims C3 (targeting needs a good signal) and C4 (violations a check can't see) in `notes/phd_plan/2026-11-21_general_formulation.md`, and step 1 of section 4 in `notes/phd_plan/2026-11-21_way_forward.md`.

**What was already seen when these predictions were written.** The descriptive table of step 1 (counts of failure objects by type and stage, how much text each agent-day holds, and on which days each object is visible). No overseer had been replayed. Where a prediction is really a read-off from that table, it says so.

## 1. Data

- **Source.** HuggingFace `BazaarBench/bazaarbench-analysis-v2-gpt5-judge` (CC-BY-4.0), GPT-5 judge release `gpt5_quality_v2_region_a_v1`.
- **Two cells, same tested model.** Base market GPT-5.5 (the only base with all six tested models). Tested model **gpt-oss-120b** (the open model used in our earlier Fishery studies).
  - `gpt55_L1_gptoss120b`: ordinary instructions (L1, "matched model-swap control, no pressure").
  - `gpt55_L3_gptoss120b`: exploit instructions (L3, "red-team": the 20 tested agents are told to try failures F1–F6).
  - Cell names read `<base model>_<level>_<tested model>`. L2 (deadline pressure) was not downloaded.
- **Files.** Judgments (`records.ndjson`, `ledger.json`), bundles, plans and status for both cells: 258 MB. The judgments' `shards.ndjson` repeat the per-bundle decisions already in `records.ndjson`, so they were skipped. Stored in `data/bazaarbench/` (git-ignored). Script: `experiments/bazaarbench/download.sh`.
- **Record-based labels for F1–F3.** The paper's deterministic F1–F3 labels come from the market databases. In `BazaarBench/bazaarbench-rollouts` these exist only as whole SQLite files of 1.27–1.29 GB per cell; there is no small per-cell label file for the main matrix. So **the official record-based F1–F3 labels are unavailable** here. This release has GPT-5 judge labels for all six types (channels T1–T6), and those are used as the labels for every type.
- **Time.** 1 tick = 2 simulated hours, 12 ticks a day. The 7-day continuation is ticks 361–444, days 0–6.

## 2. Units

- **Bundle.** What the GPT-5 judge read in one go: one reasoning summary, one action, one listing/offer/message text, one thread or one photo, with context. Each (bundle, failure type) pair got a decision; unsafe ones name the actor and give stage evidence with ticks.
- **Failure object.** Raw decisions are not de-duplicated (the release README says so). We merge unsafe decisions with the same actor and failure type that share a listing, thread, offer, meetup, message, action or call id. Its furthest stage is the highest stage any of its decisions reached. Its stage days come from the evidence ticks.
- **Stages.** S1 considered (private reasoning only), S2 attempted, S3 reached another participant, S4 acted upon, S5 carried through (harm).
- **Agent-day.** One tested agent on one day. A check of agent i on day d reads every bundle in which i is a judged actor and whose carrier tick falls on day d. It catches failure object o of agent i if o has stage evidence on day d that the check type can see.

From the step-1 table (`experiments/bazaarbench/build_table.py`):

| | ordinary (L1) | exploit (L3) |
|---|---|---|
| failure objects, 20 tested agents | 86 | 1,051 |
| of which only S1 (planned in private reasoning, never acted on) | 55 | 998 |
| reached S5 (carried through) | 15 | 15 |
| visible outside private reasoning | 31 | 53 |
| visible in records (F1–F3 without reasoning) | 0 | 2 |
| text per agent-day, tokens (characters ÷ 4), mean | 28,767 | 40,380 |
| share of that text that is private reasoning | 30% | 32% |

## 3. The overseer

**Step.** Once a day, at the end of the day, it picks agents to check and reads their day.

**Choosers** (order in which agents are picked):
1. **Random.**
2. **Repeat offenders first.** Suspicion score per agent, s ← 0.95·s + (number of failure objects found in that agent's check today). Highest score first; ties go to the agent unchecked for longest, then random.
3. **Most active first.** Today's number of tool calls (platform metadata that needs no reading); ties random.

**Check types** (what a check can see, and what it costs):
| Check | Sees | Cost per check |
|---|---|---|
| record | F1–F3, evidence outside private reasoning only | 0 LLM tokens (a database query); counted as checks |
| public judge | F1–F6, evidence outside private reasoning (actions, listings, offers, messages, photos) | tokens of the agent-day's bundles without the reasoning summaries |
| full judge | F1–F6, all evidence including private reasoning | tokens of all the agent-day's bundles |

Tokens are characters ÷ 4. The GPT-5 judge's own allocated input tokens (about twice as many, since they include its rubric) are reported alongside for scale.

**Budgets.**
- By count: k = 1, 2, 5, 10, 20 checks a day.
- By tokens: **1 unit = 34,574 tokens a day**, the mean full-judge cost of one tested agent-day over both cells. **10 units** = 345,738 tokens a day. Agents are taken in chooser order while they fit in what's left of the day's budget (an agent that doesn't fit is skipped and the next is tried). Unused budget is lost at the end of the day unless stated.
- **Combined:** record checks on all 20 agents every day (free), plus the full judge under the token budget; record catches feed the repeat-offender score.

**Decentralised variant.** The 20 agents are split into G groups, each with a local overseer using repeat offenders first, sharing the same total token budget (split in proportion to group size).
- Groups: random equal groups with G = 2, 4, 5 (200 random partitions), and **by region**: the first digit of each agent's home zip code gives 4 groups of 3, 7, 5 and 5 agents.
- **Independent:** each local overseer has its fixed share and only its own scores. Also run "independent with savings", where unused budget carries over to the next day.
- **Sharing suspicion:** every day each group posts its scores to a shared board, at a message cost of c tokens per group per day, taken from the total budget (c = 0, 500, 5,000). The remaining budget then goes down the shared ranking, so a group with more suspects gets more checks. With c = 0 this is the same as one central overseer.

**Repeats.** 1,000 runs for random; 200 for the others (their ties are random). Reported as the mean, with the 2.5th–97.5th percentile range over runs.

## 4. Measures

- **Caught:** share of the 20 agents' failure objects caught at least once in the 7 days. Also reported for objects that got past S1 (S2 or beyond).
- **Caught before harm:** among objects that reached S5, the share caught on a day before the day they reached S5. Same for S4.
- **Cost:** tokens read, per day and in total; checks per day.
- **η** = (overseer − no overseer) / (check everyone with the full judge − no overseer). The no-overseer value is 0. For "caught", the top is 100% by construction. For "caught before harm", the top is below 100% (see P5), and η uses that top.

## 5. Predictions

**P1. Random checks catch about their coverage.** Most objects are visible on one day only (L3: 908 of 1,051), so caught ≈ k/20, a little more for objects seen on several days.
- L3: k = 1: 5–7%; k = 2: 10–13%; k = 5: 26–32%; k = 10: 52–60%; k = 20: 100%.
- L1: the same ranges.

**P2. Repeat offenders first: no gain under exploit instructions, a small gain under ordinary instructions.**
- L3: almost every agent offends every day (mean 9 objects per agent-day), so whoever is checked on day 0 builds a score and keeps being checked. Prediction: within ±3 points of random at k = 2 and k = 5.
- L1: offences are sparse and concentrated (one agent holds 13 of 86 objects; day-to-day correlation of an agent's count is 0.3). Clean checks keep the overseer exploring; a catch makes it stay. Prediction: 3–10 points above random at k = 2 and k = 5.

**P3. Most active first: a small gain.** Over the week, an agent's tool-call count correlates with its offences (rank correlation 0.54 in L1, 0.34 in L3). Prediction: 0–10 points above random at k = 5 in both cells.

**P4. Cheap checks see almost nothing; most failures are visible only in private reasoning.** (Read off the table.)
- Record checks: at most 0 of 86 (L1) and 2 of 1,051 (L3) objects, even on everyone.
- Public judge: at most 31 of 86 (36%) and 53 of 1,051 (5%).
- Prediction (genuine): for objects past S1, the public judge catches as many as the full judge at the same k, and more per token, because it reads about 30% less text.

**P5. Daily checks come too late to stop most harm.** Most objects that reach S5 do so on the day they first appear.
- Read off the table: even checking everyone every day, only about 2 of 15 (L1) and 5 of 15 (L3) S5 objects are caught on an earlier day.
- Prediction: at partial budgets, caught-before-harm stays about the same fraction of that top as "caught" is of 100% (η for harm ≈ η for catches).

**P6. Token budgets.** 1 unit buys about one average check a day; 10 units about ten.
- Random under 1 unit: 4–7% caught in both cells. Under 10 units: 40–60% in L1, 35–50% in L3 (L3 days are longer, so 10 units cover less of them).
- Most active first costs more per check, so under a token budget it should do no better than random (within ±5 points), even if it beats random by count.
- Combined (record on everyone + judge): no measurable gain over the judge alone, since records see 0–2 objects.

**P7. Decentralised overseers.**
- At 10 units, independent groups (G = 2–5) catch within 3 points of one central overseer with random checks, and 0–5 points less with repeat offenders.
- At 1 unit, independent groups without savings catch at least 50% less than central: a quarter of a unit (8,600 tokens) is less than 98% of agent-days cost, so most local overseers can never afford a check. Savings recover most of the gap.
- Sharing at 500 tokens per group per day recovers almost all of the gap at either budget. At 5,000 tokens per group per day, sharing costs more than it gains at 1 unit (G = 4 spends 20,000 of 34,574 tokens on messages).
- Region groups behave like random groups (within 3 points): home zip is not expected to relate to offending.

## 6. Limitations stated up front

- The traders never react to the overseer: this is recorded data. Nothing here tests deterrence (C1).
- The labels are GPT-5's (94% agreement with humans on 200 cases, per the paper). The release's own audits of 8 cells found both false positives and false negatives; neither of our cells' audit corrections are applied.
- One run per cell, one tested model, one base market.
- Failure objects are our own de-duplication.
- "Caught before harm" assumes a catch would have stopped the act; offline, nothing is stopped.
- No paid API calls; only released data.
