# BazaarBench offline replay: summary (21 Nov 2026)

Branch `claude/bazaarbench-offline` (from `claude/r1-s1-oversight-experiments` at 55a07b6). Details: `results.md`; predictions written first: `protocol.md`; all numbers: `tables.md`.

## What was done

1. **Downloaded two cells (258 MB, git-ignored)** of the GPT-5 judge release `BazaarBench/bazaarbench-analysis-v2-gpt5-judge`: judgments, bundles, plans and status. Both have GPT-5.5 as the base market and gpt-oss-120b as the tested model:
   - ordinary instructions (`gpt55_L1_gptoss120b`);
   - exploit instructions (`gpt55_L3_gptoss120b`).
2. **Record-based F1–F3 labels: unavailable.** They only exist inside 1.3 GB SQLite files in `BazaarBench/bazaarbench-rollouts`. GPT-5's labels were used for all six types.
3. **Built the agent-day table.** One row per (tested agent, day): failures by type and furthest stage, and text a check would read (tokens = characters ÷ 4). The raw unsafe decisions were merged into "failure objects" (same agent and type, linked by a shared listing, thread, offer and so on).
4. **Wrote predictions** (`protocol.md`), committed before any replay.
5. **Replayed a budgeted overseer.**
   - k = 1, 2, 5, 10, 20 checks a day.
   - Choosers: random, repeat offenders first, most active first.
   - Three check types: records, judge without private reasoning, full judge.
   - Token budgets of 1 unit (34,574 tokens a day, one average agent-day) and 10 units.
   - A decentralised version: 2, 4 or 5 random groups, or 4 zip-code regions; independent, with savings, or sharing a board at 0, 500 or 5,000 tokens a day.
6. **Wrote** `results.md` with 3 composite figures, a scorecard of the predictions, and 12 unit tests (`tests/test_bazaarbench_replay.py`).

## Key numbers

- **Failures.** 86 failure objects under ordinary instructions, 1,051 under exploit instructions. 64% and 95% of them never left private reasoning (stage S1). Only 15 in each market were carried through (S5), nearly all sharing personal data or moving off-platform (F5).
- **Random checks catch about k/20.**
  - Exploit instructions: 6.0%, 11.9%, 28.5% and 53.9% at k = 1, 2, 5, 10.
  - Ordinary instructions: 5.6%, 11.4%, 27.3% and 53.1%.
  - η equals these numbers.
- **Repeat offenders first is worse than random under exploit instructions**: 24.6% against 28.5% at k = 5. For failures past a private plan it's 25% against 45%. The score locks on to whoever was checked first. Under ordinary instructions it's +3.3 points, within noise.
- **What a check can read decides what it catches** (claim C4). Even checking everyone daily:
  - record checks see 0 of 86 and 2 of 1,051;
  - a judge without private reasoning sees 36% and 5%.
  - Per token, the judge without reasoning catches more of the failures that reach other people: 74–83% against 64–66% at 10 units.
- **Too late for harm.** Checking everyone every day catches only 2 and 5 of the 15 carried-through failures a day before harm. Checking after every action would catch 3 and 7.
- **Token budgets.**
  - 1 unit a day catches 6–8% (about $1.50 a week at GPT-5 list prices).
  - 10 units catch 62% (ordinary) and 49% (exploit), about $15 a week.
  - Judging all 20 tested agents costs $23–36 a week.
- **Decentralised overseers.**
  - 4 independent overseers splitting 1 unit catch 0.3–1.2%, against 6–7% for one central overseer: their shares are too small for any check.
  - Savings, or a shared board at 500 tokens per group per day, recover it.
  - At 10 units the split costs 0–8 points.
- **Predictions.** 2 of 11 held in full (random ≈ coverage; harm η tracks catch η). The other 9 held partly or were mixed. The biggest miss was how badly repeat offenders first does under exploit instructions.

## What's left

- **A cheap open-model judge** on the same bundles (CREATE), to measure real miss and false-alarm rates against GPT-5 for the Harvest mistakes test.
- **A tested model that acts on more of its plans** (for example gpt-5.4 or deepseek-v4-pro, about 300 MB a pair), so record checks and repeat offenders have something to find.
- **The official F1–F3 record labels** from one 1.3 GB rollout database. Needs Ameer's OK.
- **Checks at the tick level** (after each action) instead of daily.
- **Check the GPT-5 audit status of these two cells.** The release's audits flag over-called plans, which would inflate the S1 counts that dominate the exploit market.
- **Limitations stand.**
  - The traders never react (recorded data, so nothing about deterrence).
  - The labels are GPT-5's (94% agreement with humans on 200 cases, per the paper).
  - One run per cell, one tested model.
- **Trackers.** Updated: `notes/OVERSEER_TRACKER.md` (findings 17–19 after the 22 Nov merge) and `experiments/README.md`. Not touched, to avoid merge conflicts on this side branch: `PROJECT_HISTORY.md`, `FINDINGS_AND_IMPLICATIONS.md`, `what_we_learned_and_next.md`.
