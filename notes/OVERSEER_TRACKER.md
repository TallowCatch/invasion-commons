# Overseer tracker: what we've found, in plain English

One page for the budgeted-overseer work: an overseer that can only check a few agents at a time. Each finding has its numbers, how sure we are, and where the details are. Newest changes are in the log at the bottom.

**How sure we are:**
- **Solid:** clear in every run.
- **Lead:** seen, but in few runs or still running.
- **Prediction:** not tested yet.

**The setting so far.** SocialJax Harvest Open:
- 7 agents collect apples for 1,000 steps;
- apples only regrow near other apples;
- each agent keeps its own apples.

**The rule.** Don't take an apple with fewer than 3 others within two squares.

**The overseer.** Checks k of the 7 agents each step, at random unless stated, and takes a penalty off anyone it catches. The **average penalty** is the chance of being checked (k out of 7) times the penalty, in apples.

## The findings at a glance

| # | Finding | Key numbers | How sure | Details |
|---|---|---|---|---|
| 1 | Harvest is a real social dilemma, and very fragile | all hold back: 71.5 apples each; all greedy: 9.6; one greedy bot kills the orchard in 10 of 10 runs | solid (bots) | `socialjax/week2_bots/results.md` |
| 2 | With no overseer, learning agents always strip the orchard | 68 apples a game, orchard dead in every run, 45% of apples taken break the rule | solid | `socialjax/phase2_ippo/results.md` |
| 3 | The average penalty decides, not the number of checks | 0.5 apples: nothing changes even with all 7 checked; 1 apple: the orchard always survives (441–477 apples); 2 apples: 515–530 | solid (3 runs each) | same |
| 4 | One check per step is almost as good as checking everyone (with a perfect inspector) | 515 apples with 1 check against 530 with 7, at 2 apples average penalty | solid | same |
| 5 | Freezing caught agents doesn't work | 77 apples, orchard dead | solid | same |
| 6 | Checking repeat offenders first is faster, not better, when agents start from scratch | orchard safe by about 13M steps (random: 14–57% of games); both end at 460–470 apples | solid | same |
| 7 | Targeting on a bad signal is worse than random | checking by location catches 5% of over-harvests, against 28% for random | solid | same |
| 8 | **Remove the overseer and over-harvesting comes straight back** | every setting falls back to 68 apples and a dead orchard within about 5–12M steps | lead (9 of 18 runs; the rest rerunning) | below |
| 9 | **Fixing agents that already over-harvest needs more checks, or targeting** | after 50M steps: 1 check 120 apples, 2 checks 172, 4 checks 315, 7 checks 403; repeat offenders first with 2 checks 352 | lead (1 of 3 runs per setting) | below |
| 10 | An overseer that misses half the time should need twice the penalty | prediction: average penalty × (1 − miss rate) must reach the same level | prediction (running) | `socialjax/phase2_ippo/mistakes_protocol.md` |
| 11 | False alarms should cost honest agents the same on average whatever the number of checks | prediction | prediction (running) | same |
| 12 | **BazaarBench (recorded LLM traders): random checks catch about k/20; repeat offenders first is worse than random when nearly everyone plans unsafe acts** | exploit instructions, k = 5 of 20: random 28.5%, repeat offenders 24.6% (past-plan failures: 45% vs 25%) | lead (1 market pair, offline) | `bazaarbench/offline_replay/results.md` |
| 13 | **BazaarBench: most failures exist only in private reasoning, so what a check can read decides what it catches** | exploit: 998 of 1,051 failures never leave reasoning; a check without reasoning sees at most 5%, record checks 0.2% | lead (GPT-5 labels, 1 tested model) | same |
| 14 | **BazaarBench: daily checks are too late for harm; fixed local budgets waste a small budget** | everyone checked daily catches 2 and 5 of 15 carried-through failures first; 4 independent overseers at 1 unit catch 0.3–1.2% vs 6–7% central; a 500-token shared board recovers it | lead | same |

The same patterns had already shown up in the earlier non-spatial games:
- **Fishery with LLM agents:** the expected fine decides (L4).
- **Fishery:** always checking the biggest reporter caught 25% of liars, against 100% for random checks (T1).
- **Willis et al.'s commons:** strategies obey while audited but don't change (R1b, F16).

Details are in `claude_audit_20261005/FINDINGS_AND_IMPLICATIONS.md`.

## Finding 3 in a table

Apples collected per game, average of 3 runs, at the end of training:

| Average penalty | 1 check | 2 checks | 4 checks | 7 checks | Orchard survives? |
|---|---|---|---|---|---|
| none (no overseer) | 68 | 68 | 68 | 68 | never |
| 0.5 apples | 68 | 68 | 68 | 68 | never |
| 1 apple | 441 | 458 | 476 | 477 | always |
| 2 apples | 515 | 527 | 528 | 530 | always |

For comparison, seven well-behaved scripted bots collect 500.

## Findings 8 and 9: switching the overseer off and on (stages 2 and 3, still filling in)

**Switching it off** (agents trained with the overseer, then 50M more steps without it):

| Trained with | Apples before | Apples after | Orchard |
|---|---|---|---|
| random checks, 1, 2, 4 or 7 per step | 408–482 | 68 | dead in every run |
| repeat offenders first, 2 checks | 466 | 68 | dead |
| freeze | 70 | 68 | dead (it never worked) |

**Switching it on** (agents that learned to over-harvest with no overseer, then 50M steps with it, average penalty 1 apple, 1 run each so far):

| Overseer switched on | Apples at the end | Orchard survives |
|---|---|---|
| none | 68 | 0% |
| 1 random check | 120 | 32% |
| 2 random checks | 172 | 92% |
| 4 random checks | 315 | 100% |
| 7 random checks | 403 | 100% |
| 2 checks, repeat offenders first | 352 | 100% |
| 2 checks, 50-step freeze | 76 | 0% |

**What it means.** The average penalty decides whether the overseer can work in the long run (finding 3). But how quickly it fixes agents that already misbehave depends on how often it checks, and on targeting. Frequent small penalties are a clearer signal to learn from than rare big ones. Some runs were still improving at the end, so the gap may narrow with longer training.

## What the overseer costs

So far cost is counted in checks per game: 1 check per step is 1,000 checks a game, and checking everyone is 7,000. With a perfect inspector, 1,000 checks recover about 97% of the gap between no overseer and checking everyone (finding 4). That's measured as η = (515 − 68) / (530 − 68).

## Running now (CREATE)

| What | Runs | Job |
|---|---|---|
| Stage 2 seeds 2–3, stage 3 remainder, 2 stage 1 cells | 25 | 38794336, 38794337, 38794338 |
| Mistakes test (misses and false alarms) | 24 | 38794367 |

Nodes comp035 and comp040 have broken GPUs and are excluded.

## Next

- **BazaarBench offline test:** done 21 Nov (findings 12–14, branch `claude/bazaarbench-offline`). Next there: a cheap open-model judge for real miss and false-alarm rates, and a tested model that acts on more of its plans.
- **The plan for using both testbeds:** `phd_plan/2026-11-21_way_forward.md` (written 21 Nov).

## Log

- **21 Nov (later).** BazaarBench offline replay on two markets (GPT-5.5 base, gpt-oss-120b tested, ordinary vs exploit instructions): findings 12–14. Predictions in `bazaarbench/offline_replay/protocol.md` were written before the replay; 2 of 11 held in full, the other 9 partly or mixed (scorecard in `results.md`).
- **21 Nov.** Stages 2 and 3 partly in (findings 8 and 9). The mistakes test was submitted. False alarms were redefined to fall only on honest harvests, before any run used them. This tracker was started.
- **16 Nov.** Stage 1 results (findings 2–7). Wording corrected: the average penalty is (k/7) × penalty, not k × penalty.
- **15 Nov.** Bots (finding 1) and the break-even point (greed stops paying at about 1 apple of average penalty).
