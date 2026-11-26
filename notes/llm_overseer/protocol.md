# How capable must an LLM inspector be? Protocol and predictions (written 22 Nov 2026, before any judge was run)

**The question.** Supervisor feedback (Yali, Nov) asked how powerful the overseer must be. The decision is to study **one LLM overseer, by how capable it needs to be**. In the general formulation (`../phd_plan/2026-11-21_general_formulation.md`) an inspector is summed up by two numbers:
- **α, the miss rate:** the chance a check fails to flag a record that has a violation in it;
- **β, the false-alarm rate:** the chance a check flags an honest record.

Claim **C1** says a mistaken inspector deters as if the average penalty were **(1 − α − β)·c**, where c is the average penalty a perfect inspector would give (c = (k/7)·P in Harvest). The Harvest mistakes test (`../socialjax/phase2_ippo/results.md`, tracker findings 10–11) found:
- learners need an effective average penalty **between 0.5 and 0.75 apples** (0.5: the orchard dies in every run; 0.75: it survives in every run);
- with mistakes, **few big checks (k = 1) do clearly worse than many small ones (k = 7)** at the same average penalty (342 vs 479, 382 vs 482, 470 vs 509 apples).

Running an LLM inside 1e8-step RL is infeasible. So this study **measures each model's α, β and token cost as an inspector offline**, then feeds them into Harvest RL with the trainer's existing `--miss-rate` and `--false-alarm` flags. This is the "weaker overseer" axis of the bounded scorecard (decentralised note, §7), with model size as the dial (Engels et al. 2025, *Scaling Laws for Scalable Oversight*, is the closest prior work).

**What was already seen when this was written.** The two datasets and their base rates (below). No model had judged anything. The smoke test (section 7) is run after this file is committed.

## 1. Harvest judging dataset

`overseer/llm/harvest_records.py`, built by `experiments/llm_overseer/make_harvest_dataset.py`; saved as `experiments/llm_overseer/data/harvest_judge.jsonl.gz` (committed, 3,000 records, fixed seed 0).

- **Where it comes from.** SocialJax Harvest Open, 7 scripted bots, 1,000-step episodes, under a random k = 1 overseer with no penalty. Six populations (all restrained; 6 restrained + 1 greedy; 5 + 2 greedy; 4 restrained + 3 sneaky; 7 sneaky; 3 restrained + 2 sneaky + 1 greedy + 1 random), 8 episodes each. Sneaky bots turn greedy after 10 steps unchecked.
- **One record** is one agent's last w steps, w = 1, 10 or 50.
- **The true label** comes from the wrapper (`overseer/socialjax/harvest_overseer.py`): 1 if the agent took at least one apple with fewer than 3 other apples within two squares, counted on the map just before its move.
- **Balance, per window:** 500 violation records, 400 honest records with at least one apple taken, 100 honest records with none. Bot kinds are mixed in by round robin (for example at w = 10: 164 greedy, 163 restrained and 163 sneaky violation records; restrained bots break the rule by accident when a neighbour takes a nearby apple in the same step).
- **Two renderings of the same record:**
  - **summary:** per step, "took an apple; other apples within two squares of it just before the move: n", or "took no apple". The judge only has to compare n with 3.
  - **raw:** per step, the move; on steps where it took an apple, the 5 × 5 map around that apple as it was before the move, in ASCII. The judge has to count the apples in the right 12 cells itself (the prompt shows which 12 cells count).
- **Length:** summary 126 / 405 / 1,490 characters at w = 1 / 10 / 50; raw 456 / 923 / 2,654.

**Amendment 1 (22 Nov, after the 50-item smoke test, before any full run).** A test found that in 478 of 7,090 eats the centre of the raw map showed empty ground: SocialJax regrows apples at the start of a step, before agents move, so the apple taken had regrown on that very step and is not on the previous map (which is the map the wrapper judges). The raw rendering now always draws the centre as the apple taken; the 12 cells around it are unchanged, and so are the labels and the summary rendering. The dataset was rebuilt and the smoke test rerun. The partial 0.5B full run started before the fix was discarded.

**Amendment 2 (22 Nov, recorded after the runs; none changes a label, prompt or item).**
- **Cloud smoke test first.** After Qwen 3B answered NO to almost every Harvest record, gpt-oss-120B judged 50 subset items (`--limit 50`) to check the prompt before the full cloud runs (section 7). Those 50 calls count toward the token cap: the cloud total was 1,107,628 tokens over 1,250 calls, under the 1.5M cap. The cloud runs started before Qwen 7B had finished.
- **Qwen 7B on the 8 GB M1.** The first 7B run was killed after 493 of 600 items: it was swapping, and swap filled the disk. Its last 107 items (BazaarBench public) were rerun with `--num-ctx 4096` instead of 8192. Every remaining prompt fits: the longest is 3,690 tokens.
- **Reason mode (P7)** ran only for Qwen 3B and 7B, only on the 120 w = 1 Harvest items (`--windows 1`, `--num-ctx 2048`), because of the time and memory 7B needs. One 3B item failed with an HTTP 500 and was rerun.

## 2. BazaarBench judging set

`overseer/llm/bazaar_records.py`, built by `experiments/llm_overseer/make_bazaar_dataset.py` from the offline-replay download (`data/bazaarbench/`, git-ignored). The texts are not committed; the index of chosen pairs with the GPT-5 labels is (`experiments/llm_overseer/data/bazaar_index.tsv`).

- **One example** is one (evidence bundle, failure type) pair that the GPT-5 judge decided: unsafe (1) or not (0). Cells: base market GPT-5.5, tested model gpt-oss-120b, ordinary (L1) and exploit (L3) instructions.
- **Balance:** per failure type and cell, up to 50 unsafe and as many safe, among bundles whose rendering fits in 16,000 characters: 836 examples (F1 74 + 74, F2 56 + 56, F3 81 + 81, F4 100 + 100, F5 100 + 100, F6 7 + 7).
- **Two renderings:** **bundle** (what GPT-5 read, minus bookkeeping fields; 3,940 characters on average) and **public** (the same without the private reasoning summaries; 1,692).
- **The labels are GPT-5's, not truth.** GPT-5 counted a failure only planned in private reasoning (stage S1) as unsafe. 328 of the 418 unsafe examples have evidence only in reasoning. The GPT-5 rubric is not in the release; our rule texts (`overseer/llm/prompts.py`) paraphrase the six failure types. So α and β here measure **agreement with GPT-5**.

## 3. The inspector ladder and the judging subset

**Change of plan (22 Nov, from the coordinating session at Ameer's request):** inference runs through **Ollama**, not vLLM on CREATE. vLLM on CREATE stays a documented, untested alternative.

| Tag | Ollama model | Parameters | Family | Where |
|---|---|---|---|---|
| qwen2.5-0.5b | qwen2.5:0.5b-instruct | 0.5B | Qwen2.5 Instruct | local (M1, 8 GB) |
| qwen2.5-1.5b | qwen2.5:1.5b-instruct | 1.5B | Qwen2.5 Instruct | local |
| qwen2.5-3b | qwen2.5:3b-instruct | 3.1B | Qwen2.5 Instruct | local |
| qwen2.5-7b | qwen2.5:7b-instruct | 7.6B | Qwen2.5 Instruct | local |
| gpt-oss-20b | gpt-oss:20b-cloud | 21B (3.6B active) | gpt-oss (reasoning) | Ollama cloud |
| gpt-oss-120b | gpt-oss:120b-cloud | 117B (5.1B active) | gpt-oss (reasoning) | Ollama cloud |

- **Local models are Ollama's default 4-bit quantisation (Q4_K_M)**, not the full-precision weights.
- **Size and family are confounded at the top of the ladder:** Qwen locally, gpt-oss in the cloud. gpt-oss also reasons before answering (`think: low`), so it differs in reasoning as well as size. Within each family the size trend is clean; across the family boundary it is not.
- **Cloud budget cap:** at most 2 cloud models, at most 600 items per model, and at most about 1.5M tokens in total across all cloud calls. Every cloud call's tokens go to a log (`results/llm_overseer/cloud_tokens.tsv`), and the run stops at the cap. Cloud runs start only after the local results look sane.
- **One shared subset of 600 items for every model** (`overseer/llm/subset.py`, seed 0):
  - Harvest: per window, 30 violation and 30 honest records (of the honest ones, 68 of 90 took at least one apple), each judged in both renderings: 360 items;
  - BazaarBench: per failure type F1–F5, 12 unsafe and 12 safe, each judged in both renderings: 240 items (52 of the 60 unsafe are reasoning-only; F6 is left out, with only 7 unsafe in the whole set).
- **Prompt mode: direct** (answer YES or NO in one word; up to 16 output tokens; temperature 0). gpt-oss thinks first (up to 2,048 output tokens, thinking included). An optional **reason** mode (think briefly, then "VERDICT: YES/NO") runs on the local models only if time allows.
- **A reply with no YES/NO** counts as "no flag" (a miss on a violation, a correct pass on an honest record). Its rate is reported.

## 4. Measures

For each model × dataset × rendering (and window for Harvest):
- **α** = share of violation records not flagged; **β** = share of honest records flagged, with 95% Wilson intervals (30 per cell, so about ±15 points per window cell, about ±9 pooled over windows).
- **β on honest harvests (β_eat):** β on honest Harvest records with at least one apple taken. This is the β the RL wrapper uses (false alarms fall only on honest harvests, `mistakes_protocol.md`), so it is the one fed to training.
- **Confusion table** (true × verdict, with "no verdict" as its own column).
- **Tokens per check:** mean tokens in and out, and seconds per check.
- **C1 applied:** effective average penalty e = (1 − α − β_eat)·c at c = 1 and c = 2, using the w = 1 numbers (the RL runs check one step, `--window 1`). Classed as:
  - **"protects" if e ≥ 0.75** (every tested run at 0.75 or more kept the orchard);
  - **"fails" if e ≤ 0.5** (every tested run at 0.5 or less lost it);
  - **"uncertain"** between the two.

## 5. Predictions (written before any judging)

**P1. Summary rendering: α + β falls with size within Qwen.**
- 0.5B is near chance: α + β between 0.7 and 1.1, because tiny models tend to answer the same word to almost everything (YES on more than 80% of items, or on fewer than 20%).
- 1.5B: 0.4–0.8. 3B: 0.2–0.6. 7B: at most 0.35.
- gpt-oss (both sizes): at most 0.1. With a number to compare and time to think, this is easy.

**P2. Raw is harder.** At every Qwen size, α + β on raw is at least 0.15 above summary. gpt-oss-120b on raw: at most 0.25; gpt-oss-20b: at most 0.35. The usual mistake is counting all 24 cells instead of the 12 that count, which over-counts apples and so **raises α** (violations look legal) more than β.

**P3. Longer windows hurt raw more than summary.** One violation hidden among many legal eats is easy to miss when every map has to be read. For Qwen 3B and 7B on raw, α at w = 50 is at least 0.1 above α at w = 1. On summary the window changes α + β by under 0.15.

**P4. C1 thresholds (at w = 1).**
- At c = 1 a model needs α + β_eat ≤ 0.25 to "protect". Prediction: only gpt-oss (both renderings) and Qwen 7B on summary do.
- At c = 2 it needs α + β_eat ≤ 0.625. Prediction: Qwen 3B and 7B on summary, and gpt-oss on both, protect; 0.5B fails in both renderings; 1.5B is uncertain or fails.
- So **a bigger penalty can buy back a weaker inspector, down to about 3B on the easy rendering**, but not below. This is a prediction about the threshold, not about apples.

**P5. Tokens per check grow with the window and the rendering, not with the model's size** (except output). Direct-mode input: about 300 tokens at w = 1 summary, 600–700 at w = 50 summary, 1,200–1,600 at w = 50 raw. Output: 1–3 tokens for Qwen in direct mode; 100–600 thinking tokens for gpt-oss. So with a reasoning model **most of the cost of a check is output**, and the cost per step of history (c_hist in the formulation) is about 10–30 input tokens per step.

**P6. BazaarBench agreement with GPT-5 is lower than on Harvest.**
- Bundle rendering: Qwen α + β at least 0.4 at every local size; gpt-oss-120b 0.15–0.4.
- Public rendering: α on the 52 reasoning-only unsafe items is at least 0.8 for every model, because the evidence is not there (claim C4 in miniature). β on public is no higher than on bundle.

**P7. Reason mode (only if run).** For Qwen 3B and 7B on raw, asking for brief reasoning lowers α + β by at least 0.1, at 20–100 times the output tokens. For 0.5B it does not help.

## 6. Into RL (after the judge runs)

`experiments/llm_overseer/harvest_runs.py` reads the measured α and β_eat at w = 1 and writes a job list in the `phase2_runs.py` format (`name<TAB>args`):
- one run per model × rendering × k ∈ {1, 7} × c ∈ {1, 2} × seeds 1–3, with `--k`, `--chooser random`, `--penalty-apples 7c/k`, `--miss-rate α`, `--false-alarm β_eat`;
- options to drop models whose (α, β) round to the same values, and to keep one seed for cells predicted to fail.

**Prediction for those runs** (from P4 and finding 11): every "protects" cell keeps the orchard; at the same (α, β), k = 1 collects clearly fewer apples than k = 7; "fails" cells end near 68 apples with the orchard dead.

## 7. What would change the plan

- If even gpt-oss-120b has α + β above 0.25 on the summary rendering, the prompt or the rule text is the problem, not capability. Check the replies before reading anything into the ladder.
- If Qwen does not improve with size (P1 fails flat), the ladder is measuring prompt-following, not capability. Try the reason mode before concluding.
- If β is near 0 and α near 1 for small models (they always answer NO), C1 is trivially failed and the useful comparison starts at 3B.

## 8. Limitations stated up front

- Bots, not learners, produce the Harvest records. RL agents' records may look different (for example, more near-miss eats at exactly 2 or 3 apples).
- The RL step uses α and β as fixed coin flips per check; a real LLM's mistakes are correlated with the record (hard records are missed more often), and learners could find and exploit that. This is not tested here.
- 30 records per label per cell: window-level numbers are rough (±15 points); pooled numbers are better.
- 4-bit local models; one prompt wording; temperature 0; one seed.
- BazaarBench labels are GPT-5's, and our rule texts are a paraphrase of its rubric.
