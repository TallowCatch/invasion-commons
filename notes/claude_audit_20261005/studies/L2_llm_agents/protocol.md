# Protocol L2: do language-model agents follow the same audit logic?

**Frozen:** 2026-10-07T11:50Z, before any L2 code was written or any L2 model call was made. A draft of this file existed earlier; this version replaces it.

This is a local protocol, not a public preregistration. It is not edited after
results exist; changes go under "Amendments". L2 is the last experiment for
the paper (claim 6) and the main study proposed by the L1 pilot. It adds
nothing outside claim 6.

## How many model families

The closest LLM-agent studies vary widely:

| Study | Models | Families |
| --- | ---: | ---: |
| Bhandary et al. 2026 (lying in a sustainability game) | 1 | 1, named as a limitation |
| Makins et al. 2026 (multi-agent AI control) | 2 sizes | 1 |
| Guzman Piedrahita et al. 2025 (sanctioning institutions) | 7 | 3 (OpenAI, DeepSeek, Meta) |
| Rehm 2026 (self-governance in commons) | 8 | 4, two sizes each |
| PIMMUR audit (Zhou et al. 2026) | 6 | 6 |
| GovSim (Piatti et al. 2024) | 15 | 5 |

Single-family studies treat that as a limitation. Recent studies that make
cross-model claims use **3–4 families**.

**L2 uses 3 families**, the three that Ollama Cloud's free plan serves on this
account. Tested on 2026-10-07:
- Mistral, DeepSeek, GLM, Kimi and MiniMax require paid credits;
- Qwen and Llama are not offered on Ollama Cloud.

| Family | Model (Ollama Cloud) |
| --- | --- |
| OpenAI | `gpt-oss:120b-cloud`, the L1 model |
| Google | `gemma4:31b-cloud` |
| NVIDIA | `nemotron-3-super:cloud` |

A fourth family (Mistral) can be added later under the same protocol, if
credits are bought. It would be reported as an addition.

## Question

LLM fishers in Fishery face random audits with a fine, or audits with memory.
1. Do they stop over-taking once the expected fine exceeds their gain from one
   step of over-taking (P1)?
2. Under memory without fines, do they keep over-taking (P4)?
3. Do all three families behave alike?

## Design

**Setting:**
- Fishery, MSY target, 20 steps.
- 4 of the 6 fishers are LLM agents; the other 2 follow fixed rules (L1's
  population, contexts 0–9).
- L1's v3 interface: everything in tonnes, a catch between 0 and 6 t, the
  same request and catch prompts, history of the last 3 rounds.
- Audit rate q = 1/6, independent per agent and step.

**Over-taking** means catching more than 0.06 t above the allowance, as in L1.

**Cells:**

| Cell | Rules wording | Consequence of being caught over-taking | Expected fine e = F/6 (t) |
| --- | --- | --- | --- |
| E0, E1, E2, E4, E8, E36 | explicit ("you can choose to catch more than your allowance") | fine F = 0, 1, 2, 4, 8, 36 t | 0, 0.17, 0.33, 0.67, 1.33, 6 |
| EM | explicit | no fine; the reviewer remembers by how much you went over and lowers your later allowances (S4's targeted-allowance rule) | 0 |
| S0, S36 | silent (L1's wording) | fine 0 or 36 t | 0 and 6 |
| P0, P36 | explicit, paraphrased rules (same content, different wording) | fine 0 or 36 t | 0 and 6 |

- **Contexts:** 10 per cell, the same 10 in every cell and for every model,
  so cells are paired by context.
- **Size:** 11 cells × 10 = 110 episodes per model, 330 in total.

**Comprehension check.** Once per LLM agent per episode, as in L1. The order
of the answer options is shuffled per agent with a fixed seed, which fixes
L1's always-in-the-same-place answers.

**Cost.** About 113,000 tokens per episode, so about 12.4 M tokens per model.

## Outcomes, per model and cell

- **Over-take rate:** the share of LLM agent-steps with over-taking.
- Mean tonnes above the allowance.
- Harvest per rule-following agent.
- Share of steps breaking the MSY limit.
- Final stock.
- Valid-answer rate, fallbacks and comprehension score.

**The break-even g for LLM agents.** For each model, from E0: the mean extra
catch per over-take step, in tonnes. It is computed from E0 alone, before
looking at the other cells.

## Hypotheses (each tested per model)

**Claim 6 holds if H1 holds in at least 2 of the 3 families.**

- **L2-H1 (P1).** The over-take rate is lower in cells with e ≥ g than in
  cells with e < g (the E-cells only). The test is a paired context-bootstrap
  difference of the pooled rates.
  - Falsifier: the interval includes 0 or lies above it.
- **L2-H2 (more fine, less over-taking).** The over-take rate in E36 is below
  E0.
  - Falsifier: the interval includes 0.
- **L2-H3 (P4: memory reduces harm but does not deter).** Under EM:
  - the over-take rate stays above 5%;
  - the share of steps breaking the MSY limit is lower than in E0.
  - Falsifier: either fails.
- **L2-H4 (L1 replication).** S0 has an over-take rate below 5%.
  - Falsifier: 5% or more.
- **L2-H5 (wording robustness).** The difference E0 − E36 has the same sign
  and an overlapping interval under the paraphrase (P0 − P36).
  - Falsifier: opposite sign, or intervals that do not overlap.

## Gates and stop rules

1. **Offline gate.** The runner passes a smoke test with L1's fake model,
   including stopping and resuming after a simulated usage-limit error.
   `pytest -q tests` passes.
2. **Pilot gate per model (real calls).** One context in E0 and E36.
   - Pass: at least 95% valid answers on the first try, and a mean
     comprehension score of at least 2 of 3.
   - A model that fails is excluded and reported, not replaced. Pilot
     episodes are not part of the results.
   - The prompts are frozen after the first pilot and not changed between
     models.
3. **Full run.**
   - One full run per model, in chunks. The runner stops cleanly when the
     free usage limit is reached and resumes at the next unfinished episode.
   - An episode cut off midway is rerun from its start. The calls of the
     abandoned attempt stay in the log and count towards the token total.
4. **Token cap:** 15 M per model.
5. **No changes after any full-run outcome is seen.** Any interface fault is
   handled as in L1: an amendment, a new run directory, and the failed run
   kept.

## Analysis

- Paired context-bootstrap intervals: 4,000 resamples, seed 20261019.
- Rates pooled over agent-steps within a cell.
- No pooling across models.
- All results are reported, including failed gates.

## Amendments

### Amendment 1 (2026-10-07, after the pilot gate began; before any full run)

**What was seen before this change:**
- **gpt-oss:120b** passed its pilot gate: 100% valid answers on the first try (320 decisions), and a mean comprehension score of 3 of 3.
- **gemma4:31b** failed every decision. It wraps its JSON in a markdown code fence (```` ```json … ``` ````), and the parser rejects that, so every decision fell back to a default. The pilot was stopped before Nemotron ran.
- Re-parsing Gemma's saved answers with the change below made 314 of 314 first-try decisions valid. No behaviour (over-taking) was looked at.

**Changes:**
1. **Parsing.** A reply that is exactly one JSON object inside a markdown code fence is unwrapped before parsing; everything else is parsed as before. This is an interface fix of the kind L1's amendments made. It applies to every model, and gives identical results for gpt-oss, which never used fences. Gemma's and Nemotron's pilot gates are rerun from scratch with it, in a new pilot directory; the failed Gemma pilot is kept.
2. **Where it runs.** The full run happens on GitHub Actions, not on Ameer's laptop:
   - it calls Ollama Cloud's API directly with an API key, using the API's model names for the same models (`gpt-oss:120b`, `gemma4:31b`, `nemotron-3-super`);
   - each scheduled job resumes at the next unfinished episode and stops cleanly at the usage limit, after a time limit (between episodes), or after a network failure;
   - finished episodes are committed to the `l2-results` branch after every job.
3. **Token cap.** Raised from 15 M to 20 M per model. The pilot measured about 800 tokens per call, so about 14.4 M per model is expected, and restarts add more.
4. **Which models run.** A model enters the full run only after its pilot gate has passed, as recorded in `gates.json` on the `l2-results` branch.

Nothing else changes: the cells, contexts, prompts, seeds, outcomes and hypotheses are unchanged.

### Amendment 2 (2026-10-07 ~16:15Z, during the gpt-oss full run; before any full-run outcome was looked at)

**What was seen before this change:**
- Ollama's free plan turned out to be a monthly allowance, not a 5-hour one. At 75.5% used, it could not finish L2 (about 4 months of free usage were needed). Ameer bought Ollama Pro, which also serves the paid model families and allows 3 cloud models at a time.
- gpt-oss had finished 22 of its 110 full-run games. Only the count of finished games was looked at; no outcome (over-taking, stock, harvest) was examined.
- Measured speeds from the pilots: gemma4 about 0.6 s per call, gpt-oss about 2.2 s, nemotron about 8.3 s. At that speed nemotron alone needs about 40 hours on one job.

**Changes:**
1. **A fourth family, Mistral, as the addition this protocol already allowed** ("A fourth family (Mistral) can be added later … reported as an addition"). The model is `mistral-large-4` (Ollama Cloud API name; `mistral-large-4:cloud` in the runner), Mistral's current flagship on Ollama Cloud, tested to answer on this account on 2026-10-07.
   - It passes the same pilot gate first (one context in E0 and E36; ≥ 95% valid first-try answers; mean comprehension ≥ 2 of 3) and uses the same frozen prompts, cells, contexts and seeds.
   - **Claim 6 is still judged on the three frozen families only** (H1 in at least 2 of gpt-oss, gemma4, nemotron). Mistral's H1–H5 are reported beside them, as an addition.
   - DeepSeek, GLM, Kimi and MiniMax are now also available. They are not added.
2. **Parallel jobs.** Three GitHub Actions lanes run at once, within Pro's limit of 3 models at a time. Each lane works through a fixed list of units, and no unit is in two lanes:
   - Lane A: gpt-oss (all contexts), then nemotron contexts 5–7.
   - Lane B: gemma4, then the Mistral pilot and full run, then nemotron contexts 8–9.
   - Lane C: nemotron contexts 0–4.
   The lists may be rebalanced between jobs to even out finishing times; that changes only which job runs a game, never what a game is.
3. **Splitting a model by context.** A unit that covers only some contexts writes its own log, manifest, STATUS and DONE files (suffix `_ctxLO-HI`), so parallel jobs never write the same file. The token cap (20 M per model) counts all of a model's logs. A model is complete when all 110 of its games exist. Within a unit the order is unchanged (contexts outer, cells inner), and every game is identical to what a single job would run, because seeds depend only on the context and cell.

Nothing else changes: the cells, contexts, prompts, seeds, outcomes, hypotheses and the claim-6 rule for the three frozen families are unchanged.

### Amendment 3 (2026-10-07 ~16:25Z, after an independent code review; before any outcome comparison)

**What was seen before this change:**
- An independent review of the code against this protocol was done blind: it computed no outcomes.
- A mechanical check of the 24 finished gpt-oss games looked only at the mechanics, not at over-taking by cell:
  - fines match audits and catches exactly, and EM has no fines;
  - rule-followers never over-take, and catches stay in range;
  - the audit rate is 0.18 against 1/6, and the audit draws are identical across cells within a context;
  - there are 0 invalid first-try answers, 0 fallbacks, and a mean comprehension score of 3 of 3.
- While reading the file format, the summary fields of one game were seen: gpt-oss, E0, context 0, where the lake collapsed. That game's setting matches the pilot game already reported above.
- No cell was compared with another.

**Fault found (an interface fault, handled as rule 5 says):**
- The EM cell tells the agent "the reviewer records by how much you went over and lowers your later allowances". The code records a caught over-take only as S4's share-of-the-cut, d = (catch − allowance)/(request − allowance).
- That share is undefined when the reviewer made no cut (allowance = request), so the caught over-take is silently dropped.
- An LLM can also catch more than it requested, giving d > 1. After clipping, d = 1 sets that agent's allowance to 0 for the rest of the game.
- So, in some states, EM does not do what its prompt says.
- S4's simulated cheaters always caught between the allowance and the request, so the rule never met these cases there.

**Changes:**
1. **EM is held back from the main run** (`--skip-cells EM`). The other 10 cells are unaffected and continue in `claude_l2_v1`.
   - The two gpt-oss EM games already saved stay in `claude_l2_v1`, as a kept failed run. They are not analysed, and their outcomes have not been looked at.
   - A corrected memory rule will be fixed in a later amendment, before any EM game is run again. EM then runs for every model in a new directory, `claude_l2_em_v2`.
2. **Job safety (no effect on any game):**
   - No new game starts in the last 35 minutes of a job, so a slow game is never killed by the 350-minute job timeout.
   - Game files are written atomically.
   - A log line cut off by a killed job is skipped, so it cannot crash later jobs.
3. **Blinding:** the public job log no longer prints each game's over-take rate or final stock.
4. **Analysis fixes (decided before any outcome comparison):**
   - The H3 bootstrap now counts each resampled context as often as it is drawn, as the other tests already did. The share of steps breaking the MSY limit is pooled over steps, as this protocol's pooling rule says.
   - Each model gets its own generator, seeded 20261019, so a model's intervals do not depend on which other models exist.

**Known and accepted:**
- Over-taking is defined as more than 0.06 t above the allowance, as in L1. So a catch up to 0.06 t over is "checked, no fine", although the rules text says any excess is fined. This is rare; it is reported, not changed.
- The E1 prompt reads "1 tonnes". The prompts are frozen, so this is not changed.

### Amendment 4 (2026-10-07 ~20:15Z, during the Mistral pilot; no Mistral full-run game had been played)

**What was seen before this change:**
- Gemma finished its 100 games, and its lane started the `mistral-large-4` pilot.
- In the first 6 pilot calls, Mistral Large 4 used about 400–750 tokens per comprehension answer and **5,600–6,400 tokens per decision**, nearly all hidden reasoning. Each decision took 70–90 s. The other models use about 800 tokens per call.
- At that rate one game takes about 3.5 hours and about 1 M tokens. A full run would need about 100 M tokens, five times the 20 M cap, and about two weeks of one lane.
- To get these numbers, the timing and token counts of those 6 pilot calls were read. Their answers were visible in the same rows. Pilot games are not part of the results.

**Changes:**
1. **The fourth family's model is `mistral-large-3:675b`** (`mistral-large-3:675b-cloud` in the runner), Mistral's previous flagship on Ollama Cloud. It has no hidden reasoning and answered a test call in under 1 s on 2026-10-07.
   - The choice is based on cost and speed only, measured before any Mistral full-run game.
   - It passes the same pilot gate first, with the same frozen prompts, cells, contexts and seeds.
   - Its replies come wrapped in code fences, which Amendment 1's parsing already handles.
2. **Mistral Large 4** is stopped at its pilot, and its pilot calls are kept in `claude_l2_pilot_v2/mistral-large-4_cloud`. It is reported as not run, because it would exceed the token cap.
3. Mistral remains an addition. Claim 6 is still judged on gpt-oss, gemma4 and nemotron.
4. To stop the expensive pilot at once, the running jobs were cancelled. The lanes restarted from their next unfinished game.

### Amendment 5 (2026-10-08 ~09:15Z, after the 10 other cells finished; before any analysis of them)

**What was seen before this change:**
- All 400 games of the 10 non-EM cells had finished: 100 per model, for gpt-oss, gemma4, nemotron and mistral-large-3.
- Only the counts of finished games and their token totals were looked at. No analysis had been run, and no outcome had been looked at.

**The corrected memory rule (EM).**
- When an agent is checked and has caught more than 0.06 t above its allowance, the reviewer records the excess in tonnes: catch minus allowance.
- From then on, that agent's allowance is its usual allowance minus the mean of its recorded excesses, and never below 0:

  a′ = max(0, scale · request − mean recorded excess).

- This is what the EM prompt already says: "the reviewer records by how much you went over and lowers your later allowances to make up for it". It keeps S4's idea that the reviewer cuts a known over-taker's allowance by the amount it is expected to take back. It now works when nothing was cut, and when a catch exceeds the request.
- There are still no fines in EM. The prompts are unchanged.
- The rule is set in the runner (`MEMORY_RULE = "tonnes"`) and saved in every EM game and manifest. A test replays it from the saved steps.

**How EM runs:**
- 10 contexts × 4 models = 40 games, in the new directory `claude_l2_em_v2`, with the same seeds and contexts as before. EM games are therefore still paired by context with the other cells.
- The jobs run as the "em" stage of the same workflow. Nemotron is split over two lanes; gpt-oss, gemma4 and mistral share the third.
- The EM games run after the other cells, so they were played about one day later on the same Ollama Cloud models. This is noted when the results are reported.

**Analysis:**
- H3 uses EM from `claude_l2_em_v2` and E0 from `claude_l2_v1`.
- The EM games in `claude_l2_v1`, made under S4's rule, are not used.
- Everything else in the analysis is unchanged.

### Amendment 6 (2026-10-08 ~09:30Z, found while running the analysis)

**What happened:** the call logs of the main run (`claude_l2_v1`) are incomplete.

| Model | Calls logged | Calls expected (about) |
| --- | ---: | ---: |
| gemma4 | 1,806 | 16,400 |
| gpt-oss | 5,770 | 16,400 |
| mistral-large-3 | 1,481 | 16,400 |
| nemotron | 1,336 | 16,400 |

**Cause:**
- When two lanes saved at once, `l2_save.sh` rebased the store in place. That rewrote the log files on disk.
- The runner keeps each log open, so its later lines went to a file that no longer existed. After a job's first rebase, its log lines were lost.
- Game files were not affected. Each game is written once, as a new file, when it ends.

**What is affected:**
- **Not affected:** all outcomes and hypotheses, which come from the game files, and the per-game counts of fallbacks, re-prompts and comprehension, which are stored in each game.
- **Affected:**
  - Most saved answer texts are lost, including the models' stated reasons.
  - The per-call token totals are incomplete. The STATUS token counts are kept in memory by each job, so they are close to the true values but undercount somewhat.
- **Reporting rule:** valid-answer rates are reported from the game files, where a re-prompt means an invalid first answer. Token use is reported as approximate.

**Fix:**
- The save step never touches the files being written. It copies the files a job changed (found with `git status`, which only reads) into a separate clone, and commits and pushes from there.
- A test with two live writers and simultaneous pushes kept 400 of 400 lines in both logs.
- The EM stage was stopped after 8 minutes, before any in-job save, so its logs are complete. It restarted with the fixed save step.
