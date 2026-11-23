# LLM inspector ladder: summary (22 Nov 2026)

How well do LLMs of different sizes work as the overseer's *inspector*, the part that reads a record and says whether the rule was broken? Same 600 items for every model, under a protocol written before any run (`protocol.md`, with Amendments 1–2). Full numbers: `results/tables.md` and `results/summary.json`.

## What ran

- **Items.** 360 Harvest items (windows w = 1, 10, 50; each shown as a text summary and as raw maps) and 240 BazaarBench items (full bundle, and the public part only).
- **Models.** Local Qwen2.5-Instruct 0.5B, 1.5B, 3B, 7B (direct answers); cloud gpt-oss 20B and 120B (`think: low`); Qwen 3B and 7B with brief reasoning on the 120 w = 1 Harvest items.
- **Cloud spend.** 1,107,628 tokens over 1,250 calls, under the 1.5M cap.
- **Run problems.** The first 7B run swapped and was resumed with `--num-ctx 4096`; one 3B reasoning item always returns an Ollama error, so that cell has 119 items.

α is the miss rate and β_eat the false-alarm rate on honest harvests. "Class" is what claim C1 predicts for the learning agents at an average penalty of 1 / 2 apples.

## Findings

**1. At w = 1 (the step the RL overseer checks), inspectors are either almost perfect or useless.**

| Inspector | Summary α / β_eat | Class | Raw maps α / β_eat | Class |
|---|---|---|---|---|
| Qwen 0.5B, 1.5B, 3B | same answer every time | fails / fails | same answer every time | fails / fails |
| Qwen 3B + reasoning | 0.00 / 0.23 | protects / protects | 0.90 / 0.24 | fails / fails |
| Qwen 7B | 0.00 / 0.00 | protects / protects | 1.00 / 0.04 | fails / fails |
| Qwen 7B + reasoning | 0.00 / 0.00 | protects / protects | 0.47 / 0.04 | fails / protects |
| gpt-oss 20B | 0.00 / 0.00 | protects / protects | 0.10 / 0.08 | protects / protects |
| gpt-oss 120B | 0.00 / 0.00 | protects / protects | 0.03 / 0.00 | protects / protects |

- **A step, not a slope.** Capability jumps at 7B; 3B is no better than 1.5B.
- **What the inspector is shown matters as much as its size.** On raw maps, where it has to count apples itself, plain 7B fails; only gpt-oss protects.
- **Brief reasoning helps small models, at a price.** It rescues 3B on summaries (about 35× the output tokens). 7B + reasoning on raw maps is the one inspector where a bigger penalty is predicted to turn failure into protection.
- gpt-oss-120B scores 0.00 on summaries, so the prompt is not the problem.

**2. Cost.** Input grows about 10 tokens per step of history for summaries and 16 for raw maps (235 → 749 and 459 → 1,261 from w = 1 to 50). Qwen answers in 2 tokens; gpt-oss thinks for 40–60 tokens on summaries and 170–440 on raw maps, so a check costs about 330 and 900 tokens.

**3. Longer records hurt.** Small models' one-word bias flips with record length (0.5B: NO to every w = 1 summary, YES to every w = 50 one). Even strong models degrade: Qwen 7B summaries α + β_eat 0 → 0.31 at w = 50; gpt-oss-120B raw maps 0.03 → 0.27, mostly false alarms.

**4. BazaarBench: every model agrees poorly with GPT-5's labels.** α + β is 0.70–0.83 on full bundles (gpt-oss-120B 0.70). Most GPT-5 "unsafe" labels are plans written only in private reasoning, which our models mostly don't count as violations, so this is a disagreement over the rubric as much as capability. With reasoning removed, α on reasoning-only failures rises to 0.71–1.00 (claim C4 in miniature). Size doesn't buy agreement here.

**5. Predictions.** Right: P4 at c = 1, P5 mostly, P7 for 7B. Partly right: P1, P2, P6. Wrong: P3 (window effects have a different shape), P4's claim that a bigger penalty buys back a weak inspector in direct mode, P6's 120B range, P7 for 3B on raw maps.

## Not done

- ~~RL runs with LLM inspectors~~ **done 23 Nov:** `rl_results.md`. C1's classes held in 15 of 16 cells. The miss was gpt-oss-20B at k = 1, c = 1: misses and false alarms don't trade one for one.
- Reasoning mode on longer windows, BazaarBench and the other models.
- The vLLM-on-CREATE script and the live overseer are untested or design-only.
