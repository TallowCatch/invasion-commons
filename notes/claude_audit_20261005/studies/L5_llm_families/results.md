# Results L5: more model families (screen only)

**Status: closed after the screen.** Written 27 October 2026 from the stored games. It was not written when L5 stopped on 12 October, which was a gap in the record.

- **Protocol:** `protocol.md` in this folder, with four amendments, all dated 12 October:
  1. closed families on hold;
  2. GLM-5.3 dropped for cost and latency;
  3. no second open family;
  4. DeepSeek's threshold test not run, for budget.
- **Data:** `claude_l5_v1/` on the `l2-results` branch. Prediction: `prediction.json` here, frozen 2026-10-12 15:38 UTC.

## Question

Do other open model families over-take under the explicit rules with no fine (the L2/L4 E0 cell)? If one does, does it stop at the fine its own one-round gain predicts, as gpt-oss and Nemotron did?

## What was done

- **Stage A (screen):** 10 Fishery games (E0: audits at 1/6 with a 0 t fine; 4 LLM fishers and 2 programmed; 20 rounds).
- **Gate:** the model must answer validly on the first try and pass the comprehension check.
- **Stage B** (E30 and E36, 20 games) was planned only for models that qualified.

## What came out [verified]

| Model | Screen | Outcome |
|---|---|---|
| **DeepSeek V4.1 Flash** | 10 games | Gate passed: 100% valid on the first try, comprehension 3 of 3. Over-took in **79.5% of agent-rounds** [76.4, 82.2]. One-round excess **ĝ₁ = 5.41 t** (311 over-take rounds), so the predicted stopping fine is **F\* = 36 t** (grid step below: 30 t). **The stock collapsed in all 10 games**; final stock 0; honest fishers took 1.4 t each on average. |
| **GLM-5.3** | 0 games finished | Stopped during its first game: about 12,000 reasoning tokens and about 2 minutes per decision (158 calls, about $8 of credit). A 10-game screen would have cost about $90. No outcome exists. |
| MiniMax M3 | not screened | Amendment 3 (cost). |
| Closed families (Claude Sonnet 5.5, GPT-6.1 Sol, Gemini 3.8 Flash) | not run | Amendment 1 (no paid API access). The code is built and tested. |

**Stage B for DeepSeek was not run** (Amendment 4; it would have cost about $11). So **its 36 t prediction is untested.**

## What it means

- A fifth family, DeepSeek, over-takes as readily as Nemotron (80% against 64% at E0) and destroys the resource without a fine. This widens the paper's evidence that open models over-take when told it is possible.
- It says nothing yet about whether DeepSeek stops at its predicted fine.

## What it does not show, and caveats

- One screened family, 10 games.
- The cost of heavy-reasoning models (GLM, DeepSeek) limited L5 more than anything else. That limit is a finding for planning, not a result about the models.
- The GLM spend (about $8) produced no outcome.

## Next step

Run DeepSeek's stage B (about $11) only if credit allows (see `FINDINGS_AND_IMPLICATIONS.md` and the compute memory). Otherwise, report DeepSeek as screened, with the prediction stated and untested.
