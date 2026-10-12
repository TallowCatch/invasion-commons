# Protocol L5: do more model families, open and closed, over-take, and does the predicted fine stop the ones that do?

**Frozen:** 2026-10-12T10:30Z, before any L5 code was written or any L5 game was played. The experiment freeze was lifted by Ameer on 12 Oct (see L4). This is a local protocol; changes go under "Amendments".

## Why

Only two models over-took on purpose in L2: gpt-oss-120b and Nemotron 3 Super. Gemma 4 31B mostly kept its request after a cut, and Mistral Large 3 never over-took. All four are open-weight. Every main-track paper close to ours includes at least one closed frontier model (`research/venue_readiness/2026-10-12_decision.md` §3).

## Families

**Open, through Ollama Cloud:**
- DeepSeek V4.1 Flash (`deepseek-v4.1-flash`);
- GLM-5.3 (`glm-5.3`).

**Closed, through OpenRouter (OpenAI-compatible API):**
- Anthropic Claude Sonnet 5.5 (`anthropic/claude-sonnet-5.5`);
- OpenAI GPT-6.1 Sol (`openai/gpt-6.1-sol`);
- Google Gemini 3.8 Flash (`google/gemini-3.8-flash`).

These are mid-priced, current models. The closed models run only once Ameer has put an OpenRouter key in the environment (`OPENROUTER_API_KEY`). The key is never printed or committed.

**Sampling deviations, declared now:**
- Temperature is 0.7 and a seed is sent wherever the provider accepts them.
- OpenRouter lists no temperature parameter for Claude Sonnet 5.5 or GPT-6.1 Sol, and no seed for Claude. For those models the provider default is used, and each call's seed is logged but has no effect.
- JSON mode is requested (`response_format: json_object`). Answers inside a code fence are accepted, as in L2 Amendment 1.

## Design

- **Populations and seeds:** the same game, prompts (explicit wording, q = 1/6), and new populations and seeds as L4 (population base 1,960,000,000), so L5's E0 is comparable with L4's E0 for gpt-oss and Nemotron.
- **Stage A, pilot gate and screen.** Each family plays E0 on populations 0–9 (10 games).
  - **Gate:** at least 95% valid first answers and mean comprehension of at least 2 of 3. These are L2's rules. A family that fails the gate is reported and not used.
  - **Qualifying:** a family qualifies for the threshold test if its E0 over-take rate is at least 20% of agent-rounds and the lower bound of the 95% population-bootstrap interval is above 5%. gpt-oss-120b (55%) and Nemotron 3 Super (65%) would qualify; Gemma 4 31B (4–10%) and Mistral Large 3 (0%) would not.
- **Prediction.** After stage A, for each qualifying family, `analyze_l4.py predict` writes, into L5's `prediction.json` (committed before any stage-B game):
  - $\hat g_1$: the mean tonnes above the allowance on E0's over-take steps;
  - $F^*$: the smallest fine on {6, 12, …, 48} with $F/6 > \hat g_1$;
  - $F^-$: the grid step below $F^*$.
- **Stage B, threshold test.** Each qualifying family plays E$F^-$ and E$F^*$ on populations 0–9 (20 games).

## Outcomes and hypotheses (per family)

- **L5-H1, descriptive.** The E0 over-take rate, and the kind of over-take: the catch above the request, or the request kept after a cut (as in L2's post hoc split).
- **L5-H2 (qualifying families).**
  - (a) The rate at E$F^*$ is at most 5%.
  - (b) The rate at E$F^-$ is above 5%.
  - Each part is reported separately, with falsifiers as in L4-H1.
- **How results enter the paper.**
  - Families that never over-take are reported as a result about *whether* models over-take.
  - The claim "the stopping fine follows the observed excess" counts the families for which H2(a) holds, together with L3 and L4.

## Size and cost

- Stage A: 5 families × 10 games = 50 games.
- Stage B: at most 5 × 20 = 100 games.
- **Measured or listed prices**, per game, at about 80k input and 40k output tokens:

| Model | Approximate cost per game |
|---|---:|
| DeepSeek V4.1 Flash | $0.07 |
| GLM-5.3 | $0.29 |
| Claude Sonnet 5.5 | $0.56 |
| GPT-6.1 Sol | $0.56 |
| Gemini 3.8 Flash | $0.21 |

- **Worst case:** about $20 of Ollama credit for the open families and about $70 of OpenRouter credit for the closed ones. A token cap per family (5M tokens) stops a run that costs far more than expected. It is reported as an amendment if reached.

## Gates and stop rules

- **Offline:** tests pass. The OpenRouter client is tested with a fake HTTP server, and the error path is tested with a missing key.
- **Order:** no stage-B game runs before `prediction.json` is committed. The runner prints progress only.

## Analysis

As in L4: paired bootstrap over populations, 4,000 resamples, seed 20261019, all results reported.

## Amendments

### Note 1 (2026-10-12T10:25Z): corrected freeze time
The "Frozen" line gives 10:30Z, which was a mistake made when writing it. The protocol was committed at 10:17Z (commit `c001344`), before the code (commit `30d93b2`, 10:22Z) and before the first game (10:22:42Z). Nothing else changes.

### Amendment 1 (2026-10-12T10:50Z): closed families on hold
Ameer decided not to pay for closed-model API access now. Claude Sonnet 5.5, GPT-6.1 Sol and Gemini 3.8 Flash are on hold until research credits (Anthropic External Researcher Access, Google Gemini Academic Program) or supervisor funding arrive. Their code is built and tested; if they run, they use this protocol unchanged. L5 now consists of DeepSeek V4.1 Flash and GLM-5.3 on Ollama. If the closed families never run, the paper states "open-weight models only" as a limitation. No L5 game had finished when this was decided.

### Amendment 2 (2026-10-12T20:45Z): GLM-5.3 dropped for cost and latency
- **Why.** On the remote runner, GLM-5.3 used about 12,000 completion (reasoning) tokens and about 2 minutes per decision: 158 calls, 1.81M output tokens, about $8 of Ollama credit. One 20-round game would take about 5 hours and cost about $9, so the 10-game screen alone would cost about $90, more than Ameer's monthly credit. This is the same problem as Mistral Large 4 in L2 (Amendment 4).
- **What was stopped.** Lane A was cancelled at about 20:30Z, during GLM's first game. No GLM game finished, so no GLM outcome exists. The partial call log is kept.
- **Replacements timed on one real request and one real catch prompt** (completion tokens; seconds, which may include queueing behind the 3 cloud lanes):

| Model | Request | Catch |
|---|---|---|
| Kimi K2.6 | 9,424 tokens; 604 s | 8,531 tokens; 205 s |
| GLM-5.3 Flash | 14,153 tokens; 65 s | 975 tokens; 3.5 s |
| MiniMax M3 | 3,483 tokens; 33 s | 1,019 tokens; 11 s |

- **Decision.** MiniMax M3 is the only feasible candidate, at about $0.9 per game. Whether to screen it is Ameer's budget decision, and is pending. Until then, L5 continues with DeepSeek V4.1 Flash alone. DeepSeek passed the gate and qualified: E0 over-take rate 0.80, $\hat g_1$ = 5.41 t, F* = 36 t, F⁻ = 30 t, frozen at 15:38Z.
