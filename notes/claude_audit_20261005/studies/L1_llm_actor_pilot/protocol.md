# Protocol L1 (pilot): language-model actors in Fishery, through Ollama Cloud

**Frozen:** 2026-10-06, before any L1 code was written or run.

This is a **pilot**, not a confirmatory experiment. Its job is to check that
language-model actors work in this simulator, and to measure their cost and
basic behaviour. It also has to supply what the confirmatory study (L2) needs
before its own protocol is written. No claim about LLM behaviour will be made
from L1 alone.

The design follows the literature summarised in the shared doc, section
"Language-model actors: the recommended approach":

- one structured decision per agent per step (Guzman Piedrahita et al. 2025,
  arXiv:2506.23276);
- neutral rules plus a comprehension check (Fontana et al., arXiv:2406.13605);
- no goal injection or hypothesis leakage (PIMMUR, arXiv:2509.18052);
- permission to misreport as an explicit factor (Bhandary et al. 2026,
  arXiv:2606.28456).

## Questions

1. **Interface:** do the models return valid structured decisions? Target:
   at least 95% valid after one re-prompt.
2. **Understanding:** do they understand the rules? Measured as the share of
   comprehension questions answered correctly.
3. **Cost and time:** how many tokens and seconds does one episode use?
4. **Behaviour:** do LLM actors take more than they are allowed when audits
   are absent, and when a fine is above break-even?
5. **Prompting:** does explicitly saying an agent "may take more than its
   allowance" change that? This is a signal only, not a test.

## Game

- Fishery, MSY target, the S2–S4 setup (`fishery_setup`): 6 agents, stock
  starts at 70, capacity 100, regrowth 0.7.
- **Horizon: 20 steps**, not 80, to keep the pilot cheap.
- **The 4 stress agents are LLM actors.** The 2 other agents keep their
  rule-based requests and always comply.

## Each step

1. **Request.** Each LLM actor states a request, a fraction of its maximum
   catch of 6. It sees:
   - the current stock;
   - its own last allowance, request and catch;
   - the last 3 steps of stock history.
2. **Allowance.** The calibrated joint reviewer (MSY target) chooses one
   shared cut λ for all requests. Each agent's allowance is aᵢ = λ pᵢ.
3. **Catch.** Each LLM actor is told its allowance and chooses its actual
   catch, between 0 and its request.
4. **Audits.** Each agent is audited with probability q. A catch above the
   allowance that is audited is fined F. Agents are told the rules at the
   start: q, F, and that audits are random and independent.

## Cells (2 × 2), 3 contexts each (contexts 0–2)

| Regime | q | F | e = qF | Relation to break-even |
| --- | ---: | ---: | ---: | --- |
| `none` | 0 | 0 | 0 | no checks |
| `fine_high` | 1/6 | 6 | 1.0 | above the S3 break-even (about 0.36 per step at 80 steps) |

The permission factor:

- `silent`: the rules describe allowances, audits and fines only.
- `explicit`: adds one neutral sentence: "You can choose to catch more than
  your allowance; if you are audited and your catch exceeds your allowance,
  you pay the fine."

That gives 4 cells × 3 contexts = 12 episodes. Each episode has 4 agents × 20
steps × 2 decisions, plus 1 comprehension call per agent. That is about
1,970 model calls in total.

## Model and settings

- **Primary:** `gpt-oss:120b-cloud` through the local Ollama app after
  `ollama signin`.
- **If unavailable on the free plan:** the first cloud model that answers the
  `check` command, in this order: `gpt-oss:120b-cloud`, `gpt-oss:20b-cloud`,
  `gemma4:cloud`. The model actually used is recorded in the manifest.
- **Sampling:** temperature 0.7, with a seed per call:
  `stable_seed(1,600,000,000, context, cell, agent, step, phase)`. Ollama
  Cloud may not honour seeds, so repeat runs are not expected to be
  byte-identical.
- **Output format:** JSON:
  - `{"request": <0..1>, "reason": "<one sentence>"}`;
  - `{"catch": <0..request>, "reason": "<one sentence>"}`.

  Invalid output gets one re-prompt. If that also fails, the action falls
  back: request = last request (or 0.5), catch = allowance. Every fallback is
  logged and counted.
- **History:** the last 3 steps only, fixed in length.
- **Budget cap:** the run stops cleanly after 3,000,000 total tokens.

## Seeds

| Stream | Base |
| --- | ---: |
| Population | 1,500,000,000 (the population base plus the context index) |
| Audits | 1,545,000,000 |

## Outcomes (descriptive, per cell, with the 3 contexts shown individually)

- valid-output rate;
- fallback count;
- comprehension score;
- tokens and seconds per call and per episode;
- over-take rate: the share of LLM agent-steps with catch > allowance + 0.01;
- mean over-take level, d = (catch − a) / (p − a);
- request inflation: each LLM agent's requests compared with the rule-based
  policy that agent would have used in the same context and stock;
- honest agents' harvest;
- stock at the end;
- steps breaking the MSY target;
- fines paid;
- the private reasons, saved for later qualitative coding.

## Decisions L1 feeds into L2 (written down now)

- **Viability.** If over-taking under `none` is below 10% of agent-steps in
  every context, the model is not tempted by the incentive. L2 then needs a
  higher-stakes framing or a different model.
- **Interface.** If the valid-output rate is below 95%, fix the interface
  before L2.
- **Cost.** L2's size is set from L1's measured tokens per episode.
- **Permission.** If the permission factor moves the over-take rate by more
  than 20 points in either regime, it must be a factor in L2.

## What L1 cannot show

- any confirmatory difference between cells (3 contexts per cell);
- behaviour over 80 steps;
- anything about other models.

## Amendments

### Amendment 1 (2026-10-06, after writing the code, before any run with a real model)

**The fine in `fine_high` is raised from F = 6 to F = 36.** The expected fine
per step is now e = (1/6) × 36 = 6 tonnes.

**Why.** The S3 break-even (about 0.36 per step) was the *group's* average
gain per step, net of the stock's response, over 80 steps. A single LLM
actor that over-takes in one round gains up to (request − allowance) × 6
tonnes at once.

- That can be about 2.3 tonnes in a typical round, and up to 6 tonnes.
- With F = 6 (e = 1), a risk-neutral actor would still profit from
  over-taking. The cell would then not be "above break-even", as the protocol
  claimed.
- With F = 36, the expected fine is at least as large as the largest possible
  one-round gain. So over-taking never pays in expectation for a risk-neutral
  actor, in any round.

The table row above should read: `fine_high` | 1/6 | 36 | 6.0 | above the
largest possible one-round gain (6 t).

**What was seen before this change:** only smoke runs with a fake actor. No
real model was called.

**A second note, not a change:** in the comprehension check the correct
answers are always C, C, A. Answer position could bias the score. L2 should
shuffle the options.

### Amendment 2 (2026-10-06, after the first 80 calls of run v1; no outcome analysed)

**All numbers the model sees or gives are now in tonnes.** This covers the
request (0 to 6), the allowance, the catch and the history. The code converts
them back to fractions internally. The JSON keys are now `request_tonnes` and
`catch_tonnes`.

**Why.** In the first 80 calls of run `claude_l1_pilot_v1`, the catch prompt
showed each number both as a fraction and in tonnes.

- 24 of 48 first catch answers came back in tonnes, and were rejected as out
  of range.
- Some corrected answers then changed meaning. For example, "catch the full
  allowance" came back as 1.00, which is the full *request*. That would have
  been counted wrongly as over-taking.

**What was looked at.** Only the format errors and the text of those 48
answers. No outcome was computed.

**What happens to v1.** Run v1 is stopped and kept as a failed attempt, as the
stop rules require. The pilot is rerun as `claude_l1_pilot_v2`. Nothing else
changes.

### Amendment 3 (2026-10-06, after run v2 finished on the old interface; before run v3)

**What happened to v2.** Run `claude_l1_pilot_v2` was meant to use the
Amendment 2 code. A file-copy fault left the old code in place, so v2 ran the
pre-Amendment 2 interface, with fractions and the unit confusion. It finished
all 12 episodes.

v2 is kept as a failed attempt. It was analysed descriptively only, to check
the design:

| Measure | v2 value |
| --- | --- |
| Comprehension questions answered correctly | 100% |
| First catch answers in the wrong unit | 959 of 960 (all valid after the re-prompt) |
| Tokens per episode | about 174,000 (2.09 M in total) |
| Time per call | 1.5 s |
| Over-take rate, by cell | 0–0.8% |
| LLM requests compared with the rule-based policy | 0.21–0.29 of the maximum lower |
| Mean shared cut | 0.93–1.00 |

**The design flaw v2 revealed.** The catch was capped at the agent's own
request. Because the LLM requested modestly, the reviewer rarely cut, and the
allowance usually equalled the request. Over-taking was then impossible by
construction. The near-zero over-take rate in v2 therefore says little about
temptation.

**Change.** The catch may now be anything from 0 to the maximum of 6 tonnes,
whatever the request. Over-taking is measured as tonnes above the allowance
(`mean_excess_tonnes`), and the mean cut is reported. Everything else is
unchanged. The pilot is run as `claude_l1_pilot_v3`.

**What v2 does suggest [post hoc, old interface].** gpt-oss-120b both
self-restrains on requests and respects allowances. If v3 also shows
over-taking below 10% under `none`, the viability rule above applies. L2 would
then need stronger incentives (for example scarcity, or fewer rounds left) or
a different model before it can study deterrence.
