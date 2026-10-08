# Results L3: language models stop over-taking close to their own gain, but deterrence starts well before it

**Protocol:** `protocol.md` in this folder. It was frozen at 2026-10-08T10:20Z, with the prediction written, before any code. The analysis code (`analyze_l3.py`) was committed before any game.
- Note 1 records that partial outcomes, 17 of 80 games, were seen by a metrics script while the run was going. Nothing was changed.
- The lanes were rebalanced at 12:30Z. That affects only which job played which game.

**Run:** GitHub Actions with Ollama Cloud (Pro), 2026-10-08 11:09Z to 18:22Z.
- 80 games: 40 for gpt-oss and 40 for Nemotron.
- Complete call logs: 6,560 and 6,441 calls, including re-prompts and restarted games.
- 0 fallbacks.
- Valid first answers: 100% for gpt-oss, 99.98% for Nemotron.
- Comprehension: 3 of 3 in all 320 checks.
- Raw games are on the `l2-results` branch, in `claude_l3_v1/`.
- Tables (`l3_summary.json`, `l3_curve.csv`, `l3_context_cells.csv`) are in `notes/claude_audit_20261005/runs/claude_l3_v1/`.
- Figures: `fig4_llm_agents` (panels a–b, the 10-fine curve) and `fig5_spine`.

**Status of the numbers:** the hypothesis tests are **verified** (`analyze_l3.py`, unchanged since before the run). Intervals are 95% context-bootstrap intervals (4,000 resamples). Interpretations marked **[post hoc]** were not tested.

## Question

Each model's gain from one step of over-taking, g, was measured in L2's no-fine cell. Does over-taking stop at the first tested fine above that gain, as the risk-neutral break-even rule says (P1)?

**The prediction, written before any game.** g = 4.48 t for gpt-oss and 4.74 t for Nemotron. So F = 12, 18 and 24 t (e = 2, 3, 4) should not deter, and F = 30 t (e = 5) should.

## What came out

**Over-taking (% of LLM agent-steps) at every explicit fine.** L2's cells are combined with L3's new ones. "Relative" means relative to no fine.

| F (t) | e (t) | gpt-oss: e/g | gpt-oss: over-taking | gpt-oss: relative | Nemotron: e/g | Nemotron: over-taking | Nemotron: relative |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 55.3% [52.0, 58.1] | 1 | 0 | 64.8% [60.6, 68.9] | 1 |
| 8 | 1.33 | 0.30 | 51.4% [49.4, 53.5] | 0.93 | 0.28 | 58.5% [54.2, 62.6] | 0.90 |
| **12** | 2 | 0.45 | 43.5% [40.0, 46.9] | 0.79 | 0.42 | 56.3% [52.4, 59.8] | 0.87 |
| **18** | 3 | 0.67 | 12.6% [8.1, 17.2] | 0.23 | 0.63 | 41.8% [33.6, 50.0] | 0.65 |
| **24** | 4 | 0.89 | **1.0%** [0.2, 1.9] | 0.02 | 0.84 | 8.0% [4.1, 12.2] | 0.12 |
| **30** | 5 | 1.12 | 0.5% [0.0, 1.1] | 0.01 | 1.05 | **1.8%** [0.9, 2.8] | 0.03 |
| 36 | 6 | 1.34 | 0.0% | 0 | 1.27 | 0.0% | 0 |

**Hypotheses:**

| Hypothesis | gpt-oss | Nemotron |
| --- | --- | --- |
| L3-H1 (exact): the first fine at which over-taking is at most 5% is F\* = 30 | **fails**: F\* = 24, one grid step early | **holds**: F\* = 30 |
| L3-H2 (within one grid step): F\* is 24, 30 or 36 | **holds** | **holds** |
| L3-H3: over-taking at F = 24 (e = 4, below g) is at least 25% | **fails**: 1.0% | **fails**: 8.0% |
| The curve never rises again once it has fallen below 5% | yes (monotone) | yes (monotone) |

**Claim 6, by the pre-stated rule: "within one grid step".** It is exact for Nemotron and one step early for gpt-oss.

**Harm to the rule-followers.** Catch per rule-following fisher, in tonnes, at F = 12 → 18 → 24 → 30:
- gpt-oss: 25.2 → 31.1 → 33.8 → 34.2;
- Nemotron: 9.6 → 18.8 → 31.8 → 34.6.

Lakes collapsed under Nemotron: 4, 1, 0, 0 of 10.

**Social metrics** (`runs/social_metrics/`, as GovSim and Perolat et al. 2017 define them). At F = 24–36, efficiency is 0.99 of the maximum sustainable harvest for both models, equality 0.79–0.82, and the stock is at or above half capacity in 81–100% of rounds. At F = 12, efficiency is 0.85–1.00, equality 0.67–0.78, and the stock is at or above half capacity in only 6–19% of rounds.

## What it means

**What it shows:**
- **Over-taking ends close to each model's own gain.** For both models, over-taking is at or below 2% from the first fine above g onwards. The prediction made from L2's no-fine cell located the stopping point exactly for Nemotron, and one grid step early for gpt-oss.
- **Deterrence is graded, and it starts well before the gain** (H3 failed for both). The simulated best-responders in R3 switch off in one step at e/g = 1.0–1.2. The LLMs reduce over-taking steadily from about e/g = 0.4. At e/g ≈ 0.85–0.9 over-taking has already fallen to 2–12% of the no-fine level, below what a risk-neutral agent with gain g would do.
  - Over-taking is halfway down at about e/g ≈ 0.55 for gpt-oss and 0.7 for Nemotron, interpolated **[post hoc]**.
- **Read with L2,** the dose–response has three parts:
  - fines far below the gain do nothing, or make things worse (gpt-oss at F = 1–2);
  - fines from about 0.4 g to g reduce over-taking steadily;
  - fines at or above g stop it.

**What it does not show:**
- **Why deterrence starts early.** Three explanations fit and none was tested **[post hoc]**:
  1. The models act as if risk-averse. In the classic tax-evasion model, risk aversion lowers the threshold (Allingham & Sandmo 1972; metadata depth only).
  2. g overstates the gain the model sees. An over-take also lowers the stock and so the later catch, which g as measured (tonnes above the allowance in one step) leaves out.
  3. The models react to a large stated fine as a signal in itself.
- **That the threshold sits exactly at g.** The grid steps are 1 t of expected fine, about 0.2 g, so "within one grid step" is the resolution of this test.
- **Anything about models that do not over-take on purpose.** Gemma and Mistral were not run, as the protocol says.

## Design caveats

- 10 contexts per fine. The intervals resample contexts.
- Replies use temperature 0.7 with fixed seeds, so each game is one draw per context.
- L3's games were played about one day after L2's, on the same cloud models. Comparisons across L2 and L3 cells assume the models did not change in between. The two runs agree where they overlap in kind: over-taking falls steadily from F = 8 to F = 36.

## What we now believe

- **High confidence.** For LLM agents that over-take on purpose, the expected fine at which they stop is predictable from their own no-fine behaviour, to within one grid step (about 0.2 g).
- **Moderate confidence.** LLMs are easier to deter than a risk-neutral best responder: deterrence begins at about 0.4 g and is mostly complete before g.
- **Untested:** why; whether the same holds in Forest or River; and whether fines that grow with the excess (marginal deterrence) would shift the curve.
- **Next:** none. This was the last experiment. The next step is writing.
