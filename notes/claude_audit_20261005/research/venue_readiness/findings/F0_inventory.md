# F0: What the paper already has (verified from the repository, 12 Oct 2026)

## Experiments

| Study | What it tests | Agents | Scale | Result |
|---|---|---|---|---|
| R1, R2 | Reviewer's target (one-step line or half capacity); a wrong model | programmed, honest | 64 populations × 80 rounds | Half capacity gives 1,393 against 555 harvest. A wrong model approves 82% of risky requests. A learning reviewer recovers in about 60 rounds. |
| S1, S1b, T1, C1, R2, R4, R5 | Consequences of a catch: none, memory, fine | programmed under-reporters | 12 settings on a common 2×2 grid, plus 5 wider Fishery settings | Memory beats audits without memory in 12 of 12 settings. Audits without memory remove 10–46% of the harm. |
| S2, S3, R3 | Deterrence threshold at e = qsF against g* | programmed over-takers, level chosen by search | 6 settings, q from 1/6 to 0.02, s ≤ 1 | 5 of 5 thresholds predicted in advance, within one grid step. |
| S4, S5 | Audit rules (fine, memory, extra checks, everyone's cut tightened) | adaptive programmed cheaters | Table S3 | Tightening everyone's cut makes cheating pay (142 against 113). Memory alone leaves 41% of the gain. Fine plus memory deters at q = 0.05. |
| T1, C1 | Audit targeting (random, largest report, forecast gap) and a fixed schedule | programmed | all three games | Largest report: 25% against 100% of liars caught. Forecast gap does best in Forest. A known schedule fails. |
| L1 | Pilot of the LLM actor interface | LLMs | gate | Passed. |
| L2 | LLM fines 0–36 t, memory, wording controls | gpt-oss-120b, Nemotron 3 Super, Gemma 4 31B, Mistral Large 3 | 400 games, plus 40 memory games | Over-taking stops at 36 t. A small fine raises gpt-oss over-taking. Memory cuts over-taking to 20–33%. |
| L3 | Threshold location at fines 12–30 t, prediction frozen in advance | gpt-oss, Nemotron | 80 games | Exact for Nemotron; one step early for gpt-oss. H3 (≥25% over-taking at 24 t) failed. |
| Post hoc (9–12 Oct) | Largest excess; bootstrap; split-half; wording | — | existing data | Both models stop below the risk-neutral benchmark. The stopping fine is stable. The predictor moves one step with wording in 2 of 4 cases. |

**LLM totals:** 520 games, 81,232 decisions, 0 fallbacks, 6 re-prompts, every comprehension check 3 of 3.

## Writing and infrastructure

- **Paper:** an AAAI-format paper (8 pages: 7 of content plus references) and a 7-page supplement. The supplement holds settings, key terms, controls, prompts, protocols, positioning, the wider Fishery grid and the post hoc checks.
- **Pre-registration:** a frozen protocol before the code of every study, with dated amendments. These are local, not on OSF.
- **Code and tests:** 362 passing tests. The runner on GitHub Actions resumes cleanly after an Ollama usage limit.
- **Novelty search** (`novelty/`) as of 8 Oct; scooping watch on penseralabs/becker-agents.
- **Reviews:** two AI reviews, both evaluated (`review_20261009_ai_review.md`, `review2_20261012_ai_review.md`).

## Known gaps (from the reviews and our own limitations)

1. With q fixed for the LLMs, the expected fine and the stated fine cannot be told apart.
2. Calibration and test used the same 10 populations; no fresh run exists.
3. Transfer across wordings is untested near the threshold.
4. Only 2 LLMs over-took on purpose; all models are open-weight, with no frontier closed model.
5. LLM results come from one game (Fishery).
6. The silent control still mentions a check with a 0 t fine.
7. The reasoning texts of the first LLM run were lost. Outcomes are unaffected.

## Measured compute

From the complete L3 logs:

| Model | Tokens per game | Minutes per game (serial) |
|---|---:|---:|
| gpt-oss-120b | about 126k | 4.3 |
| Nemotron 3 Super | about 178k | 25.7 |

- **L3:** 80 games used about 12.2M tokens.
- **L2 (all four families):** 400 games, which the free plan could not finish ("about 4 months of free usage"). Pro was bought for it.
