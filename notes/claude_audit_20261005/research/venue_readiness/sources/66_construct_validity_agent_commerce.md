# 66 — When Guardrails Look Effective: Construct Validity Failures in LLM Agent Commerce Evaluation

- Authors: Peiying Zhu, Sidi Chang
- URL: https://arxiv.org/abs/2609.01519
- Posted: 2026-09-01. Comment: "Submitted to the NeurIPS 2026 Trust-AI-Eval Workshop. 7 pages"
- Accessed: 2026-10-12
- Type: workshop submission (preprint)
- Credibility 3/5 · Recency 5/5 · Bias 3/5

## Verbatim quotes
- Abstract: "The four largest 14B single-generation effects averaged +229; after three generations per profile-condition, they averaged +37.6 (95% bootstrap interval [-34.2, 109.3]), while generation residuals account for 49.9% of variation in this post-hoc probe."
- Abstract: "Scripted positive controls show why this matters. A profit-maximizing seller already attains first-best welfare, so guardrails mostly redistribute and reduce welfare"
- Abstract: "We contribute a construct-validity contract separating incentive validity, protocol isolation, stochastic stability, and welfare accounting, and returning INVALID or INCONCLUSIVE before substantive policy claims."

## Threat to claim 4: medium — a ready-made checklist a reviewer can apply: are our LLM thresholds stable across generations/seeds (stochastic stability), and do all arms share the same prompt and protocol except the fine (protocol isolation)? Response: report seeds per cell and intervals; state that only the fine line changes between arms (verify). Our programmed-agent arm acts as the "scripted positive control" they recommend — say so; this is a strength.
