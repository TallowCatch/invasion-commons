# AI Control: Improving Safety Despite Intentional Subversion

- **Authors:** Ryan Greenblatt, Buck Shlegeris, Kshitij Sachan, Fabien Roger
- **URL:** https://arxiv.org/abs/2312.06942 (HTML read); OpenReview forum KviM5k8pcP
- **Venue:** "ICML 2024 Oral" (main conference). ICML 2024 reviews not public (API search returned no review notes).
- **Date accessed:** 2026-10-12
- **Source type:** peer-reviewed conference paper (primary)
- **Credibility 5 / Recency 3 / Bias 4**

## Verbatim
- Models: "using access to a powerful but untrusted model (in our case, GPT-4), access to a less powerful trusted model (in our case, GPT-3.5), and limited access to high-quality trusted labor" ("which we simulate here with GPT-4 for convenience").
- Environment: one testbed — "we want to use LLMs to solve 5,000 programming problems from the APPS dataset".
- Runs / estimation: "We actually only generate blue and red team responses for 1500 problems, and then simulate running the scenario with 5000 problems, by upsampling"; "To measure safety, we repeatedly simulate the scenario using the protocol".
- Audit budget as enforcement parameter: "In each simulation, the protocol has enough H points to use on up to 2% of problems. The central use for [H] [is] to audit submitted code."

## Relevance
ICML oral with two closed models from one provider and one environment; strength came from a precise threat model and protocol comparisons (incl. auditing thresholds), not model breadth.
