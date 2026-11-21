# Refresh targets (check before each submission)

Run `update commons_platform` to produce a delta in `diffs/`.

## Novelty watch (adversarial triggers)
- Any paper with an **overseer that audits or fines LLM agents in a spatial commons** (search: "LLM agents" + "Commons Harvest" / "SocialJax" / "Melting Pot" + audit / fine / sanction / monitor / regulator).
- Follow-ups to OpenSocialJax (arXiv:2610.13710; repo commit b5f59a3) adding institutions; new SocialJax releases (commit c64f1aa).
- Follow-ups to Large Legislative Models (arXiv:2410.08345), Bell & Madisetti (ICML 2026), Okamoto et al. (arXiv:2608.12323), CoopEval (arXiv:2604.15267).
- New Willis / Zhao / Du / Leibo papers; changes to `willis-richard/emergent_llm` (commit be78ff9).
- Multi-agent AI-control evaluations (Radev 2026 [86], Gardner-Challis 2026 [87], Nakamura 2026 [99]).

## Numbers to re-check
- OpenSocialJax cost per seed (README; [23]) and calls per step.
- Melting Pot regrowth constants in code vs papers (0.0025 vs 0.001) [01].
- AI-control audit budgets (0.5–2%) [80, 82, 84].
- FishBase r bands [79]; human CPR audit bands (10–20% weak, 50% deterrent) [77].
- Ollama cost per call ($0.00025, measured in L4).

## Hypotheses to revisit
- H3 (extend, not build) — revisit if OpenSocialJax hooks or cost fail the S0 admission test.
- H4 (cost) — revisit if per-step LLM calls get ≥10× cheaper.

## Insufficient-evidence items to upgrade
- Bell & Madisetti design details [64] (abstract only).
- Tax-experiment audit-rate range [68, 69].
- Concordia cost per step [13] (estimate).
- Venues of [94], [95].
