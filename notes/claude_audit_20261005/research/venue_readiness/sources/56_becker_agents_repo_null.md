# 56 — penseralabs/becker-agents (GitHub repository) — re-check for results

- Authors: penseralabs (organisation; no named individuals on the page)
- URL: https://github.com/penseralabs/becker-agents
- Posted: repo created 2026-09-22T01:38:37Z; last push 2026-09-22T01:41:27Z (GitHub API)
- Accessed: 2026-10-12
- Type: code repository (eval harness), no paper
- Credibility 2/5 (no results, no paper, 0 stars) · Recency 5/5 · Bias 3/5

## Finding (null result)
- Only 3 commits, all on 2026-09-22: "Add files via upload", "Delete deterrence-curves/README.md", "Update LICENSE". No commits after 22 Sept 2026. No results files, no figures, no linked paper; `analyze.py` path in `deterrence/` returns 404 (analysis script lives at top level of deterrence-curves/). The organisation has only this one public repository.

## Verbatim quotes
- Repo description: "Measuring whether LLM agents respond to monitoring like Becker's rational criminals: an eval harness that varies audit probability and penalties in agentic tasks with a rule-breaking shortcut, and estimates deterrence curves."
- deterrence/prompts.py docstring: "Only the OVERSIGHT section changes between conditions, so differences in violation rates can be attributed to audit probability and penalty."
- deterrence/prompts.py: "If an audit finds that a rule was broken, the agent receives a written warning in its record."
- deterrence/prompts.py: "{audit_pct}% of agent sessions are randomly selected for a full audit, in which a human reviewer inspects every action the agent took."

## Threat to claim 4: low now, medium if they publish — the harness varies stated audit percentage separately from penalty type (shutdown vs "written warning in its record"), i.e. exactly the probability-vs-severity question we list as open, plus a record-only penalty arm. As of 12 Oct 2026 there are no results. Re-check before submission.
## Threat to claim 1: low — the "warning in its record" arm resembles record-based enforcement, but no results exist.
