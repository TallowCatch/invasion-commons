# Plan: what it takes to get the audited-commons paper accepted (deep research, medium depth)

**Started:** 12 Oct 2026.
**Genre:** decision.
**Root:** `notes/claude_audit_20261005/research/venue_readiness/`.

## 1. Reframe

**Question.** Which venue, reachable between Oct 2026 and Jul 2027, fits the paper as it stands or with a bounded addition? What do reviewers there expect that the paper lacks? Can Ameer's Ollama Pro plan pay for the LLM runs that would close the gaps?

**Decision behind it.** There are three paths:
- (a) submit the current paper with text fixes only;
- (b) run one bounded LLM study first (L4: new populations, matched expected fine at different audit rates, a wording transfer), then submit to a main venue;
- (c) widen the paper (more model families, frontier closed models, an LLM version of another game) for a journal.

**Hypotheses (falsifiable):**
- **H1.** A main-track ML or agents venue (ICML, NeurIPS, IJCAI, AAMAS) expects at least 4–6 LLMs, including at least one frontier closed model. Two models that over-take on purpose will be called thin.
  - Refuted if accepted close papers routinely use ≤3 open models.
- **H2.** The single most-requested gap is the identification test (vary the audit rate at a fixed expected fine). Without it, reviewers will treat the LLM result as correlational.
  - Refuted if close papers were accepted without such a manipulation.
- **H3.** A workshop (AAAI-27 workshops, or ICML/NeurIPS 2027 workshops), TMLR or JAAMAS will accept the paper as it is or with text fixes.
  - Refuted if their stated criteria or recent acceptances show a higher bar.
- **H4.** Ollama Pro can cover L4 (about 200 games, ~30M tokens) within one billing month, but cannot add frontier closed models, which Ollama does not serve.
  - Refuted by Ollama's stated plan limits or by its model list.

## 2. Building blocks of the report

1. What has been done (inventory, from the repository).
2. Venue table: deadline, format, fit and bar.
3. Reviewer expectations, drawn from accepted close papers and their public reviews.
4. Gap list: required, recommended, optional.
5. Compute: Ollama plan, token cost per game, alternatives.
6. Recommended path with a timeline.
7. Adversarial section.
8. Refresh targets.

## 3. Sourcing strategy

Four parallel sub-agents, each saving sources to `sources/NN_slug.md` with verbatim quotes:

| Agent | Topic | Source numbers |
|---|---|---|
| A | Venues and deadlines, Oct 2026–Jul 2027: official calls for papers | 01–19 |
| B | Reviewer expectations: close accepted papers and their OpenReview reviews (GovSim, Ye & Steinhardt, Okamoto, Bracale, Guzman Piedrahita, SanctSim and others) | 20–39 |
| C | Ollama Cloud: plan limits, pricing, model list; and alternatives for frontier models (OpenRouter, provider APIs) | 40–54 |
| D | Adversarial: new competing work since 8 Oct 2026; becker-agents repository status; critiques of LLM-agent social-science evaluations | 55–69 |

**Opposition queries:** "LLM agents expected penalty deterrence audit probability 2026"; "LLM economic games validity critique"; "prompt sensitivity LLM agent evaluations".

**Internal evidence (verified 12 Oct).** Tokens per game, from the complete L3 call logs:
- gpt-oss-120b: 126k tokens, 4.3 min serial;
- Nemotron 3 Super: 178k tokens, 25.7 min serial.

The L3 run of 80 games used about 12.2M tokens.

## Risk register

- Deadlines for 2027 venues may not be announced yet. Estimates must be marked as estimates.
- Ollama's plan limits are stated in "usage", not tokens. Converting needs Ameer's dashboard.
- Reviews on OpenReview exist only for some venues.

## Stop criteria

The search stops when:
- every venue row has an official source, or is marked as an estimate;
- H1–H4 each have at least 3 sources, or are marked as under-determined.

## Changelog

- 12 Oct: plan written; agents A–D launched.
- 12 Oct: agents A–D returned (67 source files). Report written: `2026-10-12_decision.md`; `refresh_targets.md` and `sources.csv` added. H1 partly confirmed, H2 plausible but undocumented, H3 confirmed for the workshops and TMLR, H4 confirmed (Ollama credits cover L4; Ollama has no closed models).
