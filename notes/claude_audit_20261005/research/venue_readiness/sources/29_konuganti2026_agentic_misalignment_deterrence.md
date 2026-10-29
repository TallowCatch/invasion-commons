# Agentic Misalignment Deterrence: Incorporating Probability and Stakeholders to Decision-Making

- **Authors:** Karthik R Konuganti, Ryan Co
- **URL:** https://openreview.net/forum?id=z9Bbff0YlE ; https://icml.cc/virtual/2026/67825
- **Venue:** "FAGEN@ICML 2026 Poster" (ICML 2026 workshop; ICML virtual page lists it under "Second Workshop on Agents in the Wild"). Not main track.
- **Date accessed:** 2026-10-12
- **Source type:** workshop paper abstract (primary)
- **Credibility 3 / Recency 5 / Bias 4**

## Verbatim (abstract)
- "We test whether stating an audit probability in the system prompt deters this behavior across three frontier models, sweeping the probability across p ∈ {0, 0.05, 0.1, 0.5, 0.95}."
- "Harmful behavior declines with stated probability (18–49 percentage point reductions), with chain-of-thought analysis showing models increasingly engage with the stated probability in their reasoning."
- Models named: "Opus 4 shows minimal change from baseline, R1 reduces partially, and Sonnet 4 reduces to levels equivalent to high-probability monitoring."
- "we additionally test target-specific consequence framings ... and find no consistent rate change across models despite passing manipulation checks"

## Relevance
Direct precedent that sweeping audit probability on LLM agents is feasible and publishable; it is the manipulation our LLM arm lacks (we fix p = 1/6). Runs per condition not in abstract.
