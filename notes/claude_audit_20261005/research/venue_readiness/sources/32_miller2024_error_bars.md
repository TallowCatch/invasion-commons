# Adding Error Bars to Evals: A Statistical Approach to Language Model Evaluations

- **Author:** Evan Miller (Anthropic)
- **URL:** https://arxiv.org/abs/2411.00640 (v1, 01 Nov 2024, HTML read)
- **Venue:** arXiv preprint (stat.AP)
- **Date accessed:** 2026-10-12
- **Source type:** methods guidance
- **Credibility 4 / Recency 4 / Bias 4**

## Verbatim recommendations
"Our specific recommendations to researchers include: 1. Computing standard errors of the mean using the Central Limit Theorem 2. When questions are drawn in related groups, computing clustered standard errors 3. Reducing variance by resampling answers and by analyzing next-token probabilities 4. When two models are being compared, conducting statistical inference on the question-level paired differences, rather than the population-level summary statistics 5. Using power analysis to determine whether an eval (or a random subsample) is capable of testing a hypothesis of interest"
- Section title: "Don't touch the thermostat!" (on temperature).

## Relevance to our paper
Clustering = population/game is the unit (we bootstrap populations: good). Power analysis is the gap: 10 populations per LLM condition needs a stated minimum detectable effect.
