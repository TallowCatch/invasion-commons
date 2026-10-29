# Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents (GovSim) — paper

- **Authors:** Giorgio Piatti, Zhijing Jin, Max Kleiman-Weiner, Bernhard Schölkopf, Mrinmaya Sachan, Rada Mihalcea
- **URL:** https://arxiv.org/abs/2404.16698 (HTML v4 read); proceedings: https://proceedings.neurips.cc/paper_files/paper/2024/file/ca9567d8ef6b2ea2da0d7eed57b933ee-Paper-Conference.pdf ; OpenReview forum 0zWzJj6lO3
- **Venue:** NeurIPS 2024 (main track, poster). OpenReview venue string: "NeurIPS 2024 poster".
- **Date accessed:** 2026-10-12
- **Source type:** peer-reviewed conference paper (primary)
- **Credibility 5 / Recency 3 / Bias 4** (authors' own report)

## Facts used (verbatim)
- Models: "We test our generative agents with fifteen different LLMs, including open-weights and closed-weights models."
  (The appendix list read by a fetch tool contains GPT-3.5, GPT-4, GPT-4-turbo, GPT-4o, Claude-3 Haiku/Sonnet/Opus, Llama-2 7B/13B/70B, Llama-3 8B/70B, Mistral 7B / 8x7B, Qwen 72B/110B — i.e. both closed frontier and open models. List not re-verified line by line.)
- Runs: "Each simulation was repeated with five random seeds. The average scores for each metric are presented in the main text, while the standard deviations are in the appendix."
- Decoding: "we ensure reproducibility by setting the text generation temperature to zero, i.e., greedy decoding."
- Stats: "We report the survival rate, the mean, and 95% confidence intervals of survival time (Surv.), total gain (Gain), efficiency (Eff.), equality (Eq.), and Over-usage." Also t-tests: "agents without communication tend to overuse the common resource by 22% (t-test; ...)" [p-value not rendered in HTML text].
- Environments: "we develop three common pool resource dilemmas" (fishery, pasture, pollution).
- Cost: "We spent in total 1500 USD across OpenAI API and Anthropic API."
- Enforcement: no sanction/fine parameter was varied (perturbations were communication on/off, universalization prompt, greedy newcomer). [From tool summary of full text; no verbatim sentence because it is an absence.]
