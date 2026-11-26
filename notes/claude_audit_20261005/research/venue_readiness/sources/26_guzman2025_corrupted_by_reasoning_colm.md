# Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games — paper + COLM 2025 reviews

- **Authors:** David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin
- **URL:** https://arxiv.org/abs/2506.23276 (HTML read); OpenReview forum https://openreview.net/forum?id=kH6LOHGjEl (via API search)
- **Venue:** COLM 2025 (main conference; venue string "COLM 2025")
- **Date accessed:** 2026-10-12
- **Source type:** peer-reviewed conference paper + public reviews (primary)
- **Credibility 5 / Recency 4 / Bias 3**

## Paper facts (verbatim)
- Models: "traditional models (DeepSeek-V3 ..., GPT-4o ..., GPT-4o-mini ..., and Llama-3.3-70B ...) and reasoning-focused models (o1-mini ..., o1-preview ..., and o3-mini variants ..., each tested at low, medium, and high reasoning settings)."
- Runs: "For each model configuration, we conducted 5 independent runs, except for high-cost models (o1-preview and o3-mini-high) where we performed a single run due to budget constraints."
- Stats: "Results represent averages across 5 independent runs"; "Contribution Mean ± Std"; reasoning-strategy CIs via "hierarchical bootstrapping, which accounts for the nested structure of our data (models within archetypes, runs within models)."
- Temperature: "For traditional models, we used a temperature setting of 1.0".
- One environment (public goods game with sanctioning institution, after Gürerk et al. 2006).

## Reviews (verbatim)
- Reviewer gBFf (rating 7): "The paper doesn't clearly specify how sensitive outcomes are to the way prompts and agent communication are designed, prompt phrasing could critically bias model behavior." Question: "How robust are the results to different phrasings of the public goods and sanctioning tasks? ... Would slightly different language alter cooperation rates?" Also: "the study does not provide side-by-side comparisons to real human experiments under the same setup."
- Authors' general response lists a shared concern: "Sensitivity to Experimental Setup (Game Parameters & Prompt Phrasing) (R-WHGr, R-cqML, R-gBFf)" — i.e. three reviewers raised it.
- Rebuttal added enforcement-parameter variation: "we conducted targeted experiments altering key game parameters (public good multiplier, Punishment Cost & Effect, Endowment) and testing an alternative prompt with a narrative framing on representative models (Llama-3.3-70B and o1-mini)." Table note: "Ablation results are from single simulation runs."
- Scope criticism (R-cqML, per author reply heading): "Limited Scope (Public Goods Games)"; reviewers asked "Do you plan to test the models in other cooperation scenarios".

## Relevance
Main-track acceptance with 5 runs per model, one game, ~7 models mixing closed frontier (GPT-4o, o1) and open. Prompt-wording sensitivity was the single most repeated reviewer request; punishment-parameter variation was added only in rebuttal, single runs.
