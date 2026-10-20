# GovSim — NeurIPS 2024 official reviews and author responses (OpenReview)

- **Paper:** Piatti et al., "Cooperate or Collapse" (Submission 1329)
- **URL:** https://openreview.net/forum?id=0zWzJj6lO3 (web page behind a browser challenge; review text retrieved through the public OpenReview API search endpoint, api2.openreview.net/notes/search)
- **Venue:** NeurIPS 2024 poster (accepted)
- **Date accessed:** 2026-10-12
- **Source type:** public peer reviews (primary)
- **Credibility 5 / Recency 3 / Bias 3** (individual reviewers)

## Ratings seen
Reviewer 9ZQN rating 6; tYtP rating 7; dG5Q rating 6 (confidence 1); KPE1 rating 4 (confidence 5). Meta-review not retrieved.

## Verbatim criticisms
- KPE1 (model coverage): "I regard the primary flaw of this article as it does not test the performance of the GPT-4 model. ... GPT-4-turbo and GPT-4 are entirely different models"
- KPE1 (prompt sensitivity): "The paper does not demonstrate the generalizability of its conclusions through prompt sensitivity analysis (e.g., paraphrasing). We cannot be sure whether specific prompts influenced the simulation results or if there is any data leakage involved."
- KPE1 (mechanism): "the article does not profoundly demonstrate how to enhance cooperation or explain why cooperation cannot be achieved."
- KPE1 (environment generality): "the performance of the third scenario (River Pollution) is significantly worse than the other two. ... there is no data analysis or log analysis to support this claim."
- KPE1 (cherry-picking, quoted in author reply): "Merely presenting numerical results is insufficient, as potential cherry-picking in prompt settings could raise concerns among readers."
- dG5Q (environments): "Are these three scenarios too similar? they seems to be one task but with different name."
- 9ZQN (construct validity): "The study assumes that current LLMs can approximate human-like strategic reasoning and negotiation skills, which might be an overestimation of their current capabilities."
- tYtP (human baseline): "The paper would be strengthened if the authors incorporated human participants in the study"
- Praise on coverage — tYtP: "Experiments are conducted over most state-of-the-art LLMs." KPE1: "Experiments on different models are adequate."

## Note
No reviewer asked for more seeds than 5; no reviewer asked for varying a sanction parameter (the paper has none).
