# Why Do AI Agents Break Rules? How Framing, Context, and Social Signals Shape Compliance

- **Authors:** Mika Okamoto, Ansel Kaplan Erol, Kutluhan Erol
- **URL:** https://arxiv.org/abs/2608.12323 (v2, 20 Aug 2026, HTML read); OpenReview forum VmhbSWpZp6
- **Venue:** "COLM 2026 WAB Poster" (workshop). Reviews not public via API.
- **Date accessed:** 2026-10-12
- **Source type:** preprint + workshop acceptance (primary)
- **Credibility 4 / Recency 5 / Bias 4**

## Verbatim
- Models: "We evaluate twelve instruction-tuned language models: ten open-weights candidates representing the realistic pool for enterprise fine-tuning, plus two leading closed-source models (Gemini 3 Flash and Grok 4.1 Fast)". Table: "Evaluated models (accessed via OpenRouter). Ten are open-weights; Gemini and Grok are API-only." Includes GPT-OSS-120B, Qwen 3.5 Flash, Gemma 4 31B, GLM 4.7 Flash, DeepSeek.
- Runs: "We complete each request 25 times." "single conditions are 25 trials and are not interpreted on their own."
- Stats: "At a single cell carries a 95% confidence interval roughly 20 pp wide, so we base claims about a model on rates pooled across conditions rather than on individual cells." "we report Wilson score intervals"; "Differences of proportions use Newcombe's hybrid-score interval".
- Temperature: "All other models run in default configurations at temperature 1.0".
- Enforcement varied, but audit likelihood and fine move together: "2. Small Fine : "Unlikely" audit, $2,400 fine. ... 3. Medium Fine : "Possible" audit, $4,800 fine. ... 4. Large Fine : "Likely" audit, $7,200 fine."
- Environment: one scenario ("enterprise procurement chatbots"), single-agent.

## Relevance
Single-turn single-agent design, 12 models, 25 trials per cell. Its deterrence axis confounds probability and size just like ours — so this confound is common in accepted (workshop) work, though a careful reviewer may still flag it.
