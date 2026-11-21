# 57 — Why Do AI Agents Break Rules? (re-read for model overlap with our LLM set)

- Authors: Mika Okamoto, Ansel Kaplan Erol, Kutluhan Erol
- URL: https://arxiv.org/abs/2608.12323 (HTML: https://arxiv.org/html/2608.12323)
- Posted: 2026-05-29 (v1); v2 2026-08-20. Comment: "Published at 2026 AAAI/ACM Conference on AI, Ethics, and Society and 2026 COLM Workshop on Agent Behavior"
- Accessed: 2026-10-12
- Type: peer-reviewed conference paper (AIES 2026)
- Credibility 4/5 · Recency 5/5 · Bias 3/5
- Already cited. New point: it tests GPT-OSS-120B, Nemotron 3 Super, Gemma 4 and Mistral Small, overlapping our LLM set, and some of its model-level results point the other way from ours.

## Verbatim quotes
- "GPT-OSS, Qwen 3.5 Flash, and Llama 4 Maverick comply at or above 90% under imperative framing across all enforcement levels and at or above 84% under informational framing in all but one configuration"
- "compliance drops relative to the no-enforcement baseline in half the models tested: ... Nemotron drops 31 pp (81% → 50%). This confirms the Gneezy-Rustichini effect (Gneezy and Rustichini 2000) in AI agents: specifying a penalty converts a prohibition into a cost-benefit calculation."
- "Grok 4.1 Fast, Gemma 4, and GLM 4.7 Flash lose 40–68 pp when imperative phrasing is replaced with informational phrasing alone"
- Design: "Small Fine : “Unlikely” audit, $2,400 fine. ... Medium Fine : “Possible” audit, $4,800 fine." (audit likelihood and fine rise together; they are not varied separately)

## Threat to claim 4: medium — a reviewer can point out that in Okamoto et al. GPT-OSS-120B is "rule-anchored" (complies regardless of fines), Gemma 4 is framing-fragile, and Nemotron becomes LESS compliant once a fine is named, whereas we report gpt-oss and Nemotron over-taking with no fine and being deterred by fines, and Gemma 4 rarely over-taking. Response: say plainly that model rankings depend on task and wording; our games have an explicit allowance and payoff, not a legal rule; report our no-fine over-take rate under each wording; avoid model-level "trait" language.
## Note for open question: Okamoto ties audit wording to fine size (both rise together), so probability vs severity is NOT separated there — our open question is still open in this paper.
