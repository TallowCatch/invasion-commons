# Norm Enforcement for AI Agents: Robustly Shaping Behavior in Multi-Agent Systems

- **Authors:** Yaowen Ye, Jacob Steinhardt (UC Berkeley)
- **URL:** https://arxiv.org/abs/2607.09766 (HTML read); OpenReview forums vcOopIy8DX and hkzXHguaMP
- **Venue:** Workshops only, per OpenReview: "COLM 2026 WAB Poster" and "AI4GOOD Workshop 2026 Spotlight" (ICML 2026 workshop). No main-track acceptance found. Workshop reviews not public (API search found only the submission note).
- **Date accessed:** 2026-10-12
- **Source type:** preprint + workshop acceptance (primary)
- **Credibility 4 / Recency 5 / Bias 4**

## Verbatim
- Environments: sections "Social media marketing", "Chatbot service competition", "Fishery commons management" (three environments).
- Models: "use Qwen3-NEXT-80B-A3B for both types of bad actors (we consider GPT-OSS-120B in Section D.4 )"; "We simulate persona-based users with gpt-5-nano"; "We implement the ground-truth norm violation verifier ... with gpt-5-mini"; "We use Qwen3-0.6B as the verifier, as large verifiers would be prohibitively expensive at scale."; validation judge "gpt-5.4 , medium reasoning effort".
- Replication model: "To check that mechanism abuse is not specific to Qwen3, we replicate Section 5 's setup with GPT-OSS-120B as the base model".
- Runs: "Table 1 : Mechanism performance across settings, averaged over 8 seeds."
- Stats: "Shaded regions and error bars represent 95% CI."
- Enforcement varied: mechanisms compared (naive reporting, centralized verification, RepVote, EscRepVote) and "we further ablate the penalty factor to isolate the contribution of asymmetric weighting."
- Limitations: "Our environments are only abstractions of real-world systems and consider binary good vs bad agents".

## Relevance
Closest enforcement paper: agent behaviour driven essentially by one open model family (Qwen3) with one open replication (GPT-OSS-120B), 8 seeds, workshop venues.
