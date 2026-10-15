# 22. OpenSocialJax: Evaluating Social Adaptation in Sequential Social Dilemmas with Open-Ended Environmental Dynamics

- **Authors:** Zihao Guo, Meng Fang, Joel Z. Leibo, Yali Du (KCL, Liverpool, Turing, Google DeepMind)
- **Year / venue:** 2026, arXiv:2610.13710v1 [cs.MA], 9 Oct 2026 (preprint; paper licence CC BY 4.0)
- **URL:** https://arxiv.org/html/2610.13710
- **Source type:** academic
- **Accessed:** 14 Oct 2026 (full HTML downloaded, text extracted)

## Verbatim quotes

1. Calls per step (Sec. 3.3): "At every step, each agent is queried independently with a fixed system prompt, which explains the rules and the reply format, and a user message built by the harness."
2. Reply (Sec. 3.3): "The model replies with a JSON object that specifies one action and at most one edit to the notes; an invalid reply leaves the agent in place for that step."
3. Protocol (Sec. 4.1): "Our primary evaluation uses homogeneous teams of five instances of the same model. For each environment, every model is evaluated on 3 seeds; each seed samples one latent configuration and defines one episode of five trials of at most 400 steps, and an OpenHarvest trial ends early once no apple remains."
4. Schelling with prompted roles (App. C.1): "We instantiate cooperating and defecting policies with GPT-6-Luna. Each agent receives one of two roles, stated as item 1.5 ( Your role ) of its system prompt" and "For $k=0,\dots,5$ cooperators, all five agents are played by GPT-6-Luna, and which agents are cooperators rotates across three seeds."
5. Role text (App. C.1, OpenHarvest defector): "1.5 YOUR ROLE : You care only about your own score . Your goal is to earn as many points as you can yourself ."
6. Result: "No model keeps the orchard alive for a full trial." and in the arena "One restrained agent cannot protect a shared resource from four agents that deplete it."

## Other facts from the paper (paraphrased)

- Hidden rules: OpenCleanup has P(16,7) = 57,657,600 tool rule sets; OpenHarvest has 6^10 = 60,466,176 colour/ripeness reward-and-regrowth rule sets.
- Metrics: mean per-agent return per trial; hidden rules known; effective number of cleaners 1/Σ s_i²; orchard lifetime (step the last apple is eaten); Gini of returns; a scripted greedy baseline.
- Zapped agents are frozen 12 steps and the model is not called (App. A).
- Models: GPT-6-Luna, GPT-6-Sol, Claude Opus 5.5 (API); Qwen3.8-27B, Qwen3.5-4B, Gemma 4 31B (vLLM); DeepSeek-V4.1-Flash, GLM-5.3-Flash (API). Table 1 lists thinking and temperature settings.
- No cost or latency figures in the paper; these are in the repo README (source 23). No communication between agents.

## Relevance

This is the closest existing harness to what Ameer needs: LLM agents taking primitive actions in a 2D Commons Harvest from an 11×11 labelled text view, with individual rewards, harness-kept memory, and an LLM-based Schelling diagram (role prompts rather than trained policies). It has no overseer, allowances, audits or fines. Its Schelling method (prompted cooperator/defector roles, k = 0..N, rotated seats, 3 seeds) can be copied directly for an LLM commons game.
