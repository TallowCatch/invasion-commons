# 31. Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas

- **Authors:** Víctor Gallego
- **Year / venue:** 2026, arXiv:2603.19453 (v1 19 Mar 2026, v3 30 Jun 2026); arXiv comment "Accepted to NExT-Game 2026: New Frontiers in Game-Theoretic Learning, ICML 2026 Workshop. Camera-ready version". OpenSocialJax cites an earlier title, "Cooperation and exploitation in llm policy synthesis for sequential social dilemmas".
- **URL:** https://arxiv.org/html/2603.19453 ; code github.com/vicgalle/llm-policies-social-dilemmas (not inspected)
- **Source type:** academic (workshop)
- **Accessed:** 14 Oct 2026

## Verbatim quotes

1. "We propose an LLM harness that generates code-based policy functions for multi-agent environments, evaluates them with self-play, and refines them using feedback from previous iterations."
2. "we compare sparse feedback (scalar reward only) with dense feedback (reward plus social metrics: efficiency, equality, sustainability, peace)."
3. "In two Sequential Social Dilemmas (Gathering and Cleanup) and with two frontier LLMs (Claude Sonnet 4.6, Gemini 3.1 Pro), dense feedback improves over or matches sparse feedback on all metrics."
4. "Agents navigate a 2D gridworld and collect apples ( $+1$ reward). Apples respawn every 25 steps."

## Relevance

A third way to put LLMs into a 2D SSD: the LLM writes a Python policy once per iteration and the policy runs at simulator speed, so cost is per iteration, not per step. All agents share one program (homogeneous self-play) and the program sees full state with BFS helpers. An overseer study could use LLM-written policies as cheap, inspectable agent populations, but this removes per-step LLM decision-making, which is what an overseer of LLM agents is meant to test.
