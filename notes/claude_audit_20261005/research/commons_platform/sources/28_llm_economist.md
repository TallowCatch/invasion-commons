# 28. LLM Economist: Large Population Models and Mechanism Design in Multi-Agent Generative Simulacra

- **Authors:** Seth Karten, Wenzhe Li, Zihan Ding, Samuel Kleiner, Yu Bai, Chi Jin
- **Year / venue:** 2025, arXiv:2507.15815 (21 Jul 2025); listed on neurips.cc virtual pages for NeurIPS 2025 (I did not verify which track)
- **URL:** https://arxiv.org/html/2507.15815v1 ; code github.com/sethkarten/LLM-Economist (MIT per GitHub API, last push 11 Mar 2026)
- **Source type:** academic
- **Accessed:** 14 Oct 2026

## Verbatim quotes

1. "At the upper level, a planner agent employs in - context reinforcement learning to propose piecewise - linear marginal tax schedules anchored to the current U.S. federal brackets."
2. Interface: "Simulator state, history, and objectives are rendered as text; actions are JSON snippets parsed by the environment."
3. Planner cadence: "The Planner updates only after a "tax year" of worker adaptation, inducing a Stackelberg game whose equilibrium we solve via alternating ICRL."
4. Setting: "All experiments use meta-llama/Llama-3.1-8B-Instruct queried at temperature 0.7 hosted on a single H100."
5. Time-scale: "Very short tax years ( $K\!=\!5$ or $10$ ) leave the workforce too little time to adapt and stall below 65% of the Stackelberg optimal $\texttt{SWF}^{*}$ ."

## Relevance

An LLM planner over LLM workers, but with **no spatial world**: each worker only returns `{"LABOR": X}`. It is not a Foundation/Gather-Trade-Build port. Useful for Ameer as evidence that an LLM overseer needs a long enough period between policy updates (K = 128 steps here) for agents to respond, and as a planner-evaluation template (compare with Saez and U.S. federal baselines).
