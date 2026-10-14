# 30. Incentives to Build Houses, Trade Houses, or Trade House Building Skills in Simulated Worlds under Various Governing Systems or Institutions: Comparing MARL to Generative Agent-based Model

- **Authors:** Aslan S. Dizaji (AutocurriculaLab)
- **Year / venue:** 2024, arXiv:2411.17724 (21 Nov 2024), preprint. arXiv comment notes "substantial text overlap with arXiv:2108.02755, arXiv:2312.03664 by other authors".
- **URL:** https://arxiv.org/html/2411.17724 ; code github.com/aslansd/modified-ai-economist-gabm and github.com/aslansd/modified-concordia-marl (not inspected)
- **Source type:** academic (preprint)
- **Accessed:** 14 Oct 2026

## Verbatim quotes

1. "Here, in a curiosity-driven project, I simulate two somewhat similar worlds using multi-agent reinforcement learning (MARL) framework of the AI-Economist and generative agent-based model (GABM) framework of the Concordia."
2. "a special agent called game master (which its role is similar to the central planner in the AI-Economist) translates their actions into appropriate implementations."
3. On the AI Economist planner: "The planner observations include each agent’s endowment but not build-skill level. The planner does not observe the spatial state of the world."
4. Limitation: "the number of simulations for each set of input parameters of the extended AI-Economist and the extended Concordia is one"

## Relevance

The only work I found that pairs the AI Economist with LLM agents, and it does so by **switching to Concordia** (text, no grid) for the LLM side; the Foundation grid stays RL-only. One run per setting, so weak evidence. Confirms nobody has yet run LLM agents or an LLM planner inside Foundation's 2D world.
