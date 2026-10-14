# Can LLM-Augmented autonomous agents cooperate? An evaluation of their cooperative capabilities through Melting Pot

- **Authors:** Manuel Mosquera, Juan Sebastian Pinzón, Yesid Fonseca, Manuel Ríos, Nicanor Quijano, Luis Felipe Giraldo, Rubén Manrique
- **Year:** 2024
- **Venue / identifier:** arXiv:2403.11381 (preprint); code github.com/Cooperative-IA/CooperativeGPT
- **URL:** https://arxiv.org/abs/2403.11381
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. Macro-actions, not primitive moves:
   > "The LLM agents possess the capacity to execute high-level actions in each ... go to position (x, y), stay put, and explore (x, y)."
2. Bots run on a different clock:
   > "bots are in constant motion, executing a move for every two moves made by any of the agents (note that a high-level action typically comprises more than one movement)."
3. Grid-to-text problem:
   > "it proved challenging for LLMs like GPT-3.5 or GPT-4 to interpret and reason about the spatial information provided by the position of objects in the matrix."
4. Bots reused from the benchmark:
   > "These bots are part of scenario 0 of the commons harvest open scenario"
5. Cost (Appendix G, Table G.2): "Set 1 - All Coop with def 15.63(5.69)" US$ per simulation and "Set 2 - Agents vs. Bots 1.88(0.66)" (mean, SD).

## Relevance
This is the clearest existing Melting Pot-to-LLM bridge, built by the authors themselves; it is not part of Melting Pot or Concordia. It supports H4. To make LLMs usable in Commons Harvest, the authors had to (a) give them high-level navigation macros, (b) write a text observation generator, and (c) slow the bots down. Runs cost roughly $1–16 each with GPT-3.5/GPT-4. Results are not comparable with standard Melting Pot scores.
