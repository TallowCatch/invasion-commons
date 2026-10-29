# Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia

- **Authors:** Alexander Sasha Vezhnevets, John P. Agapiou, Avia Aharon, Ron Ziv, Jayd Matyas, Edgar A. Duéñez-Guzmán, William A. Cunningham, Simon Osindero, Danny Karmon, Joel Z. Leibo
- **Year:** 2023
- **Venue / identifier:** arXiv:2312.03664 (tech report, Google DeepMind)
- **URL:** https://arxiv.org/abs/2312.03664
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. > "A special agent called the Game Master (GM), which was inspired by tabletop role-playing games, is responsible for simulating the environment where the agents interact."
2. > "Agents take actions by describing what they want to do in natural language. The GM then translates their actions into appropriate implementations."
3. GM duties include:
   > "Resolving what happens when actions submitted by multiple agents conflict with one another."
4. Grounded variables:
   > "The most important responsibility of the GM is to provide the grounding for particular experimental variables, which are defined on a per-experiment basis."
5. Non-LLM logic is allowed inside GM components:
   > "the components can have their internal logic written using any existing modelling tools (ODE, graphical models, finite state machines, etc.)"
6. Commons and institutions are a named use case (Sec. 4.3):
   > "With Concordia we will be able to investigate whether the demands of sharing a resource give rise to rules, laws and norms capable of governing that resource"

## Relevance
In Concordia, the GM is already a central authority. It keeps the true state, decides outcomes and sends observations, so an overseer that audits, fines or sets allowances maps onto it directly as a GM component. The resource stock and fines can be grounded Python variables, not LLM judgements, because the paper allows ODE and state-machine components. The paper names a "lobster fishery" with resource and political variables as an intended experiment.
