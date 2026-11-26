# 32. Training Small LLMs as Spatial Multi-Agent Policies

- **Authors:** Yi Mao, Andrew Perrault
- **Year / venue:** 2026, arXiv:2608.01425 (v1 2 Aug 2026), preprint
- **URL:** https://arxiv.org/abs/2608.01425
- **Source type:** academic (preprint)
- **Accessed:** 14 Oct 2026 (abstract page only; full text not read)

## Verbatim quotes (abstract)

1. "We take up both threads in spatial cooperative games, where small frozen LLMs prompted with low-level actions fail outright, earning zero reward."
2. "we equip each game with a library of symbolic \emph{options}: typed, state-feasible, short-horizon behaviors executed by a symbolic planner."
3. "Each library is drafted by a frontier coding model from the game's source code"
4. "Reward alone is thus an unreliable readout of cooperation; behavioral evaluation must sit alongside it."

## Relevance

Evidence that small (cheap, local) LLMs cannot play primitive-action 2D games, and that a macro-action (options) layer is needed for them. A web-search summary said the paper includes Commons Harvest with patches depleted by about step 200 for frozen bases; I did not verify this in the full text, so treat it as unconfirmed. Relevant to Ameer's budget (Ollama-hosted open models): primitive actions as in OpenSocialJax may only work with frontier models.
