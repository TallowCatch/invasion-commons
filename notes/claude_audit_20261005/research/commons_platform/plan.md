# Plan: which commons platform, and which studies, for scalable oversight of AI agents

Deep-research run, started 14 October 2026. Depth: **deep**. Genre: **decision** (with a landscape section and a study list).

## 1. Reframe

**Ameer's question.** How are existing sequential commons games built, and how are their settings chosen? Are my settings grounded? Should I build a 2D moving-agent game on an existing platform, or my own? Which is more novel? How was Richard Willis's work done? Give the full list of studies needed to evaluate an overseer, including an LLM overseer and scalable-oversight ideas, the way the literature evaluates such things.

**Underlying decision.** Which environment the next two to three years of the PhD are built on, and what evaluation protocol makes the results credible to a multi-agent RL and AI-safety audience. The decision is expensive to reverse: the engineering, the compute budget ($60/month of Ollama credit) and the papers all depend on it.

**Falsifiable hypotheses.**
- **H1 (grounding).** The current Fishery/Forest/River settings are grounded for the resource model, but not for the agent population or the evaluation protocol. The resource model is logistic growth and an MSY line, as in fisheries science. The SSD literature's conventions are Schelling diagrams, focal and background populations, and social-outcome metrics. *Refuted if* the SSD literature has no settled protocol, or if the current design already follows it.
- **H2 (how settings are chosen).** Existing SSD substrates choose parameters by hand-tuning so that a dilemma appears, checked with Schelling diagrams or empirical game-theoretic analysis, not from ecological data. *Refuted if* the substrates cite ecological or behavioural data for their parameters (for example, human-lab foraging games).
- **H3 (build vs extend).** Building the overseer on an existing, cited substrate (Melting Pot or Concordia, SocialJax or OpenSocialJax, Foundation/AI Economist, emergent_llm) is more evaluable and no less novel than a from-scratch engine. The reason is that the novelty is in the overseer and the oversight protocol, not the grid. *Refuted if* no existing substrate can host an external overseer with audits, fines and allowances at acceptable cost, or if reviewers value new environments as such (for example, benchmark tracks).
- **H4 (cost).** LLM agents choosing primitive actions in a 2D game cost too much for this budget. Published LLM-in-grid work uses macro-actions or text abstractions. *Refuted if* published systems run LLMs per primitive step cheaply.

## 2. Genre and building blocks

- **Landscape:** an architecture table of each platform: engine, how agents act, observation, parameters, how parameters are justified, evaluation, licence, LLM support, cost.
- **Decision:** options against criteria (novelty, evaluability, cost, citation value, engineering risk, fit to the feedback), with a recommendation.
- **Rundown:** Willis et al. method and code.
- **Grounding audit:** each current setting, its stated justification, and the literature analogue.
- **Study list:** an extensive, ordered list with question, design, evaluation protocol and cost.

## 3. Sourcing strategy

Channels: arXiv/HTML full text, proceedings, GitHub source code (cloned to the scratchpad, read only, never executed), the local repo, and the existing notes `literature_review_ai.md`, `spatial_game_options.md` and `literature_20260921/LITERATURE_AND_GAME_DECISION.md`.

Parallel sub-agents, each saving one file per source to `sources/` (number ranges in brackets):
1. **Melting Pot and Concordia** [01–19]: Lab2D substrate architecture, prefabs and components, how `commons_harvest` and `clean_up` parameters were set, scenarios and bots, the focal/background protocol, the Melting Pot contest, and Concordia's game master and any Concordia–Melting Pot bridge.
2. **SocialJax, OpenSocialJax, JaxMARL, Foundation/AI Economist** [20–39]: architecture, how parameters were chosen, Schelling diagrams, evaluation, LLM harness, licences; and the AI Economist's 2D Gather-Trade-Build world with a planner as an existing "overseer in a grid".
3. **Richard Willis's work** [40–59]: clone `willis-richard/emergent_llm`, read the code, the paper method and his other papers; give a rundown and say how it could be extended with an overseer.
4. **Text commons and the human-lab grounding** [60–79]:
   - GovSim architecture and parameters, SanctSim, CoopEval, Bell & Madisetti;
   - Perolat 2017, Leibo 2017, Hughes 2018;
   - Janssen et al. 2010 (*Science*) and the human spatial foraging CPR experiments, as the behavioural source of settings;
   - Schelling diagrams (Schelling 1973; Leibo 2017) and empirical game-theoretic analysis.
5. **How scalable oversight is evaluated** [80–99]: AI control (trusted and untrusted monitoring, audit budget, safety and usefulness, red and blue teams); weak-to-strong and sandwiching; debate; Engels et al. scaling laws; LLM monitors and judges; multi-agent oversight; how to transplant these into a commons.
6. **Local grounding audit** [L01–L19]: every Fishery/Forest/River/Harvest parameter in the code and its stated justification in notes, compared with the literature conventions found above.

**Opposition queries (for the adversarial pass).**
- Arguments that from-scratch environments are valued (benchmark and environment tracks).
- Criticisms of Melting Pot and SocialJax: hand-tuned, "toy", poor external validity.
- Evidence that LLM agents in grids are dominated by prompt format.
- Evidence that overseers in grids add little beyond text games.

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Fabricated or misread code facts | Read cloned source; quote file and line |
| Platforms moving fast (OpenSocialJax is 5 days old) | Record the commit hash and access date |
| Confirmation bias toward "extend an existing platform" | Run the opposition queries; adversarial pass |
| Cost estimates wrong | Tie each estimate to calls × tokens and the observed $0.00025 per call (L4) |
| Budget for the investigation | Six parallel agents, then one synthesis |

## 5. Stop criteria

- Every hypothesis is confirmed, refuted or marked under-determined, with at least three independent sources of different types where possible.
- Every row of the architecture table has a source file.
- The study list maps every feedback item to at least one study.

## Changelog

- 14 Oct 2026: plan written; sub-agents 1–6 launched.
- 14 Oct 2026: all six sub-agents returned (93 sources). Adversarial pass: Large Legislative Models and a 2026 search for LLM overseers in grid commons checked by hand. Report `2026-10-14_decision.md`, `sources.csv`, `refresh_targets.md` written.
