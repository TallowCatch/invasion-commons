# Plan: define the overseer as an agent, then test it (23 Nov 2026, revised)

**Goal.** Invent and justify a formulation of an overseer for sequential social dilemmas: an agent that is part of the multi-agent system, sees only part of it, and acts on other agents. Single or several overseers, of different capability. Agree on precise hypotheses before running anything else.

**Framing (Ameer, 23 Nov).** Game theory and multi-agent RL, in the scalable-oversight direction. The overseer is a player in the game, not an economics device. Economics terms (deterrence, expected fine, Becker) are used only where unavoidable, and then translated into the game's terms: what the agents learn to do, given what the overseer can see and do.

## The question each paper is read for

**How is the overseer (monitor, regulator, planner, sanctioner, protocol) modelled as an agent?**

- **Where it sits.** Is it a player in the game, outside it, or part of the environment?
- **What it observes.** Its observation function: full, partial, local, delayed, noisy, costly.
- **What it can do.** Its action space: inspect, sanction, reward, reshape payoffs, broadcast information, shut down.
- **How it decides.** Fixed rule, learned policy, LLM, solver.
- **How many there are.** One or several, and whether and how they communicate.
- **How capable it is relative to the agents.** The scalable-oversight question: a weaker overseer watching stronger agents.
- **The timing and solution concept.** Simultaneous play, leader-follower (Stackelberg), worst-case adversary, co-learning.
- **How the agents react.** Fixed, best-responding, learning, adversarial.
- **How the setup is evaluated.**
- **What we take, and what is missing for sequential social dilemmas.**

## Phase 0: pause new experiments (done)

- The 18 worst-case runs are held on CREATE.
- Running batches finish by themselves.
- Nothing new is designed until the hypotheses are agreed.

## Phase 1: the literature, read through that lens

**The overseer as an agent in a multi-agent system** (the core):
1. Griffin et al. 2024, *Games for AI Control* (read). The protocol vs the untrusted model as a partially observable stochastic game. The monitor is noisy and budgeted. There are two modes, honest and scheming.
2. Greenblatt et al. 2024, *AI Control*. Trusted monitoring: a weaker trusted model overseeing a stronger untrusted one.
3. Makins et al. 2026, *Multi-agent AI Control*. Per-agent monitors fail against attacks split across agents.
4. Engels et al. 2025, *Scaling laws for scalable oversight* (matrix, read). Overseer vs overseen capability. This is Yali's size and intelligence point.
5. Parisi et al. 2024, *Monitored MDPs*. Observing is an action, and it can cost something.
6. Sun et al. 2019, *A regulation enforcement solution for MARL* (read). A central agent that monitors and enforces in a multi-agent RL system.
7. Zheng et al. 2021, *The AI Economist*. A learned planner agent shaping other learners (two-level learning).
8. Yang et al. 2020, *LIO*, and Baumann et al. 2020, *Adaptive mechanism design*. Agents or a designer that learn to shape others' incentives.
9. Vinitsky et al. 2023 (sanctioning and norms) and Ye & Steinhardt 2026, *Norm enforcement*. Enforcement *by* agents inside the population.
10. Wang et al. 2019, *Deep RL for green security games*. A budgeted patroller against adaptive attackers, on a map.

**Decentralised structure** (Yali's two papers): Du et al. 2022, *DMPO*; Ma et al. 2024, *Nature MI*. How one formulation covers networks of local agents, with one measurable property that says when it works.

**Testbeds:** SocialJax and OpenSocialJax (Guo et al. 2026), BazaarBench (Wang et al. 2026), Willis et al. 2025, Leibo et al. 2017. What each testbed lets an overseer see and do.

**Background only** (cited, not built on): inspection games, Gans & Holden 2026, the economics of enforcement.

**Output:** `01_literature_map.md`.
- A table: rows are papers, columns are the questions above.
- A paragraph per core paper.
- A "design space" summary: every way the literature has modelled an overseer, laid out as choices.
- Unread matrix papers worth reading, flagged with one line each.

**Checkpoint:** you check it.

## Phase 2: define the overseer and choose the formulation to try

`02_overseer_definition.md`:

1. **The game.**
   - N agents in a partially observable Markov game, each with its own observations and rewards.
   - The social dilemma: individual and group interests conflict.
   - A rule that marks some actions as violations.
2. **The overseer as a player.**
   - **Its observation function:** what it can see of the state and of the agents' histories, and at what cost.
   - **Its actions:** whom to inspect (a budget of k agents per step), and what to do on detection (sanction, or other interventions).
   - **Its inspection capability:** how reliably an inspection reveals a violation, as a function of what it reads and who is inspecting. This is where LLM size and the patterned misses come in.
   - **Its policy:** a fixed rule, learned, or an LLM.
3. **Several overseers.**
   - M overseers, each observing a region or a subset of agents.
   - What they share and at what cost.
   - M = 1 is the central case.
4. **Timing and the agents' response.**
   - The overseer commits; agents learn a response (leader-follower).
   - Ordinary learners vs worst-case agents (who see the inspections, or target the inspector's blind spots), following the honest/scheming modes of AI-Control Games.
5. **Objectives.**
   - **The group outcome:** the commons' welfare and survival.
   - **The cost of mistakes to compliant agents.**
   - **The overseer's own resource use:** inspections, tokens, messages.

   Stated as a trade-off frontier, as with AI Control's safety and usefulness.
6. **Evaluation.**
   - The bounds: no overseer, an all-seeing overseer, worst-case agents.
   - What a scorecard reports.
7. **What makes a game a valid testbed.**
   - A social dilemma.
   - A rule the overseer can check from what it observes.
   - Measurable harm and welfare.
   - The properties that predict how hard oversight is (how visible violations are to an inspection, how local agents' movement and harm are).
8. **The formulation we choose to try, and its justification.**
   - Why this point in the design space.
   - What is new compared with each core paper (one line each).
   - What it predicts that the alternatives don't.

Format for every piece: a plain sentence, then the symbol, then a Harvest or BazaarBench example with real numbers, then its source. A notation table goes at the end, and there is a one-page version for Yali.

## Phase 3: hypotheses, before any experiment

`03_hypotheses.md`. Each hypothesis is written precisely, with:
- the part of the formulation it tests;
- the prediction;
- what result would refute it;
- the cheapest experiment that could;
- whether an existing result already answers it.

Existing experiments are mapped to hypotheses. Those that serve none are dropped. New runs (including the held worst-case and the separate-network reruns) happen only for hypotheses that need them.

## How I'll write: the `formulation-explainer` skill

- Words before symbols. One symbol per idea, defined once.
- Game-theory and MARL terms (observation function, policy, best response, leader-follower, partially observable stochastic game), each defined in one plain clause the first time.
- No economics jargon unless unavoidable, and then translated.
- A worked example with real numbers for every definition.
- Each piece tied to a checked source.
- "Assumed" kept separate from "derived".
- A notation table.

## Devil's advocate (short)

- **Steel-man.** Treating the overseer as an agent puts the work in MARL and scalable oversight, where SocialJax, AI Control and Yali's group already are. It also opens questions the economics view hides: what the overseer observes, several overseers, the capability gap.
- **Risk 1: "the overseer is an agent" is already common** (AI Economist planners, LIO, Sun et al. 2019, AI-Control Games). *Mitigation:* the novelty must come from the combination these lack: a budgeted, partially observing, possibly weaker inspector, in sequential social dilemmas with learning agents, single or several. Phase 2 §8 has to state this concretely.
- **Risk 2: dropping the economics view entirely loses a correct result.** The "expected sanction decides" finding holds in our data whatever we call it. *Mitigation:* keep the result, restated in game terms. It becomes the agents' best response to the overseer's inspection rate and sanction, not a borrowed theory.
- **Risk 3: the design space is huge** (observation, actions, number, capability, learning). *Mitigation:* Phase 2 picks one formulation to try and says why. The rest are listed as alternatives.
- **Reversibility:** fully reversible. These are documents, and experiments are only paused.

**Verdict:** go ahead.
