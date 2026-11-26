# Findings: Melting Pot (1.0, 2.0) and Concordia

Sub-agent 1, deep-research run of 14 Oct 2026. The numbers in square brackets point to files in `../sources/` (01–19). Code was read from shallow clones and never installed or run: Melting Pot at commit `bc456f6` (13 Oct 2026), Concordia at `c18c968` (13 Oct 2026, v2.4.0), and the Lab2D README and licence at `f6de2d6`.

## Short answer

- **How the settings were chosen.** Melting Pot's commons settings are hand-set constants. They come with no numeric justification in the code or in any of the four papers. The only rationale given is qualitative: the mechanism comes from Janssen et al. (2010), a human-lab experiment, adapted by Perolat et al. (2017). Maps are drawn so that one agent cannot empty the commons but a group can. Each paper then checks after the fact that the game is a dilemma, using empirical Schelling diagrams. The values differ in every paper (table below), and the shipped Open code does not match its own papers. This supports H2.
- **The intended evaluation.** Train a focal population on the substrate. Then test it zero-shot in held-out scenarios, against background bots it has never met, in both resident and visitor modes. Score focal per-capita return, min-max normalised between a random agent and an exploiter trained on that scenario. Report background per-capita return and inequality as secondary metrics. Both contests kept this structure, and the Melting Pot contest's own post-mortem says the focal-only score was gamed.
- **Concordia.** Its Game Master (GM) already acts as a central authority over a grounded state. Concordia has a GovSim-style fishery with elected leaders, but no audits or enforced fines exist anywhere in its code. There is no Melting Pot or grid bridge.

## 1. How a Melting Pot substrate is built

- **Engine.** The engine is DeepMind Lab2D, with a C++ core, Lua game logic and a Python `dm_env` API, all under Apache-2.0 [06]. The world is a grid with any number of named layers, and pieces have states and orientations [04].
- **Building a substrate.** A substrate is a Python config that returns a dictionary [01, 03, 04]:
  - an **ASCII map**;
  - a `charPrefabMap` that maps each character to one or more **prefabs**;
  - **prefabs**, which are GameObject templates. Every GameObject has a `StateManager` and a `Transform`, plus components such as `Appearance`, `Edible`, `DensityRegrow`, `Zapper` and `Cleaner`;
  - a **scene** object for global logic such as `StochasticIntervalEpisodeEnding`, `DirtSpawner` and `RiverMonitor`;
  - one **avatar** object per player.
  Component logic is written in Lua (`meltingpot/lua/levels/<substrate>/components.lua` plus a shared `component_library.lua` and `avatar_library.lua`) [04].
- **Actions.** Actions are discrete and egocentric. Commons Harvest has 8: no-op, forward, back, step left, step right, turn left, turn right, zap. Clean Up adds a clean action, for 9 [01, 03].
- **Observations.** Each player sees an RGB window of 88×88×3 pixels: 11×11 cells of 8-pixel sprites, 9 cells ahead, 1 behind and 5 to each side. There is also a `READY_TO_SHOOT` flag. Clean Up adds `NUM_OTHERS_WHO_CLEANED_THIS_STEP`, which the puppet bots read. `WORLD.RGB` is for debugging only [01, 03, 04].
- **Rewards.** Any component may call `Avatar:addReward(amount)` [04]. In the commons substrates the only reward is +1 per apple eaten. Zapping has a 0 cost and a 0 penalty, so a zap costs the target only the time it is out of the game [01, 03].
- **What this means for an overseer.** An audit-and-fine overseer would be a new scene-level Lua component that reads events or state and calls `addReward` with a negative amount. The engine allows this, but it means writing Lua inside a CPU-bound C++ engine [04, 06].

## 2. Commons settings, and how they were chosen

| Setting | Perolat 2017 [09] | Hughes 2018 "Harvest" [10] | MP papers (v1 and v2) [07, 08] | MP code: `commons_harvest__open` [01] | MP code: `__closed` / `__partnership` [02] |
|---|---|---|---|---|---|
| Regrowth by apples within radius 2 (0 / 1 / 2 / ≥3) | 0 / 0.01 / 0.01 / 0.05 (3–4), 0.1 (>4) | 0 / 0.005 / 0.02 / 0.05 (ℓ1 radius) | 0 / 0.001 / 0.005 / 0.025 | **0 / 0.0025 / 0.005 / 0.025** | 0 / 0.001 / 0.005 / 0.025 |
| Sanction | time-out beam, 25 steps | **fine beam: −1 to user, −50 to target** | zap removes the player | zap: 4 frames out, cooldown 2, length 3, radius 1 | zap: 100 frames out, cooldown 1, length 4 |
| Episode | 1000 steps | 1000 steps | stochastic | min 1000, then a 0.15 chance of ending per 100 frames, cap 5000 | same |
| Players | up to 12 | 5 | v1 Open: 16; v2: 7 | 7 | 7 |
| Map | several, with an "open" map | — | — | 24×18 cells, 64 apple cells | walled rooms, 64 apple cells |

**Clean Up** in the code [03]:
- Each potential apple grows with probability 0.05 × (1 − dirt fraction / 0.4) per step, which is zero once 40% of the river is dirty.
- Dirt spawns with probability 0.5 per step, starting after 50 frames.
- A zap removes the target for 50 frames, with a cooldown of 10.
- The clean beam has length 3 and a cooldown of 2.
- There are 7 players on a 30×21 map. The episode runs at least 1000 frames, then ends with a 0.2 chance per 100 frames, up to 5000.

Hughes 2018's Cleanup [10] had the same 0.5 waste rate and 40% saturation point, with apples spawning at 0.125·x, plus the fine beam (−1 to the user, −50 to the target).

**How the values were justified** (searched in code comments, the CHANGELOG and all four papers):
- **Code:** nothing beyond citations. The docstring says only that the mechanism "was first described in Janssen et al (2010) and adapted … in Perolat et al (2017)" [01]. A grep for tuned, calibrated, chosen or Schelling in the Melting Pot repo finds nothing. The one setting with a stated reason is random episode ending, which avoids "trivial backwards induction" [04].
- **Perolat 2017:** the map was "designed so that a single agent, acting alone cannot remove all the apples but with several agents harvesting simultaneously it becomes relatively easy" [09]. The game is "quite similar to the dynamic game used in human behavioral experiments". The human-lab fine was **replaced on purpose** by a time-out beam [09]. Incentives were analysed with Schelling diagrams built during training (empirical game-theoretic analysis) [09].
- **Hughes 2018:** the dilemma is validated empirically. Some agents are forced to cooperate or defect by changing the environment, and Schelling diagrams are plotted. The paper says it is "not feasible to analytically determine cooperating and defecting policies" [10]. No source is given for the rates.
- **Melting Pot 1.0 and 2.0:** they state the rates and cite Perolat and Janssen, nothing more. The only "tuning" reported is of agent hyper-parameters, explicitly not using generalisation scores [07, 08].
- **Code does not match the papers:** the Open file ships a one-neighbour regrowth rate of 0.0025, while both papers print 0.001 [01, 07, 08]. The closed and partnership files use 0.001 [02]. No changelog entry explains the change.
- **Conclusion for H2:** confirmed for this lineage. Settings are hand-set to produce the qualitative dilemma (one agent cannot deplete the commons, a group can), then checked with Schelling diagrams. They are not taken from ecological data. The only link to human data is that the mechanism, not the numbers, comes from Janssen et al.'s lab foraging game (Agent 4 covers Janssen).

## 3. Scenarios, background bots and the evaluation protocol

- **Scenarios.** A scenario is a substrate plus a held-out background population plus a focal/background mask (`is_focal`) [05]. Melting Pot 2.0 has about 312 scenario configs across more than 50 substrates. The commons ones are Open 2, Closed 4, Partnership 6 and Clean Up 23 [05].
  - Example: `commons_harvest__open_0` has 5 focal players against 2 bots that zap and harvest unsustainably [05].
  - **Resident mode** means focal players outnumber the background. **Visitor mode** is the reverse [08].
  - **Universalization mode** fills every slot with copies of one focal policy [07].
- **How the bots are built** [07, 08, 05]:
  - **Saved models.** These are TensorFlow SavedModel RL policies, mostly trained with ACB, sometimes with **pseudorewards** added at training time. Examples: a negative reward for zapping produces "pacifists", and a −10 tile on the room midline produces "good partners" in Partnership [02, 08].
  - **Puppets.** A goal-conditioned FeUdal-style network is trained to follow goals such as EAT, CLEAN and SANCTION. A hand-written Python "puppeteer" then switches goals according to events, for example a reciprocator that cleans only when others clean [05, 07, 08].
  - **Quality control.** Candidate bots play "10–30 episodes" with fixed familiar and unfamiliar bots. Bots that do not do what they were trained for are rejected [07].
- **Score** [08]:
  - The headline score is **focal per-capita return**: the mean return over focal players only.
  - It is **min-max normalised** per scenario between a random agent (0) and an exploiter (1), an RL agent trained directly on that test scenario. Scenarios are then averaged per substrate.
  - Admitted problems: in Clean Up SC1 and Commons Harvest Closed SC3 the exploiter was no better than random, and the score is "insensitive to the distribution of rewards over players" [08].
  - Proposed secondary metrics: **background per-capita return**, to catch "negative externalities", and the **inequality** of background returns, following Perolat [08].
  - Learning curves are averaged over 3 independent runs [08].
- **Melting Pot Contest, NeurIPS 2023** [11, 19]:
  - Substrates: Prisoner's Dilemma in the Matrix Arena (8 players), Clean Up (7), Allelopathic Harvest (16) and Territory Rooms (9).
  - Phases: 22 public validation scenarios, then 12 sampled held-out scenarios, then all 51 held-out scenarios. Ranking was by mean focal per-capita return, normalised by the Melting Pot 2.0 baselines. More than 600 participants in more than 100 teams.
  - Lessons from the post-mortem: high-ranking teams had low background returns. Teams excelled on a few substrates. Rule-based agents learned to kill background agents and share territory among themselves.
  - Organisers' recommendations: rank per substrate, and use "multiple social welfare metrics".

## 4. Concordia

- **Architecture** [12, 13]:
  - Everything is an *entity* built from *components*.
  - An *engine* (sequential, simultaneous or asynchronous) asks player entities for actions, and asks GM entities for observations and to resolve those actions.
  - Agents act in natural language. The GM turns each attempt into an *event statement*, updates *grounded variables*, emits observations and settles conflicts.
  - GM components include event resolution, inventory (a "grounded inventory tracking amounts of items in python"), payoff matrix, scene tracker, terminate, next-acting and questionnaires.
  - GM components may hold non-LLM logic: "ODE, graphical models, finite state machines" [12].
- **How actions are resolved:** an LLM call conditioned on the GM's components (world state) and the attempted action. A simultaneous engine collects all actions, then resolves them [12, 13].
- **Melting Pot or grid bridge:** none. The current repo contains no Melting Pot or Lab2D code; the only match is `/lab2d/` in `.gitignore` [13]. The only Melting Pot-to-LLM bridge found is outside both projects: Mosquera et al. 2024 [16]. They needed high-level macros ("go to position (x, y)"), a hand-written text observation generator, and bots slowed to one move per two LLM moves. Runs cost about $1–16 each with GPT-3.5/GPT-4.
- **Concordia Contest, NeurIPS 2024** [15, 19]:
  - Five text substrates: Reality Show, Pub Coordination, Haggling, Labor Collective Action and State Formation. None is a renewable-resource commons.
  - Melting Pot-style resident and visitor scenarios with held-out background agents designed "intuitively" (naive altruism, stubborn, conditional cooperators).
  - One fixed LLM, Gemma-2-9B-it, chosen for cost. Ranking by Elo, checked with maximal lotteries, Copeland and ranked pairs.
  - Main finding: agents were weakest at "persuasion and norm enforcement" [15].
  - Results were published at NeurIPS 2025 D&B (arXiv:2512.03318).
- **Cost per step:** not documented. From reading the code, the default `basic` agent prefab makes about 4 LLM calls per action (three self-questions plus the act call), and the GM adds at least one resolution call per event [13]. A `CallLimitLanguageModel` wrapper defaults to 1200 calls [13]. This is an estimate, not a measurement.
- **Could a GM act as the overseer?** Yes, very naturally. The GM already holds the true state and decides outcomes, so audits (which observations to reveal or check), fines (inventory changes) and allowances (limits enforced at resolution) map onto GM components [12, 13]. The resource stock and fines can be plain Python, so an LLM is needed only where judgement is the point.
- **Existing institutions or sanctions in Concordia:**
  - `examples/resource_dilemma` (Pasture, Irrigation, Network, Fishery) has "standard" and "election" modes. The fishery stock doubles each cycle up to a capacity of 120, starting at 100, over 6 cycles. Leaders propose a "penalty schedule" [14]. This follows Faulkner et al. 2026, where elected leadership raised social welfare by 55.4% [17].
  - **The penalties exist only in prompt text.** No GM code detects violations or applies fines; a grep for penalty, violat, enforc and sanction finds only prompt strings [14].
  - Other examples (labor strike, state formation, general store "crime and punishment") involve institutions only in their narratives [13, 15].
  - An enforced audit-and-fine overseer in Concordia would therefore be new.

## 5. Criticism of Melting Pot

- **Searches.** Targeted searches found no published paper that criticises Melting Pot as "toy", hand-tuned or externally invalid [18; search log].
- **Most of the criticism comes from the authors themselves:**
  - normalisation breaks where exploiters do no better than random, or where a random agent's score is extreme (Coins) [08];
  - the per-capita score ignores how reward is distributed [08];
  - focal-only scoring was gamed in the contest, and the organisers recommend several welfare metrics [11].
- **External critiques:**
  - fixed background agents (Molten Pot, ICML 2026 workshop) [18];
  - compute cost and speed, which motivated the SocialJax JAX ports. Those sources belong to Agent 2.
- **LLM-specific:** using LLMs in Lab2D substrates needs non-standard macros and text interfaces, which breaks comparability with standard Melting Pot scores [16].
- **Inference (not from a source):** the settings differ across Perolat, Hughes and Melting Pot, and the code differs from the papers [01, 07, 09, 10]. So a reviewer cannot treat any specific number as "the" literature value. Results in the literature are claimed to hold qualitatively, not at particular parameter values.

## 6. Licences

- Melting Pot: Apache-2.0 [01, 05]. Bots ship as model assets (SavedModels) downloaded by the package; their licence was not separately checked.
- DeepMind Lab2D: Apache-2.0 [06].
- Concordia: Apache-2.0 [13].

## What this means for Ameer's choices (inference, for the synthesis)

1. Ameer's Fishery/Forest/River settings do not need to match Melting Pot's numbers, because the literature has no fixed numbers to match. What the literature does consistently is: (a) state the qualitative design principle (one agent cannot deplete the resource, a group can); (b) show a Schelling diagram or empirical game-theoretic analysis that confirms the dilemma; and (c) evaluate against held-out background populations in resident and visitor modes, reporting focal, background and overall welfare.
2. Hughes 2018's fine (−1 to the user, −50 to the target) is the cited gridworld precedent for fines. Perolat 2017 explicitly dropped fines for time-outs.
3. Concordia's GM is the closest existing "overseer slot", and its resource-dilemma example is GovSim-like with election-only governance. An enforced audit-and-fine GM component is an open niche.
4. Using Lab2D with LLM agents means building macros, a text renderer and bot pacing yourself [16]. This is consistent with H4.
