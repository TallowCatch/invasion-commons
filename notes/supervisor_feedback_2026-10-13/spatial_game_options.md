# Spatial commons games for LLM agents with audits and fines: what exists, and three design options

Research note, 13 Oct 2026. I checked every external citation on the web: arXiv pages, the publisher or proceedings pages, and the GitHub source. Where a number comes from source code and not from a paper, I say so. Nothing in the local repo was changed.

---

## 1. SocialJax

**Citation (verified).** Zihao Guo, Shuqing Shi, Richard Willis, Tristan Tomilin, Joel Z. Leibo, Yali Du. *SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas.* arXiv:2503.14576. v1 18 Mar 2025, v2 19 May 2025, v3 17 Mar 2026. Published at ICLR 2026; the repo's BibTeX key is `guo2025socialjax`, booktitle ICLR 2026. Code: github.com/cooperativex/SocialJax, Apache-2.0. Main branch head as of today: `c64f1aa`, 29 Sep 2026.
- Note: Yali Du is a co-author here and on OpenSocialJax (section 3). PROJECT_HISTORY.md mentions "Yali's question". If that is the same person, the student has a direct line to the SocialJax authors.
- The repo's `paper_v2` and `paper_v4` bib files cite this paper as `socialjax2025` with a 2025 arXiv entry. That entry should be updated to ICLR 2026.

**Environments** (README list; core mechanic from the paper and source):

| Environment | Core mechanic |
|---|---|
| Coins | 2 players. Collecting any coin gives +1. Collecting a coin of the other player's colour costs that player −2. |
| Commons Harvest: Open | Apples regrow only near other apples, so a patch eaten bare is lost. Open map with only outer walls. |
| Commons Harvest: Closed | Same engine on a map with walled rooms, each with one entrance. An agent can guard a room by zapping intruders. |
| Commons Harvest: Partnership | Rooms have two entrances, so two agents must guard one room together. |
| Clean Up | Public good. A river fills with dirt, and apples grow only while the river is clean enough. Cleaning means leaving the orchard. |
| Territory | Claiming and defending territory. The README gives no Schelling-diagram proof for it. |
| Coop Mining | Iron gives +1 to one miner. Gold needs 2–4 players within 3 steps and gives +8 each. |
| Mushrooms | Each colour pays differently: red +1 to the eater; green +2 split among all; blue +3 to everyone except the eater; orange −0.2 to all. Eating a mushroom freezes the eater for 10–20 steps. |
| Gift Refinement | Gifting a token refines it and triples the quantity. The consume action turns tokens into reward. |
| Prisoner's Dilemma: Arena | Agents collect cooperate or defect resources. Zapping another agent plays a PD on their inventories, then both respawn frozen for 10–100 steps. |

The source tree also contains an `lb_foraging` environment that the README does not list.

**Commons Harvest details.** Source: `socialjax/environments/common_harvest/harvest_open.py`. In SocialJax, Closed and Partnership are the same engine run on different ASCII maps; the maps are set in `algorithms/*/config/*harvest_closed.yaml` and `*harvest_partnership.yaml`.
- **Regrowth.** Each empty apple cell counts the apples in the 12 cells within distance 2 (the 4 adjacent, the 4 at two steps, the 4 diagonal). The per-step regrowth chance is:
  - 0.025 with 3 or more nearby apples;
  - 0.005 with exactly 2;
  - 0.001 with exactly 1;
  - **0 with none.**

  The paper's appendix gives the same numbers.
- **Full depletion.** A patch eaten bare has zero regrowth for the rest of the episode. The paper says depleted patches stop respawning. This harm is irreversible within an episode, which is exactly the "environment destruction" mechanic.
- **Zapping.** There is one `zap_forward` action, and any agent can use it. In the code I read, a zapped agent is sent back to a player spawn point. I found no fixed timeout in this path: the state has a `freeze` field, but the zap path uses respawn. Neither side gains or loses reward from a zap. The only cost is lost time: walking back for the zapped agent, and aiming and chasing for the zapper. The paper says only that defenders in Closed and Partnership zap intruders, who respawn. **Read the zap as "sent home", not a long removal.** This differs from Melting Pot (below).
- **Observation.** Partial, egocentric, 11×11 cells (`obs_size=11`). The window turns with the agent's facing and is shifted forward. By default it is a CNN-style grid tensor (`cnn=True`), not text.
- **Actions.** 8 discrete actions: turn left, turn right, move left/right/up/down, stay, zap forward.
- **Episode length.** `num_inner_steps=1000` by default and in every shipped config.
- **Agents.** 7 by default and in the configs; configurable.
- **Map.** The Open map is 16×22.
- **Reward.**
  - The paper says +1 per apple.
  - The code default is `shared_rewards=True`: every agent gets the group's total apples that step.
  - With shared rewards off, the "individual" reward is multiplied by `num_agents` (×7). The local Clean Up adapter notes found the same ×7 in Clean Up: "pinned individual mode pays seven native units per apple".
  - Optional reward shaping: inequity aversion, SVO, and others.
  - **For an audit/fine study you must set individual rewards. The default shared reward removes the dilemma.**

**Clean Up details.** Source: `socialjax/environments/cleanup/clean_up.py`.
- **Apple growth.** Chance per empty orchard cell = 0.05 × clip((f − 0.4)/(0 − 0.4), ≤ 1), where f is the dirty share of river cells. So growth is 0.05 per step when the river is clean and falls linearly to 0 at 40% dirt.
- **Dirt.** No dirt appears for the first 50 steps. After that, each step has a 0.5 chance that one river cell gets dirty.
- **Clean beam.** Action 8, `zap_clean`, removes dirt in front of the agent.
- **Zap and episode.** The zap (action 7) works as in Harvest: the hit agent respawns. 9 actions, 7 agents, 1000 steps, 11×11 view.
- **Depletion.** No "patch death" here. Growth recovers if the river is cleaned again.
- **Local finding.** The native starting map already contains dirt, although the paper describes a clean river (CLEANUP_ADAPTER.md).

**Speed / JAX** (paper, Table 1; random actions on one A100 GPU and 14 CPU cores):

| Environment | Original, 1 env (steps/s) | SocialJax, 4096 JAX envs (steps/s) |
|---|---:|---:|
| Harvest | 3.7×10³ | 7.9×10⁵ |
| Clean Up | 2.7×10³ | 6.1×10⁵ |
| Coins | 1.2×10⁴ | 3.4×10⁶ |

The training pipeline claims at least 50× the real-time speed of Melting Pot's RLlib baselines.

**LLM / text agents.** SocialJax itself has none: no LLM, text or prompt files in the repo (262 files checked). The paper says LLMs were used only to polish the writing. The speed edge is irrelevant to LLM agents, because the LLM call, not the simulator, is the bottleneck. LLM agents were added later by the same group in OpenSocialJax (section 3).

---

## 2. Melting Pot

**Citations (verified).**
- Leibo, Duéñez-Guzmán, Vezhnevets, Agapiou, Sunehag, Koster, Matyas, Beattie, Mordatch, Graepel. *Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot.* ICML 2021, PMLR 139:6187–6199, arXiv:2107.06857. It has 21 substrates and "over 80" test scenarios.
- Agapiou, Vezhnevets, Duéñez-Guzmán et al., 17 authors, Leibo last. *Melting Pot 2.0.* arXiv:2211.13746, v1 24 Nov 2022, v6 31 Oct 2023. It adds asymmetric roles. The README says "over 50" substrates and "over 256" scenarios.
- The mechanics come from Perolat et al. 2017 for Harvest and Hughes et al. 2018 for Clean Up; both are cited in the substrate docstrings. Perolat et al. took the idea from the human lab experiments of Janssen, Holahan, Lee & Ostrom 2010 (*Science* 328:613–617).

**Commons Harvest** (source: `meltingpot/configs/substrates/commons_harvest__open.py`, `__closed.py`, `__partnership.py`):
- **Regrowth.** Depends on the uneaten apples within L2 radius 2, which is the same 12 cells as SocialJax. Chances for [0, 1, 2, ≥3] nearby apples:
  - Open: [0, 0.0025, 0.005, 0.025]
  - Closed and Partnership: [0, 0.001, 0.005, 0.025]

  SocialJax's 0.001 for one neighbour matches Closed, not Open.
- **Full depletion.** The docstring says a patch with no nearby apples "can be irrevocably lost if all apples in the patch are consumed", and that each agent has an incentive to eat the last apple before someone else does.
- **Zapper settings** (code). There is no reward penalty for being zapped and no reward for zapping in any variant.

  | Variant | Cooldown | Beam (length × radius) | Removed for |
  |---|---:|---:|---:|
  | Open | 2 steps | 3 × 1 | 4 frames |
  | Closed | 1 step | 4 × 1 | 100 frames |
  | Partnership | 1 step | 4 × 1 | 100 frames |

- **Observation.** 88×88 RGB pixels, i.e. 11×11 cells of 8 pixels. The view is not centred: the agent sees 9 rows ahead, 1 behind and 5 to each side. No text.
- **Actions.** 8: no-op, forward, backward, step left, step right, turn left, turn right, fire zap.
- **Episode length.** At least 1000 frames. After that, each 100-frame interval ends the episode with chance 0.15 (Harvest) or 0.2 (Clean Up). Hard cap 5000 frames.
- **Agents.** 7 by default.
- **Reward.** +1 per apple, individual.

**Clean Up** (`clean_up.py`):
- **Growth.** Same parameters as SocialJax: maximum 0.05, depletion threshold 0.4, restoration threshold 0.0; dirt spawn chance 0.5 per step after a 50-step delay.
- **Zap.** Cooldown 10, beam 3 × 1, removes the hit player for 50 frames. The docstring says "removes any player hit by it from the game for 50 steps".
- **Clean beam.** Cooldown 2, beam 3 × 1.
- **Actions and agents.** 9 actions (the Harvest 8 plus fire clean), 7 players.

**Evaluation protocol** (Leibo et al. 2021, verified on ar5iv):
- A *substrate* is the physical game.
- A *scenario* is a substrate plus a fixed *background population* of pre-trained bots. It is used only for testing, and agents never train in it.
- The *focal population* is the agents being tested. The test is zero-shot.
- **Resident mode:** focal players outnumber background players. **Visitor mode:** background players outnumber focal players.
- The main metric is the **focal per-capita return**: the mean return of the focal players only. It is min-max normalised to 0–1 across agents.

**Melting Pot Contest 2023** (verified):
- Run at NeurIPS 2023 by the Cooperative AI Foundation with Google DeepMind and MIT. Over 600 participants in 100+ teams; $10k in prizes.
- Development phase 31 Aug – 10 Nov 2023; evaluation 14–20 Nov.
- Four substrates: `allelopathic_harvest__open`, `clean_up`, `prisoners_dilemma_in_the_matrix__arena`, `territory__rooms`. These come from the contest repo, rstrivedi/Melting-Pot-Contest-2023.
- Ranking: mean focal per-capita return over held-out scenarios, in resident and visitor modes.
- Report: Trivedi, Khan, Clifton, Hammond, Duéñez-Guzmán et al. (25 authors). *Melting Pot Contest: Charting the Future of Generalized Cooperative Intelligence.* NeurIPS 2024, Datasets & Benchmarks track (per the PDF file name).

---

## 3. LLM agents in grid social dilemmas

| Paper (verified) | Game | Grid → text | Action interface and LLM calls | Cost / latency | Main finding |
|---|---|---|---|---|---|
| **Mosquera, Pinzón, Ríos, Fonseca, Giraldo, Quijano, Manrique.** *Can LLM-Augmented autonomous agents cooperate?, An evaluation of their cooperative capabilities through Melting Pot.* arXiv:2403.11381 (v1 18 Mar 2024, v2 19 Jun 2024). Final venue unclear: the v2 header names *Artificial Intelligence Journal*, an author profile says *IEEE Trans. AI*. | Melting Pot Commons Harvest. 3 LLM agents + 2 bots, 6 trees, ~100 rounds; also one-tree and agents-vs-bots variants. | **A symbol matrix failed**: GPT-3.5/4 could not reason spatially from it. They switched to sentences, one per object, with coordinates ("Observed an apple at position [9, 20]. This apple belongs to tree 6."), keeping only the 10 nearest objects. | High-level actions (`go to (x,y)`, `immobilize player X at (x,y)`, `explore`, `stay put`), turned into several low-level moves. GPT-3.5 for most modules, GPT-4 for the action. Reflection every ~3 rounds. Calls per step not reported. | Mean per simulation: **$0.78–$15.63 and 14–213 min** (Table 2). | Agents tended to "cooperate" mainly by not attacking ("LLMs appear to equate cooperation with refraining from attacking"). They went for the last apple of a tree at similar rates in every condition: poor grasp of depletion. When told one agent was selfish, agents with no assigned personality aimed "more than 80%" of their attacks at it. |
| **Chacon-Chamorro, Pinzón, Manrique, Giraldo, Quijano.** *Evaluating Cooperative Resilience in Multiagent Systems: A Comparison Between Humans and LLMs.* arXiv:2512.11689 (12 Dec 2025). | Melting Pot "tragedy of the commons" environment; humans and LLM agents, with and without communication. Shocks: a persistent unsustainable bot and random resource removal. | Not stated on the abstract page. | Not stated. | Not stated. | Humans with communication were the most resilient. Communication helped LLM agents, but they stayed below humans. |
| **Cross, Xiang, Bhatia, Yamins, Haber.** *Hypothetical Minds: Scaffolding Theory of Mind for Multi-Agent Tasks with Large Language Models.* ICLR 2025, arXiv:2407.07086. | Melting Pot: Running With Scissors (repeated and arena), Collaborative Cooking Asymmetric, PD in the Matrix Repeated. No commons game. | Pixels cut into 8×8 patches and labelled by matching hand-labelled reference patches. Text = **list of entity types with coordinates**, plus own position, facing and inventory. This format beat the alternatives. | The LLM writes 3–6 subgoal calls (`move_to`, `fire_at`, …). An **A\* planner** does the moves. The theory-of-mind module runs only after events, up to 3 + top_k (= 8) calls each time. | GPT-4: **~$10 and ~45 min per 1000-step episode** (Table 5). GPT-3.5: ~$1, 24 min. Llama-3-70B: 58 min. | Beats LLM-agent and RL baselines. Testing and refining hypotheses matters most. |
| **Guo, Fang, Leibo, Du.** *OpenSocialJax: Evaluating Social Adaptation in Sequential Social Dilemmas with Open-Ended Environmental Dynamics.* arXiv:2610.13710 (9 Oct 2026; four days old). | SocialJax Clean Up and Commons Harvest with hidden rules and procedural maps. 5-agent teams, 5 trials × up to 400 steps, 8 LLMs, plus a mixed-model arena. | **11×11 egocentric text grid**, north-up, rows and columns labelled, two-character cell codes (`ME`, `@B` for agent B, `AP` apple, `~~` water, `##` wall, `..` ground, `,,` orchard soil, waste as a colour letter + digit). The prompt lists each legal action with the cell it leads to ("1 south -> (11, 20): ground"). The agent keeps notes, rewrites them into lessons between trials, and is shown what it has seen others do. | **Primitive actions, 1 call per agent per step** (5 calls per step). Zapped agents are frozen 12 steps and not called. Reply is JSON with one action plus an optional note edit; an invalid reply means the agent stays put. Up to ~10,000 calls per episode. | No cost or latency reported in the HTML I read. | In OpenHarvest every team eventually ate the orchard bare ("No model keeps the orchard alive for a full trial"). In OpenCleanup cleaning fell on a few agents. In the mixed arena one cooperative agent (Claude Opus 5.5) kept the river clean at its own cost. |
| **Vezhnevets, Agapiou, Aharon, Ziv, Matyas, Duéñez-Guzmán, Cunningham, Osindero, Karmon, Leibo.** *Generative agent-based modeling with actions grounded in physical, social, or digital space using Concordia.* arXiv:2312.03664 (Dec 2023). | Concordia library. | **No grid.** A text "Game Master" LLM checks and resolves each agent's natural-language action. Space exists only as the Game Master describes it. | Several LLM calls per agent per step (components + memory), plus Game Master calls. | Not given on the abstract page. | It shows the other route to "space": narrate it and drop the grid. Proximity is then only as reliable as the Game Master's consistency. |
| **Park, O'Brien, Cai, Morris, Liang, Bernstein.** *Generative Agents: Interactive Simulacra of Human Behavior.* UIST 2023, arXiv:2304.03442. | Smallville: a 2D tile town with 25 agents. | The world is a **tree of areas and objects** in natural language (town → house → kitchen → stove), not a map. | Memory stream, reflection, planning: several calls per agent per step. | "costing thousands of dollars in token credits and taking multiple days to complete", for 25 agents over two game days. | Believable behaviour. Not a dilemma study. |

**Lessons for this project.**
1. Every LLM-in-a-grid paper either (a) turned the grid into an object list with coordinates and moved agents with a path planner, or (b) gave a small labelled text grid with one call per step (OpenSocialJax). Mosquera found raw symbol matrices did not work with GPT-3.5/4. OpenSocialJax shows that 2026 models can work from a labelled 11×11 grid.
2. The cost per game ranges from about $1 to $15. Wall-clock is 15 min to 3.5 h. That is 20–300× the current ~$0.05 per Fishery/Forest game.
3. Both commons papers found LLM agents **do not avoid eating the last apple**. Destruction is therefore a real, measurable failure in these games, not a corner case.
4. I found no paper that puts an **external overseer with random audits and fines** into a spatial grid with LLM agents. The nearest relatives:
   - Mosquera's LLM agents zap one another, which is peer sanction only.
   - The AI Economist (Zheng, Trott, Srinivasa, Parkes, Socher, *Science Advances* 8(18), 2022) has a planner setting taxes in a 2D gather-and-build world, but its agents are RL agents.
   - GovSim (Piatti, Jin, Kleiman-Weiner, Schölkopf, Sachan, Mihalcea, *Cooperate or Collapse*, NeurIPS 2024, arXiv:2404.16698; a copy is already in `external/GovSim`) is text-only, covering fishery, pasture and pollution.

   This looks like an open niche. The caveat is that OpenSocialJax, from the SocialJax group, appeared this month and is the obvious base, so the overseer is what would set new work apart.

---

## 4. Sanctioning, zapping and norms in these games (brief)

- **Perolat, Leibo, Zambaldi, Beattie, Tuyls, Graepel.** *A multi-agent reinforcement learning model of common-pool resource appropriation.* NeurIPS 2017, arXiv:1707.06600. This is the original Harvest.
  - A "time-out beam" removes the hit agent for 25 steps, and neither side gets reward or penalty. The paper contrasts this with fee-to-fine punishment in human experiments.
  - **Key result for identification:** conflict "lowering the effective population size" relieves pressure on the stock, so tagging raised sustainability while "peace" fell.
  - In other words, zapping protects the resource partly just by removing harvesters, whatever its deterrent effect.
- **Vinitsky, Köster, Agapiou, Duéñez-Guzmán, Vezhnevets, Leibo.** *A learning agent that acquires social norms from public sanctions in decentralized multi-agent settings.* *Collective Intelligence* 2(2), May 2023, doi:10.1177/26339137231162025 (arXiv:2106.09012).
  - The environments are Melting Pot Allelopathic Harvest and "Clean Up with Startup Problem".
  - Sanctions are zaps, and **a sanctioning opportunity exists only when the sanctioner is physically in beam range** of the target. Every sanction event is public.
  - Their Classifier Norm Model learns which behaviour is approved and punishes in line with the group. The resulting "bandwagon" effect helps agents settle on a norm.
- **Köster, Hadfield-Menell, Everett, Weidinger, Hadfield, Leibo.** *Spurious normativity enhances learning of compliance and enforcement behavior in artificial agents.* *PNAS* 119(3):e2106028118, Jan 2022. Earlier arXiv title: "Silly rules improve the capacity of agents to learn stable enforcement and compliance behaviors", arXiv:2001.09318.
  - Setup: berry foraging, where eating a taboo berry gets you punished (zapped) by others.
  - Adding a pointless taboo on a harmless berry made punishment and compliance faster to learn and more stable, because agents get more practice at enforcing.
  - Compliance was learned on top of others' enforcement.
- **Janssen, Holahan, Lee, Ostrom** (2010, *Science* 328:613–617, doi:10.1126/science.1183532). The human lab version of a spatial renewable commons. Costly punishment occurred but **did not raise harvest without communication**. This is a useful human baseline for "peer sanction in space".

---

## 5. Local repo (read only)

### (a) SocialJax Clean Up adapter and why it stopped

**Files.**
- Code: `/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/cleanup_oversight.py` (adapter) and `fishery_sim/cleanup_policies.py` (observation-limited heuristic policies).
- Scripts: `experiments/archive/harvest_2026q2/smoke_cleanup_oversight.py` and `admit_cleanup_policies.py`.
- Tests: `tests/test_cleanup_oversight.py` and `tests/test_cleanup_policy_admission.py`.
- Requirements: `requirements-cleanup.txt`, pinned to SocialJax commit `9df972d`, JAX 0.6.2 on CPU.
- Notes, all in `notes/research_review/`: `CLEANUP_ADAPTER.md`, `CLEANUP_POLICY_ADMISSION_PROTOCOL.md`, `CLEANUP_POLICY_ADMISSION_CLOSEOUT.md`, `CLEANUP_PARSER_REPAIR_PROTOCOL.md`, `CLEANUP_PARSER_REPAIR_CLOSEOUT.md`.
- Runs: `results/runs/cleanup_oversight_smoke_20260922/` and `results/runs/cleanup_policy_admission_v2.json`.
- History: `PROJECT_HISTORY.md`, the 22–23 Sep row and the game table.

**What it did.**
- The adapter wraps the unmodified native `Clean_up` with 7 agents, the native map, individual rewards and a 300-step horizon.
- "Safe" means productive capacity: dirt fraction ≤ 0.2, i.e. apple growth at least half of maximum.
- The only intervention is to replace a rejected proposal with `stay`.
- Snapshots are canonical JSON.
- The mechanics smoke passed: noop never recovered, and a privileged full-state script with 4 cleaners and 3 harvesters reached the safe set by step 19 and harvested 655 apples.

**Why it was stopped.** The policies could not supply the decisions an overseer must make.
1. **First admission attempt** (23 Sep). It crashed on observation decoding ("native observation does not identify a unique ego row") because another visible agent could occupy the candidate ego row. No result.
2. **Parser-repair attempt.** Both reset seeds *started outside the safe set*. Neither the "productive" nor the "free-rider" heuristic reached it or harvested a single apple in 180 steps. The productive one cleaned (106 and 85 clean actions) but too little to restore growth.
3. **Outcome.** There was no productive-versus-free-rider contrast and no safe-to-unsafe onset to judge, so the stop rule ended the sequence.
4. **Recorded lesson.** In history: "A third game is useful only if it supplies meaningful oversight decisions". In the parser-repair closeout: "repeated attempts until the gate passes would be selection on the result". Separately, the heuristic's first heading is a guess, because the native observation does not give absolute facing.

**Relevance.** This is direct local evidence that the hard part of a grid game is getting **competent, contrasting policies** from partial egocentric views, not writing the adapter. LLM agents could be those policies, but competence would need its own gate.

### (b) The existing Harvest environment

**Files.**
- Engine: `/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest.py` (816 lines). `fishery_sim/orchard.py` is a deprecated alias.
- Related: `fishery_sim/harvest_evolution.py`, `harvest_rl.py`, `harvest_llm_population.py`, `harvest_benchmarks.py`, `harvest_decision_cases.py`.
- Runners and plots: `experiments/archive/harvest_2026q2/` (`run_harvest_study.py`, `run_orchard_study.py`, `run_harvest_llm_turnover.py`, …).
- Added 11–16 Mar 2026 (PROJECT_HISTORY.md).

**What it is: a patch network on a ring. There is no movement and no grid.**
- Each of `n_agents` (default 6) owns one patch with health 0–20. Initial health is 14; growth is logistic, `regen_rate * r * (1 - r/20)`, plus Gaussian weather (sd 0.3).
- **Neighbours.** `_neighbors(i) = [(i-1)%n, (i+1)%n]`, a ring.
- **Neighbour effect.** Over-harvest above `sustainable_harvest_frac * max_harvest` (0.35 × 6 = 2.1) costs each neighbour's patch `neighbor_externality` (0.12, or 0.22 in the Forest prompt) per unit of excess.
- **Observation.** Own patch health, mean of the neighbours' patches, last credit received, and the government cap.
- **Messages.** Numeric only (`announced_restraint`, `requested_credit`), sent to the two neighbours. Toggled by `communication_enabled`.
- **Credit transfers.** An agent can offer credit (cap 1.0) to neighbours who announced restraint ≥ 0.6 and actually harvested ≤ 60% of the maximum. Toggled by `side_payments_enabled`.
- **Collapse.** A patch below 4 has failed. If at least half the patches fail for 5 rounds in a row, the episode ends with a −25 penalty.
- **Patch death is only nearly absorbing.** At health 0 the logistic growth is 0, but positive weather noise can restart it. So true irreversible destruction is *not* currently modelled.
- **Overseer hooks.** `GovernmentAgent` sets caps from mean patch health and can target agents whose neighbourhood is aggressive (with an option to expand targeting to neighbours). Local checks: `harvest_local_safety_mask`. Global check: `harvest_global_safe`.

### (c) The Forest game in L6

**Files.**
- Runner: `/Users/ameerfiras/invasion-commons/invasion-commons/experiments/oversight/run_l6_llm_forest.py` (342 lines).
- Setup: `experiments/oversight/claude_oversight_common.py::harvest_setup`, the `forest_co_management` preset with regrowth × 0.85 and **communication and side payments turned off**.
- Study notes: `notes/claude_audit_20261005/studies/L6_llm_forest_spillover/` (`protocol.md`, `results.md`). Analysis: `analyze_l6.py`.

**What it is.** Forest is the Harvest engine with its rules written for foresters.
- **Players.** Six plots on a ring, so ring distance from position 0 is 1, 2 or 3. Positions 1–5 are LLMs and position 0 is a programmed agent. 20 rounds, maximum cut 6, audit chance q = 1/6, fine F.
- **Each round, per LLM.** (1) A request call, which shows the own plot and **both neighbours' plot health**. (2) The joint reviewer scales all requests by one shared factor. (3) A cut call. An audit then fines any cut above the allowance.
- **"Visible" condition.** Adds the neighbours' last cuts and allowances.
- **Calls.** About **5 agents × 20 rounds × 2 calls = 200 calls**, plus comprehension checks. That is roughly $0.05 per game, or about $2 for the 40 gpt-oss games according to the protocol cost line.
- **L6 result.** One over-cutter raised forester over-taking by about 25 points all around the ring. Visibility added nothing. Post hoc, the effect ran through the reviewer's shared cut, not imitation.

**So the repo already has:** fixed positions, neighbour adjacency, local damage spillovers, local visibility of neighbours (as a treatment), numeric messaging and credit transfers (both off in Forest). It has **no movement, no line of sight, no peer sanction action, and no truly irreversible destruction.**

---

## 6. Design assessment: three options

Baseline for comparison. Current Forest/Fishery games cost about 200 calls and $0.05 per game. Over-taking is one number per round: cut minus allowance. The Becker threshold is F* = (gain from over-taking)/q, and it is known to the experimenter. L4 confirmed it out of sample: the prediction was 30 t; Nemotron stopped at 30 t and gpt-oss one step early, at 24 t.

### Option (i): patch network or ring with proximity (extend Forest/Harvest)

**Changes.** All of these are small edits to `harvest.py` or new wrappers; the first is a one-line change to the growth rule.
1. **True patch death.** Below health h₀ a plot never regrows (zero growth and no weather). This copies the Commons Harvest rule.
2. **Local auditing.** Only foresters within ring distance d, or the overseer's "camera" sector, can observe a cut, so audit chance depends on position.
3. **Peer report and peer sanction.** A neighbour can file a report, which triggers an audit, or pay c to fine a neighbour f.
4. **Movement-lite.** Each round an agent picks which plot to cut (its own or an adjacent one), so poaching becomes possible.

**New questions it answers.**
- Does a known chance of *irreversible* loss deter over-cutting better than a fine?
- Do agents cut more when they believe no one is watching locally? This is proximity-dependent detection.
- Do LLMs use costly peer sanctions, and do sanctions crowd out or complement central audits?
- Does poaching move to the plots that are least watched?

**Identification cost: low.**
- Over-taking, allowance and gain stay one number per round, so F* can still be written down and tested as in L4.
- If audit chance depends on position, q differs across agents but is still known.
- Peer sanction adds a strategic second actor, so report the sanctioners' costs separately.

**Calls.** 200–300 per game: one extra short call for a report or sanction decision, or folded into the cut call. Cost stays roughly $0.05–0.08.

**Risks.**
- Reviewers may say it is "still not spatial": a ring is a graph, not space.
- L6 H3 was null (visibility added nothing), which may repeat for local detection.
- Answer for the paper: frame it as a network commons, which is how many real forests, fisheries and irrigation systems are governed, and cite Janssen et al. 2010 for the spatial human version.

### Option (ii): a small text-rendered grid with movement (in-house, Python)

**Design.**
- A 12×12 to 15×15 grid with 3–5 apple patches, using the Melting Pot/SocialJax regrowth rule (12-cell neighbourhood; [0, 0.001, 0.005, 0.025]), so a patch eaten bare is lost.
- 4–6 LLM agents and 100–200 steps.
- Text as in OpenSocialJax (labelled 7×7 or 11×11 egocentric view) or as in Mosquera (an object list with coordinates).
- **Macro-actions** run by an A\* executor ("go to patch B", "harvest up to k apples here", "move away"), with the LLM called every 5–10 steps or at events.
- The overseer works on *harvest rounds*:
  - every K steps it reads each agent's harvest log, sets each agent's apple allowance for the next K steps, and audits at random;
  - optionally it sees only agents inside its view radius (audit chance depends on proximity);
  - fines are in apples.
- An optional zap action with a fixed timeout (Melting Pot uses 4–100 frames).

**New questions it answers.**
- Proximity as a channel: does being near the overseer or near other agents reduce over-taking?
- Local visibility of cheating: agents see others eat the last apple.
- **Environment destruction as irreversible harm**, measured as patches lost.
- **Zapping as peer sanction**, compared with central fines.
- Displacement: do audits push over-taking to patches the overseer cannot see?

**Identification cost: high.**
- The **Becker threshold is no longer a single number**. Gain from a violation depends on position, travel time, the apples left, competitors nearby and future regrowth, so F* has to be estimated per state. A sharp test like L4's 30 t prediction becomes a band.
- A "violation" must be defined by a rule, e.g. taking more than the allowance in a window, or taking the last apples of a patch. Separate rules for over-taking and destruction are needed.
- **Navigation competence confound.** Agents that fail to reach apples look restrained, and agents that bump into the last apple look greedy. A competence gate is needed, like the one Clean Up failed: with no oversight and an instruction to be sustainable, agents must harvest well and keep patches alive.
- **Zapping confound** (Perolat 2017). Removing agents protects the stock mechanically. Report agent-steps removed and add a matched control that removes agents at random.
- More randomness per game (spawn, regrowth, paths) means more games for the same power. Patches-lost counts are small integers with high variance.

**Calls.** About 5 agents × (150 steps / 5–10 steps per call) ≈ 75–150 calls, plus overseer and report calls, ≈ **100–300 calls per game**. Prompts are longer (the map is ~300–600 tokens). Cost ≈ $0.10–0.30 per game, about 2–6× now. Wall-clock ~10–30 min. With primitive actions and one call per step (OpenSocialJax style) it becomes 750–1,000 calls per game.

**Risks.**
- Building and validating a new engine.
- Reviewers will ask why not SocialJax. The answer: macro-actions plus round-level oversight need a different time scale.
- Results may mostly reflect prompt format: Mosquera's matrix-versus-object-list failure.

### Option (iii): full SocialJax or Melting Pot integration with LLM agents

**Design.**
- Native SocialJax `harvest_open`, `harvest_closed` or Clean Up, 7 agents and 1000 steps, with individual rewards (the code defaults to shared rewards, which must be changed).
- Text view as in OpenSocialJax.
- The overseer sits *outside* the environment. Every K steps it reads apple counts per agent, sets allowances and audits; fines are subtracted from return.
- Optionally use the shipped IPPO/MAPPO checkpoints, or the bots in Melting Pot scenarios, as a **background population**, and report focal per-capita return in resident and visitor modes.

**New questions it answers.**
- The same as (ii), plus comparability with the MARL literature and Schelling diagrams.
- Whether audits and fines transfer to the canonical benchmark.
- Mixed LLM–RL populations: does an overseer protect against RL agents that learned to over-harvest?

**Identification cost: highest.**
- Everything in (ii) applies, plus primitive actions on a 1000-step horizon.
- Zap is "sent home" in SocialJax but "removed for 4/100 frames" in Melting Pot, so results will depend on the platform.
- Apple rewards are ×7 in individual mode.
- A fine has no natural place in the action space, so oversight rounds must be added on top, which is non-native.
- The local Clean Up attempt already failed to produce a usable safe-versus-risky contrast from native resets: the native start state was already dirty, and the policies were weak.

**Calls.**
- Primitive actions: 7 × 1000 = **7,000 calls per episode**. OpenSocialJax used up to 5 × 400 × 5 = 10,000.
- With a macro-action executor, ~700–1,400.
- At this project's ~$0.00025 per call, that is ~$2–5 per primitive-action episode, so a 40-game cell costs $80–200. That is above the $60/month Ollama Pro credit.
- Sequential steps mean hours per game. Published comparisons: Hypothetical Minds about $10 and 45 min per 1000-step episode with GPT-4; Mosquera up to $15.63 and 213 min.

**Risks.**
- **Novelty.** OpenSocialJax (9 Oct 2026) already put LLM agents into SocialJax Harvest and Clean Up. New work must stand on the overseer, not the integration.
- JAX speed gives nothing to LLM runs.
- High engineering cost and a known failed admission gate.
- Also: a fixed horizon combined with "eat the last apple near the end" incentives (MP docstring); and dependence on a moving upstream (the SocialJax head changed after the pinned commit, including movement and respawn fixes).

### Bottom line

- **For the 20 Nov workshop and the January full paper, option (i) is the only one that keeps the clean identification** (a known F*, matched cells, ~$0.05 per game). Add *true patch death* plus *position-dependent detection* and *peer report or sanction*. This answers the irreversibility and proximity questions with the current machinery.
- **Option (ii) is the right next paper** if spatial movement is the point. Keep oversight at the round level and movement under macro-actions, so a per-round over-take number still exists. Gate on navigation competence before any oversight comparison.
- **Treat option (iii) as a collaboration or extension, not a main line.** Use OpenSocialJax's harness, and talk to its authors (the SocialJax group includes Yali Du). Cost and the earlier Clean Up admission failure argue against starting there.

---

## Sources (all checked 13 Oct 2026)

- SocialJax: arxiv.org/abs/2503.14576; arxiv.org/html/2503.14576v3; github.com/cooperativex/SocialJax (`socialjax/environments/common_harvest/harvest_open.py`, `cleanup/clean_up.py`, `algorithms/IPPO/config/*.yaml`)
- Melting Pot: arxiv.org/abs/2107.06857 (ar5iv HTML); proceedings.mlr.press/v139/leibo21a.html; arxiv.org/abs/2211.13746; github.com/google-deepmind/meltingpot (`meltingpot/configs/substrates/commons_harvest__{open,closed,partnership}.py`, `clean_up.py`, `utils/substrates/specs.py`)
- Melting Pot Contest: cooperativeai.com/contest/melting-pot-2023; github.com/rstrivedi/Melting-Pot-Contest-2023; mlanthology.org/neurips/2024/trivedi2024neurips-melting
- Mosquera et al.: arxiv.org/abs/2403.11381, arxiv.org/html/2403.11381v2
- Chacon-Chamorro et al.: arxiv.org/abs/2512.11689
- Hypothetical Minds: arxiv.org/abs/2407.07086; huggingface.co/papers/2407.07086.md; ICLR 2025 proceedings
- OpenSocialJax: arxiv.org/abs/2610.13710, arxiv.org/html/2610.13710
- Concordia: arxiv.org/abs/2312.03664
- Generative Agents: arxiv.org/abs/2304.03442 (HTML v2 for the cost quote)
- Perolat et al. 2017: arxiv.org/abs/1707.06600 (ar5iv); NeurIPS 2017 proceedings
- Vinitsky et al. 2023: arxiv.org/abs/2106.09012; doi:10.1177/26339137231162025
- Köster et al. 2022: doi:10.1073/pnas.2106028118; pmc.ncbi.nlm.nih.gov/articles/PMC8784148
- Janssen et al. 2010: doi:10.1126/science.1183532
- AI Economist: Zheng et al., *Science Advances* 8(18), 2022, pmc.ncbi.nlm.nih.gov/articles/PMC9067926
- GovSim: arxiv.org/abs/2404.16698; mlanthology.org/neurips/2024/piatti2024neurips-cooperate
