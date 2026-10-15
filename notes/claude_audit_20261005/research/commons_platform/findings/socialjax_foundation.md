# Findings: SocialJax, OpenSocialJax, JaxMARL, Foundation / AI Economist

Sub-agent 2 of the commons-platform deep research, 14 Oct 2026. Source numbers in [brackets] refer to files in `../sources/` (20–33). The earlier note `notes/supervisor_feedback_2026-10-13/spatial_game_options.md` already covers SocialJax game parameters, Melting Pot parameters and the LLM-in-grid table; this file does not repeat them. All repos were cloned shallow and read only; nothing was installed or run.

| Repo | Commit read | Licence | Status (GitHub API, 14 Oct 2026) |
|---|---|---|---|
| cooperativex/SocialJax | `c64f1aa` (29 Sep 2026) | Apache-2.0 | active |
| cooperativex/OpenSocialJax | `b5f59a3` (8 Oct 2026) | Apache-2.0 | active, 6 days old |
| FLAIROx/JaxMARL | `976aeb1` (10 Sep 2026) | Apache-2.0 | active |
| salesforce/ai-economist | `a84d5f3` (9 May 2022) | BSD-3-Clause | **archived** |

---

## 1. SocialJax architecture, and where an overseer would go

**How an environment is built** [21]
- **Base class.** Every game subclasses `MultiAgentEnv` (copied from the JaxMARL API [24]). The base `step()` calls the game's `step_env`, then auto-resets: when `dones["__all__"]` is true it swaps in a fresh state using `jax.lax.select`.
- **State is a small pytree.** It is a flax `@dataclass` of arrays: agent locations (row, col, facing), inventories, inner and outer step counters, the grid, apple cells, freeze flags and respawn locations.
- **Map.** Written as an ASCII list (`'A'` apple, `'P'` spawn, `'W'` wall, `'Q'` inside spawn). It is converted to an int16 grid, and the spawn and apple cell lists are fixed when the game is built. Closed and Partnership are the same class with a different `map_ASCII`, passed in through the config.
- **Step and reset are jitted closures** written inside `__init__` (`self.step_env = jax.jit(_step)`), not methods. Movement conflicts are settled by a shared module, `movement.py`.
- **Vectorisation lives in the trainer, not the game.** IPPO runs `jax.vmap(env.reset)` and `jax.vmap(env.step)` over `NUM_ENVS = 256` copies and rolls out with `jax.lax.scan` for 1000 steps. The whole `make_train` is jitted, and can be vmapped over seeds.
- **Rewards are computed inside `_step`.** An apple is eaten when an agent's new cell holds one. The reward mode then decides who gets paid:
  - shared mode (`shared_rewards=True`): every agent gets the team total, and `info["original_rewards"]` holds each agent's own apples;
  - individual mode: an apple pays `1 × num_agents` to the eater, and `info` is empty;
  - optional reshaping: inequity aversion, SVO, or "interest" mixing.
- **Config gotchas** [21]:
  - `ippo_cnn_harvest_open.yaml` defaults to `reward: common`.
  - The per-agent inventory `agent_invs` is int8. By my reading of the code (not run) it would wrap after 127 apples.
  - The README says results after `v1.1.0` may differ, because an agent-overlap bug was fixed.

**Adding a new game** [21]
1. Write `socialjax/environments/<name>/<name>.py` with a `State` dataclass, an ASCII map, and jitted `_step`/`reset` closures.
2. Add its `observation_space` and `action_space`.
3. Export it in `environments/__init__.py`, then add it to `REGISTERED_ENVS` and `make()` in `registration.py`.
4. Add a per-game trainer script and a Hydra yaml under `algorithms/<ALGO>/`.

**Adding an overseer layer (every K steps: read harvest, set allowances, audit, fine)**
- **RL agents (stays in JAX).** Write a wrapper like `LogWrapper` [21]:
  - Give it its own `@struct.dataclass` state: allowance vector, per-window harvest counter (int32), cumulative fines, and the step within the window.
  - Each step, add `reward / num_agents` (individual mode) to the window counter.
  - When `step % K == 0`, use `jnp.where` to:
    - draw audits with `jax.random.bernoulli(q)`;
    - fine audited agents for (harvest − allowance)⁺ × F;
    - set the next allowances, for example from the apple count on the map;
    - reset the counter.
  - Subtract fines from the reward.
  - For RL agents to respond, the allowance and window position must be in their observation. The default view is an 11×11 CNN grid, so that means extra channels or a separate input. This is the one change that touches the network.

  Everything stays `jit`/`vmap`-able. In effect this is the AI Economist two-level design [25] on a commons. Expect to need its curriculum: train agents without the overseer first, then anneal it in.
- **LLM agents or an LLM overseer (leaves JAX).** An LLM call cannot sit inside `jit`. The loop must be in Python, calling one jitted `step_env` per step, which is what OpenSocialJax does [23]. JAX's speed then buys nothing, because the LLM call dominates.

## 2. How SocialJax and OpenSocialJax showed each game is a real social dilemma (Schelling diagrams)

**Definition used by both** [20, 22]
- R_c(l) and R_d(l) are the mean returns of cooperating and defecting policies when there are l cooperators out of N.
- The game is a social dilemma if all of these hold:
  - R_c(N) > R_d(0): mutual cooperation beats mutual defection;
  - R_c(N) > R_c(0): mutual cooperation beats being exploited;
  - fear (R_d(i) > R_c(i) for small i), greed (the same for large i), or both.
- The plotted curves are R_c(l+1) and R_d(l) against the number of *co-players* who cooperate (Leibo et al. 2017; Hughes et al. 2018).

**SocialJax procedure: RL policies** [20]
- **Cooperator** = a policy from IPPO trained with **common reward**. **Defector** = a policy from IPPO trained with **individual reward**.
- Quote: "The cooperative policies are sampled from the agents that used a common reward, while the defector policies originate from agents trained with independent rewards. We evaluate the environments over 30 episodes and compute the average rewards for the cooperative and defecting agents."
- **Mixes.** l cooperator policies and N − l defector policies are put in the same game (N = 7 for Harvest and Clean Up). The figure caption adds a dotted line for the overall average return.
- **Not stated in the paper:** how many checkpoints were sampled, how seats were assigned, and confidence intervals.
- **No Schelling script is in the repo.** A grep for "schelling" matches only the README [21].
- **Findings:**
  - Coins, all three Harvests, Gift and Mushrooms show fear.
  - Clean Up, Coop Mining and PD Arena mainly show greed.
  - In Harvest Open one defector is enough to collapse returns, "because defector agents typically harvest the last apple in a patch, permanently depleting it."
  - Territory is marked ✗.
- **Caveat** from the README: a Schelling diagram can prove a game *is* a dilemma, not that it is not, because trained agents may be suboptimal [21].

**OpenSocialJax procedure: prompted LLM roles** [22, 23]
- **One model in every seat** (GPT-6-Luna). Each agent gets item "1.5 YOUR ROLE" in its system prompt; otherwise the prompts are identical, and the game pays everyone individually in both roles.
  - Cooperator: cares about the group total and will give up some of its own score.
  - Defector: "You care only about your own score."
  - The defector role does not forbid cleaning. Agents are not told anyone else's role.
- **k = 0…5 cooperators.** Seat i cooperates when (i + j) mod 5 < k, where j is the seed index, so seats rotate across 3 seeds.
- Each run is one episode of 5 trials of 400 steps, with the normal harness. Returns are averaged over the 5 trials.
- C(x) = R_c(x+1) and D(x) = R_d(x) are plotted against x, the number of the other four agents who cooperate.
- **Findings:**
  - OpenHarvest: defecting pays at every x, so fear and greed both hold.
  - OpenCleanup: greed holds but fear does not, since apples grow only if someone cleans.

**For Ameer.** The OpenSocialJax role-prompt method transfers directly to LLM Fishery/Forest/River or to a new 2D game. It costs 6 compositions × seeds games, and it gives the "is the un-overseen game really a dilemma" check that the SSD literature expects.

## 3. OpenSocialJax (arXiv:2610.13710, 9 Oct 2026) [22, 23]

**What it is**
- Clean Up and Commons Harvest rebuilt in JAX, with procedural maps and hidden rules drawn per episode:
  - OpenCleanup: tool recipes, 57,657,600 rule sets;
  - OpenHarvest: colour × ripeness pay and regrowth, 60,466,176 rule sets.
- The rules stay fixed across the 5 trials of an episode.
- 5 agents and individual rewards; the harness refuses common reward.

**Observation**
- The **11×11 view turns with the agent** (9 cells ahead, 1 behind) but is **drawn north-up** with row and column labels and two-character cell codes.
- The message has a fixed set of blocks, kept up to date by the harness:
  - progress line;
  - RULES YOU KNOW (built from the agent's own experience);
  - YOUR NOTES;
  - EARLIER TRIALS;
  - OTHER AGENTS (only what was in view);
  - YOUR PLAN;
  - LAST STEP / RECENT STEPS (last 15);
  - NOW (position, view, patches seen);
  - ACTIONS NOW (each legal action with its result).

**Actions and calls**
- Primitive actions: 0–3 compass moves, 4–5 turn, 6 zap or fire, 7 stay, 8 pick up (Clean Up only).
- **One call per agent per step**, 5 in parallel. The reply is a JSON object: why, mode, goal, one notes edit, and the action.
- An invalid reply means the agent stays put. A zapped agent is frozen for 12 steps and not called.
- At each trial end there is one extra call per agent to rewrite its notes into lessons. Agents never communicate.

**Models** (Table 1)
- API: GPT-6-Luna, GPT-6-Sol, Claude Opus 5.5, DeepSeek-V4.1-Flash (thinking off), GLM-5.3-Flash.
- Local via vLLM: Qwen3.8-27B, Qwen3.5-4B, Gemma 4 31B.

**Metrics**
- Mean per-agent return per trial.
- Hidden rules known.
- Effective number of cleaners, 1/Σ s_i².
- Total cleaning.
- Orchard lifetime: the step at which the last apple is eaten.
- Gini of returns.
- A scripted greedy baseline.
- Arena: each model's return and its share of the cleaning.

**Protocol**
- Homogeneous teams: 8 models × 3 seeds (906–908, the same maps and rules for every model) × 5 trials.
- Mixed arena: 5 models with seats rotated.
- Schelling runs (section 2).
- Rule-reveal ablation.
- **Inconsistency:** the paper says trials last at most 400 steps, but `open_harvest_5ag/config.json` uses `inner: 300`, and the Schelling config notes "the formal Harvest runs use 300" [23].

**Code and cost**
- Public, Apache-2.0. Runs resume from recorded replies, and a SLURM driver is included.
- Cost per OpenCleanup seed (about 10k calls): about $4 with gpt-6-luna, $7 with GLM-5.3-Flash, $70 with gpt-6-sol and $220 with Claude Opus 5.5. An OpenHarvest seed costs about a quarter of that, because trials stop once the orchard is dead [23].
- The paper itself reports no cost.

**What an overseer would need** [23]
1. **A ledger hook that can change scores.** It would go in `run_episode` in `llm_policy/open_harvest_policy.py`. The existing `on_step` hook must not mutate anything, so a new hook is needed.
   - Every K steps it reads `apples_in` / `points_in`, or the per-step `info["eaten"]`.
   - It draws audits and subtracts fines from `points_in`.
   - It logs the overseer's decision.
2. **A new prompt block**, "OVERSEER: your allowance this window, apples you have taken, fines so far", added through `compose_user`. The rules in the system prompt (`open_harvest_prompt.SYSTEM`) must describe audits and fines.
3. **For an LLM overseer**, one extra call every K steps. Its view would be a per-agent harvest table, possibly with the map. It would return allowances and an audit list. This is cheap next to the 5 calls per step the agents make.
4. **Choices to make:**
   - whether fines are in points (the hidden colour pay makes "points" unequal across apples) or in apples;
   - whether the hidden-rules layer should be switched off (`reveal_rules: true`), so that over-taking is not confounded with rule discovery.

   Engineering is moderate, perhaps a few hundred lines. Running cost is the barrier (section 6).

## 4. JaxMARL [24]

JaxMARL (Rutherford et al., NeurIPS 2024 Datasets & Benchmarks) contains no commons or renewable-resource game. Its social-dilemma games are **Coin Game** (2 players) and **STORM** (spatial in-the-matrix games with a payoff matrix). The rest is MPE, Overcooked v1/v2, MABrax, Hanabi, SMAX, JaxNav, Switch Riddle and Robotarium. It matters only because SocialJax follows its API.

## 5. Foundation / AI Economist [25, 26]

**Architecture**
- A Gym-style Python/NumPy simulator, with CUDA ports through WarpDrive. It is built from two parts:
  - **Scenarios** set the world, regrowth, rewards and metrics. Gather-Trade-Build is `layout_from_file/simple_wood_and_stone` on 25×25 maps such as `quadrant_25x25_20each_30clump.txt`.
  - **Components** add actions and dynamics: Build, Gather (move), ContinuousDoubleAuction (trade) and PeriodicBracketTax.
- Every component has the same interface: `get_n_actions`, `get_additional_state_fields`, `component_step`, `generate_observations` and `generate_masks` [26].

**The planner (`BasicPlanner`)**
- It is **not embodied**: it has no location.
- **Action space:** one discrete head per tax bracket, each choosing a rate from {0, 0.05, …, 1.0}. The defaults are 7 US-federal-style brackets, or `n_brackets=5`.
- **How often it acts:** it can act only at the first step of each 100-step tax period. At every other step its actions are masked to no-op, so 10 decisions per 1000-step episode. Taxes are collected at period end and shared out as a lump sum [25, 26].
- **What it observes:** where it is in the tax period, the last period's incomes (sorted), and current rates. It does not see the map [26, 30].

**How the planner is trained** [25]
- Two-level PPO with shared-weight agents.
- Phase 1: agents alone with no tax for 50M steps, with the labour cost annealed in.
- Phase 2: taxes on for 1B steps, with the maximum tax rate annealed and the planner's entropy annealed. The agents first meet near-random taxes; the planner starts to improve at about 30M steps.
- Learning rates were set by grid search, and curriculum settings by grid plus manual search.

**How parameters were chosen** [25]
- Build-skills, labour costs and the utility curvature η were tuned so that the income distribution under the US Federal schedule (4 agents) roughly matches US incomes in 2018.
- This is **calibration to an outside target**, unlike the hand-tuned SSD games.

**Evaluation** [25, 26]
- Two welfare objectives:
  - **equality × productivity**, where equality = 1 − Gini of coin and productivity = total coin (code: `coin_eq_times_productivity`);
  - inverse-income-weighted utility.
- Baselines: free market, US Federal 2018, and the Saez formula (elasticity picked by grid search; about 3 was best, against about 0.87 from regression).
- 10 seeds with standard errors and significance tests.
- Behavioural analyses: specialisation, tax gaming across years, and policy similarity across seeds.
- Result: +8% utilitarian welfare and +12% equality × productivity over Saez (Open-Quadrant, 4 agents).

**Not a commons**
- With the default `regen_halfwidth=0`, an empty resource cell on a source block regrows with chance `resource_regen_prob = 0.01`, and source blocks never disappear [26]. Over-gathering therefore causes no lasting harm.
- To study environment destruction the regrowth step (`scenario_step`) must be rewritten, for example so that regrowth depends on the neighbours and source blocks can die.

**LLMs and Foundation**
- I found **no work that runs LLM agents or an LLM planner inside Foundation's 2D world.** The nearest relatives:
  - **Large Legislative Models** (Gasztowtt et al., arXiv:2410.08345, preprint) [27]: an LLM principal (GPT-4o mini, Gemini-1.5 flash) sets a three-tier tax on recent harvest in **Melting Pot Commons Harvest Open**, and incentives in Clean Up. The agents are RL, and the principal acts once per episode. It beat the AI Economist planner and bandits on sample efficiency. Runs were capped at 90 LLM actions because of API cost. This is **the closest prior art to an LLM overseer on a 2D commons**, but it has no audits, no imperfect monitoring and no LLM agents.
  - **LLM Economist** (Karten et al., arXiv:2507.15815; NeurIPS 2025 listing) [28]: an LLM planner over 100 LLM workers who choose only labour hours. There is no space. It found that tax years of K = 5–10 steps are too short; 128 steps were used.
  - **TaxAgent** (Wang et al., ICME 2025 oral) [29]: an LLM government over LLM households in a non-spatial macroeconomy, with the AI Economist baselines.
  - **Dizaji 2024** [30]: extended the AI Economist for RL, and used Concordia (text) for the LLM side, with one run per setting.
- **Licence and maintenance:** BSD-3-Clause; repo archived, last commit May 2022, Python 3.7-era dependencies [26].

## 6. Other 2D commons or grid platforms with LLM support, 2024–2026

Beyond the earlier note's table (Mosquera, Hypothetical Minds, OpenSocialJax, Concordia, Generative Agents):
- **Gallego 2026** [31] (ICML 2026 workshop). The LLM writes a Python policy for Gathering and Cleanup grids, and the policy runs in self-play. The LLM is called once per iteration, not per step. Dense social-metric feedback (efficiency, equality, sustainability, peace) helps.
- **Mao & Perrault 2026** [32] (preprint, abstract only). Small frozen LLMs with primitive actions score zero in spatial cooperative games; options run by a symbolic planner plus per-agent LoRA training fix this. Its Commons Harvest numbers were not checked.
- **Nguyen et al., IJCAI 2025** [33]. An LLM "Common Harvest" with two regions and navigation between them (more abstract than a grid). Llama-3.1-70B and Qwen-2.5-72B, 200 steps, no overseer.
- **Large Legislative Models** [27], above.

**Pattern.** LLMs act in a 2D commons in one of four ways:
1. a primitive action per step from a labelled text view (OpenSocialJax), which works only with strong models;
2. macro-actions or options run by a planner (Mosquera, Hypothetical Minds, Mao & Perrault);
3. policies written as code (Gallego);
4. dropping the grid (Concordia, GovSim, LLM Economist).

An overseer that audits and fines **LLM agents in a 2D commons** still appears to be unclaimed. An **LLM overseer of RL agents** in Commons Harvest has been done once, in a preprint, as a tax-setter [27].

## Bottom line: how hard is an overseer on each

| Platform | Agents | Overseer engineering | Run cost | Main catch |
|---|---|---|---|---|
| SocialJax | RL (PPO) | **Low to moderate.** A JAX wrapper with its own state; allowances and window position must go into observations. Stays jittable. | GPU hours (1e8 steps per run) | Two-level training is unstable (AI Economist needed curricula). Must set `shared_rewards: False`. Rewards are scaled by N. |
| OpenSocialJax | LLM, primitive actions | **Moderate.** A ledger hook that can change scores, a prompt block, optionally an LLM overseer call every K steps. | **High**: about $1 per OpenHarvest seed with gpt-6-luna, up to tens or hundreds of dollars for frontier models [23] | Hidden rules confound over-taking with exploration. 6 days old, little use yet. |
| Foundation | RL | **Low for the mechanism.** Copy `PeriodicBracketTax` into an audit-and-fine component; the planner head and masks already exist. **High for the game**: regrowth must be rewritten so depletion can happen. | GPU or CPU hours, 1B steps | Not a commons. Archived. No LLM support. |
| JaxMARL | RL | Not applicable: no commons game. | — | — |
