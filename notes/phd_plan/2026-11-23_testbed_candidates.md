# Testbed candidates for decentralised overseers (23 Nov 2026)

**Purpose.** Pick 3–4 testbeds that differ in two properties:
- **locality:** does the damage stay near the cheater, and do agents stay in one region?
- **visibility:** is a violation visible in one step, or only over many?

The formulation note (`2026-11-23_decentralised_overseer_formulation.md`) predicts that these two properties decide whether local overseers can match a central one:
- locality is measured by ξ_loc, built from the crossing rate χ and the harm containment ζ (§5 there);
- visibility is claim C4.

**What this table gives for each testbed.**
- **Cheat:** what "cheating" means there.
- **Record?:** can a check decide it from a record of what the agent saw and did?
- **1 step?:** is it visible within one step?
- **Damage:** is the damage shared with others, and is it permanent?
- **Local?:** what it means for decentralisation: harm containment (ζ) and how much agents roam (χ). All are expectations, not measurements.
- **Agents:** how many by default, and whether that scales.
- **Code / effort:** is code available, and how much work is it to add the overseer, given the existing JAX wrapper `overseer/socialjax/harvest_overseer.py`?

**How the facts were checked.** On 21 Nov 2026:
- **SocialJax:** read from the source at github.com/cooperativex/SocialJax, commit 18e2171 (17 Oct 2026, Apache-2.0).
- **Melting Pot:** read from the substrate configs at github.com/google-deepmind/meltingpot (Apache-2.0).
- **LLM testbeds:** checked against their arXiv abstract pages and repositories.

"Not stated" means I could not find the number.

## Three code facts that matter before choosing

1. **Every SocialJax game except LB-Foraging and PD Arena defaults to `shared_rewards=True`.** In that mode every agent is paid the group total, so nobody has a private reason to cheat. Every overseer study must set `shared_rewards=False`; the Harvest wrapper already does.
2. **SocialJax ships only Harvest Open.**
   - `harvest_common_closed` is commented out in `socialjax/registration.py`, and there is no Partnership file.
   - The SocialJax paper (Guo et al. 2025) does list Closed and Partnership.
   - `Harvest_open` takes a custom `map_ASCII` with walls (`W`), so a map with rooms can be passed in without new code.
3. **Zapping in SocialJax Harvest Open appears to do nothing.** The freeze check in `_step` is commented out (harvest_open.py, around line 1024). So Closed's "defend your room" mechanic would need restoring. Territory does apply the freeze.

## Table 1: SocialJax (JAX; the overseer wrapper already exists for Harvest)

| Env (registry id) | Cheat | Record? | 1 step? | Damage | Local? | Agents | Code / effort |
|---|---|---|---|---|---|---|---|
| **Harvest Open** (`harvest_common_open`) | eating an apple with fewer than 3 apples within two squares (regrowth chance 2.5% / 0.5% / 0.1% / 0 per step for 3+ / 2 / 1 / 0 neighbours) | **yes**, given the map the agent saw | **yes** | shared by whoever uses that patch; **permanent** for a patch stripped to 0 | ζ high: the harm radius is 2 and each of the 6 patches lies on one side of a left/right split. χ probably high: one open 16×22 map, all spawns along the bottom edge | 7; `num_agents` param, 60 spawn cells | done (all phase 2 results) |
| **Harvest with rooms** (`harvest_common_open` with a walled `map_ASCII`, like Melting Pot's Closed) | as Open | yes | yes | shared within a room; permanent | **ζ high, χ low**: rooms become overseer regions | 7 (scales with spawn cells) | **small**: map only. Restoring zap/freeze (medium) is needed only if rooms must be defended |
| **Clean Up** (`clean_up`) | free-riding: harvesting while never cleaning the river | **partly**: clean actions are logged (`info["clean_action_info"]`), but "not enough cleaning" needs a quota over w steps | **no**, many steps | **shared by all**, reversible. Regrowth is 0.05 × a linear function of the global dirt fraction, and 0 once dirt exceeds 0.4. Dirt spawns with chance 0.5 after step 50 | **ζ low**: one dirt fraction drives every orchard cell. River and orchard are separate bands | 7; param, 19 spawns, 19×28 map | **small–medium**: write the w-step rule |
| **Mushrooms** (`mushrooms`) | eating red (+1 to self only) instead of blue (+3 split among the others) | yes: the type eaten is on the grid; `info["eat_blue_mushrooms"]` | yes | shared and immediate (a lost gain to others); not permanent. Orange costs every agent 0.2, the eater included, so it is no temptation | **ζ low** (global payoff split) | 6; param; 12×23 map | small. A weak dilemma: there is no collapse |
| **Territory Open** (`territory_open`) | zapping or repainting others' claimed walls; zapping players (two hits remove a player for good) | partly: zap and claim actions are visible, but ownership needs extra wall state | single acts yes; "invasion" over many steps | hurts neighbours; **permanent** for destroyed walls and removed players | **ζ high**: damage at borders; players hold areas | 9; param, 18 spawns, 23×39 map | **medium**: track wall ownership |
| **Gift** (`gift`) | hoarding: consuming tokens instead of gifting them (a gift triples them) | partly: gifts are logged (`info["give_actions"]`), but hoarding needs a window | no, many steps | hurts would-be recipients; not permanent | weakly spatial (a gift goes to the agent in front); 25×25 map | 6; param | small–medium |
| **Coop Mining** (`coop_mining`) | starting gold ore (needs 2–4 miners within 3 steps, +8 each) then leaving, or always mining iron (+1) | partly: intent is hard to define | many steps | hurts one partner; ore respawns | local; 27×27 map | 4; param | medium |
| **PD Arena** (`pd_arena`) | collecting "defect" resources, then interacting (payoffs [[3, −1], [5, 1]]) | yes: the inventory at interaction | yes | pairwise | local; 24×25 map | 2; param | medium; pairwise, not a commons |
| **Coins** (`coin_game`) | picking up the other player's coin | yes | yes | one victim | 16×11 map, too small to split | **2, fixed** | small, but cannot be decentralised |
| **LB-Foraging** (`lb_foraging`) | none: cooperative food loading | – | – | – | 8×8 map | 4 | not relevant |

## Table 2: Melting Pot substrates (Python + Lua on DeepMind Lab2D)

- Melting Pot is from Leibo et al. 2021 (arXiv 2107.06857) and Agapiou et al. 2022, "Melting Pot 2.0" (arXiv 2211.13746).
- Player counts are `default_player_roles` from each config.
- **Effort:** large to port into JAX. Medium for a Python-loop overseer reading the debug observations, but too slow for IPPO at the 1e8-step scale.

| Substrate | Cheat | Record? | 1 step? | Damage | Local? | Players |
|---|---|---|---|---|---|---|
| commons_harvest__open | eating the last apples of a patch (same regrowth rule) | yes | yes | shared; permanent per patch | roaming; 18×24 map | 7 |
| commons_harvest__closed | as Open; rooms can be defended by zapping | yes | yes | shared within a room | **rooms** | 7 |
| commons_harvest__partnership | as Closed, with two-door rooms shared by pairs | yes | yes | as Closed | rooms per pair | 7 |
| clean_up | not cleaning | partly | many steps | global; reversible | bands; 21×30 map | 7 |
| territory__open / __rooms / __inside_out | invading or destroying others' walls | partly | many steps | local; permanent wall loss | **rooms** (one per player in Rooms) | 9 / 9 / 5 |
| allelopathic_harvest__open | eating instead of planting; planting a colour the group didn't agree on | partly | many steps | global (growth depends on colour shares) | 30×29 map | 16 |
| externality_mushrooms__dense | eating mushrooms that harm others | yes | yes | shared, instant | global | 5 |
| prisoners_dilemma_in_the_matrix__arena | defect inventory | yes | yes | pairwise | 24×25 map | 8 |
| coop_mining | leaving a partner | partly | many steps | pairwise | 27×27 map | 6 |
| hidden_agenda | impostor sabotage (hidden role) | no, hidden by design | many steps | shared | spatial | 5 |
| gift_refinements, fruit_market, factory_commons, daycare, boat_race, predator_prey, chemistry, paintball | low relevance to a commons with an enforceable rule | – | – | – | – | 2–16 |

## Table 3: LLM and text commons

| Testbed | Cheat | Record? | 1 step? | Damage | Local? | Agents | Code / effort |
|---|---|---|---|---|---|---|---|
| **Willis et al. commons** (Willis, Zhao, Du, Leibo 2026, arXiv 2602.16662) | choosing D, intensive extraction | yes | yes, one round | shared; stock 0 is absorbing | **not spatial**. Regions are fixed groups of agents: χ = 0, ζ ≈ 1/M | 4, 16, 64, 256 (evolution runs: 512) | github.com/willis-richard/emergent_llm, MIT. LLMs write strategies as code, so play is cheap. **Small**: adapter `overseer/adapters/emergent_llm.py` and the R1a/R1b work exist |
| **GovSim** (Piatti et al. 2024, arXiv 2404.16698): fishery, pasture, pollution | catching more than the sustainable share, (pool/2)/N, which is 10 when the pool is full and N = 5 | yes | yes, one round (a month) | shared; permanent below 5 units (the stock doubles, capped at 100) | not spatial | 5 (config `num_agents`) | github.com/giorgiopiatti/GovSim, MIT; **medium** (LLM cost per round) |
| **OpenSocialJax** (Guo, Fang, Leibo, Du 2026, arXiv 2610.13710): OpenHarvest, OpenCleanup | over-harvesting; skipping cleaning | yes / partly | yes / many steps | within a trial, "every team eventually exhausts" the resource | **spatial gridworld**; hidden rules sampled per episode | teams of 5; a 1-vs-4 arena | github.com/cooperativex/OpenSocialJax, Apache-2.0; JAX env plus a Python LLM loop. **Small–medium**: closest to the Harvest wrapper |
| **BazaarBench** (Wang, Shi, Oldfield, Marro, Yu, Torr, Du 2026, arXiv 2610.06748) | F1 misstated condition; F2 selling an item not owned; F3 promising one item to several buyers; F4 confirming too early; F5 sharing personal data or moving off-platform; F6 unsupported reputation claims | F2, F3, F5 yes from records; F1, F6 need ground truth | F2 and F5 yes; F3 and F4 many steps | hurts trading partners; no resource collapse | **a trade network, not a map**. Regions are groups of traders (by category or trading partners); harm stays local if partners are mostly in-group | 100 traders per market, 20 running the tested model | github.com/ziyan-wang98/BazaarBench (the LICENSE file says Apache 2.0; GitHub's API reports NOASSERTION). **Medium** for offline replay; large live |
| Mosquera et al. 2024 (arXiv 2403.11381): LLM agents in Melting Pot Commons Harvest | last apples of a patch | yes | yes | permanent per patch | spatial | the repo default is 3 LLMs plus bots | github.com/Cooperative-IA/CooperativeGPT, no licence shown |
| Rehm 2026 (arXiv 2609.22600): GovSim with self-governance | a group catch above 50 | yes | yes | permanent; starvation | not spatial | 5 | no code link found |
| Guzman Piedrahita et al. 2025 (arXiv 2506.23276): public goods with punishment | free-riding | yes | yes | shared, no stock | not spatial | 7 | github.com/davidguzmanp/SanctSim |
| Emergence World (Akkil et al. 2026, arXiv 2606.08367) | not a commons; 120+ tools | partly | many steps | one world ended in population collapse | spatial | **unverified** | not checked |

## How the candidates spread over the two properties

| | Visible in one step | Visible only over many steps |
|---|---|---|
| **Local harm, agents stay (low χ)** | Harvest with rooms; Territory single acts | Territory invasion |
| **Local harm, agents roam (high χ)** | Harvest Open; OpenHarvest | Harvest with a w-step rate rule (variant) |
| **Global harm (low ζ)** | Willis commons; GovSim; Mushrooms | Clean Up; OpenCleanup; Allelopathic Harvest |
| **Network-local harm** | BazaarBench F2, F5 | BazaarBench F3, F4 |

## Recommendation: four testbeds

1. **Harvest, Open and with rooms (SocialJax).** Local harm and one-step visibility.
   - The same mechanics with two maps turns crossing rate χ up and down with nothing else changed.
   - It carries the existing M = 1 results as the baseline.
   - It tests D1 (Δ_dec ≈ 0 at w = 1, and > 0 at w = 10 on Open but not on rooms), D2 (migration, boundary-hopping) and D3 (sharing with κ = 0, 1, 2).
   - Effort: small.
2. **Clean Up (SocialJax).** Global harm and many-step visibility, the hard case on both properties.
   - **Prediction:** decentralising fails here unless overseers share, and C4 and D1 compound.
   - It needs `shared_rewards=False` and a w-step cleaning rule.
   - Effort: small–medium.
3. **Willis et al.'s commons.** Global harm, one-step visibility, no space.
   - Fixed groups give χ = 0 and ζ = 1/M, so it isolates harm containment from crossing.
   - **Predictions:** with a fixed rule, decentralising costs nothing. When each overseer optimises its own group's welfare, overseers under-spend.
   - It scales to 256 agents cheaply, so M can go to 16 or more. That is the closest we get to NMI's 199–436 agents.
   - Effort: small (the adapter exists).
4. **BazaarBench.** LLM traders on a trade network, with one-step and many-step failures in the same market, and real token costs.
   - It tests whether D1–D3 hold when regions are groups of agents rather than map areas, and when checks have a real price.
   - Start with offline replay, which is already planned.
   - Effort: medium.

**Reserves:**
- **Territory:** local harm, many steps, permanent damage. Medium effort.
- **OpenSocialJax:** if LLM agents in a gridworld are wanted.
- **Mushrooms:** cheap, but a weak dilemma.

**Not recommended:**
- **Coins and PD Arena:** 2 agents, which cannot be split into regions.
- **LB-Foraging:** no dilemma.
- **Melting Pot ports:** large effort for the same games SocialJax has in JAX.

## Sources (checked 21 Nov 2026)

- **Verified on arXiv** (title, first author and year match):
  - 2503.14576 SocialJax, Guo 2025
  - 2107.06857 Melting Pot, Leibo 2021
  - 2211.13746 Melting Pot 2.0, Agapiou 2022
  - 2404.16698 GovSim, Piatti 2024
  - 2610.13710 OpenSocialJax, Guo 2026
  - 2602.16662 Willis 2026
  - 2610.06748 BazaarBench, Wang 2026
  - 2403.11381 Mosquera 2024
  - 2609.22600 Rehm 2026
  - 2506.23276 Guzman Piedrahita 2025
  - 2606.08367 Emergence World, Akkil 2026
  - 1702.03037 Leibo 2017
- **Code read directly:**
  - SocialJax at commit 18e2171: registration, defaults, the Harvest map and zap code, Clean Up thresholds, Mushrooms rewards;
  - Melting Pot substrate configs.
- **Correction to the reading list:** arXiv 2609.15293 is Yuhang Wang 2026, "Why LLM Agents Collapse Without Oversight: The Enforcement Gap…", an enforcement paper with no environment of its own. It is **not** the Emergence World platform (2606.08367).
- **Not verified:** the Emergence World agent count; the CooperativeGPT licence; a code link for Rehm 2026.
- **Repo note:** `external/GovSim` and `external/emergent_llm` are empty submodule folders in a fresh worktree. Run `git submodule update --init` before use.
