# GovSim official repository: configuration and environment code

- **Authors:** Giorgio Piatti et al. (repository owner: giorgiopiatti/GovSim)
- **Year:** 2024–2025 (local clone at commit 1d11adf047b24fa2ba0d44a1d4931015ea2e5210, committed 19 Jan 2025)
- **Venue / identifier:** GitHub, https://github.com/giorgiopiatti/GovSim
- **URL:** local clone /Users/ameerfiras/invasion-commons/invasion-commons/external/GovSim (read only)
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026

## Verbatim code lines

1. `simulation/scenarios/fishing/conf/experiment/fish_baseline_concurrent.yaml`
   ```
   14:  max_num_rounds: 12
   15:  initial_resource_in_pool: 100
   16:  poupulation_change_after_round: double_100_cap
   19:  num_agents: 5
   20:  harvesting_order: concurrent # random-sequential, concurrent
   21:  assign_resource_strategy: stochastic # proportional, stochastic
   55:    max_conversation_steps: 10
   ```
   (The sheep and pollution baseline YAMLs use the same values.)
2. `simulation/scenarios/common/environment/concurrent_env.py`, initial threshold:
   ```
   216:            "sustainability_threshold": (
   217:                10
   218:            ),  # each day the fish double and cap at 100, so maximum 50 can be fished
   ```
3. Same file, end-of-month dynamics:
   ```
   428:                        < 5  # less than 5 fish remain, so we collapse
   ...
   434:                self.internal_global_state["resource_in_pool"] = min(
   435:                    self.cfg.initial_resource_in_pool,
   436:                    self.internal_global_state["resource_in_pool"] * 2,
   437:                )  # Double the fish in the lake, but cap at 100
   ...
   441:                self.internal_global_state["sustainability_threshold"] = int(
   442:                    (self.internal_global_state["resource_in_pool"] // 2)
   443:                    // self.internal_global_state["num_agents"]
   ```
4. `simulation/conf/config.yaml`: `temperature: 0.0`, `seed: 42`.
5. `simulation/scenarios/fishing/conf/experiment/fish_perturbation_outsider.yaml`: `num_agents: 4`, `max_num_rounds: 15`, perturbation `round: 3`, `num_agents: 5`, `type: insert_outsider`.

## Relevance
The code confirms that the carrying capacity is the initial stock (cap = `initial_resource_in_pool`), growth is an exact doubling, and the threshold used for over-usage is **per agent** (half the pool divided by agents), not the group-level f(t) written in the paper. No config or comment gives a reason for these numbers beyond "fish double and cap at 100, so maximum 50 can be fished". Note: GovSim-SelfGovern (65) describes GovSim's individual over-usage as flagging "a single agent's catch exceeds the whole group's sustainable share"; the code (line 441–443) divides by the number of agents, so that description looks inaccurate.
