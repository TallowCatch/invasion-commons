# 26. Foundation / ai-economist source code (Salesforce)

- **Authors:** Salesforce Research (Zheng, Trott, Srinivasa et al.)
- **Year:** 2020–2022 (last commit 9 May 2022)
- **Identifier:** github.com/salesforce/ai-economist, commit `a84d5f3fdcabb207d9fde7754d34906903b3e184` (9 May 2022). Cloned `--depth 1`, read only, nothing executed. Licence BSD-3-Clause (LICENSE.txt). GitHub API on 14 Oct 2026: `archived= True`, last push 20 Aug 2023.
- **Source type:** primary-code
- **Accessed:** 14 Oct 2026

## Exact code lines

1. `ai_economist/foundation/base/base_component.py`: "Environments expand the agents' state/action spaces by querying: get_n_actions get_additional_state_fields" and "Environments expand their dynamics by querying: component_step generate_observations generate_masks".
2. `ai_economist/foundation/components/redistribution.py`, `PeriodicBracketTax`: docstring "period (int): Length of a tax period in environment timesteps. Taxes are updated at the start of each period and collected/redistributed at the end of each period. Must be > 0. Default is 100 timesteps."
3. Planner action space (same file): `return [("TaxIndexBracket_{:03d}".format(int(r)), self.n_disc_rates) for r in self.bracket_cutoffs]`, with `rate_disc=0.05` giving rates "[0.0, 0.05, 0.10, ... 1.0]".
4. Planner acts only on the first step of a period, via masks: `if self.tax_cycle_pos != 1 or self.disable_taxes: masks[self.world.planner.idx] = self._planner_masks["zeros"]`.
5. Planner observation: `obs[self.world.planner.idx] = dict(is_tax_day=..., is_first_day=..., tax_phase=..., last_incomes=self._last_income_obs_sorted, curr_rates=self._curr_rates_obs)`.
6. `ai_economist/foundation/agents/planners.py`: "Unlike the "mobile" agent, the planner does not represent an embodied agent in the world environment."
7. `ai_economist/foundation/scenarios/utils/rewards.py`: `def coin_eq_times_productivity(coin_endowments, equality_weight):` ... `return equality * prod`.
8. Scenario defaults, `scenarios/simple_wood_and_stone/layout_from_file.py`: `env_layout_file="quadrant_25x25_20each_30clump.txt"`, `resource_regen_prob=0.01`, `isoelastic_eta=0.23`, `energy_cost=0.21`; regrowth in `scenario_step` uses `health = np.maximum(resource_map, resource_source_blocks)` so a source block never loses its ability to regrow.

## Relevance

Foundation already has the shape of an overseer layer: a component with its own period, a planner-only action head masked except at period start, planner observations of per-agent income, and payoff changes applied at period end. An audit-and-fine component would copy `PeriodicBracketTax`. But Gather-Trade-Build is not a commons: resource sources regrow at a fixed chance forever, so over-harvest causes no lasting harm unless `scenario_step` is rewritten. The repo is archived (no maintenance), Python/NumPy (plus WarpDrive CUDA ports), Python 3.7-era dependencies.
