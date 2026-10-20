# 21. SocialJax source code (architecture)

- **Authors:** cooperativex (Guo et al.)
- **Year:** 2025–2026
- **Identifier:** github.com/cooperativex/SocialJax, main branch, commit `c64f1aaa70a0f5d71d0a57ee54ab6880e0ee1f19` (29 Sep 2026). Cloned `--depth 1` to scratchpad, read only, nothing executed. Licence Apache-2.0 (LICENSE file; GitHub API: not archived, last push 29 Sep 2026).
- **Source type:** primary-code
- **Accessed:** 14 Oct 2026

## Exact code lines

1. `socialjax/environments/multi_agent_env.py`, base class: `class MultiAgentEnv(object):` / `"""Jittable abstract base class for all SocialJax Environments."""`; `step()` calls `obs_st, states_st, rewards, dones, infos = self.step_env(key, state, actions, timestep)` and auto-resets with `jax.tree.map(lambda x, y: jax.lax.select(dones["__all__"], x, y), states_re, states_st)`.
2. `socialjax/environments/common_harvest/harvest_open.py`, state pytree:
   ```
   @dataclass
   class State:
       agent_locs: jnp.ndarray
       agent_invs: jnp.ndarray
       inner_t: int
       outer_t: int
       grid: jnp.ndarray
       apples: jnp.ndarray
       freeze: jnp.ndarray
       reborn_locs: jnp.ndarray
   ```
   Map encoding: `char_to_int = {'W': 1, ' ': 0, 'A': 3, 'P': 4, 'Q': 5}`; `nums_map = ascii_map_to_matrix(map_ASCII, char_to_int)`; `self.SPAWNS_APPLE = find_positions(nums_map, 3)`.
3. Step is a jitted closure built in `__init__`: `if jit: self.step_env = jax.jit(_step)`; `self.reset = jax.jit(reset)`.
4. Reward computed inside `_step` (harvest_open.py ~L1180–1243): shared mode `rewards_sum = jnp.sum(original_rewards)` paid to every agent, with `info["original_rewards"]` per agent; individual mode `rewards = jnp.where(apple_matches, 1, rewards) * self.num_agents` and `info = {}`.
5. Inventory counter: `new_invs = state.agent_invs + apple_matches`, with reset `agent_invs=jnp.array([(0,0)]*num_agents, dtype=jnp.int8)`.
6. Vectorisation lives in the trainer, not the env: `algorithms/IPPO/ippo_cnn_harvest_open.py` uses `obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)` and `jax.vmap(env.step, in_axes=(0, 0, 0))`, rolled out with `jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])`; base config `"NUM_ENVS": 256`, `"NUM_STEPS": 1000`, `"TOTAL_TIMESTEPS": 1e8`.
7. Wrapper pattern for extra state (`socialjax/wrappers/baselines.py`): `@struct.dataclass class LogEnvState: env_state: State; episode_returns: float; ...` and `class LogWrapper(JaxMARLWrapper)` whose jitted `step` calls `self._env.step(key, state.env_state, action)`.
8. README.md, on Schelling diagrams: "*Due to algorithmic limitations, agents may not always learn the optimal actions. As a result, Schelling diagrams can prove that the environment is social dilemmas, but they cannot definitively prove that the environment is not social dilemmas.*" No Schelling-diagram script exists in the repo (grep for "schelling" matches only README.md).
9. Config gotcha: `algorithms/IPPO/config/ippo_cnn_harvest_open.yaml` has `- reward: common  # override on CLI: reward=individual`, and `reward/common.yaml` sets `shared_rewards: True`.

## Relevance

An overseer that reads per-agent harvest every K steps, sets allowances, audits and fines can be written as a `LogWrapper`-style JAX wrapper with its own `struct.dataclass` state (allowance vector, per-window harvest counter, fines), using `inner_t % K` with `jnp.where`; it then stays jittable and vmappable for RL. Per-agent apples must be read as `reward / num_agents` in individual mode (info is empty there). Do not use `agent_invs` as the ledger: it is int8 and, by my reading, would wrap past 127 apples (not run). An LLM overseer cannot sit inside `jit`; it needs a Python loop like OpenSocialJax's harness (source 23).
