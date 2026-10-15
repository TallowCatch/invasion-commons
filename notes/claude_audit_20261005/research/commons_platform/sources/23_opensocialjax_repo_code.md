# 23. OpenSocialJax source code and README (LLM harness, cost)

- **Authors:** cooperativex (Guo, Fang, Leibo, Du)
- **Year:** 2026
- **Identifier:** github.com/cooperativex/OpenSocialJax, commit `b5f59a38aba19b23b304d8dad693a3eea78fd5b0` (8 Oct 2026). Cloned `--depth 1`, read only, nothing executed. Licence Apache-2.0 (LICENSE file and README badge; GitHub API: not archived, 2 stars on 14 Oct 2026).
- **Source type:** primary-code (plus README docs)
- **Accessed:** 14 Oct 2026

## Verbatim quotes and exact code lines

1. README, cost: "One OpenCleanup seed is 5 trials × 400 steps × 5 agents, about 10k calls: in the paper's runs it cost about $4 with gpt-6-luna, $7 with GLM-5.3-Flash, $70 with gpt-6-sol and $220 with Claude Opus 5.5; an OpenHarvest seed is roughly a quarter of that, since a trial ends once the orchard is dead."
2. README, demo: "The demo's 405 calls took about ten minutes and cost $0.17 with gpt-6-luna (the five agents are queried in parallel)."
3. README, providers: "`kind: api` for any OpenAI-compatible endpoint, `kind: anthropic` for the Anthropic API, `kind: vllm` for a local open-weight model served with vLLM."
4. `llm_policy/open_harvest_policy.py`, `run_episode` (main loop, Python, not jitted as a whole): `step = jax.jit(env.step_env); own_scale = 1.0 / n   # the environment pays an apple as n to the agent that ate it`; agents decided in parallel with `with ThreadPoolExecutor(max_workers=n) as ex: chosen = list(ex.map(decide, range(n)))`; per-agent ledger updated with `points_in[i][trial] += float(own[i]); apples_in[i][trial] += int(eaten[i])`; hook `on_step(trial, t, state, agents, obs, ks)` documented as "optional hook called right before the agents decide at each step (snapshot tooling; it must not mutate anything)".
5. `Agent.decide` builds the prompt: `content = P.compose_user(progress, self.ledger.render(live), self.social.render(t), self.plan.render(t), last, "\n".join(self.recent), obs_text, actions_now(facts, env, zap_cd), earlier="\n".join(self.earlier), notes=self.notebook.render())` then `raw = self.provider.complete(self.system, messages, self.max_tokens, schema=ACTION_SCHEMA)`.
6. `llm_policy/run_experiment.py`: individual rewards enforced, `raise SystemExit(f'setting says reward={s["reward"]!r}: rewards are individual only')`; `experiments/open_harvest_5ag/config.json` setting has `'inner': 300` (the paper text says trials of "at most 400 steps"; the Schelling config's "about" note says "the formal Harvest runs use 300").
7. `llm_policy/run_schelling.py` docstring: "seat i (agent letter A-E and its spawn slot) is a cooperator when (i + j) % 5 < k, where j is the seed's index in setting.seeds, so the cooperators' seats shift by one from seed to seed. Nobody is told the others' roles."

## Relevance

An overseer fits at the Python loop level in `run_episode`: every K steps read `apples_in`/`points_in` (or the per-step `info["eaten"]`, `info["original_rewards"]`), compute allowances, draw audits, subtract fines from the ledger, and add an "OVERSEER" block to `compose_user` so agents see their allowance and fines. That is roughly a new prompt block, a ledger hook that may mutate scores (the existing `on_step` must not), and optionally an LLM overseer call per K steps. Cost scale per primitive-action game is dollars, not cents (gpt-6-luna about $1 per OpenHarvest seed of 5 trials by the README's "quarter of $4").
