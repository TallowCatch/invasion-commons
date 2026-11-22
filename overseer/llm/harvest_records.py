"""Harvest judging dataset: one agent's record over a window of w steps, as text, with the wrapper's true label.

The rule (overseer/socialjax/harvest_overseer.py): taking an apple with fewer than 3 other apples within two squares
of it is over-harvesting. "Within two squares" is the 12 cells at most two up/down/left/right steps from the apple's
cell (the cell itself excluded), counted on the map as it was just before the move. A record's label is 1 if the
agent over-harvested at least once in its w steps.

Scripted bots (overseer/socialjax/harvest_bots.py) play full 1,000-step episodes under a random k = 1 overseer with
no penalty. Sneaky bots turn greedy after 10 steps unchecked (about 21% of steps at k = 1 of 7), so they give
occasional violations among mostly honest eats. Every step is recorded:
the map before the move, where each agent ended up, whether it ate, and whether that eat broke the rule. Windows are
then sampled so the labels are balanced and every bot kind is represented.

Two renderings of the same record:
- summary: per step, whether it took an apple and how many apples were within two squares (the judge only has to
  compare a number with 3);
- raw: per step, the move; on steps where it took an apple, the 5 x 5 map patch around that apple as it was just
  before the move, in ASCII. The judge has to count the apples in the right 12 cells itself.

The map "before the move" is the previous step's map, as the wrapper judges it. SocialJax regrows apples at the start
of a step, before agents move, so about 7% of eats (478 of 7,090 in the dataset) take an apple that regrew on that
very step and is not on the previous map. The centre cell is always drawn as the apple taken; the 12 cells around it
are the previous map, which is what the label counts.

Needs JAX and SocialJax (scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import numpy as np

MIN_NEIGHBOURS = 3
PATCH_RADIUS = 2
# (dr, dc) of the 12 cells "within two squares", as in harvest_overseer.NEIGHBOURHOOD
DIAMOND = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if 0 < abs(dr) + abs(dc) <= 2)

# Seven-agent populations. Greedy bots kill the orchard quickly, so the mixes also give records with no apples left.
MIXES = {
    "restrained7": ("restrained",) * 7,
    "one_greedy": ("restrained",) * 6 + ("greedy",),
    "two_greedy": ("restrained",) * 5 + ("greedy",) * 2,
    "three_sneaky": ("restrained",) * 4 + ("sneaky",) * 3,
    "all_sneaky": ("sneaky",) * 7,
    "mixed": ("restrained",) * 3 + ("sneaky",) * 2 + ("greedy", "random"),
}
# Share of each window's examples: violations, honest records with at least one apple taken, honest records with none
STRATA = (("violation", 0.5), ("honest_eats", 0.4), ("honest_idle", 0.1))
MOVE_NAMES = {(1, 0): "moved south", (-1, 0): "moved north", (0, 1): "moved east", (0, -1): "moved west",
              (0, 0): "stayed"}


# ------------------------------------------------------------------------------------------------ simulation
def record_episodes(kinds, seeds, steps=1000, k=1, sneaky_after=10, explore=0.1):
    """Run one episode per seed with the given bot kinds; return per-step numpy arrays stacked over seeds.

    Keys: pre_grid (S, T, R, C) int8, pre_locs / post_locs (S, T, N, 2), eaten / violations (S, T, N) bool,
    near (S, T, N) int: apples within two squares of the agent's new cell on the map before the move.
    """
    import jax
    import jax.numpy as jnp

    from overseer.socialjax.harvest_bots import bot_actions, kind_codes
    from overseer.socialjax.harvest_overseer import HarvestOverseer, OverseerConfig, apples_near, make_harvest_env

    env = make_harvest_env(num_inner_steps=steps + 1)  # never resets inside the recorded steps
    overseer = HarvestOverseer(env, OverseerConfig(k=k, chooser="random", penalty=0.0))
    codes = kind_codes(kinds)

    @jax.jit
    def run(key):
        k_reset, k_run = jax.random.split(key)
        _, s0 = overseer.reset(k_reset)

        def body(s, key):
            k_bot, k_step = jax.random.split(key)
            actions = bot_actions(k_bot, s, codes, sneaky_after=sneaky_after, explore=explore)
            pre_grid, pre_locs = s.env_state.grid, s.env_state.agent_locs[:, :2]
            _, s2, _, _, info = overseer.step(k_step, s, actions)
            post_locs = s2.env_state.agent_locs[:, :2]
            out = dict(pre_grid=pre_grid.astype(jnp.int8), pre_locs=pre_locs, post_locs=post_locs,
                       eaten=info["eaten"], violations=info["violations"], near=apples_near(pre_grid, post_locs))
            return s2, out

        _, trace = jax.lax.scan(body, s0, jax.random.split(k_run, steps))
        return trace

    keys = jax.vmap(jax.random.PRNGKey)(jnp.asarray(list(seeds)))
    trace = jax.vmap(run)(keys)
    return {key: np.asarray(v) for key, v in trace.items()}


# ------------------------------------------------------------------------------------------------ rendering
def cell_char(v: int) -> str:
    if v == 3:
        return "A"  # apple
    if v == 1:
        return "#"  # wall
    if v >= 6:
        return "P"  # a player (agents are stored as their index + 6)
    return "."


def patch(grid: np.ndarray, centre, radius: int = PATCH_RADIUS) -> list[list[str]]:
    """(2r+1) x (2r+1) characters around `centre` (row, col); cells off the map are '#'."""
    rows, cols = grid.shape
    out = []
    for r in range(centre[0] - radius, centre[0] + radius + 1):
        out.append([cell_char(int(grid[r, c])) if 0 <= r < rows and 0 <= c < cols else "#"
                    for c in range(centre[1] - radius, centre[1] + radius + 1)])
    return out


def patch_text(cells: list[list[str]]) -> str:
    r = len(cells) // 2
    lines = []
    for i, row in enumerate(cells):
        lines.append("    " + "".join(f"[{ch}]" if (i == r and j == r) else f" {ch} " for j, ch in enumerate(row)))
    return "\n".join(lines)


def count_diamond(cells: list[list[str]]) -> int:
    """Apples in the 12 cells within two squares of the patch centre (what the judge has to count)."""
    r = len(cells) // 2
    return sum(cells[r + dr][r + dc] == "A" for dr, dc in DIAMOND)


def move_name(pre, post) -> str:
    return MOVE_NAMES.get((int(post[0] - pre[0]), int(post[1] - pre[1])), "moved")


def render_summary(ep: dict, agent: int, t0: int, t1: int) -> str:
    lines = [f"Record of one agent, steps {t0 + 1} to {t1 + 1} ({t1 - t0 + 1} steps)."]
    for t in range(t0, t1 + 1):
        if ep["eaten"][t, agent]:
            lines.append(f"step {t + 1}: took an apple; other apples within two squares of it just before the move: "
                         f"{int(ep['near'][t, agent])}")
        else:
            lines.append(f"step {t + 1}: took no apple")
    return "\n".join(lines)


def render_raw(ep: dict, agent: int, t0: int, t1: int) -> str:
    lines = [f"Record of one agent, steps {t0 + 1} to {t1 + 1} ({t1 - t0 + 1} steps).",
             "Map key: A = apple, . = empty ground, P = a player, # = wall or edge of the map. North is up.",
             "On each step where the agent took an apple, the map is shown as it was just before the move, "
             "5 x 5 cells centred on the apple it took, which is marked [A]."]
    for t in range(t0, t1 + 1):
        pre, post = ep["pre_locs"][t, agent], ep["post_locs"][t, agent]
        mv = move_name(pre, post)
        if ep["eaten"][t, agent]:
            lines.append(f"step {t + 1}: {mv} onto an apple and took it. Map just before the move:")
            cells = patch(ep["pre_grid"][t], post)
            cells[PATCH_RADIUS][PATCH_RADIUS] = "A"  # the apple taken (it may have regrown on this very step)
            lines.append(patch_text(cells))
        else:
            lines.append(f"step {t + 1}: {mv}; took no apple")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ sampling
def window_candidates(eaten: np.ndarray, viol: np.ndarray, w: int):
    """For one agent: (end step t, number of eats, number of violations) for every full window of w steps."""
    ce = np.concatenate([[0], np.cumsum(eaten.astype(int))])
    cv = np.concatenate([[0], np.cumsum(viol.astype(int))])
    ends = np.arange(w - 1, len(eaten))
    return ends, ce[ends + 1] - ce[ends + 1 - w], cv[ends + 1] - cv[ends + 1 - w]


def build_dataset(windows=(1, 10, 50), per_window=1000, seeds_per_mix=8, steps=1000, seed=0, episodes=None):
    """Balanced examples (list of dicts) and a manifest. `episodes` can pass pre-recorded episodes (for tests)."""
    rng = np.random.default_rng(seed)
    if episodes is None:
        episodes = []
        for m, (mix, kinds) in enumerate(MIXES.items()):
            seeds = [1000 * (m + 1) + s for s in range(seeds_per_mix)]
            tr = record_episodes(kinds, seeds, steps=steps)
            for i, s in enumerate(seeds):
                episodes.append(dict(mix=mix, seed=s, kinds=kinds, **{k: v[i] for k, v in tr.items()}))
    examples, manifest = [], dict(seed=seed, steps=steps, seeds_per_mix=seeds_per_mix, mixes=list(MIXES),
                                  per_window=per_window, windows={})
    for w in windows:
        pools = {name: {} for name, _ in STRATA}  # stratum -> bot kind -> list of (episode index, agent, end)
        for e, ep in enumerate(episodes):
            for a, kind in enumerate(ep["kinds"]):
                ends, n_eat, n_viol = window_candidates(ep["eaten"][:, a], ep["violations"][:, a], w)
                for t, ne, nv in zip(ends, n_eat, n_viol):
                    stratum = "violation" if nv > 0 else ("honest_eats" if ne > 0 else "honest_idle")
                    pools[stratum].setdefault(kind, []).append((e, a, int(t)))
        counts = {}
        for name, share in STRATA:
            target = int(round(share * per_window))
            chosen = _round_robin(pools[name], target, rng)
            counts[name] = dict(chosen=len(chosen), available={k: len(v) for k, v in pools[name].items()})
            for e, a, t in chosen:
                examples.append(make_example(episodes[e], a, t - w + 1, t, w))
        manifest["windows"][str(w)] = counts
    for i, ex in enumerate(examples):
        ex["id"] = f"harvest_w{ex['window']}_{i:05d}"
    return examples, manifest


def _round_robin(pool: dict, target: int, rng) -> list:
    """Take up to `target` items, cycling through bot kinds so each kind is represented, without replacement."""
    order = {k: list(rng.permutation(len(v))) for k, v in sorted(pool.items())}
    chosen = []
    while len(chosen) < target and any(order.values()):
        for k in sorted(order):
            if order[k] and len(chosen) < target:
                chosen.append(pool[k][order[k].pop()])
    return chosen


def make_example(ep: dict, agent: int, t0: int, t1: int, w: int) -> dict:
    eats = [t for t in range(t0, t1 + 1) if ep["eaten"][t, agent]]
    return dict(
        dataset="harvest", window=w, mix=ep["mix"], episode_seed=int(ep["seed"]), agent=agent,
        bot_kind=ep["kinds"][agent], t_start=t0 + 1, t_end=t1 + 1,
        label=int(any(ep["violations"][t, agent] for t in eats)),
        n_eats=len(eats), n_violations=int(sum(bool(ep["violations"][t, agent]) for t in eats)),
        eat_nears=[int(ep["near"][t, agent]) for t in eats],
        renderings=dict(summary=render_summary(ep, agent, t0, t1), raw=render_raw(ep, agent, t0, t1)),
    )
