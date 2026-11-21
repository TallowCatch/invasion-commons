"""The rule each overseer enforces, per SocialJax game (GameRules for overseer/socialjax/decentralised.py).

A GameRules object holds the SocialJax env and says, after every step, which agents broke the game's rule:
  init(env_state) -> aux              the rule's own state (e.g. steps since each agent last cleaned)
  judge(aux, prev_env_state, actions, env_state, raw_reward, info) -> (aux, signals)
signals:
  violation (N,) bool   the rule was broken this step (ground truth)
  suspect   (N,) bool   an act that is a violation unless something in the last `lookback` steps clears it
  clear     (N,) bool   an act that clears the agent for the next `lookback` steps (e.g. cleaning)
  honest    (N,) bool   an honest act a false alarm can fall on (an apple or coin taken within the rule)
  taken     (N,) bool   goods taken this step (apples, coins)
  location_score (N,)   higher = standing where violations happen (the "location" chooser)
  resource  ()          goods left on the map
  extra     dict        game-specific per-step logs
`lookback` is how many steps of an agent's record a check needs to prove one violation (1 = the act alone).
Rewards are individual in every game (shared_rewards=False), so each is a social dilemma.

Games:
- harvest: SocialJax Harvest Open. Rule: don't eat an apple that leaves fewer than m apples within two squares.
  `scale=2` tiles the map side by side (16 x 44) with 14 agents, for more agents per region.
- coins: SocialJax Coin Game (2 agents; the env hard-codes two players). Rule: don't take a coin of the other
  player's colour. A coin pays its taker +1; a coin of the other's colour also costs the other player 2
  (payoff_matrix [[1, 1, -2], [1, 1, -2]], times num_agents = 2 with individual rewards: +2 to the taker, -4 to the
  owner). So taking the other's coin gains the taker 2 and loses the group 2.
- cleanup: SocialJax Clean Up. Rule: don't eat an apple while the river's pollution is above `threshold` unless you
  cleaned (fired the cleaning beam and removed at least one piece of dirt) in the last `clean_window` (W) steps.
  This is free-riding: apples regrow only while pollution is below 0.4 (clean_up.py), and cleaning pays nothing.
  One check sees one act of eating, but proving the violation needs W steps of the record, so a short look cannot
  tell a free-rider from an honest agent who cleaned before the look started.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from overseer.socialjax.harvest_overseer import APPLE, apple_count_grid, apples_near, make_harvest_env

HARVEST_MAP = [
    "AAA    A      A    AAA",
    "AA    AAA    AAA    AA",
    "A    AAAAA  AAAAA    A",
    "      AAA    AAA      ",
    "       A      A       ",
    "  A                A  ",
    " AAA  Q        Q  AAA ",
    "AAAAA            AAAAA",
    " AAA              AAA ",
    "  A                A  ",
    "                      ",
    "                      ",
    "                      ",
    "  PPPPPPPPPPPPPPPPPP  ",
    " PPPPPPPPPPPPPPPPPPPP ",
    "PPPPPPPPPPPPPPPPPPPPPP",
]  # harvest_open.py default map (16 x 22)

GAMES = ("harvest", "coins", "cleanup")


class HarvestRules:
    name = "harvest"
    lookback = 1
    stay = 6

    def __init__(self, num_inner_steps=1000, min_neighbours=3, scale=1, num_agents=None):
        if scale == 1:
            kw = {}
            n = num_agents or 7
        else:
            amap = [row * scale for row in HARVEST_MAP]
            kw = dict(map_ASCII=amap, grid_size=(len(amap), len(amap[0])))
            n = num_agents or 7 * scale
        self.env = make_harvest_env(num_inner_steps=num_inner_steps, num_agents=n, **kw)
        self.shape = (16, 22 * scale)
        self.m = min_neighbours
        self.reward_unit = float(self.env.num_agents)  # one apple pays num_agents

    def positions(self, env_state):
        return env_state.agent_locs[:, :2].astype(jnp.int32)

    def init(self, env_state):
        return {}

    def judge(self, aux, prev, actions, env_state, raw, info):
        eaten = raw > 0
        locs = self.positions(env_state)
        # judged on the map the agent saw when it moved (harvest_overseer.py)
        violation = eaten & (apples_near(prev.grid, locs) < self.m)
        near = apples_near(env_state.grid, locs)
        apples = jnp.sum(env_state.grid == APPLE)
        return aux, dict(violation=violation, suspect=violation, clear=jnp.zeros_like(violation), honest=eaten,
                         taken=eaten, location_score=-near.astype(jnp.float32), resource=apples,
                         extra=dict(eaten=eaten, apples=apples))


class CoinsRules:
    """Coin Game. Agent 0 owns red coins (grid code 3), agent 1 green coins (code 4)."""
    name = "coins"
    lookback = 1
    stay = 6
    OWN = (3, 4)

    def __init__(self, num_inner_steps=1000):
        import socialjax
        self.env = socialjax.make("coin_game", num_inner_steps=num_inner_steps, shared_rewards=False)
        self.shape = (self.env.GRID_SIZE_ROW, self.env.GRID_SIZE_COL)
        self.reward_unit = 2.0  # a coin pays its taker 1 x num_agents

    def positions(self, env_state):
        return env_state.agent_locs[:, :2].astype(jnp.int32)

    def init(self, env_state):
        return {}

    def judge(self, aux, prev, actions, env_state, raw, info):
        own = info["eat_own_coins"].astype(jnp.float32) > 0  # took a coin of its own colour
        # raw = 2 own + 2 other_taken - 4 stolen_from_me, and an agent takes at most one coin a step, so with no own
        # coin, raw is +2 or -2 exactly when it took the other's (+2 - 4 if its own coin was taken in the same step)
        took_other = ~own & (jnp.isclose(raw, 2.0) | jnp.isclose(raw, -2.0))
        stolen_from = took_other[::-1]
        grid = env_state.grid
        other_colour = jnp.array(self.OWN[::-1])
        locs = self.positions(env_state)
        rows, cols = grid.shape
        rr, cc = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")

        def near_other(loc, colour):
            d = jnp.abs(rr - loc[0]) + jnp.abs(cc - loc[1])
            return jnp.sum((grid == colour) & (d <= 2))

        coins = jnp.sum((grid == 3) | (grid == 4))
        return aux, dict(violation=took_other, suspect=took_other, clear=jnp.zeros_like(took_other), honest=own,
                         taken=own | took_other,
                         location_score=jax.vmap(near_other)(locs, other_colour).astype(jnp.float32),
                         resource=coins, extra=dict(own_coins=own, stolen_from=stolen_from, coins=coins))


# Clean Up constants (clean_up.py at 18e2171)
CU_APPLE, CU_DIRT, CU_ZAP_CLEAN = 3, 8, 8
CU_STEP = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])  # facing direction 0-3 -> (row, col) step


def beam_cells(loc, d):
    """The 4 cells a cleaning beam fired from loc facing d reaches: 1 and 2 ahead, and ahead-right, ahead-left."""
    step = jnp.asarray(CU_STEP)
    ahead = loc + step[d]
    return jnp.stack([ahead, loc + 2 * step[d], ahead + step[(d + 1) % 4], ahead + step[(d - 1) % 4]])


class CleanupRules:
    name = "cleanup"
    stay = 6

    def __init__(self, num_inner_steps=1000, clean_window=100, threshold=0.2, num_agents=7):
        import socialjax
        self.env = socialjax.make("clean_up", num_inner_steps=num_inner_steps, num_agents=num_agents,
                                  shared_rewards=False)
        self.shape = (self.env.GRID_SIZE_ROW, self.env.GRID_SIZE_COL)
        self.lookback = int(clean_window)
        self.threshold = float(threshold)
        self.denominator = len(self.env.POTENTIAL_DIRT) + len(self.env.DIRT) + len(self.env.RIVER)
        self.orchard_top = int(np.asarray(self.env.POTENTIAL_APPLE)[:, 0].min())
        self.reward_unit = float(num_agents)  # one apple pays num_agents

    def positions(self, env_state):
        return env_state.agent_locs[:, :2].astype(jnp.int32)

    def dirt_grid(self, env_state):
        locs = env_state.potential_dirt_and_dirt_locs
        return jnp.zeros(self.shape, bool).at[locs[:, 0], locs[:, 1]].set(
            env_state.potential_dirt_and_dirt_label == CU_DIRT)

    def pollution(self, env_state):
        return jnp.sum(env_state.potential_dirt_and_dirt_label == CU_DIRT) / self.denominator

    def init(self, env_state):
        n = self.env.num_agents
        return dict(since_clean=jnp.full(n, self.lookback, jnp.int32))  # nobody has cleaned yet

    def judge(self, aux, prev, actions, env_state, raw, info):
        rows, cols = self.shape
        cleaned_cells = self.dirt_grid(prev) & ~self.dirt_grid(env_state)  # dirt only leaves by cleaning

        def effective(loc_dir, a):
            cells = beam_cells(loc_dir[:2].astype(jnp.int32), loc_dir[2].astype(jnp.int32))
            inside = (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)
            hit = cleaned_cells[jnp.clip(cells[:, 0], 0, rows - 1), jnp.clip(cells[:, 1], 0, cols - 1)]
            return (a == CU_ZAP_CLEAN) & jnp.any(hit & inside)

        clean = jax.vmap(effective)(env_state.agent_locs, jnp.asarray(actions))
        since_clean = jnp.where(clean, 0, jnp.minimum(aux["since_clean"] + 1, 1_000_000))
        eaten = raw > 0
        polluted = self.pollution(prev) > self.threshold
        suspect = eaten & polluted
        violation = suspect & (since_clean >= self.lookback)
        locs = self.positions(env_state)
        apples = jnp.sum(env_state.grid == CU_APPLE)
        in_orchard = (locs[:, 0] >= self.orchard_top - 1).astype(jnp.float32)
        score = in_orchard * (1.0 + jnp.minimum(since_clean, self.lookback) / self.lookback)
        return dict(since_clean=since_clean), dict(
            violation=violation, suspect=suspect, clear=clean, honest=eaten & ~violation, taken=eaten,
            location_score=score, resource=apples,
            extra=dict(eaten=eaten, apples=apples, cleaned=clean, pollution=self.pollution(env_state)))


def make_rules(game, num_inner_steps=1000, **kw):
    if game == "harvest":
        return HarvestRules(num_inner_steps=num_inner_steps, **kw)
    if game == "harvest_large":
        return HarvestRules(num_inner_steps=num_inner_steps, scale=2, **kw)
    if game == "coins":
        return CoinsRules(num_inner_steps=num_inner_steps, **kw)
    if game == "cleanup":
        return CleanupRules(num_inner_steps=num_inner_steps, **kw)
    raise ValueError(f"unknown game {game!r}; choose from harvest, harvest_large, coins, cleanup")


__all__ = ["GAMES", "HarvestRules", "CoinsRules", "CleanupRules", "make_rules", "beam_cells", "apple_count_grid"]
