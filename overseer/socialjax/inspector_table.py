"""A patterned inspector for Harvest at w = 1: miss and false-alarm chances that depend on the situation (23 Nov 2026).

The coin-flip inspector (`miss_rate`, `false_alarm_rate` in harvest_overseer.py / decentralised.py) misses a checked
violation with one fixed chance alpha and flags a checked honest harvest with one fixed chance beta. A real LLM's
mistakes depend on the record it reads (notes/llm_overseer/error_patterns/protocol.md). This module holds a table of
alpha(s) and beta(s) fitted to an LLM's verdicts, indexed by a few cheap features of the situation, and looks them up
in JAX for every agent each step.

Features (all on the map just before the move, in the 5 x 5 patch around the apple taken, which is what the raw
rendering shows the LLM; overseer/llm/harvest_records.patch):
  near     apples in the 12 cells within two squares (the rule's count; violation iff near < m)
  outer    apples in the other 12 cells of the patch (the corners and the cells next to them)
  players  players in the patch, the centre cell excluded (the eater itself is usually one of them)
  walls    walls or off-map cells in the patch
Off-map cells count as walls, as in the renderer. Agents are stored in the grid as their index + 6 (SocialJax
harvest_open.py: len(Items) = 6); walls are 1; apples 3.

Table JSON (written by experiments/llm_overseer/fit_error_model.py):
  {"features": [{"name": "near", "edges": [1, 2, 3, ...]}, {"name": "outer", "edges": [2, 5]}],
   "alpha": nested lists, one axis per feature, "beta": the same shape, ...metadata}
A feature's bin is the number of edges it is at or above (edges [2, 5]: bin 0 for 0-1, 1 for 2-4, 2 for 5+).
alpha is used for checked violations and beta for checked honest eats; entries outside their label's cells are never
read (alpha cells with near >= m, beta cells with near < m) and may be anything.
"""
from __future__ import annotations

import json

import jax.numpy as jnp
import numpy as np

from overseer.socialjax.harvest_overseer import APPLE, NEIGHBOURHOOD

WALL = 1
FIRST_AGENT = 6  # len(harvest_open.Items)
FEATURES = ("near", "outer", "players", "walls")
BOX = tuple((dr, dc) for dr in range(-2, 3) for dc in range(-2, 3) if (dr, dc) != (0, 0))
OUTER = tuple(d for d in BOX if d not in NEIGHBOURHOOD)  # 12 cells in the patch that the rule does not count


def _shift_sum(mask, offsets):
    rows, cols = mask.shape
    padded = jnp.pad(mask, 2)
    return sum(padded[2 + dr:2 + dr + rows, 2 + dc:2 + dc + cols] for dr, dc in offsets)


def feature_grids(grid):
    """dict name -> (rows, cols) int32 grid of each feature for an apple taken at that cell."""
    apple = (grid == APPLE).astype(jnp.int32)
    player = (grid >= FIRST_AGENT).astype(jnp.int32)
    rows, cols = grid.shape
    wall = jnp.pad((grid == WALL).astype(jnp.int32), 2, constant_values=1)  # off the map counts as a wall
    walls = sum(wall[2 + dr:2 + dr + rows, 2 + dc:2 + dc + cols] for dr, dc in BOX)
    return dict(near=_shift_sum(apple, NEIGHBOURHOOD), outer=_shift_sum(apple, OUTER),
                players=_shift_sum(player, BOX), walls=walls)


def features(grid, locs):
    """dict name -> (N,) int32 features for agents at (row, col) `locs` (N, 2) on map `grid`."""
    g = feature_grids(grid)
    return {k: v[locs[:, 0], locs[:, 1]] for k, v in g.items()}


class InspectorTable:
    """alpha(s), beta(s) lookup. Build from a JSON string (DecConfig.inspector_table holds the text)."""

    def __init__(self, text: str):
        spec = json.loads(text)
        self.names = tuple(f["name"] for f in spec["features"])
        if not self.names or any(n not in FEATURES for n in self.names):
            raise ValueError(f"inspector table features must be among {FEATURES}, got {self.names}")
        self.edges = tuple(tuple(int(e) for e in f["edges"]) for f in spec["features"])
        shape = tuple(len(e) + 1 for e in self.edges)
        alpha, beta = np.asarray(spec["alpha"], np.float32), np.asarray(spec["beta"], np.float32)
        if alpha.shape != shape or beta.shape != shape:
            raise ValueError(f"alpha and beta must have shape {shape} (bins per feature), got {alpha.shape}, "
                             f"{beta.shape}")
        if not ((0 <= alpha).all() and (alpha <= 1).all() and (0 <= beta).all() and (beta <= 1).all()):
            raise ValueError("table entries must be probabilities")
        self.shape = shape
        self.alpha, self.beta = jnp.asarray(alpha.ravel()), jnp.asarray(beta.ravel())
        self.strides = tuple(int(np.prod(shape[i + 1:])) for i in range(len(shape)))

    def cell(self, feats):
        """(N,) flat cell index from a features dict."""
        idx = 0
        for name, edges, stride in zip(self.names, self.edges, self.strides):
            x = feats[name]
            b = sum((x >= e).astype(jnp.int32) for e in edges) if edges else jnp.zeros_like(x)
            idx = idx + b * stride
        return idx

    def rates(self, grid, locs):
        """(miss chance, false-alarm chance), each (N,), for agents that took an apple at `locs` on map `grid`."""
        c = self.cell(features(grid, locs))
        return self.alpha[c], self.beta[c]


def constant_table(alpha: float, beta: float, near_edges=(1, 2, 3)) -> str:
    """A table with the same alpha and beta in every cell (for tests: it must equal the coin flip)."""
    shape = (len(near_edges) + 1,)
    return json.dumps(dict(features=[dict(name="near", edges=list(near_edges))], alpha=[alpha] * shape[0],
                           beta=[beta] * shape[0]))
