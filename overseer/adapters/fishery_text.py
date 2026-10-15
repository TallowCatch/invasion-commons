"""Adapter for the text Fishery (fishery_sim and the L2/L4 game loop).

Units: requests, allowances and takes are fractions of the per-agent maximum (cfg.max_harvest_per_agent = 6 t), as in
the runners. Totals are in tonnes. The state is a fishery_sim.fishery_oversight.FisherySnapshot.
"""
from __future__ import annotations

import numpy as np

from fishery_sim.calibrated_oversight import fishery_target_ok
from fishery_sim.fishery_oversight import FisherySnapshot, transition
from overseer.core import ResourceModel, WindowObs


class FisheryResource(ResourceModel):
    """The Fishery's stock under logistic regrowth.

    target "msy" (the paper's): the stock left after fishing must stay at or above K/2.
    `within_target` calls the runners' own test (fishery_target_ok) so that SharedCut picks exactly the same scale as
    fishery_choose_scale(cfg, stock, requests, "joint", target, collapsed); a collapsed stock allows nothing.
    """

    def __init__(self, cfg, target="msy"):
        self.cfg, self.target = cfg, target

    def safe_total(self, state):
        if state.collapsed:
            return 0.0
        if self.target == "msy":
            return max(0.0, float(state.stock) - self.cfg.stock_max / 2)
        raise NotImplementedError(f"safe_total for target {self.target}")

    def within_target(self, state, total):
        return (not state.collapsed) and fishery_target_ok(self.cfg, state.stock, total, self.target)

    def total(self, takes):
        # same operation order as fishery_predicted_total: (take * max).sum()
        return float((np.asarray(takes, float) * self.cfg.max_harvest_per_agent).sum())

    def harm(self, state, takes):
        """Harm line (L2's msy_break): the stock left after fishing is below K/2."""
        return bool(state.stock - np.asarray(takes, float).sum() * self.cfg.max_harvest_per_agent
                    < self.cfg.stock_max / 2 - 1e-9)

    def first_best_per_window(self, state=None):
        """rK/4 tonnes per round (17.5 t with r = 0.7, K = 100)."""
        return self.cfg.regen_rate * self.cfg.stock_max / 4


class FisheryText:
    """The game side: the starting state, one round's stock update, and the overseer's observation."""

    def __init__(self, cfg, target="msy"):
        self.cfg = cfg
        self.resource = FisheryResource(cfg, target)

    def initial_state(self):
        return FisherySnapshot(self.cfg.stock_init)

    def observe(self, t, state, requests):
        return WindowObs(window=t, state=state, requests=np.asarray(requests, float))

    def step(self, state, taken):
        """(next state, per-agent payoffs in tonnes). Uses what was taken; fines never enter the stock."""
        future, payoffs, _ = transition(self.cfg, state, taken)
        return future, payoffs
