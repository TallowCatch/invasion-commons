"""Allowance policies: how the overseer sets each agent's cap for a window.

Each policy has one method, `allowances(obs, resource, memory) -> Allowances`.
"""
from __future__ import annotations

import numpy as np

from overseer.core import Allowances

SCALES = (1.0, 0.75, 0.5, 0.25, 0.0)  # the shared-cut menu (fishery_sim/calibrated_oversight.py SCALES)


class NoCap:
    """No allowance: every agent may take what it likes."""

    def allowances(self, obs, resource, memory=None):
        return Allowances(None)


class SharedCut:
    """The current reviewer. It cuts every request by the same share: the largest scale in {1, 3/4, 1/2, 1/4, 0}
    whose planned total keeps the target (resource.within_target). Allowance = scale * request.

    With a `Memory` (sanction.py), agents with a record get the memory rule's allowance instead (L2 EM, S4 memory).
    This is fishery_choose_scale(cfg, stock, requests, "joint", target, collapsed) in the runners.
    """

    def __init__(self, scales=SCALES):
        self.scales = tuple(scales)

    def choose_scale(self, obs, resource):
        req = np.asarray(obs.requests, float)
        for scale in self.scales:
            if resource.within_target(obs.state, resource.total(req * scale)):
                return scale
        return 0.0

    def allowances(self, obs, resource, memory=None):
        req = np.asarray(obs.requests, float)
        scale = self.choose_scale(obs, resource)
        caps = scale * req
        if memory is not None:
            caps = memory.adjust(caps, scale, req)
        return Allowances(caps, scale)


class TargetedCut:
    """S5's memory_cut: the shared cut plans for what known over-takers are expected to take.

    A flagged agent (one with a 'share' record of mean dbar) is expected to take a + dbar * (p - a); every other agent
    is expected to take a = scale * p. The scale is the largest in the menu whose expected total keeps the target.
    Every agent, flagged or not, gets the plain allowance scale * p. With no flagged agent this is SharedCut.
    Copied from experiments/oversight/run_s5_stronger_attackers.py (memory_cut_scale).
    """

    def __init__(self, scales=SCALES):
        self.scales = tuple(scales)

    def allowances(self, obs, resource, memory=None):
        if memory is None or memory.rule != "share":
            raise ValueError("TargetedCut needs a Memory with the 'share' rule")
        p = np.asarray(obs.requests, float)
        flagged = memory.flagged()
        d = np.where(flagged, memory.mean_record(), 0.0)
        chosen = 0.0
        for scale in self.scales:
            a = p * scale
            expected = np.where(flagged, a + d * (p - a), a) if flagged.any() else a
            if resource.within_target(obs.state, resource.total(expected)):
                chosen = scale
                break
        return Allowances(chosen * p, chosen)
