"""Audit budget: how many audits the overseer may make, what they cost, and the share of agent-windows checked.

W1 (studies/W1_welfare_cost_rescoring/protocol.md §2): audit budget = audits / (N * T), as a % of agent-rounds;
monitoring cost = c per audit.
"""
from __future__ import annotations

import numpy as np


class Budget:
    """Counts audits and agent-windows; optionally caps audits per window or per episode.

    When a cap binds, the audits kept are the first ones in agent order (v0; no study so far uses a cap).
    """

    def __init__(self, per_window=None, per_episode=None, cost_per_audit=0.0):
        self.per_window, self.per_episode, self.cost_per_audit = per_window, per_episode, float(cost_per_audit)
        self.audits = 0
        self.agent_windows = 0
        self.windows = 0

    def limit(self, audited):
        audited = np.asarray(audited, bool).copy()
        room = []
        if self.per_window is not None:
            room.append(int(self.per_window))
        if self.per_episode is not None:
            room.append(max(0, int(self.per_episode) - self.audits))
        if room and audited.sum() > min(room):
            keep = np.flatnonzero(audited)[:min(room)]
            audited[:] = False
            audited[keep] = True
        return audited

    def charge(self, audited):
        """Record one window's audits; returns their cost."""
        k = int(np.asarray(audited, bool).sum())
        self.audits += k
        self.agent_windows += len(audited)
        self.windows += 1
        return self.cost_per_audit * k

    @property
    def cost(self):
        return self.cost_per_audit * self.audits

    @property
    def percent(self):
        """Audits as a % of agent-windows so far."""
        return 100.0 * self.audits / self.agent_windows if self.agent_windows else 0.0
