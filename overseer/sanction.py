"""Sanctions: what happens to an agent found over-taking. One method, `apply(obs, ledger, findings) -> Sanctions`.

A fine changes only the agent's score. It never changes the resource: the game's stock update uses what was taken.
"""
from __future__ import annotations

import numpy as np

from overseer.core import Sanctions


class Fine:
    """A flat fine F per catch (the current reviewer)."""

    def __init__(self, amount):
        self.amount = float(amount)

    def apply(self, obs, ledger, findings):
        s = Sanctions.zero(len(findings.over))
        s.fines[findings.over] += self.amount
        return s


# ------------------------------------------------------------------ memory
def targeted_allowance(scale, dbar, p):
    """Allowance whose expected take, for an agent that keeps over-taking a share dbar of the cut, is scale * p.
    Copied from experiments/oversight/run_s4_adaptive_memory_cost.py."""
    if dbar >= 1 - 1e-12:
        return 0.0
    return max(0.0, (scale - dbar) / (1 - dbar)) * p


def observed_overtake(taken, allowance, p):
    """Share of the cut that was taken anyway; None if nothing was cut. Copied from run_s4_adaptive_memory_cost.py."""
    return (taken - allowance) / (p - allowance) if p - allowance > 1e-12 else None


class Memory:
    """The overseer remembers caught over-takes and lowers the agent's later allowances. No fine.

    Rules (run_l2_llm_agents.py, MEMORY_RULE):
      - "share" (S4; L2's first EM games): record the share of the cut taken anyway, d = (taken - a) / (p - a),
        capped at 1 (nothing is recorded if nothing was cut). Later allowance = targeted_allowance(scale, mean d, p).
      - "excess" (L2 Amendment 5, called "tonnes" in the episode files): record taken - a, in the game's units.
        Later allowance = max(0, scale * p - mean excess).
    Allowances are adjusted by the allowance policy (SharedCut) through `adjust`.
    """

    RULES = ("share", "excess")

    def __init__(self, rule="excess", n_agents=None, cap_share=True):
        rule = {"tonnes": "excess"}.get(rule, rule)
        if rule not in self.RULES:
            raise ValueError(f"Unknown memory rule {rule}")
        self.rule, self.cap_share = rule, cap_share
        self.record = None if n_agents is None else [[] for _ in range(n_agents)]

    def _ensure(self, n):
        if self.record is None:
            self.record = [[] for _ in range(n)]

    def flagged(self):
        return np.array([bool(r) for r in self.record], bool)

    def mean_record(self):
        return np.array([float(np.mean(r)) if r else 0.0 for r in self.record])

    def apply(self, obs, ledger, findings):
        n = len(findings.over)
        self._ensure(n)
        req = None if ledger.requests is None else np.asarray(ledger.requests, float)
        for i in np.flatnonzero(findings.over):
            taken, a = float(ledger.taken[i]), float(ledger.allowance[i])
            if self.rule == "share":
                d = observed_overtake(ledger.taken[i], ledger.allowance[i], req[i])
                if d is not None:
                    self.record[i].append(min(1.0, d) if self.cap_share else d)
            else:
                self.record[i].append(taken - a)
        return Sanctions.zero(n)

    def adjust(self, caps, scale, requests):
        """This window's allowances for agents with a record (others keep `caps`)."""
        self._ensure(len(caps))
        caps = np.array(caps, float)
        for i, r in enumerate(self.record):
            if not r:
                continue
            if self.rule == "share":
                caps[i] = targeted_allowance(scale, float(np.mean(r)), requests[i])
            else:
                caps[i] = max(0.0, caps[i] - float(np.mean(r)))
        return caps


class Exclude:
    """A caught agent is shut out (allowance 0) from the next window on, for `windows` windows (None = for good)."""

    def __init__(self, windows=None):
        self.windows = windows
        self.until = None

    def apply(self, obs, ledger, findings):
        n = len(findings.over)
        if self.until is None:
            self.until = np.full(n, -np.inf)
        end = np.inf if self.windows is None else obs.window + self.windows
        self.until[findings.over] = end
        s = Sanctions.zero(n)
        s.excluded = self.until > obs.window
        return s


class Leniency:
    """Stub for L7: an agent who reports another's over-take has its own fine waived. It needs a talk or report
    channel, which v0 does not have, so `apply` refuses to run rather than guess the rule."""

    def __init__(self, amount):
        self.amount = float(amount)

    def apply(self, obs, ledger, findings):
        raise NotImplementedError("Leniency needs a report channel (L7); not in overseer v0")
