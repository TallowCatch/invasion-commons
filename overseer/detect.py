"""Detectors: what an audit finds. One method, `detect(obs, ledger, plan) -> Findings`.

An agent is over-taking when taken > allowance + tol. The runners use two tolerances:
  - L2-L5: tol = OVER_T / MAX_CATCH = 0.06 / 6 (0.06 t above the allowance), run_l2_llm_agents.py;
  - S2-S5: tol = 1e-12 (any excess), run_s2_compliance_deterrence.py.
"""
from __future__ import annotations

import numpy as np

from overseer.core import Findings
from overseer.seeding import stable_seed


def _over(ledger, tol):
    taken = np.asarray(ledger.taken, float)
    if ledger.allowance is None:
        return np.zeros(len(taken), bool), np.zeros(len(taken))
    allowance = np.asarray(ledger.allowance, float)
    return taken > allowance + tol, taken - allowance


class Perfect:
    """Every audited over-taker is found; nobody else is."""

    def __init__(self, tol=1e-12):
        self.tol = tol

    def detect(self, obs, ledger, plan):
        over, excess = _over(ledger, self.tol)
        found = plan.audited & over
        return Findings(found, np.where(found, excess, 0.0))


class Noisy:
    """An audited over-taker is missed with probability `miss`; an audited agent within its allowance is wrongly
    flagged with probability `false_alarm`.

    Draws: rng = default_rng(stable_seed(seed, *prefix, window, *suffix)); the first n uniforms decide detection
    (found if u < 1 - miss), the next n decide false alarms (flagged if u < false_alarm). With prefix=(context,) and
    miss = 1 - s this is S3 Part B's detection (run_s3_threshold_timing_memory.py, Auditor.audit).
    """

    def __init__(self, miss, false_alarm, seed, prefix=(), suffix=(), tol=1e-12):
        self.miss, self.false_alarm, self.tol = float(miss), float(false_alarm), tol
        self.seed, self.prefix, self.suffix = int(seed), tuple(prefix), tuple(suffix)

    def detect(self, obs, ledger, plan):
        over, excess = _over(ledger, self.tol)
        n = len(over)
        u = np.random.default_rng(stable_seed(self.seed, *self.prefix, obs.window, *self.suffix)).random(2 * n)
        found = plan.audited & ((over & (u[:n] < 1 - self.miss)) | (~over & (u[n:] < self.false_alarm)))
        return Findings(found, np.where(found & over, excess, 0.0))
