"""Audit policies: which agents the overseer checks after a window.

Each policy has one method, `choose(obs, ledger, overseer) -> boolean array`.

Seeds. Every draw comes from numpy.random.default_rng(stable_seed(seed, *prefix, window, *suffix)). `prefix` usually
holds the study's context (population); `suffix` holds anything else the runner put after the window index.
  - L2-L5 (run_l2_llm_agents.py):  UniformRandom(q, SEEDS["audit"], prefix=(context,))
  - S3/S4 (run_s3_threshold_timing_memory.py):  UniformRandom(q, SEEDS["audit"], prefix=(context,), suffix=(round(q, 6),))
  - S2 'rand k' (run_s2_compliance_deterrence.py):  FixedCount(k, SEEDS["audit"], prefix=(game, context), suffix=(protocol,))
"""
from __future__ import annotations

import numpy as np

from overseer.seeding import stable_seed


class _Seeded:
    def __init__(self, seed, prefix=(), suffix=()):
        self.seed, self.prefix, self.suffix = int(seed), tuple(prefix), tuple(suffix)

    def rng(self, window):
        return np.random.default_rng(stable_seed(self.seed, *self.prefix, window, *self.suffix))


class UniformRandom(_Seeded):
    """Each agent is audited with probability q, independently: one uniform per agent and window, audited if u < q.

    Because the draw does not depend on q (unless q is put in `suffix`, as S3 did), the agents audited at a smaller q
    are a subset of those audited at a larger q (L4's nested audits).
    """

    def __init__(self, q, seed, prefix=(), suffix=()):
        super().__init__(seed, prefix, suffix)
        self.q = float(q)

    def choose(self, obs, ledger, overseer):
        return self.rng(obs.window).random(overseer.n_agents) < self.q


class FixedCount(_Seeded):
    """Exactly k agents audited per window, chosen at random (S2's 'rand1', 'rand2')."""

    def __init__(self, k, seed, prefix=(), suffix=()):
        super().__init__(seed, prefix, suffix)
        self.k = int(k)

    def choose(self, obs, ledger, overseer):
        n = overseer.n_agents
        audited = np.zeros(n, bool)
        audited[self.rng(obs.window).permutation(n)[:self.k]] = True
        return audited


class Targeted(_Seeded):
    """The k most suspect agents are audited (S2's 'targ1'). Ties are broken by a tiny random jitter, as in S2.

    `score(obs, ledger, overseer) -> array` gives each agent's suspicion; by default, its number of past catches.
    """

    def __init__(self, k, seed, prefix=(), suffix=(), score=None):
        super().__init__(seed, prefix, suffix)
        self.k = int(k)
        self.score = score

    def choose(self, obs, ledger, overseer):
        n = overseer.n_agents
        s = overseer.catches.astype(float) if self.score is None else np.asarray(self.score(obs, ledger, overseer), float)
        jitter = self.rng(obs.window).uniform(0, 1e-9, n)
        audited = np.zeros(n, bool)
        audited[np.argsort(-(s + jitter))[:self.k]] = True
        return audited
