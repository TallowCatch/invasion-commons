"""The overseer's data classes and the `Overseer` itself (overseer/DESIGN.md, "The oversight cycle").

Units are the game's own. In the text Fishery a request, allowance or take is a fraction of the 6 t per-agent maximum
(0 to 1), as in the runners; `ResourceModel.total` turns a vector of takes into the game's total (tonnes).

The overseer acts once per window:

    decision = overseer.begin_window(obs)          # allowances for this window
    outcome  = overseer.end_window(obs, ledger)    # audits, findings, sanctions; updates memory and budget
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ------------------------------------------------------------------ what the game supplies
class ResourceModel:
    """The game's resource, as the overseer sees it. An adapter subclasses this; the overseer never imports a game."""

    def safe_total(self, state) -> float:
        """The largest total take this window that keeps the game's target."""
        raise NotImplementedError

    def within_target(self, state, total: float) -> bool:
        """True if a total take of `total` keeps the target. Default: total <= safe_total(state).
        An adapter overrides it with the game's own test when the two differ in floating point."""
        return float(total) <= self.safe_total(state)

    def total(self, takes) -> float:
        """Total take of a vector of per-agent takes, in the units of safe_total."""
        return float(np.asarray(takes, float).sum())

    def harm(self, state, takes) -> bool:
        """The paper's harm line, for the window that starts in `state` and has these takes."""
        raise NotImplementedError

    def first_best_per_window(self, state) -> float:
        """The welfare a planner could take per window for ever (used for efficiency)."""
        raise NotImplementedError


@dataclass
class WindowObs:
    """What the overseer sees at the start of a window."""
    window: int
    state: Any                              # the game's resource state (Fishery: a FisherySnapshot)
    requests: np.ndarray | None = None      # per-agent requests, if the game has them
    positions: Any = None                   # spatial games (not used in v0)
    messages: Any = None                    # talk channels (not used in v0)


@dataclass
class Allowances:
    """The overseer's decision for one window."""
    caps: np.ndarray | None                 # per-agent cap; None = no cap
    scale: float | None = None              # the shared cut, if the policy uses one
    excluded: np.ndarray | None = None      # agents shut out this window (cap forced to 0)
    text: str | None = None                 # rules or announcements for the agents


@dataclass
class Ledger:
    """What happened in the window, per agent."""
    taken: np.ndarray
    allowance: np.ndarray | None
    requests: np.ndarray | None = None
    reported: np.ndarray | None = None      # reporting games only (not used in v0)


@dataclass
class AuditPlan:
    audited: np.ndarray                     # boolean per agent
    cost: float = 0.0                       # cost of these audits (Budget.cost_per_audit times their number)


@dataclass
class Findings:
    over: np.ndarray                        # over-taking detected (False for every agent not audited)
    excess: np.ndarray                      # taken - allowance for detected agents, else 0


@dataclass
class Sanctions:
    fines: np.ndarray                       # score change (positive = paid), never a change in the resource
    excluded: np.ndarray                    # agents to exclude from later windows

    @classmethod
    def zero(cls, n):
        return cls(np.zeros(n), np.zeros(n, bool))

    def __add__(self, other):
        return Sanctions(self.fines + other.fines, self.excluded | other.excluded)


@dataclass
class WindowOutcome:
    plan: AuditPlan
    findings: Findings
    sanctions: Sanctions

    @property
    def audited(self):
        return self.plan.audited

    @property
    def caught(self):
        return self.findings.over


# ------------------------------------------------------------------ the overseer
@dataclass
class Overseer:
    """A rule brain: allowances, audits, detection and sanctions composed from the parts in this package.

    Parts left as None do nothing: no allowance policy means no cap, no audit policy means nobody is audited.
    The first `Memory` among `sanctions` is the memory that the allowance policy reads.
    """
    n_agents: int
    resource: ResourceModel
    allowance: Any = None                   # an AllowancePolicy (allowance.py)
    audit: Any = None                       # an AuditPolicy (audit.py)
    detector: Any = None                    # a Detector (detect.py); None = Perfect()
    sanctions: tuple = ()                   # Sanction objects (sanction.py), applied in order and summed
    budget: Any = None                      # a Budget (budget.py); None = unlimited and free
    fines: np.ndarray = field(init=False)   # total fines per agent so far
    catches: np.ndarray = field(init=False)  # catches per agent so far (read by Targeted audits)
    excluded: np.ndarray = field(init=False)
    _decision: Allowances | None = field(init=False, default=None)

    def __post_init__(self):
        from overseer.budget import Budget
        from overseer.detect import Perfect
        from overseer.sanction import Memory
        self.detector = Perfect() if self.detector is None else self.detector
        self.budget = Budget() if self.budget is None else self.budget
        self.sanctions = tuple(self.sanctions)
        self.memory = next((s for s in self.sanctions if isinstance(s, Memory)), None)
        self.fines = np.zeros(self.n_agents)
        self.catches = np.zeros(self.n_agents, int)
        self.excluded = np.zeros(self.n_agents, bool)

    def begin_window(self, obs: WindowObs) -> Allowances:
        """Set this window's allowances."""
        if self.allowance is None:
            decision = Allowances(None)
        else:
            decision = self.allowance.allowances(obs, self.resource, self.memory)
        if self.excluded.any():
            caps = np.zeros(self.n_agents) if decision.caps is None else decision.caps.copy()
            caps[self.excluded] = 0.0
            decision = Allowances(caps, decision.scale, self.excluded.copy(), decision.text)
        self._decision = decision
        return decision

    def end_window(self, obs: WindowObs, ledger: Ledger) -> WindowOutcome:
        """Audit the window's ledger, detect over-taking, sanction it, and update memory and budget."""
        n = self.n_agents
        if self.audit is None:
            audited = np.zeros(n, bool)
        else:
            audited = np.asarray(self.audit.choose(obs, ledger, self), bool)
        audited = self.budget.limit(audited)
        plan = AuditPlan(audited, self.budget.charge(audited))
        findings = self.detector.detect(obs, ledger, plan)
        total = Sanctions.zero(n)
        for s in self.sanctions:
            total = total + s.apply(obs, ledger, findings)
        self.fines += total.fines
        self.catches += findings.over
        self.excluded = total.excluded.copy()  # an Exclude sanction keeps its own clock
        return WindowOutcome(plan, findings, total)

    @property
    def decision(self):
        """The allowances set by the last begin_window."""
        return self._decision
