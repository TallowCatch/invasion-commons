"""Fixed observation-only Cleanup actor policies, not oversight monitors.

The native tensor is the sole sensory input. Channel indices follow the pinned
SocialJax one-hot representation; agent positions, dirt labels and metrics from
the simulator state are deliberately absent from this API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fishery_sim.cleanup_oversight import CLEAN, NUM_AGENTS


@dataclass(frozen=True)
class PolicyMemory:
    step: int = 0
    heading_guess: int | None = None


def _heading(view: Any, old: int | None) -> tuple[int, int]:
    # Upstream's rotated, forward-offset view puts the agent at one of these
    # two locations. Its self/other bit is inconsistent, so accept either.
    top = bool(view[1, 5, 9] or view[1, 5, 10])
    bottom = bool(view[9, 5, 9] or view[9, 5, 10])
    if not top and not bottom:
        raise ValueError("native observation does not identify an ego row")
    # Another agent can occupy the alternative ego coordinate. In that case
    # use remembered orientation, or a fixed initial tie-break, not hidden state.
    row = (1 if old is None or old % 2 == 0 else 9) if top and bottom else (1 if top else 9)
    candidates = (0, 2) if row == 1 else (1, 3)
    return row, old if old in candidates else candidates[0]


def _absolute_action(view_delta: tuple[int, int], heading: int) -> int:
    row, col = view_delta
    # Inverse of upstream's k*90-degree observation rotation.
    dr, dc = ((row, col), (col, -row), (-row, -col), (-col, row))[heading]
    if abs(dr) >= abs(dc):
        return 4 if dr > 0 else 5
    return 3 if dc > 0 else 2


def decide(
    observation: Any, agent_id: int, memory: PolicyMemory,
    *, variant: str = "productive",
) -> tuple[int, PolicyMemory]:
    """Return one native action and all recurrent state, without evaluator data."""
    if variant not in ("productive", "free_rider"):
        raise ValueError("unknown policy variant")
    if type(agent_id) is not int or not 0 <= agent_id < NUM_AGENTS:
        raise ValueError("invalid agent ID")
    if not isinstance(memory, PolicyMemory) or memory.step < 0:
        raise ValueError("invalid policy memory")
    import numpy as np

    view = np.asarray(observation)
    if view.shape != (11, 11, 19):
        raise ValueError("expected pinned native Cleanup observation shape")
    ego_row, heading = _heading(view, memory.heading_guess)
    ego = (ego_row, 5)
    cleaner = variant == "productive" and agent_id < 4
    target_channel = 7 if cleaner else 2  # Native dirt / apple.
    targets = np.argwhere(view[:, :, target_channel] != 0)
    targets = sorted(
        ((int(r), int(c)) for r, c in targets),
        key=lambda p: (abs(p[0] - ego[0]) + abs(p[1] - ego[1]), p),
    )

    if cleaner and targets and (
        abs(targets[0][0] - ego[0]) + abs(targets[0][1] - ego[1]) <= 3
    ):
        # Sweep the native four-cell beam through every heading; visible dirt
        # alone does not prove that a particular beam will hit it.
        action = CLEAN if memory.step % 2 == 0 else 1
    elif targets:
        target = targets[0]
        action = _absolute_action((target[0] - ego[0], target[1] - ego[1]), heading)
    else:
        # Fixed exploration pattern, shared by both variants. It can stall at
        # walls; the admission gate, not an oracle, determines competence.
        action = (4, 3, 5, 2)[(memory.step // 8 + agent_id) % 4]

    next_heading = (heading - 1) % 4 if action == 1 else heading
    return action, PolicyMemory(memory.step + 1, next_heading)
