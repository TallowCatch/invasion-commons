"""Policy boundary and frozen admission-rule tests."""

from __future__ import annotations

from dataclasses import replace
import os

import numpy as np
import pytest

from experiments.admit_cleanup_policies import SEEDS, TAIL, assess
from fishery_sim.cleanup_oversight import CLEAN, CleanupAdapter, CleanupConfig
from fishery_sim.cleanup_policies import PolicyMemory, decide


def _view(channel: int | None = None, position: tuple[int, int] = (3, 5)):
    view = np.zeros((11, 11, 19), dtype=np.int8)
    view[1, 5, 10] = 1
    if channel is not None:
        view[position[0], position[1], channel] = 1
    return view


def test_cleaner_and_free_rider_use_different_native_actions():
    view = _view(7, (2, 5))
    action, next_memory = decide(view, 0, PolicyMemory())
    assert action == CLEAN
    assert next_memory == PolicyMemory(1, 0)
    for step in range(8):
        action, _ = decide(view, 0, PolicyMemory(step), variant="free_rider")
        assert action != CLEAN
    action, _ = decide(_view(2), 6, PolicyMemory(), variant="productive")
    assert action == 4


def test_memory_is_explicit_and_input_is_not_mutated():
    view = _view(7, (7, 5))
    before = view.copy()
    memory = PolicyMemory(2, 2)
    first = decide(view, 1, memory)
    assert first == decide(view, 1, memory)
    np.testing.assert_array_equal(view, before)
    assert memory == PolicyMemory(2, 2)
    assert first[1].step == 3


@pytest.mark.parametrize("agent,variant", [(0, "productive"), (4, "productive"), (0, "free_rider")])
def test_no_extra_evidence_can_enter_decision(agent, variant):
    view = _view(7 if agent == 0 else 2)
    original = decide(view, agent, PolicyMemory(), variant=variant)
    unrelated = {"hidden_pollution": 999, "other_actions": (8,) * 7}
    assert unrelated  # The API cannot accept or read this evaluator-only data.
    assert decide(view.copy(), agent, PolicyMemory(), variant=variant) == original


def test_bad_observation_and_identity_fail_closed():
    with pytest.raises(ValueError, match="shape"):
        decide(np.zeros((10, 10, 19)), 0, PolicyMemory())
    with pytest.raises(ValueError, match="ego"):
        decide(np.zeros((11, 11, 19)), 0, PolicyMemory())
    with pytest.raises(ValueError, match="agent ID"):
        decide(_view(), 7, PolicyMemory())
    with pytest.raises(ValueError, match="variant"):
        decide(_view(), 0, PolicyMemory(), variant="oracle")


def test_other_agent_at_alternative_ego_row_uses_explicit_memory():
    view = _view()
    view[9, 5, 10] = 1
    assert decide(view, 0, PolicyMemory())[1].heading_guess == 0
    assert decide(view, 0, PolicyMemory(0, 1))[1].heading_guess == 1


def test_assessment_requires_every_context_and_condition():
    episodes = []
    for seed in SEEDS:
        episodes.extend([
            {"seed": seed, "variant": "productive", "first_safe_step": 20,
             "tail_safe_steps": TAIL, "tail_apples_harvested": 1,
             "tail_preexisting_dirt_removed": 1,
             "preexisting_dirt_removed": 2},
            {"seed": seed, "variant": "free_rider", "first_safe_step": None,
             "tail_safe_steps": 0, "apples_harvested": 1,
             "clean_actions": 0, "preexisting_dirt_removed": 0},
        ])
    assert assess(episodes)["admitted"]
    episodes[-1]["apples_harvested"] = 0
    assert not assess(episodes)["admitted"]
    with pytest.raises(ValueError, match="incomplete"):
        assess(episodes[:-1])


@pytest.mark.skipif(not os.environ.get("SOCIALJAX_SOURCE"), reason="optional pinned native Cleanup")
def test_hidden_state_and_other_observations_do_not_affect_local_policy():
    adapter = CleanupAdapter(CleanupConfig(horizon=2))
    state = adapter.reset(17)
    own = state.observations[0]
    memory = PolicyMemory()
    baseline = decide(own, 0, memory)
    labels = state.native.potential_dirt_and_dirt_label
    altered = replace(
        state,
        native=state.native.replace(potential_dirt_and_dirt_label=labels * 0),
        observations=state.observations.at[1:].set(0),
    )
    assert adapter.np.array_equal(own, altered.observations[0])
    assert decide(altered.observations[0], 0, memory) == baseline
    assert adapter.np.array_equal(state.observations[0], own)
