"""Pure contract tests always run; native tests require the optional checkout.

SOCIALJAX_SOURCE being explicitly set opts into strict native tests: broken
dependencies/revision are then failures, not silently skipped integration.
"""

from __future__ import annotations

from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import fishery_sim.cleanup_oversight as cleanup
from fishery_sim.cleanup_oversight import (
    CLEAN, NOOP, NUM_AGENTS, CleanupAdapter, CleanupConfig, CleanupState,
    CleanupUnavailable, effective_spawn_probability, replace_with_noop,
    validate_actions,
)


def test_import_needs_no_optional_dependencies():
    root = str(Path(__file__).resolve().parents[1])
    code = (
        f"import sys; sys.path.insert(0, {root!r}); "
        "import fishery_sim.cleanup_oversight; "
        "assert not any(x in sys.modules for x in ('jax', 'numpy', 'flax', 'socialjax'))"
    )
    subprocess.run([sys.executable, "-S", "-c", code], check=True)


def test_optional_source_guard(monkeypatch):
    monkeypatch.delenv("SOCIALJAX_SOURCE", raising=False)
    with pytest.raises(CleanupUnavailable, match="requirements-cleanup.txt"):
        CleanupAdapter()


def test_optional_import_guard_cleans_up_search_path(monkeypatch, tmp_path):
    monkeypatch.setattr(cleanup, "_verified_source", lambda _: tmp_path)

    def missing(name):
        raise ModuleNotFoundError(f"No module named {name}")

    before = sys.path[:]
    monkeypatch.setattr(cleanup.importlib, "import_module", missing)
    with pytest.raises(CleanupUnavailable, match="optional dependency import failed"):
        CleanupAdapter()
    assert sys.path == before


@pytest.mark.parametrize("revision,dirty,message", [
    ("wrong", "", "revision mismatch"),
    (cleanup.SOCIALJAX_REVISION, " M socialjax/environments/movement.py", "modified"),
])
def test_source_pin_and_cleanliness_enforced(monkeypatch, tmp_path, revision, dirty, message):
    def git_output(command, **kwargs):
        return revision if "rev-parse" in command else dirty

    monkeypatch.setattr(cleanup.subprocess, "check_output", git_output)
    with pytest.raises(CleanupUnavailable, match=message):
        cleanup._verified_source(tmp_path)


@pytest.mark.parametrize("actions", [(0,) * 6, (0,) * 8, (9,) * 7, (-1,) * 7, (1.0,) * 7, (True,) * 7])
def test_invalid_actions(actions):
    with pytest.raises(ValueError):
        validate_actions(actions)


def test_only_noop_replacements_are_authorized():
    proposals = (0, 1, 2, 3, 4, 7, CLEAN)
    for mask in range(2**NUM_AGENTS):
        rejected = [i for i in range(NUM_AGENTS) if mask & (1 << i)]
        executed = replace_with_noop(proposals, rejected)
        assert all(a == (NOOP if i in rejected else proposals[i]) for i, a in enumerate(executed))
    assert proposals == (0, 1, 2, 3, 4, 7, CLEAN)
    for bad in (-1, 7, True, 1.5):
        with pytest.raises(ValueError):
            replace_with_noop(proposals, [bad])


def test_spawn_target_matches_native_threshold_test():
    assert effective_spawn_probability(0, 100) == 0.05
    assert effective_spawn_probability(20, 100) == 0.025
    assert effective_spawn_probability(40, 100) == 0
    assert effective_spawn_probability(80, 100) == 0
    with pytest.raises(ValueError):
        effective_spawn_probability(1, 0)
    with pytest.raises(ValueError):
        effective_spawn_probability(101, 100)


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"horizon": True}, {"minimum_spawn_fraction": 0}, {"minimum_spawn_fraction": float("nan")}])
def test_config_validation(kwargs):
    with pytest.raises(ValueError):
        CleanupConfig(**kwargs)


@pytest.fixture(scope="module")
def adapter():
    if not os.environ.get("SOCIALJAX_SOURCE"):
        pytest.skip("optional real SocialJax tests: set SOCIALJAX_SOURCE")
    return CleanupAdapter(CleanupConfig(horizon=3))


def test_real_reset_step_snapshot_parity(adapter):
    state = adapter.reset(17)
    frozen = adapter.snapshot(state)
    assert adapter.snapshot(adapter.reset(17)) == frozen
    restored = adapter.restore(frozen)
    assert adapter.snapshot(restored) == frozen
    actions = (0, 1, 2, 3, 4, 5, CLEAN)
    next_state, rewards, done = adapter.transition(state, actions, 29)
    replay, replay_rewards, replay_done = adapter.transition(restored, actions, 29)
    assert adapter.snapshot(next_state) == adapter.snapshot(replay)
    assert (rewards, done) == (replay_rewards, replay_done)
    obs, native, reward, native_done, _ = adapter.env.step_env(
        adapter.jax.random.PRNGKey(29), state.native, adapter.jax.numpy.asarray(actions)
    )
    assert adapter.snapshot(next_state) == adapter.snapshot(CleanupState(native, obs, state.contract))
    assert rewards == tuple(float(r) for r in reward)
    assert done is False and not bool(native_done["__all__"])
    assert adapter.snapshot(state) == frozen
    assert set(json.loads(frozen)["native"]) == {f.name for f in fields(state.native)}
    for seed in (30, 31):
        next_state, rewards, done = adapter.transition(next_state, actions, seed)
        replay, replay_rewards, replay_done = adapter.transition(replay, actions, seed)
        assert (adapter.snapshot(next_state), rewards, done) == (adapter.snapshot(replay), replay_rewards, replay_done)


def test_real_noop_enum_and_action_space(adapter):
    native = cleanup.importlib.import_module("socialjax.environments.cleanup.clean_up")
    assert NOOP == int(native.Actions.stay)
    assert CLEAN == int(native.Actions.zap_clean)
    assert adapter.env.action_space(0).n == 9


def test_horizon_preserves_native_final_transition_without_autoreset(adapter):
    state = adapter.reset(17)
    for step in range(3):
        state, _, done = adapter.transition(state, (NOOP,) * NUM_AGENTS, step)
        assert done == (step == 2)
    assert int(state.native.inner_t) == 3
    assert int(state.native.outer_t) == 0
    assert adapter.snapshot(adapter.restore(adapter.snapshot(state))) == adapter.snapshot(state)
    with pytest.raises(ValueError, match="terminal"):
        adapter.transition(state, (NOOP,) * NUM_AGENTS, 99)


def test_safe_uses_native_pollution_labels_not_visible_grid_or_stale_apples(adapter):
    state = adapter.reset(17)
    assert not adapter.safe(state)  # Native reset is a recovery case.
    labels = state.native.potential_dirt_and_dirt_label
    clean = labels * 0 + adapter._items.potential_dirt
    healthy = replace(state, native=state.native.replace(potential_dirt_and_dirt_label=clean))
    assert adapter.safe(healthy)
    assert adapter.metrics(healthy)["apples_on_grid"] == 0
    assert adapter.metrics(healthy)["effective_spawn_probability"] == 0.05
    river_cells = adapter.metrics(state)["river_cells"]
    for count in (int(river_cells * 0.2), int(river_cells * 0.2) + 1):
        changed = clean.at[:count].set(adapter._items.dirt)
        candidate = replace(state, native=state.native.replace(potential_dirt_and_dirt_label=changed))
        assert adapter.safe(candidate) == (count / river_cells <= 0.2)


def test_local_evidence_does_not_include_hidden_state_or_other_proposals(adapter):
    state = adapter.reset(17)
    first = adapter.local_evidence(state, (NOOP,) * NUM_AGENTS, 0)
    changed = (NOOP,) + (CLEAN,) * (NUM_AGENTS - 1)
    second = adapter.local_evidence(state, changed, 0)
    assert {f.name for f in fields(first)} == {"agent_id", "observation", "proposed_action"}
    adapter.np.testing.assert_array_equal(first.observation, state.observations[0])
    adapter.np.testing.assert_array_equal(first.observation, second.observation)
    assert first.proposed_action == second.proposed_action == NOOP
    assert adapter.full_state_evidence(state, changed).proposed_actions == changed


def test_snapshot_rejects_incompatible_contract_and_array_schema(adapter):
    state = adapter.reset(17)
    payload = json.loads(adapter.snapshot(state))
    payload["manifest"]["source_revision"] = "wrong"
    with pytest.raises(ValueError, match="contract"):
        adapter.restore(json.dumps(payload).encode())
    payload = json.loads(adapter.snapshot(state))
    payload["native"]["grid"]["shape"] = [1, 1]
    with pytest.raises(ValueError, match="schema"):
        adapter.restore(json.dumps(payload).encode())
    with pytest.raises(ValueError, match="different Cleanup contract"):
        adapter.safe(replace(state, contract="other"))


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5])
def test_invalid_seeds(adapter, seed):
    with pytest.raises(ValueError):
        adapter.reset(seed)
