"""Bounded real-Cleanup mechanics probe, never a trained-policy benchmark.

Run as a module in the optional isolated environment. The scripted controller
is privileged, stateless, and makes proposals; it is NOT an oversight monitor.
"""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import sys
import time

from fishery_sim.cleanup_oversight import (
    CLEAN, NOOP, NUM_AGENTS, CleanupAdapter, CleanupConfig, CleanupState,
    CleanupUnavailable, replace_with_noop,
)


# These are native absolute movement deltas, not the intuitive screen labels.
MOVES = ((4, (1, 0)), (2, (0, 1)), (5, (-1, 0)), (3, (0, -1)))
DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))


def dirt_cells(adapter: CleanupAdapter, state: CleanupState) -> set[tuple[int, int]]:
    """Full-state diagnostic, preserving the native coordinate/label pairing."""
    return {
        tuple(map(int, loc))
        for loc, label in zip(
            adapter.np.asarray(state.native.potential_dirt_and_dirt_locs),
            adapter.np.asarray(state.native.potential_dirt_and_dirt_label),
        )
        if int(label) == int(adapter._items.dirt)
    }


def _paths(start, shape, blocked):
    paths = {start: (0, NOOP)}
    queue = deque([start])
    while queue:
        row, col = queue.popleft()
        distance, first_action = paths[row, col]
        for action, (dr, dc) in MOVES:
            dest = row + dr, col + dc
            if (0 <= dest[0] < shape[0] and 0 <= dest[1] < shape[1]
                    and dest not in blocked and dest not in paths):
                paths[dest] = (distance + 1, action if distance == 0 else first_action)
                queue.append(dest)
    return paths


def scripted_actions(adapter: CleanupAdapter, state: CleanupState) -> tuple[int, ...]:
    """Four full-state greedy cleaners plus three nearest-apple harvesters.

    Breadth-first paths and native four-cell beam geometry only; no simulator
    lookahead, policy search, training, or hidden future random seeds.
    """
    np = adapter.np
    grid = np.asarray(state.native.grid)
    locs = np.asarray(state.native.reborn_locs)
    occupied = {tuple(map(int, loc[:2])) for loc in locs}
    walls = {tuple(map(int, loc)) for loc in np.argwhere(grid == adapter._items.wall)}
    dirty = dirt_cells(adapter, state)
    apples = {tuple(map(int, loc)) for loc in np.argwhere(grid == adapter._items.apple)}
    reserved = set()
    actions = []
    for agent, (row, col, heading) in enumerate(locs.tolist()):
        start = row, col
        paths = _paths(start, grid.shape, (occupied - {start}) | walls | reserved)
        action = NOOP
        if agent < 4 and dirty:
            poses = {}
            for target in sorted(dirty):
                for direction, (dr, dc) in enumerate(DIRECTIONS):
                    sr, sc = DIRECTIONS[(direction + 1) % 4]
                    for offset in ((dr, dc), (2 * dr, 2 * dc), (dr + sr, dc + sc), (dr - sr, dc - sc)):
                        pose = target[0] - offset[0], target[1] - offset[1], direction
                        if pose[:2] in paths:
                            poses.setdefault(pose, set()).add(target)
            if poses:
                def cost(pose):
                    delta = (pose[2] - heading) % 4
                    turns = min(delta, 4 - delta)
                    distance = paths[pose[:2]][0]
                    return ((distance + turns + 1) / len(poses[pose]), distance, pose)

                pose = min(poses, key=cost)
                distance, first = paths[pose[:2]]
                delta = (pose[2] - heading) % 4
                action = first if distance else (CLEAN if not delta else (0 if delta in (1, 2) else 1))
                # Avoid assigning multiple cleaners the same immediately cleaned dirt.
                if action == CLEAN:
                    dirty -= poses[pose]
        elif agent >= 4:
            reachable = apples & paths.keys()
            if reachable:
                target = min(reachable, key=lambda p: (paths[p][0], p))
                action = paths[target][1]
                apples.discard(target)
            else:
                orchard = np.asarray(adapter.env.POTENTIAL_APPLE).tolist()
                target = tuple(orchard[(agent - 4) * len(orchard) // 3])
                if target in paths:
                    action = paths[target][1]
        delta = dict(MOVES).get(action, (0, 0))
        reserved.add((row + delta[0], col + delta[1]))
        actions.append(action)
    return tuple(actions)


def _rollout(adapter, initial, seed, scripted):
    state = initial
    trace = []
    returns = [0.0] * NUM_AGENTS
    was_safe = adapter.safe(state)
    first_safe_step = 0 if was_safe else None
    for step in range(adapter.config.horizon):
        actions = scripted_actions(adapter, state) if scripted else (NOOP,) * NUM_AGENTS
        dirty_before = dirt_cells(adapter, state)
        state, rewards, done = adapter.transition(state, actions, (seed + 10_000 + step) % 2**32)
        now_safe = adapter.safe(state)
        if now_safe and first_safe_step is None:
            first_safe_step = step + 1
        returns = [total + reward for total, reward in zip(returns, rewards)]
        trace.append({
            **adapter.metrics(state), "safe": now_safe,
            "unsafe_onset": was_safe and not now_safe,
            "actions": actions, "rewards": rewards,
            "apples_harvested": sum(rewards) / NUM_AGENTS,
            "clean_actions": actions.count(CLEAN),
            "preexisting_dirt_removed": len(dirty_before - dirt_cells(adapter, state)),
        })
        was_safe = now_safe
        if done:
            break
    tail = trace[-100:]
    return {
        "controller": "scripted_full_state_four_cleaners" if scripted else "noop_mechanics_control",
        "steps": len(trace), "initial": adapter.metrics(initial),
        "initial_safe": adapter.safe(initial), "final": adapter.metrics(state),
        "first_safe_step": first_safe_step,
        "unsafe_steps": sum(not row["safe"] for row in trace),
        "unsafe_onsets": sum(row["unsafe_onset"] for row in trace),
        "native_individual_returns": returns,
        "apples_harvested": sum(row["apples_harvested"] for row in trace),
        "clean_actions": sum(row["clean_actions"] for row in trace),
        "preexisting_dirt_removed": sum(row["preexisting_dirt_removed"] for row in trace),
        "tail_steps": len(tail), "tail_all_safe": all(row["safe"] for row in tail),
        "tail_apples_harvested": sum(row["apples_harvested"] for row in tail),
        "tail_preexisting_dirt_removed": sum(row["preexisting_dirt_removed"] for row in tail),
        "trace": trace,
    }, state


def run_smoke(steps=300, seed=17, output_dir: Path | None = None):
    wall_start, cpu_start = time.perf_counter(), time.process_time()
    adapter = CleanupAdapter(CleanupConfig(horizon=steps))
    if adapter.manifest["backend"] != "cpu":
        raise RuntimeError("smoke must run on CPU; set JAX_PLATFORMS=cpu before starting")
    initial = adapter.reset(seed)
    frozen = adapter.snapshot(initial)
    assert frozen == adapter.snapshot(adapter.reset(seed)), "reset is not repeatable"
    restored = adapter.restore(frozen)
    assert frozen == adapter.snapshot(restored), "snapshot is not canonical"
    noop = (NOOP,) * NUM_AGENTS
    first, rewards, done = adapter.transition(initial, noop, seed)
    replay, replay_rewards, replay_done = adapter.transition(restored, noop, seed)
    assert (adapter.snapshot(first), rewards, done) == (adapter.snapshot(replay), replay_rewards, replay_done)
    obs, native, native_rewards, native_done, _ = adapter.env.step_env(
        adapter.jax.random.PRNGKey(seed), initial.native, adapter.jax.numpy.asarray(noop)
    )
    assert adapter.snapshot(first) == adapter.snapshot(CleanupState(native, obs, initial.contract))
    assert rewards == tuple(float(r) for r in native_rewards)
    assert not bool(native_done["__all__"])
    assert replace_with_noop((CLEAN,) * NUM_AGENTS, range(NUM_AGENTS)) == noop
    startup = {"wall_seconds": time.perf_counter() - wall_start,
               "cpu_seconds": time.process_time() - cpu_start}

    control, _ = _rollout(adapter, initial, seed, scripted=False)
    scripted, final = _rollout(adapter, initial, seed, scripted=True)
    final_snapshot = adapter.snapshot(final)
    assert adapter.snapshot(adapter.restore(final_snapshot)) == final_snapshot
    assert frozen == adapter.snapshot(initial), "native step mutated its input"
    mechanism_pass = (
        steps >= 200 and scripted["tail_all_safe"]
        and scripted["tail_apples_harvested"] > 0
        and scripted["tail_preexisting_dirt_removed"] > 0
    )
    report = {
        "kind": "bounded_native_mechanics_smoke_not_benchmark_evaluation",
        "manifest": adapter.manifest, "seed": seed,
        "repeatability": {"reset": True, "step": True, "snapshot": True,
                          "native_step_parity": True, "input_unchanged": True},
        "authority": "proposal suppression to native stay only; controller is not a monitor",
        "policy_gate": {
            "mechanism_demonstration_passed": bool(mechanism_pass),
            "policy_ready": False,
            "missing": ["validated observation-limited maintenance/harvesting policy",
                        "validated productive free-rider and damaging proposals",
                        "held-out policy contexts and seed replication",
                        "matched monitor budget, hidden-state bounds and independent risk labels"],
        },
        "noop_control": control, "scripted_controller": scripted,
        "snapshot_sha256": {"initial": hashlib.sha256(frozen).hexdigest(),
                            "final": hashlib.sha256(final_snapshot).hexdigest()},
        "native_transition_calls": 3 + control["steps"] + scripted["steps"],
        "native_reset_calls": 2,
        "timing": {"startup_and_first_compile": startup,
                   "total_wall_seconds": time.perf_counter() - wall_start,
                   "total_cpu_seconds": time.process_time() - cpu_start},
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024),
        "platform": platform.platform(), "python": platform.python_version(),
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "initial.snapshot.json").write_bytes(frozen)
        (output_dir / "final.snapshot.json").write_bytes(final_snapshot)
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, choices=range(1, 501), metavar="1..500")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cpu-seconds", type=int, default=180, choices=range(1, 901), metavar="1..900")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/commons-cleanup-smoke"))
    args = parser.parse_args()
    # Set before JAX import. Limit CPU time across all JAX worker threads too.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    used = math.ceil(time.process_time())
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    ceiling = used + args.cpu_seconds
    if hard != resource.RLIM_INFINITY:
        ceiling = min(ceiling, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (ceiling, ceiling))
    signal.alarm(min(args.cpu_seconds, 900))
    try:
        report = run_smoke(args.steps, args.seed, args.output_dir)
    except CleanupUnavailable as exc:
        print(json.dumps({"status": "unavailable", "policy_ready": False, "reason": str(exc)}))
        return 2
    finally:
        signal.alarm(0)
    summary = {k: v for k, v in report.items() if k not in ("noop_control", "scripted_controller")}
    for key in ("noop_control", "scripted_controller"):
        summary[key] = {k: v for k, v in report[key].items() if k != "trace"}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
