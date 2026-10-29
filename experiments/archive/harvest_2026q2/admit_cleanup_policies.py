"""Bounded, evaluator-only Cleanup policy-admission smoke; not monitor comparison."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import signal
import time

from fishery_sim.cleanup_oversight import CLEAN, NUM_AGENTS, CleanupAdapter, CleanupConfig
from fishery_sim.cleanup_policies import PolicyMemory, decide


SEEDS = (17, 43)
VARIANTS = ("productive", "free_rider")
STEPS = 180
TAIL = 60


def _dirty(adapter: CleanupAdapter, state) -> set[tuple[int, int]]:
    return {
        tuple(map(int, loc))
        for loc, label in zip(
            adapter.np.asarray(state.native.potential_dirt_and_dirt_locs),
            adapter.np.asarray(state.native.potential_dirt_and_dirt_label),
        )
        if int(label) == int(adapter._items.dirt)
    }


def _episode(adapter: CleanupAdapter, seed: int, variant: str) -> dict:
    state = adapter.reset(seed)
    initial_safe = adapter.safe(state)
    memory = [PolicyMemory() for _ in range(NUM_AGENTS)]
    individual_returns = [0.0] * NUM_AGENTS
    safe_steps = clean_actions = removed = apples = onsets = 0
    tail_safe = tail_clean = tail_removed = tail_apples = 0
    first_safe_step = 0 if initial_safe else None
    was_safe = initial_safe
    for step in range(STEPS):
        actions = []
        for agent in range(NUM_AGENTS):
            action, memory[agent] = decide(
                state.observations[agent], agent, memory[agent], variant=variant,
            )
            actions.append(action)
        before = _dirty(adapter, state)
        state, rewards, done = adapter.transition(state, actions, seed + 10_000 + step)
        if done != (step == STEPS - 1):
            raise RuntimeError("unexpected native episode termination")
        now_safe = adapter.safe(state)
        harvested = sum(rewards) / NUM_AGENTS
        cleaned = len(before - _dirty(adapter, state))
        for agent, reward in enumerate(rewards):
            individual_returns[agent] += reward
        safe_steps += int(now_safe)
        clean_actions += actions.count(CLEAN)
        removed += cleaned
        apples += harvested
        onsets += int(was_safe and not now_safe)
        if now_safe and first_safe_step is None:
            first_safe_step = step + 1
        if step >= STEPS - TAIL:
            tail_safe += int(now_safe)
            tail_clean += actions.count(CLEAN)
            tail_removed += cleaned
            tail_apples += harvested
        was_safe = now_safe
    return {
        "seed": seed, "variant": variant, "initial_safe": initial_safe,
        "first_safe_step": first_safe_step, "safe_steps": safe_steps,
        "unsafe_onsets": onsets, "clean_actions": clean_actions,
        "preexisting_dirt_removed": removed, "apples_harvested": apples,
        "native_individual_returns": individual_returns,
        "native_total_return": sum(individual_returns),
        "tail_safe_steps": tail_safe, "tail_clean_actions": tail_clean,
        "tail_preexisting_dirt_removed": tail_removed,
        "tail_apples_harvested": tail_apples,
        "final": adapter.metrics(state),
    }


def assess(episodes: list[dict]) -> dict:
    by_key = {(row["seed"], row["variant"]): row for row in episodes}
    if len(by_key) != len(SEEDS) * len(VARIANTS):
        raise ValueError("incomplete or duplicate episode matrix")
    checks = {}
    for seed in SEEDS:
        productive = by_key[seed, "productive"]
        rider = by_key[seed, "free_rider"]
        checks[str(seed)] = {
            "productive_recovered": productive["first_safe_step"] is not None,
            "productive_tail_safe": productive["tail_safe_steps"] == TAIL,
            "productive_tail_harvest": productive["tail_apples_harvested"] > 0,
            "productive_tail_maintenance": productive["tail_preexisting_dirt_removed"] > 0,
            "free_rider_harvest": rider["apples_harvested"] > 0,
            "free_rider_no_clean": rider["clean_actions"] == 0,
            "free_rider_less_capacity": rider["tail_safe_steps"] < productive["tail_safe_steps"],
            "free_rider_no_more_maintenance": rider["preexisting_dirt_removed"]
            <= productive["preexisting_dirt_removed"],
        }
    return {"admitted": all(all(row.values()) for row in checks.values()), "checks": checks}


def run() -> dict:
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    adapter = CleanupAdapter(CleanupConfig(horizon=STEPS))
    if adapter.manifest["backend"] != "cpu":
        raise RuntimeError("CPU backend required")
    episodes = [_episode(adapter, seed, variant) for seed in SEEDS for variant in VARIANTS]
    return {
        "protocol": "notes/research_review/CLEANUP_POLICY_ADMISSION_PROTOCOL.md",
        "policy_ready": assess(episodes)["admitted"],
        "assessment": assess(episodes), "episodes": episodes,
        "transition_calls": len(episodes) * STEPS,
        "wall_seconds": time.perf_counter() - start_wall,
        "cpu_seconds": time.process_time() - start_cpu,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "adapter_manifest": adapter.manifest,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/tmp/commons-cleanup-policy-admission.json"))
    args = parser.parse_args()
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    used = math.ceil(time.process_time())
    _, hard = resource.getrlimit(resource.RLIMIT_CPU)
    ceiling = used + 120 if hard == resource.RLIM_INFINITY else min(used + 120, hard)
    resource.setrlimit(resource.RLIMIT_CPU, (ceiling, ceiling))
    signal.alarm(120)
    try:
        report = run()
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if len(encoded.encode()) > 5_000_000:
            raise RuntimeError("report exceeds frozen output cap")
        args.output.write_text(encoded)
        print(encoded)
    finally:
        signal.alarm(0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
