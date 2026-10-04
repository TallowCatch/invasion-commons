"""Scientific invariants of the bounded actor-search pilot."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from experiments.run_actor_pressure_pilot import (
    all_jobs, candidates_for, choose_candidates, episode_block, mask_identity,
    partners_for, plan,
)
from fishery_sim.budgeted_oversight import mask_requests


def test_plan_has_nested_search_and_disjoint_seed_ranges():
    cfg = plan("pilot")
    assert cfg["actor_budgets"] == [1, 8, 32]
    assert max(cfg["actor_budgets"]) == cfg["max_candidates"]
    train = {cfg["train_weather_seed_base"] + 1000 * context + weather
             for context in range(cfg["contexts"])
             for weather in range(cfg["train_weather_streams"])}
    test = {cfg["test_weather_seed_base"] + 1000 * context + weather
            for context in range(cfg["contexts"])
            for weather in range(cfg["weather_streams"])}
    assert not train & test
    assert len(all_jobs(cfg)) == 4 * 3 * 2 * (1 + 3 * 3)


def test_candidate_pool_and_test_partners_are_fixed_across_budgets():
    cfg = plan("smoke")
    assert [asdict(spec) for spec in candidates_for(cfg, 0)] == [
        asdict(spec) for spec in candidates_for(cfg, 0)]
    assert [asdict(spec) for spec in partners_for(cfg, 0, test=True)] == [
        asdict(spec) for spec in partners_for(cfg, 0, test=True)]
    assert [asdict(spec) for spec in partners_for(cfg, 0, test=True)] != [
        asdict(spec) for spec in partners_for(cfg, 0, test=False)]


def test_selection_uses_training_scores_only_and_is_nested(monkeypatch):
    cfg = plan("smoke")
    seen_seeds = []

    def fake_run(game, agents):
        seen_seeds.append(game.seed)
        candidate_id = int(agents[0].spec.strategy_id.split("_")[-1])
        return dict(final_payoffs=np.array([candidate_id] + [0] * (game.n_agents - 1)),
                    garden_failure_event=0)

    monkeypatch.setattr("experiments.run_actor_pressure_pilot.run_harvest_episode", fake_run)
    selected = choose_candidates(cfg, 0)
    assert selected["selected"] == {"1": 0, "8": 7}
    assert selected["train_scores"] == list(range(8))
    assert set(seen_seeds) == set(selected["train_weather_seeds"])
    assert all(seed < cfg["test_weather_seed_base"] for seed in seen_seeds)


def test_inspection_positions_match_across_actor_budgets():
    cfg = plan("smoke")
    identity = mask_identity(0, 0, 3)
    low = mask_requests(np.linspace(0.1, 0.6, 6), 3, identity)
    high = mask_requests(np.linspace(0.4, 0.9, 6), 3, identity)
    assert np.array_equal(np.isfinite(low), np.isfinite(high))


def test_no_review_and_reviewer_use_identical_policies():
    cfg = plan("smoke")
    selection = choose_candidates(cfg, 0)
    baseline = episode_block(cfg, (0, 1, 0, "none", 0), selection)
    reviewed = episode_block(cfg, (0, 1, 0, "joint", 6), selection)
    assert baseline["config"] == reviewed["config"]
    assert baseline["policies"] == reviewed["policies"]
    assert reviewed["metrics"]["request_inspections"] >= 0
    assert all(0 <= row["scale"] <= 1 for row in reviewed["trace"])


def test_invalid_profile_rejected():
    with pytest.raises(ValueError):
        plan("full")
