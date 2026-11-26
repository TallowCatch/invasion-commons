"""Tests for the H5 local-pass/global-fail measures (notes/formulation/h5_lpgf/protocol.md), on hand-built maps.

Run: .venv-socialjax/bin/python -m pytest -q tests/test_lpgf.py
"""
import numpy as np
import pytest

from experiments.socialjax import measure_lpgf as M


def maps(shape, *apple_sets):
    """(T+1, rows, cols) bool maps, one per set of apple cells."""
    out = np.zeros((len(apple_sets),) + shape, bool)
    for t, cells in enumerate(apple_sets):
        for r, c in cells:
            out[t, r, c] = True
    return out


def eats(steps, *per_step):
    """eat_pos (T, N, 2), eaten (T, N), violation (T, N) from per-step lists of (row, col, is_violation)."""
    n = max(1, max(len(s) for s in per_step))
    pos = np.zeros((steps, n, 2), int)
    eaten = np.zeros((steps, n), bool)
    viol = np.zeros((steps, n), bool)
    for t, step in enumerate(per_step):
        for i, (r, c, v) in enumerate(step):
            pos[t, i] = (r, c)
            eaten[t, i] = True
            viol[t, i] = v
    return pos, eaten, viol


def rule_ok(apples_before, r, c):
    """The Harvest rule on the map before the step: 3 or more apples within two squares."""
    return M.shift_sum(apples_before, M.NEIGHBOURHOOD)[r, c] >= 3


def test_patches_of_the_harvest_map():
    from overseer.socialjax.games import HARVEST_MAP
    spawn = np.array([[ch == "A" for ch in row] for row in HARVEST_MAP])
    labels, k = M.patches(spawn)
    assert k == 6
    assert sorted(np.bincount(labels[spawn])) == [6, 6, 13, 13, 13, 13]


def test_legal_eat_that_kills_a_cell():
    # e = (2, 4) has 3 apples within two squares, so eating it is legal; (2, 2) had e as its only support
    shape, spawn = (5, 9), np.ones((5, 9), bool)
    before = [(2, 4), (2, 5), (2, 6), (1, 5)]
    a = maps(shape, before, before[1:])
    assert rule_ok(a[0], 2, 4)
    out = M.harvest_measures(a, *eats(1, [(2, 4, False)]), spawn)
    assert out["violations"] == 0 and out["eats"] == 1
    assert M.dead_cells(a[1], spawn)[2, 2] and not M.dead_cells(a[0], spawn)[2, 2]
    assert out["dead_new"] > 0 and out["dead_legal_solo"] == out["dead_new"]
    assert out["dead_violation"] == 0 and out["dead_no_eat"] == 0
    assert out["dead_new_allpass"] == out["dead_new"]  # the step had no violation
    assert out["legal_eats_killing"] == 1 and out["ext_cells"] == 0


def test_joint_legal_kill():
    # a = (2, 3) and b = (2, 5) are each legal (4 apples near); together they were (2, 4)'s whole support
    shape, spawn = (5, 9), np.ones((5, 9), bool)
    before = [(2, 3), (2, 5), (0, 3), (4, 3), (2, 1), (0, 5), (4, 5), (2, 7)]
    after = before[2:]
    a = maps(shape, before, after)
    assert rule_ok(a[0], 2, 3) and rule_ok(a[0], 2, 5)
    assert not M.dead_cells(maps(shape, before[1:])[0], spawn)[2, 4]  # either eat alone leaves (2, 4) alive
    out = M.harvest_measures(a, *eats(1, [(2, 3, False), (2, 5, False)]), spawn)
    assert M.dead_cells(a[1], spawn)[2, 4]
    assert out["dead_legal_joint"] >= 1 and out["dead_violation"] == 0


def test_violating_kill_and_patch_extinction():
    # two patches; patch A's last apple (2, 2) is eaten with no apple near: a violation that ends the patch
    shape = (5, 11)
    spawn = np.zeros(shape, bool)
    for cell in [(2, 1), (2, 2), (2, 3), (2, 8), (2, 9)]:
        spawn[cell] = True
    a = maps(shape, [(2, 2), (2, 8), (2, 9)], [(2, 8), (2, 9)])
    assert not rule_ok(a[0], 2, 2)
    out = M.harvest_measures(a, *eats(1, [(2, 2, True)]), spawn)
    assert out["ext_patches"] == 1 and out["ext_cells"] == 3 and out["ext_cells_violation"] == 3
    assert out["ext_cells_allpass"] == 0
    assert out["dead_new"] == 3 and out["dead_violation"] == 3 and out["dead_legal_solo"] == 0
    assert out["violating_eats_killing"] == 1 and out["patches_alive_end"] == 1


def test_joint_legal_patch_extinction():
    # a 2 x 2 patch: each apple has the other 3 within two squares, so four simultaneous eats are all legal
    shape = (4, 4)
    spawn = np.zeros(shape, bool)
    block = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for cell in block:
        spawn[cell] = True
    a = maps(shape, block, [])
    assert all(rule_ok(a[0], r, c) for r, c in block)
    out = M.harvest_measures(a, *eats(1, [(r, c, False) for r, c in block]), spawn)
    assert out["ext_cells"] == 4 and out["ext_cells_legal_joint"] == 4 and out["ext_cells_allpass"] == 4


def test_steps_after_collapse_are_not_counted():
    # the last apple goes at step 0 (a violation); steps 1-4 are empty, nobody eats, everyone "passes"
    shape = (5, 5)
    spawn = np.zeros(shape, bool)
    spawn[2, 2] = spawn[2, 3] = True
    a = maps(shape, [(2, 2)], [], [], [], [], [])
    out = M.harvest_measures(a, *eats(5, [(2, 2, True)], [], [], [], []), spawn)
    assert out["ext_patches"] == 1 and out["ext_steps"] == 1 and out["dead_steps"] == 1
    assert out["dead_new"] == 2 and out["dead_new_allpass"] == 0
    for tau in M.TAUS:
        assert out[f"tau{tau}_cross"] == 1 and out[f"tau{tau}_cross_allpass"] == 0
        assert out[f"tau{tau}_june_steps"] == 4  # June's raw form counts every dead step as local pass, global fail
    assert out["alive_end"] == 0 and out["ever_empty"] == 1


def test_cleanup_window_pass():
    # pollution crosses 0.4 at step 3 and again at step 8; a violation at step 1 is inside a W = 5 window of the
    # first crossing only
    pol = np.array([0.1, 0.2, 0.3, 0.45, 0.3, 0.3, 0.3, 0.3, 0.5, 0.5])
    viol = np.zeros((10, 2), bool)
    viol[1, 0] = True
    out = M.cleanup_measures(pol, np.full(10, 50.0), 50, viol, window=5)
    assert out["pollution_cross"] == 2 and out["pollution_cross_allpass"] == 2
    assert out["pollution_cross_windowpass"] == 1


def test_overseer_settings_from_stage1_names(tmp_path):
    pytest.importorskip("socialjax")
    env, rules_kw, ov = M.overseer_settings(tmp_path / "B_random_k7_c2.0_seed1")
    assert env == "harvest" and ov["k"] == 7 and ov["chooser"] == "random" and ov["penalty"] == pytest.approx(14.0)
    _, _, ov = M.overseer_settings(tmp_path / "A_none_seed1")
    assert ov["k"] == 0 and ov["chooser"] == "none" and ov["penalty"] == 0.0


def test_lasting_deaths_exclude_revived_cells():
    # (2, 2) dies at step 0 (legal eat at (2, 4)); at step 1 an apple regrows at (2, 3) and revives it
    shape, spawn = (5, 9), np.ones((5, 9), bool)
    before = [(2, 4), (2, 5), (2, 6), (1, 5)]
    a = maps(shape, before, before[1:], before[1:] + [(2, 3)])
    out = M.harvest_measures(a, *eats(2, [(2, 4, False)], []), spawn)
    assert M.dead_cells(a[1], spawn)[2, 2] and not M.dead_cells(a[2], spawn)[2, 2]
    assert out["dead_revived"] >= 1
    lasting = sum(out[f"dead_lasting_{k}"] for k in M.CLASSES)
    assert 0 < lasting < out["dead_new"]  # (4, 4) stays dead; (2, 2) does not count
    assert lasting <= out["dead_end"]
