"""Tests for the patterned inspector (23 Nov 2026; notes/llm_overseer/error_patterns/protocol.md): the inspector
table, its JAX features, the overseer and trainer option --inspector-table, and the error-model fit.
Skipped without JAX and SocialJax; run with
.venv-socialjax/bin/python -m pytest -q tests/test_inspector_table.py (scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("socialjax")

import jax.numpy as jnp  # noqa: E402

from overseer.llm.harvest_records import DIAMOND, PATCH_RADIUS, patch  # noqa: E402
from overseer.socialjax.decentralised import DecConfig, DecentralisedOverseer  # noqa: E402
from overseer.socialjax.games import make_rules  # noqa: E402
from overseer.socialjax.harvest_bots import bot_actions, kind_codes  # noqa: E402
from overseer.socialjax.inspector_table import InspectorTable, constant_table, features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "c0f39d9"  # before the inspector table
INFO = ("taken", "violations", "checked", "caught", "detected", "false_alarms", "presumed_wrong", "detected_count",
        "missed_count", "raw_reward", "penalty", "frozen", "resource", "cost")
CODES = ("greedy", "greedy", "sneaky", "restrained", "restrained", "random", "greedy")


def _play(ov, steps, seed=0, record=None):
    """Bots (some greedy, so there are violations) under `ov`; yields (state before, actions, step outputs)."""
    _, s = ov.reset(jax.random.PRNGKey(seed))
    codes = kind_codes(CODES)
    key = jax.random.PRNGKey(seed + 1)
    for _ in range(steps):
        key, k_bot, k_step = jax.random.split(key, 3)
        act = bot_actions(k_bot, s, codes)
        prev = s
        out = ov.step(k_step, s, act)
        s = out[1]
        yield prev, act, out


# ------------------------------------------------------------------------------------------------ features
def test_features_match_the_raw_rendering_patch():
    """near / outer / players / walls equal what the LLM sees in the 5 x 5 patch (harvest_records.patch)."""
    rules = make_rules("harvest", num_inner_steps=200)
    ov = DecentralisedOverseer(rules, DecConfig(k=0, chooser="none"))
    checked = 0
    for prev, _, (_, s, _, _, info) in _play(ov, 60):
        grid = np.asarray(prev.env_state.grid)
        locs = np.asarray(s.env_state.agent_locs[:, :2]).astype(int)
        f = {k: np.asarray(v) for k, v in features(prev.env_state.grid, jnp.asarray(locs)).items()}
        np.testing.assert_array_equal(f["near"], np.asarray(info["near_prev"]))
        for a in range(7):
            cells = patch(grid, locs[a])
            r = PATCH_RADIUS
            ring = [(i, j) for i in range(5) for j in range(5) if (i, j) != (r, r)]
            diamond = {(r + dr, r + dc) for dr, dc in DIAMOND}
            assert f["near"][a] == sum(cells[i][j] == "A" for i, j in diamond)
            assert f["outer"][a] == sum(cells[i][j] == "A" for i, j in ring if (i, j) not in diamond)
            assert f["players"][a] == sum(cells[i][j] == "P" for i, j in ring)
            assert f["walls"][a] == sum(cells[i][j] == "#" for i, j in ring)
            checked += 1
    assert checked == 420


def test_lookup_picks_the_right_cell():
    spec = dict(features=[dict(name="near", edges=[1, 2, 3]), dict(name="outer", edges=[2, 5])],
                alpha=[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2], [3.0, 3.1, 3.2]],
                beta=[[0.5, 0.6, 0.7], [0.0] * 3, [0.0] * 3, [0.9, 0.8, 0.7]])
    spec["alpha"] = [[v / 4 for v in row] for row in spec["alpha"]]  # probabilities
    t = InspectorTable(json.dumps(spec))
    near = jnp.asarray([0, 0, 1, 2, 3, 7, 12, 0])
    outer = jnp.asarray([0, 2, 4, 5, 1, 9, 3, 12])
    c = np.asarray(t.cell(dict(near=near, outer=outer)))
    np.testing.assert_array_equal(c, [0, 1, 4, 8, 9, 11, 10, 2])
    np.testing.assert_allclose(np.asarray(t.alpha)[c], np.asarray(spec["alpha"]).ravel()[c])
    np.testing.assert_allclose(np.asarray(t.beta)[c], [0.5, 0.6, 0.0, 0.0, 0.9, 0.7, 0.8, 0.7])
    with pytest.raises(ValueError):
        InspectorTable(json.dumps(dict(spec, alpha=[[0.1]])))
    with pytest.raises(ValueError):
        InspectorTable(json.dumps(dict(spec, features=[dict(name="colour", edges=[])])))


# ------------------------------------------------------------------------------------------------ overseer
def test_constant_table_equals_the_coin_flip():
    """A table with alpha = a and beta = b everywhere gives exactly the run of miss_rate = a, false_alarm = b."""
    rules = make_rules("harvest", num_inner_steps=300)
    coin = DecentralisedOverseer(rules, DecConfig(k=3, chooser="random", penalty=7.0, miss_rate=0.3,
                                                  false_alarm_rate=0.2))
    table = DecentralisedOverseer(rules, DecConfig(k=3, chooser="random", penalty=7.0,
                                                   inspector_table=constant_table(0.3, 0.2)))
    totals = dict(detected=0, false_alarms=0, missed=0)
    for (_, _, a), (_, _, b) in zip(_play(coin, 150), _play(table, 150)):
        for x, y in zip(a[:3], b[:3]):
            jax.tree.map(np.testing.assert_array_equal, x, y)
        for f in INFO:
            np.testing.assert_array_equal(a[4][f], b[4][f], err_msg=f)
        totals["detected"] += int(a[4]["detected"].sum())
        totals["false_alarms"] += int(a[4]["false_alarms"].sum())
        totals["missed"] += int((a[4]["checked"] & a[4]["violations"] & ~a[4]["detected"]).sum())
    assert min(totals.values()) > 0, totals  # the test exercised catches, false alarms and misses


def test_patterned_table_applies_by_situation():
    """alpha = 1 only at near = 2 and beta = 1 only at near = 3: checked near-2 violations are never caught, other
    violations always; checked near-3 honest eats are always flagged, others never."""
    spec = dict(features=[dict(name="near", edges=[1, 2, 3, 4])], alpha=[0.0, 0.0, 1.0, 0.0, 0.0],
                beta=[0.0, 0.0, 0.0, 1.0, 0.0])
    rules = make_rules("harvest", num_inner_steps=400)
    ov = DecentralisedOverseer(rules, DecConfig(k=7, chooser="all", inspector_table=json.dumps(spec)))
    seen = dict(v2=0, v_other=0, h3=0, h_other=0)
    for _, _, (_, _, _, _, info) in _play(ov, 300, seed=4):
        near, eat, viol = (np.asarray(info[k]) for k in ("near_prev", "taken", "violations"))
        det, fa = np.asarray(info["detected"]), np.asarray(info["false_alarms"])
        assert not det[viol & (near == 2)].any() and det[viol & (near != 2)].all()
        honest = eat & ~viol
        assert fa[honest & (near == 3)].all() and not fa[honest & (near != 3)].any()
        seen["v2"] += int((viol & (near == 2)).sum())
        seen["v_other"] += int((viol & (near != 2)).sum())
        seen["h3"] += int((honest & (near == 3)).sum())
        seen["h_other"] += int((honest & (near != 3)).sum())
    assert min(seen.values()) > 0, seen


def test_table_only_where_it_is_defined():
    with pytest.raises(ValueError):
        DecConfig(window=2, inspector_table=constant_table(0.1, 0.1))
    with pytest.raises(ValueError):
        DecConfig(miss_rate=0.1, inspector_table=constant_table(0.1, 0.1))
    with pytest.raises(ValueError):
        DecentralisedOverseer(make_rules("coins", num_inner_steps=10), DecConfig(inspector_table=constant_table(0, 0)))


def _module_at(commit, relpath, name, tmp_path):
    try:
        src = subprocess.run(["git", "show", f"{commit}:{relpath}"], cwd=ROOT, capture_output=True, text=True,
                             check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git history not available")
    path = tmp_path / f"{name}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_overseer_is_unchanged(tmp_path):
    """With no table, the overseer and the Harvest rule give the same steps as before the change."""
    old_games = _module_at(BASE_COMMIT, "overseer/socialjax/games.py", "old_games", tmp_path)
    old_dec = _module_at(BASE_COMMIT, "overseer/socialjax/decentralised.py", "old_dec", tmp_path)
    kw = dict(k=2, chooser="random", penalty=7.0, miss_rate=0.2, false_alarm_rate=0.1)
    new = DecentralisedOverseer(make_rules("harvest", num_inner_steps=300), DecConfig(**kw))
    old = old_dec.DecentralisedOverseer(old_games.make_rules("harvest", num_inner_steps=300), old_dec.DecConfig(**kw))
    for (_, _, a), (_, _, b) in zip(_play(new, 120), _play(old, 120)):
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[2], b[2])
        assert set(b[4]) <= set(a[4])
        for f in b[4]:
            np.testing.assert_array_equal(a[4][f], b[4][f], err_msg=f)


# ------------------------------------------------------------------------------------------------ trainer
TINY = dict(NUM_STEPS=20, NUM_MINIBATCHES=4, UPDATE_EPOCHS=1)


@pytest.fixture()
def tiny(monkeypatch):
    trainer = importlib.import_module("experiments.socialjax.train_ippo_overseer")
    for k, v in TINY.items():
        monkeypatch.setitem(trainer.HPARAMS, k, v)
    return trainer


def _cfg():
    return dict(env="harvest", rules=dict(min_neighbours=3), total_steps=80, num_envs=2, seed=1)


def test_trainer_default_path_matches_the_trainer_before(tiny, tmp_path, monkeypatch):
    base = _module_at(BASE_COMMIT, "experiments/socialjax/train_ippo_overseer.py", "base_trainer_t", tmp_path)
    for k, v in TINY.items():
        monkeypatch.setitem(base.HPARAMS, k, v)
    ov = DecConfig(k=2, chooser="random", penalty=3.5 * 7, miss_rate=0.1, false_alarm_rate=0.08)
    new_p, new_m = jax.jit(tiny.make_train(_cfg(), ov)[0])(jax.random.PRNGKey(1))
    old_p, old_m = jax.jit(base.make_train(_cfg(), ov)[0])(jax.random.PRNGKey(1))
    for a, b in zip(jax.tree.leaves(new_p), jax.tree.leaves(old_p)):
        np.testing.assert_array_equal(a, b)
    assert set(old_m) <= set(new_m)
    for k in old_m:
        np.testing.assert_array_equal(new_m[k], old_m[k], err_msg=k)
    # the new measures add up: eats by near sum to apples eaten, violations by near to violations
    np.testing.assert_allclose(new_m["eats_by_near"].sum(-1), new_m["apples_eaten"], rtol=1e-5)
    np.testing.assert_allclose(new_m["violations_by_near"].sum(-1), new_m["violations"], rtol=1e-5)
    assert (new_m["checks_on_violations"] >= new_m["detected"]).all()


def test_trainer_constant_table_equals_coin_flip_flags(tiny, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(dict(json.loads(constant_table(0.1, 0.08)), note="metadata is dropped")))
    a = tiny.parser().parse_args(["--out", "x", "--k", "2", "--chooser", "random", "--inspector-table", str(path)])
    text = tiny.read_table(a.inspector_table)
    assert "note" not in text
    coin = DecConfig(k=2, chooser="random", penalty=3.5 * 7, miss_rate=0.1, false_alarm_rate=0.08)
    tab = DecConfig(k=2, chooser="random", penalty=3.5 * 7, inspector_table=text)
    p1, m1 = jax.jit(tiny.make_train(_cfg(), coin)[0])(jax.random.PRNGKey(1))
    p2, m2 = jax.jit(tiny.make_train(_cfg(), tab)[0])(jax.random.PRNGKey(1))
    for x, y in zip(jax.tree.leaves(p1), jax.tree.leaves(p2)):
        np.testing.assert_array_equal(x, y)
    for k in m1:
        np.testing.assert_array_equal(m1[k], m2[k], err_msg=k)


def test_error_pattern_job_list_parses_with_the_trainer():
    """Every line parses; table arms point at a loadable table and leave the coin flip at 0; coin arms have no
    table; the penalty gives an average penalty of c apples under random checks; names are unique."""
    import shlex
    trainer = importlib.import_module("experiments.socialjax.train_ippo_overseer")
    from experiments.llm_overseer import error_pattern_runs as R
    tags = [t for t in ("gpt-oss-20b_direct", "qwen2.5-7b_reason") if (ROOT / R.TABLES / f"{t}.json").exists()]
    if not tags:
        pytest.skip("no fitted tables")
    names = []
    for name, args in R.runs(tags, tables=str(ROOT / R.TABLES)):
        a = trainer.parser().parse_args(["--out", "x"] + shlex.split(args))
        c = float(re.search(r"_k\d+_c([\d.]+)_seed", name).group(1))
        assert a.k / 7 * a.penalty_apples == pytest.approx(c)
        if "_table_" in name:
            text = trainer.read_table(a.inspector_table)
            InspectorTable(text)
            DecConfig(k=a.k, chooser=a.chooser, inspector_table=text)
            assert a.miss_rate == 0 and a.false_alarm == 0
        else:
            assert not a.inspector_table and 0 < a.miss_rate < 1
        names.append(name)
    assert len(names) == len(set(names))


def test_inverted_table_h3b():
    """H3b: the inverted table swaps gpt-oss's near=0 and near=2 miss rates (blind spot on the most harmful
    violations), keeps beta, and is scaled so its bank-average alpha equals the matched coin flip; every job line
    parses and points at it."""
    import gzip
    import json
    import shlex
    from experiments.llm_overseer import error_pattern_runs as R
    tables = ROOT / R.TABLES
    if not (tables / "gpt-oss-20b_inverted.json").exists():
        pytest.skip("no inverted table")
    orig = json.loads((tables / "gpt-oss-20b_direct.json").read_text())
    inv = json.loads((tables / "gpt-oss-20b_inverted.json").read_text())
    InspectorTable(json.dumps(inv))
    assert inv["beta"] == orig["beta"]
    a0, a2 = orig["alpha"][0], orig["alpha"][2]
    f = inv["alpha"][2][0] / a0[0]
    assert inv["alpha"][0] == pytest.approx([x * f for x in a2]) and inv["alpha"][2] == pytest.approx([x * f for x in a0])
    rows = [json.loads(l) for l in gzip.open(ROOT / "experiments/llm_overseer/data/error_bank.jsonl.gz", "rt")]
    viol = [r for r in rows if r["near"] < 3]
    mean = sum(inv["alpha"][min(r["near"], 2)][0 if r["outer"] < 2 else 1] for r in viol) / len(viol)
    assert mean == pytest.approx(orig["matched_coin_flip"]["alpha"], abs=1e-4)
    trainer = importlib.import_module("experiments.socialjax.train_ippo_overseer")
    names = []
    for name, args in R.inverted(tables=str(tables)):
        a = trainer.parser().parse_args(["--out", "x"] + shlex.split(args))
        assert a.inspector_table.endswith("gpt-oss-20b_inverted.json") and a.miss_rate == 0
        names.append(name)
    assert len(names) == len(set(names)) == 12
