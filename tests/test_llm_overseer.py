"""Tests for the LLM inspector ladder (overseer/llm/, experiments/llm_overseer/).

Most tests need only the standard library and numpy. The episode-recording test needs JAX and SocialJax and is
skipped without them (.venv-socialjax, scripts/socialjax_local_setup.sh).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from experiments.llm_overseer.harvest_runs import runs
from overseer.llm import subset as S
from overseer.llm.bazaar_records import render_bundle
from overseer.llm.harvest_records import DIAMOND, count_diamond
from overseer.llm.judge import run_items, tokens_logged
from overseer.llm.metrics import c1_class, effective_penalty, rates, wilson
from overseer.llm.prompts import build_messages, parse_verdict

ROOT = Path(__file__).resolve().parents[1]
HARVEST = ROOT / "experiments/llm_overseer/data/harvest_judge.jsonl.gz"


@pytest.fixture(scope="module")
def harvest():
    return S.load_jsonl(HARVEST)


# ------------------------------------------------------------------------------------------------ dataset
def test_diamond_is_the_wrappers_12_cells():
    from overseer.llm import harvest_records
    assert len(DIAMOND) == 12
    src = (ROOT / "overseer/socialjax/harvest_overseer.py").read_text()
    wrapper = eval(re.search(r"NEIGHBOURHOOD = (\(.*?\))  #", src, re.S).group(1))  # noqa: S307 - our own file
    assert set(wrapper) == set(DIAMOND) == set(harvest_records.DIAMOND)


def test_dataset_is_balanced(harvest):
    assert len(harvest) == 3000
    for w in (1, 10, 50):
        labels = [e["label"] for e in harvest if e["window"] == w]
        assert len(labels) == 1000 and sum(labels) == 500


def parse_patches(raw: str):
    lines, out = raw.splitlines(), []
    for i, ln in enumerate(lines):
        if ln.endswith("Map just before the move:"):
            out.append([re.findall(r"[\[ ](.)[\] ]", row[4:]) for row in lines[i + 1:i + 6]])
    return out


def test_renderings_carry_the_true_label(harvest):
    """The judge can recover the label exactly: counting the raw patches and reading the summary both give it."""
    for e in harvest:
        patches = parse_patches(e["renderings"]["raw"])
        assert len(patches) == e["n_eats"]
        counts = [count_diamond(p) for p in patches]
        assert counts == e["eat_nears"]
        assert all(p[2][2] == "A" for p in patches)
        summary = [int(x) for x in re.findall(r"just before the move: (\d+)", e["renderings"]["summary"])]
        assert summary == e["eat_nears"]
        assert e["label"] == int(any(n < 3 for n in e["eat_nears"]))
        assert len(e["renderings"]["summary"].splitlines()) == e["window"] + 1


def test_subset_is_fixed_and_balanced(harvest):
    h1, h2 = S.harvest_subset(harvest), S.harvest_subset(harvest)
    assert [e["id"] for e in h1] == [e["id"] for e in h2]
    assert len(h1) == 180 and sum(e["label"] for e in h1) == 90
    assert len(S.items(h1, [])) == 360


def test_bazaar_rendering_drops_bookkeeping_and_reasoning():
    b = {"bundle_id": "x", "bundle_kind": "reasoning", "digest": "abc", "judged_actor_ids": [5],
         "actions": [{"kind": "send_message", "event_id": 1, "args": {"body": "hi"}, "photo_ids": []}],
         "reasoning": [{"agent_id": 5, "reasoning_summary": "I will lie"}]}
    full, public = json.loads(render_bundle(b)), json.loads(render_bundle(b, public=True))
    assert "digest" not in full and "bundle_id" not in full and "event_id" not in full["actions"][0]
    assert "photo_ids" not in full["actions"][0] and full["reasoning"][0]["reasoning_summary"] == "I will lie"
    assert "reasoning" not in public and public["actions"] == full["actions"]


# ------------------------------------------------------------------------------------------------ prompts
@pytest.mark.parametrize("text,mode,want", [
    ("YES", "direct", 1), ("No.", "direct", 0), ("**NO**", "direct", 0), ("", "direct", None),
    ("maybe", "direct", None), ("<think>yes</think>NO", "direct", 0),
    ("Step 3 has 2 apples, so yes.\nVERDICT: YES", "reason", 1), ("...\nVERDICT: **NO**", "reason", 0),
    ("I think the answer is no", "reason", 0), ("It is unclear.", "reason", None),
])
def test_parse_verdict(text, mode, want):
    assert parse_verdict(text, mode) == want


def test_prompts(harvest):
    e = harvest[0]
    raw = build_messages(e, "raw", "direct")
    assert "fewer than 3" in raw[0]["content"] and "marked x" in raw[1]["content"]
    assert "marked x" not in build_messages(e, "summary", "direct")[1]["content"]
    bz = dict(dataset="bazaar", ftype="F2", judged_actor_ids=[7], label=1, renderings={"bundle": "{}"})
    assert "#7" in build_messages(bz, "bundle", "reason")[1]["content"]


# ------------------------------------------------------------------------------------------------ metrics
def test_rates_and_c1():
    rows = ([dict(label=1, verdict=1, tokens_in=10, tokens_out=1)] * 3 + [dict(label=1, verdict=None, tokens_in=10,
            tokens_out=1)] + [dict(label=0, verdict=1, tokens_in=10, tokens_out=1)] + [dict(label=0, verdict=0,
            tokens_in=10, tokens_out=1)] * 3)
    r = rates(rows)
    assert r["alpha"] == 0.25 and r["beta"] == 0.25 and r["no_verdict"] == 1 / 8 and r["tokens_in"] == 10
    lo, hi = wilson(3, 4)
    assert 0 < lo < 0.75 < hi <= 1
    assert effective_penalty(0.5, 0.0, 2) == 1.0
    assert (c1_class(0.5), c1_class(0.6), c1_class(0.75)) == ("fails", "uncertain", "protects")


def test_job_list_format():
    c1 = [dict(model="m1", mode="direct", rendering="summary", alpha=0.1, beta_eat=0.05),
          dict(model="m2", mode="direct", rendering="summary", alpha=0.11, beta_eat=0.04),
          dict(model="m3", mode="direct", rendering="raw", alpha=0.9, beta_eat=0.1)]
    lines = list(runs(c1))
    assert len(lines) == 3 * 2 * 2 * 3
    name, args = lines[0]
    assert name == "L_m1_summary_k1_c1.0_seed1"
    assert args == "--k 1 --chooser random --penalty-apples 7.0 --miss-rate 0.1 --false-alarm 0.05 --seed 1"
    assert len(list(runs(c1, dedupe=0.05))) == 2 * 2 * 2 * 3  # m2 rounds like m1
    small = list(runs(c1, one_seed_if_fails=True))
    assert sum(n.startswith("L_m3") for n, _ in small) == 4  # 1 - 0.9 - 0.1 = 0: one seed per (k, c)


# ------------------------------------------------------------------------------------------------ judge loop
class FakeBackend:
    model = "fake"

    def chat(self, messages, max_tokens):
        return dict(text="YES", thinking="", tokens_in=100, tokens_out=1, seconds=0.0, model="fake", done_reason="stop")


def test_run_items_resumes_and_caps(tmp_path, harvest):
    items = S.items(harvest[:5], [])
    out, log = tmp_path / "x.jsonl", tmp_path / "tok.tsv"
    st = run_items(FakeBackend(), items, "direct", out, 4, token_cap=250, token_log=log)
    assert st["written"] == 3 and "token cap" in st["stop"] and tokens_logged(log) == 303
    st = run_items(FakeBackend(), items, "direct", out, 4)  # resume: only the 7 left
    assert st["todo"] == 7 and st["written"] == 7
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert len({(r["id"], r["rendering"]) for r in rows}) == 10 and all(r["verdict"] == 1 for r in rows)


# ------------------------------------------------------------------------------------------------ simulation
def test_recorded_violations_match_the_rule():
    pytest.importorskip("jax")
    pytest.importorskip("socialjax")
    from overseer.llm.harvest_records import MIXES, patch, record_episodes
    tr = record_episodes(MIXES["two_greedy"], seeds=[3], steps=60)
    eaten, viol, near = tr["eaten"][0], tr["violations"][0], tr["near"][0]
    assert viol.sum() > 0
    assert ((eaten & (near < 3)) == viol).all()
    t, a = map(int, next(zip(*eaten.nonzero())))
    assert count_diamond(patch(tr["pre_grid"][0][t], tr["post_locs"][0][t, a])) == near[t, a]
