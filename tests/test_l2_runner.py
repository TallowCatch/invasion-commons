import json

from experiments.oversight import run_l2_llm_agents as l2
from fishery_sim import llm_actor as L


def test_comprehension_answers_follow_the_shuffle():
    text, ans = l2.comprehension("fine", 8.0, 42)
    for k, right in (("q1", "About 17%"), ("q2", "You pay a fine of 8 tonnes"), ("q3", "The fish you catch minus any fines")):
        line = [x for x in text.split("\n") if x.startswith(k.upper())][0]
        assert f"{ans[k]}) {right}" in line


def test_quota_stop_then_resume(tmp_path):
    cells, ctx = ("E0", "EM"), (0,)
    assert l2.run(l2.QuotaAfter(10), tmp_path, cells, ctx, "fake", horizon=2) == "quota"
    assert json.loads((tmp_path / "STATUS").read_text())["status"] == "quota"
    assert l2.run(L.FakeClient(), tmp_path, cells, ctx, "fake", horizon=2) == "done"
    assert (tmp_path / "DONE").exists() and len(list((tmp_path / "episodes").glob("*.json"))) == 2


def test_memory_cell_lowers_allowance_after_a_catch(tmp_path):
    l2.run(L.FakeClient(over=1.0), tmp_path, ("EM",), (0, 1, 2), "fake", horizon=20)
    eps = [json.loads(p.read_text()) for p in (tmp_path / "episodes").glob("*.json")]
    assert all(e["fines"] == [0.0] * 6 for e in eps)
    caught_any = any(any(s["caught"]) for e in eps for s in e["steps"])
    assert caught_any


def test_fenced_json_is_accepted_and_plain_json_unchanged():
    fenced = '```json\n{"request_tonnes": 3, "reason": "x"}\n```'
    assert l2.parse_decision(fenced, "request_tonnes", 0, 6) == (3.0, None)
    assert l2.parse_decision('{"request_tonnes": 2.5}', "request_tonnes", 0, 6) == L.parse_decision('{"request_tonnes": 2.5}', "request_tonnes", 0, 6)
    assert l2.parse_decision("not json", "request_tonnes", 0, 6)[1] == "parse"


def test_time_limit_stops_between_games(tmp_path):
    assert l2.run(L.FakeClient(), tmp_path, ("E0", "E36"), (0, 1), "fake", horizon=2, max_minutes=0) == "time"
    assert not list((tmp_path / "episodes").glob("*.json"))


def test_api_mode_uses_api_names_and_auth(monkeypatch):
    c = l2.Client("gemma4:31b-cloud", base_url="https://ollama.com", api_key="k")
    assert c.remote and l2.API_NAMES[c.model] == "gemma4:31b"


def test_split_lanes_keep_separate_files_and_share_the_token_count(tmp_path):
    cells = ("E0",)
    assert l2.run(L.FakeClient(), tmp_path, cells, (0,), "fake", horizon=2, tag="ctx0-0") == "done"
    assert l2.run(L.FakeClient(), tmp_path, cells, (1,), "fake", horizon=2, tag="ctx1-1") == "done"
    for f in ("calls_ctx0-0.jsonl", "calls_ctx1-1.jsonl", "DONE_ctx0-0", "DONE_ctx1-1", "manifest_ctx1-1.json"):
        assert (tmp_path / f).exists()
    assert not (tmp_path / "calls.jsonl").exists()
    assert json.loads((tmp_path / "STATUS_ctx1-1").read_text())["tokens"] > json.loads(
        (tmp_path / "STATUS_ctx0-0").read_text())["tokens"]


def test_a_cut_off_last_log_line_does_not_stop_the_run(tmp_path):
    assert l2.run(L.FakeClient(), tmp_path, ("E0",), (0,), "fake", horizon=2) == "done"
    with open(tmp_path / "calls.jsonl", "a") as f:
        f.write('{"phase": "catch", "prompt_tok')  # a job killed mid-write
    assert l2.run(L.FakeClient(), tmp_path, ("E0", "E36"), (0,), "fake", horizon=2) == "done"
    assert not list((tmp_path / "episodes").glob("*.tmp"))


def test_tonnes_memory_rule_replays_from_the_saved_steps(tmp_path):
    """Amendment 5: later allowance = max(0, scale * request - mean recorded excess), also when nothing was cut."""
    assert l2.MEMORY_RULE == "tonnes"
    l2.run(L.FakeClient(over=1.0), tmp_path, ("EM",), (0, 1, 2), "fake", horizon=20)
    eps = [json.loads(p.read_text()) for p in (tmp_path / "episodes").glob("*.json")]
    lowered = 0
    for e in eps:
        assert e["memory_rule"] == "tonnes" and e["fines"] == [0.0] * 6
        record = {i: [] for i in e["llm_agents"]}
        for s in e["steps"]:
            for i in e["llm_agents"]:
                base = s["scale"] * s["requests"][i]
                want = max(0.0, base - sum(record[i]) / len(record[i])) if record[i] else base
                assert abs(s["allowance"][i] - want) < 1e-9
                lowered += s["allowance"][i] < base - 1e-9
            for i in e["llm_agents"]:
                if s["caught"][i]:
                    record[i].append(s["taken"][i] - s["allowance"][i])
    assert lowered > 0


def test_l3_cells_change_only_the_fine_and_are_not_in_the_l2_set():
    import re
    assert set(l2.L3_CELLS).isdisjoint(l2.CELLS) and len(l2.CELLS) == 11
    base = l2.rules_text("explicit", "fine", 36.0)
    for cell, (wording, consequence, fine) in l2.L3_CELLS.items():
        text = l2.rules_text(wording, consequence, fine)
        assert re.sub(r"fine of [0-9.]+ tonnes", "", text) == re.sub(r"fine of [0-9.]+ tonnes", "", base)
        assert f"fine of {fine:g} tonnes" in text


def test_l3_cell_runs(tmp_path):
    assert l2.run(L.FakeClient(over=0.5), tmp_path, ("E24",), (0,), "fake", horizon=3) == "done"
    e = json.loads((tmp_path / "episodes" / "E24__0.json").read_text())
    assert e["fine"] == 24.0


def test_l3_first_deterring_fine():
    from experiments.oversight import analyze_l3 as a3
    assert a3.first_deterring({0: .5, 24: .3, 30: .0, 36: .0}) == (30, False)
    assert a3.first_deterring({0: .5, 12: .01, 24: .3, 30: .0, 36: .0}) == (30, True)
    assert a3.first_deterring({0: .5, 30: .2, 36: .1}) == (None, False)
