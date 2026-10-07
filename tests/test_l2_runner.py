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
