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
