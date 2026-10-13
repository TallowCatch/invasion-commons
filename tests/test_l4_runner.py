"""Gate tests for L4 and L5 (protocols: studies/L4_llm_identification, studies/L5_llm_families)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from experiments.oversight import llm_clients as C
from experiments.oversight import run_l2_llm_agents as l2
from experiments.oversight import run_l4_llm_identification as R
from fishery_sim import llm_actor as L

REF = json.loads((Path(__file__).parent / "data_l2_prompts_ref.json").read_text())


def test_l2_prompts_unchanged_byte_for_byte():
    for cell, (w, c, f) in l2.ALL_CELLS.items():
        text, answers = l2.comprehension(c, f, 12345)
        assert l2.rules_text(w, c, f) == REF[cell]["rules"]
        assert [text, answers] == REF[cell]["comp"]


@pytest.mark.parametrize("q,shown", [(1 / 3, "0.333 (about 33 in 100)"), (1 / 12, "0.083 (about 8 in 100)")])
def test_only_the_audit_rate_changes(q, shown):
    for wording in ("explicit", "paraphrase"):
        base, new = l2.rules_text(wording, "fine", 24.0).split("\n"), l2.rules_text(wording, "fine", 24.0, q).split("\n")
        diff = [(a, b) for a, b in zip(base, new) if a != b]
        assert len(base) == len(new) and len(diff) == 1
        a, b = diff[0]
        assert b == a.replace("0.167 (about 17 in 100)", shown).replace("0.167 chance (about 17 in 100)",
                                                                         shown.replace(" (", " chance ("))
    text, answers = l2.comprehension("fine", 24.0, 7, q)
    assert f"About {round(100 * q)}%" in text


def test_no_check_cell_has_no_audits(tmp_path):
    rules = l2.rules_text("explicit", "none", 0.0, 0.0)
    assert "No one checks the actual catches." in rules and "probability" not in rules
    e = l2.episode(L.FakeClient(over=0.5), 0, "N0", open(tmp_path / "log", "w"), dict(tokens=0), horizon=4,
                   spec=("explicit", "none", 0.0, 0.0), seeds=R.SEEDS)
    assert not any(any(s["audited"]) for s in e["steps"]) and sum(e["fines"]) == 0


def test_audits_are_nested_across_audit_rates(tmp_path):
    log, used = open(tmp_path / "log", "w"), dict(tokens=0)
    run = {q: l2.episode(L.FakeClient(over=0.5), 3, f"x{q}", log, used, horizon=6, spec=("explicit", "fine", 12.0, q),
                         seeds=R.SEEDS) for q in (1 / 12, 1 / 6, 1 / 3)}
    for t in range(6):
        a12, a6, a3 = (run[q]["steps"][t]["audited"] for q in (1 / 12, 1 / 6, 1 / 3))
        assert all((not x or y) and (not y or z) for x, y, z in zip(a12, a6, a3))


def test_new_populations_differ_from_l2():
    from experiments.oversight.claude_oversight_common import fishery_setup
    _, p_l2, _ = fishery_setup(0, l2.SEEDS["pop"], 20)
    _, p_l4, _ = fishery_setup(0, R.SEEDS["pop"], 20)
    assert (p_l2["thresholds"] != p_l4["thresholds"]).all()


def test_threshold_fines_and_stage2_cells():
    assert R.threshold_fines(4.48) == (30, 24) and R.threshold_fines(3.96) == (24, 18)
    assert R.threshold_fines(0.5) == (6, None) and R.threshold_fines(9.0) == (None, 48)
    groups = R.stage2_groups("L4", dict(g1_E=4.48, g1_P=3.96))
    assert [sorted(g[1]) for g in groups] == [["E24", "E30", "P18", "P24"],
                                             ["E12", "E18", "Q12_24", "Q12_36", "Q3_6", "Q3_9"]]
    assert [len(g[0]) for g in groups] == [10, 20]
    assert {g[1]["Q3_6"][3] for g in groups if "Q3_6" in g[1]} == {1 / 3}
    for c, (_, _, F, q) in groups[1][1].items():  # matched cells: e = 2 or 3
        assert round(q * F, 9) in (2, 3)
    # a threshold cell that is also a matched cell is played once, on 20 populations
    only = R.stage2_groups("L4", dict(g1_E=2.5))  # F* = 18, F- = 12: both already matched cells
    assert len(only) == 1 and len(only[0][0]) == 20 and {"E12", "E18"} <= set(only[0][1])
    assert [sorted(g[1]) for g in R.stage2_groups("L5", dict(g1_E=3.9))] == [["E18", "E24"]]


def test_openrouter_missing_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(C, "read_key_file", lambda path=None: None)
    with pytest.raises(C.MissingKey):
        C.OpenRouterClient("anthropic/claude-sonnet-5.5")


def _server(status, payload):
    seen = {}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"] = self.headers["Authorization"]
            self.send_response(status)
            self.end_headers()
            self.wfile.write(json.dumps(payload).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, seen


def test_openrouter_client_reply_and_parameters():
    srv, seen = _server(200, {"model": "anthropic/claude-sonnet-5.5", "choices": [{"message": {"content": '{"ok": 1}'}}],
                              "usage": {"prompt_tokens": 11, "completion_tokens": 3, "cost": 0.001}})
    try:
        c = C.OpenRouterClient("anthropic/claude-sonnet-5.5", api_key="test-key",
                               url=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
        r = c.chat([{"role": "user", "content": "hi"}], seed=5)
    finally:
        srv.shutdown()
    assert r.content == '{"ok": 1}' and (r.prompt_tokens, r.completion_tokens, r.cost) == (11, 3, 0.001)
    assert "temperature" not in seen["body"] and "seed" not in seen["body"]  # not accepted by this model
    assert seen["body"]["response_format"] == {"type": "json_object"} and seen["auth"] == "Bearer test-key"


def test_openrouter_no_credit_stops_cleanly():
    srv, _ = _server(402, {"error": {"code": 402, "message": "insufficient credits"}})
    try:
        c = C.OpenRouterClient("google/gemini-3.8-flash", api_key="k",
                               url=f"http://127.0.0.1:{srv.server_port}/v1/chat/completions")
        with pytest.raises(l2.QuotaStop):
            c.chat([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()


def test_smoke_both_stages(tmp_path):
    s1 = R.run_stage(L.FakeClient(over=0.5), "L4", "fake", 1, tmp_path, horizon=2)
    pred = tmp_path / "prediction.json"
    pred.write_text(json.dumps({"models": {"fake": dict(g1_E=4.48, g1_P=3.96)}}))
    s2 = R.run_stage(L.FakeClient(over=0.5), "L4", "fake", 2, tmp_path, horizon=2, prediction_path=pred,
                     contexts_arg="0-0")
    names = sorted(p.stem for p in (tmp_path / "episodes").glob("*.json"))
    assert s1 == s2 == "done"
    assert {"E0__0", "P0__0", "N0__0", "E24__0", "E30__0", "P18__0", "P24__0", "Q3_6__0", "Q12_36__0"} <= set(names)


def test_stage_time_budget_spans_both_groups(tmp_path):
    """13 Oct fix: one time budget per stage. Before, each group restarted it and remote jobs overran their limit."""
    import time as _t
    pred = tmp_path / "prediction.json"
    pred.write_text(json.dumps({"models": {"fake": dict(g1_E=4.48, g1_P=3.96)}}))
    t0 = _t.time()
    st = R.run_stage(L.FakeClient(over=0.5), "L4", "fake", 2, tmp_path, horizon=20, prediction_path=pred,
                     max_minutes=0.002)
    assert st == "time" and _t.time() - t0 < 30
