"""Tests for R1b: audit-aware (Pool A) and blind (Pool B) strategies in Willis et al.'s CPR. No LLM or network calls:
every generation test uses run_r1b_generate.FakeGenClient.

Protocol: notes/claude_audit_20261005/studies/R1b_audit_aware_strategies/protocol.md
Run: PYTHONPATH=. python3 -m pytest -q tests/test_r1b.py tests/test_r1a.py
"""
import inspect
import json
import shutil

import numpy as np
import pytest

E = pytest.importorskip("overseer.adapters.emergent_llm")

from emergent_llm.common import Action, Attitude, PlayerHistory  # noqa: E402
from emergent_llm.games.common_pool import CommonPoolDescription  # noqa: E402
from emergent_llm.generation import create_strategies as W  # noqa: E402
from emergent_llm.generation import prompts as WP  # noqa: E402
from emergent_llm.players import SimplePlayer  # noqa: E402

from experiments.oversight import analyze_r1b as A  # noqa: E402
from experiments.oversight import run_r1b_generate as G  # noqa: E402
from experiments.oversight import run_r1b_parts as P  # noqa: E402


class NoNetwork(Exception):
    pass


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any attempt to open a URL fails the test."""
    import urllib.request

    def boom(*a, **k):
        raise NoNetwork("network call attempted in a test")
    monkeypatch.setattr(urllib.request, "urlopen", boom)


@pytest.fixture(scope="module")
def pilot_store(tmp_path_factory):
    out = tmp_path_factory.mktemp("r1b") / "strategies"
    status = G.run("pilot", out, G.FakeGenClient(), workers=1)
    assert status["stopped"] == "done"
    return out


# ------------------------------------------------------------------ prompts
def test_system_prompts_are_theirs():
    assert E.DESCRIPTION_SYSTEM_PROMPT in inspect.getsource(W.generate_strategy_description)
    assert f'"{E.CODE_SYSTEM_PROMPT}"' in inspect.getsource(W.generate_strategy_code)


@pytest.mark.parametrize("att", list(Attitude))
def test_pool_b_prompts_are_unchanged(att):
    assert E.strategy_prompt("B", att) == WP.create_strategy_user_prompt(att, "common_pool")
    assert E.code_prompt("B", "some strategy") == WP.create_code_user_prompt("some strategy", "common_pool")
    assert "AUDITS AND FINES" not in E.strategy_prompt("B", att) + E.code_prompt("B", "x")


@pytest.mark.parametrize("att", [Attitude.SELFISH, Attitude.ALTRUISTIC])
def test_pool_a_prompts_add_exactly_one_section(att):
    theirs = WP.create_strategy_user_prompt(att, "common_pool")
    ours = E.strategy_prompt("A", att)
    assert ours.replace(E.AUDIT_SECTION, "", 1) == theirs and ours.count("AUDITS AND FINES:") == 1
    game = WP.format_game_description("common_pool")
    assert ours.index(E.AUDIT_SECTION) == ours.index(game) + len(game)
    for word in ("audit_prob", "game_description.fine", "last round", "do not change the stock", "q × F"):
        assert word in E.AUDIT_SECTION


def test_pool_a_code_prompt_and_interface():
    desc = "Defect when the gain exceeds q*F.\n" + WP.format_game_description("common_pool")  # adversarial text
    theirs = WP.create_code_user_prompt(desc, "common_pool")
    ours = E.code_prompt("A", desc)
    face = E.pool_a_interface_text()
    their_face = WP.get_interface_description(CommonPoolDescription)
    assert ours.replace(E.AUDIT_SECTION, "", 1).replace(face, their_face) == theirs
    assert "    audit_prob: float\n    fine: float\n" in face
    assert "my_audited: NDArray[np.bool_]" in face and "my_fines: NDArray[np.float64]" in face
    # nothing removed from their interface block
    for line in their_face.splitlines():
        assert line in face


# ------------------------------------------------------------------ the audit-aware game
def _rule_players(n, rule):
    return [SimplePlayer(f"p{i}", (lambda h, i=i: Action.C if rule(i, h.round_number) else Action.D))
            for i in range(n)]


def _mixed(i, t):
    return (i + t) % 3 != 0


def test_aware_game_same_stock_audits_and_fines_as_r1a():
    n = 64
    a = E.AuditedCommonPoolGame(_rule_players(n, _mixed), E.r1b_description("B", n, 0.25, 3.0), key=(1, 2, 3))
    b = E.AuditAwareCommonPoolGame(_rule_players(n, _mixed), E.r1b_description("A", n, 0.25, 3.0), key=(1, 2, 3))
    ra, rb = a.play_game(), b.play_game()
    for k in ("stock_path", "extraction", "audited", "fines", "caught"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert ra.total_payoffs == rb.total_payoffs and ra.defections_nonfinal == rb.defections_nonfinal
    assert np.allclose(b.history.payoffs, b.extraction - b.fines)      # Pool A sees net payoffs
    assert np.array_equal(a.history.payoffs, a.extraction)            # Pool B sees the extraction (R1a)
    assert not b.audited[-1].any()


def test_pool_a_strategy_reads_audit_prob_fine_and_history():
    n, q, F = 4, 0.5, 2.5
    seen = []

    class Probe:
        def __init__(self, game_description):
            self.gd = game_description

        def __call__(self, history, current_stock):
            seen.append((self.gd.audit_prob, self.gd.fine, history.round_number,
                         history.my_audited.copy(), history.my_fines.copy(), history.my_payoffs.copy()))
            return Action.D

    from emergent_llm.common import Gene
    from emergent_llm.players import LLMPlayer
    desc = E.r1b_description("A", n, q, F)
    view = desc.strategy_view()
    assert isinstance(view, CommonPoolDescription) and (view.audit_prob, view.fine) == (q, F)
    assert not hasattr(E.r1b_description("B", n, q, F).strategy_view(), "audit_prob")
    focal = LLMPlayer("focal", Gene("x", Attitude.SELFISH), view, Probe, max_errors=0)
    g = E.AuditAwareCommonPoolGame([focal] + _rule_players(n - 1, _mixed), desc, key=(0, 1, 2))
    g.play_game()
    assert len(seen) == 20
    for qq, ff, t, aud, fin, pay in seen:
        assert (qq, ff) == (q, F) and len(aud) == t
        assert np.array_equal(aud, g.audited[:t, 0]) and np.array_equal(fin, g.fines[:t, 0])
        assert np.allclose(pay, g.extraction[:t, 0] - g.fines[:t, 0])
    assert g.fines[:, 0].sum() > 0          # some audits fined the always-D probe


def test_aware_history_is_read_only_and_a_player_history():
    g = E.AuditAwareCommonPoolGame(_rule_players(4, _mixed), E.r1b_description("A", 4, 1.0, 1.0), key=(0, 0, 0))
    g.play_game()
    h = g.player_history(0)
    assert isinstance(h, PlayerHistory)
    with pytest.raises(ValueError):
        h.my_fines[0] = 5.0


# ------------------------------------------------------------------ generation: budget, call cap, resume, logging
def test_pilot_plan_is_balanced():
    slots = G.pilot_slots()
    assert len(slots) == 10
    bases = [a.to_base_attitude() for _, a in slots]
    assert bases.count(Attitude.COLLECTIVE) == 5 and bases.count(Attitude.SELFISH) == 5
    assert {a for _, a in slots} == set(Attitude)          # all 8 synonyms
    full = G.plan(16)
    assert len(full) == 128 and full[:10] == slots and len(set(full)) == 128


def test_pilot_writes_registry_and_log(pilot_store):
    for pool in E.POOLS:
        reg = E.load_pool(pilot_store, pool)
        assert sum(reg.count_strategies(g) for g in reg.available_genes) == 10
        assert len(reg.available_genes) == 8
    rows = G.read_jsonl(pilot_store / "calls.jsonl")
    assert len(rows) == 40                                  # 20 strategies x (description + code)
    prompts = {r["hash"]: r["text"] for r in G.read_jsonl(pilot_store / "prompts.jsonl")}
    for r in rows:
        for k in ("prompt_hash", "system_hash", "response", "prompt_eval_count", "eval_count", "cost_usd",
                  "cum_cost_usd", "pool", "attitude", "index", "step"):
            assert k in r
        assert G.sha(prompts[r["prompt_hash"]]) == r["prompt_hash"]
        assert ("AUDITS AND FINES:" in prompts[r["prompt_hash"]]) == (r["pool"] == "A")
    tok = sum(r["prompt_eval_count"] + r["eval_count"] for r in rows)
    assert rows[-1]["cum_cost_usd"] == pytest.approx(tok * 0.32 / 1e6)
    rep = G.pilot_report(pilot_store)
    assert rep["finished"] == 20 and rep["projected_usd_256"] == pytest.approx(rep["usd_per_strategy"] * 256)
    assert rep["gate_passes"] == (rep["projected_usd_256"] <= 2.5)


def test_call_cap_stops_and_records(tmp_path):
    st = G.run("pilot", tmp_path, G.FakeGenClient(), call_cap=5)
    assert st["stopped"] == "call_cap" and st["calls"] == 5
    assert len(G.read_jsonl(tmp_path / "calls.jsonl")) == 5
    assert json.loads((tmp_path / "status.json").read_text())["stopped"] == "call_cap"
    st2 = G.run("pilot", tmp_path, G.FakeGenClient(), call_cap=5)      # prior calls are read from the log
    assert st2["stopped"] == "call_cap" and st2["calls_this_run"] == 0


def test_budget_cap_stops_before_reaching_it(tmp_path):
    # each fake call costs 100,000 tokens = $0.032; cap $0.10 -> 3 calls (0.096 + one more call would reach 0.128)
    st = G.run("pilot", tmp_path, G.FakeGenClient(prompt_tokens=50_000, completion_tokens=50_000), cap_usd=0.10)
    assert st["stopped"] == "budget_usd" and st["calls"] == 3 and st["spent_usd"] < 0.10
    # a second run reads the spend from the log and makes no call
    st2 = G.run("pilot", tmp_path, G.FakeGenClient(prompt_tokens=50_000, completion_tokens=50_000), cap_usd=0.10)
    assert st2["calls_this_run"] == 0 and st2["stopped"] == "budget_usd"


def test_resume_matches_uninterrupted_run(tmp_path, pilot_store):
    out = tmp_path / "s"
    caps = [3, 8, 17, 1000]
    for cap in caps:
        st = G.run("pilot", out, G.FakeGenClient(), call_cap=cap, workers=2)
    assert st["stopped"] == "done"
    rows = G.read_jsonl(out / "calls.jsonl")
    keys = [(r["pool"], r["attitude"], r["index"], r["step"], r["attempt"]) for r in rows]
    assert len(keys) == len(set(keys)) == 40               # nothing generated twice
    ref = {(r["pool"], r["attitude"], r["index"], r["step"]): r["prompt_hash"]
           for r in G.read_jsonl(pilot_store / "calls.jsonl")}
    assert {(r["pool"], r["attitude"], r["index"], r["step"]): r["prompt_hash"] for r in rows} == ref
    a = (out / "pool_A/common_pool/gpt-oss-120b.py").read_text().split('"""', 2)[2]
    b = (pilot_store / "pool_A/common_pool/gpt-oss-120b.py").read_text().split('"""', 2)[2]
    assert a == b


def test_quota_stop_is_not_an_attempt_and_resumes(tmp_path):
    st = G.run("pilot", tmp_path, G.FakeGenClient(raise_after=3))
    assert st["stopped"] == "quota"
    recs = G.all_records(tmp_path, "A") + G.all_records(tmp_path, "B")
    assert all(len(r["attempts"]) == 0 or r["status"] == "ok" for r in recs)
    st = G.run("pilot", tmp_path, G.FakeGenClient())
    assert st["stopped"] == "done" and st["counts"]["A"]["ok"] == 10 and st["counts"]["B"]["ok"] == 10


def test_retries_then_failure_is_excluded(tmp_path):
    st = G.run("pilot", tmp_path, G.FakeGenClient(fail_codes=2))
    assert st["counts"]["A"]["ok"] == 10                    # third attempt passes
    assert all(len(r["attempts"]) == 3 for r in G.all_records(tmp_path, "A"))
    out2 = tmp_path / "fail"
    st = G.run("pilot", out2, G.FakeGenClient(fail_codes=3))
    assert st["counts"]["A"]["failed"] == 10 and st["calls"] == 20 * 4
    assert "class Strategy_" not in (out2 / "pool_A/common_pool/gpt-oss-120b.py").read_text()
    calls = st["calls"]
    G.run("pilot", out2, G.FakeGenClient())                 # failed strategies are never regenerated
    assert json.loads((out2 / "status.json").read_text())["calls"] == calls


def test_full_mode_needs_pilot_gate(tmp_path):
    with pytest.raises(SystemExit):
        G.main(["full", "--out", str(tmp_path), "--client", "fake"])


def test_pool_a_harness_rejects_division_by_fine():
    code = '''class Strategy(BaseStrategy):
    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        ratio = current_stock / self.game_description.fine
        return Action.C if ratio > 1 else Action.D'''
    with pytest.raises(Exception):
        G.check_code("A", "```python\n" + code + "\n```", Attitude.SELFISH, 1)


# ------------------------------------------------------------------ Parts 1 and 2
@pytest.fixture(scope="module")
def parts_root(pilot_store, tmp_path_factory):
    root = tmp_path_factory.mktemp("r1b_parts")
    shutil.copytree(pilot_store, root / "strategies")
    for pool in E.POOLS:
        for n in (4, 64):
            for e in P.P1_E:
                assert P.part1_unit(pool, n, e, root / "strategies", root, games=3)[0] == "done"
    return root


def test_part1_pool_b_blind_by_construction(parts_root):
    p1 = A.load_part1(parts_root)
    for n in (4, 64):
        h = A.h2(p1)[str(n)]
        assert h["holds"] and h["max_abs_change_any_strategy"] == 0.0
        # the fake blind strategies do defect (so the check is not vacuous)
        assert h["mean_by_e"]["0.0"] > 0


def test_part1_pool_a_responds_to_expected_fine(parts_root):
    p1 = A.load_part1(parts_root)
    h = A.h1(p1)["64"]
    assert h["rate_e0"] > 0.9 and h["rate_e2"] == 0.0 and h["verdict"] == "holds"
    # files are skipped when they exist
    assert P.part1_unit("A", 64, 2.0, parts_root / "strategies", parts_root, games=3)[0] == "exists"


def test_prediction_and_grid(parts_root):
    P.main(["prediction", "--root", str(parts_root), "--pred-games", "40"])
    with pytest.raises(SystemExit):
        P.main(["prediction", "--root", str(parts_root), "--pred-games", "40"])
    e, rec = P.read_e_pred(parts_root)
    conds = P.conditions(e) if e > 0 else None
    if conds is None:
        pytest.skip("fake pool gave e_pred <= 0")
    names = [c["name"] for c in conds]
    assert names[:4] == ["A_e1.25", "B_e1.25", "A_none", "B_none"] and len(conds) == 12  # Amendment 3 baselines
    none = next(c for c in conds if c["name"] == "A_none")
    assert none["q"] == 0.0 and none["F"] == 0.0
    st, _ = P.run_one(none, 0, parts_root / "strategies", parts_root, generations=2)
    assert st == "done"
    c = next(c for c in conds if c["name"] == "A_e1.25")
    st, _ = P.run_one(c, 0, parts_root / "strategies", parts_root, generations=2)
    assert st == "done" and P.run_one(c, 0, parts_root / "strategies", parts_root, generations=2)[0] == "exists"
    run = A.read_gz(P.run_path(parts_root, c, 0))
    assert run["prediction_utc"] is None or isinstance(run["prediction_utc"], str)
    assert len(run["generation_stats"]) == 2 and run["condition"]["F"] == pytest.approx(1.25 * e / 0.1)


def test_evolution_uses_aware_game_for_pool_a(pilot_store):
    used = []
    orig = E.AuditAwareCommonPoolGame.player_history

    def spy(self, i):
        used.append(i)
        return orig(self, i)
    reg = E.load_pool(pilot_store, "A")
    desc = E.r1b_description("A", 4, 0.1, 10.0)
    E.seed_globals(1)
    ev = E.AuditedCulturalEvolution(E.evolution_config(desc, 1, population_size=16), reg, run=0,
                                    audit_seed=E.R1B_BASE)
    try:
        E.AuditAwareCommonPoolGame.player_history = spy
        stats = ev.run_tournament()
    finally:
        E.AuditAwareCommonPoolGame.player_history = orig
    assert used and len(stats) == 1


def test_r1a_evolution_still_builds_r1a_games(monkeypatch):
    assert E.AuditedCommonPoolDescription.standard(4).game_type() is E.AuditedCommonPoolGame
    ev_sig = inspect.signature(E.AuditedCulturalEvolution.__init__)
    assert ev_sig.parameters["audit_seed"].default == E.R1A_BASE


# ------------------------------------------------------------------ Part 3 heuristic
def test_part3_ast_heuristic():
    pos = G.FAKE_AWARE_CODE.format(base="selfish", selfish=True).split("```python\n")[1].split("```")[0]
    r = A.analyse_code(pos)
    assert r["reads_q_or_F"] and r["reads_audit_history"] and r["ev_comparison"]
    indirect = '''class S:
    def __call__(self, history, current_stock):
        q = self.game_description.audit_prob
        F = self.game_description.fine
        cost = q * F
        gain = current_stock / 8
        if gain - cost > 0:
            return 1
        return 0'''
    assert A.analyse_code(indirect)["ev_comparison"]
    reads_only = '''class S:
    def __call__(self, history, current_stock):
        if self.game_description.fine > 0:
            return 1
        return 0'''
    r = A.analyse_code(reads_only)
    assert r["reads_q_or_F"] and not r["ev_comparison"]
    blind = G.FAKE_BLIND_CODE.format(base="selfish", selfish=True).split("```python\n")[1].split("```")[0]
    assert not A.analyse_code(blind)["ev_comparison"]


def test_amendment_1_holds_the_exact_added_text():
    from pathlib import Path
    proto = (Path(__file__).resolve().parents[1] / "notes" / "claude_audit_20261005" / "studies"
             / "R1b_audit_aware_strategies" / "protocol.md").read_text()
    assert "Amendment 1: exact Pool A prompt additions" in proto
    assert E.AUDIT_SECTION.strip("\n") in proto


def test_part1_no_audit_level(parts_root):
    """Amendment 3: Part 1's no-audit level (q = 0) writes its own file, which the H1-H3 loader does not read."""
    st, _ = P.part1_unit("A", 4, None, parts_root / "strategies", parts_root, games=1)
    assert st == "done"
    rec = A.read_gz(P.part1_path(parts_root, "A", 4, None))
    assert rec["e"] == "none" and rec["q"] == 0.0 and rec["F"] == 0.0
    assert all(f == 0 for row in rec["focal_fines"] for f in row)
