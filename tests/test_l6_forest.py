"""Gate tests for L6 (protocol: studies/L6_llm_forest_spillover)."""
import json

from experiments.oversight import run_l6_llm_forest as F


class Recorder(F.FakeForest):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.prompts = []

    def chat(self, messages, seed=None, json_mode=True):
        self.prompts.append(([m["content"] for m in messages if m["role"] == "user"][0], seed))
        return super().chat(messages, seed, json_mode)


def play(cell, tmp_path, horizon=4, ctx=0):
    rec = Recorder(over=0.5)
    e = F.episode(rec, ctx, cell, open(tmp_path / f"{cell}.log", "w"), dict(tokens=0, cap=10 ** 9), 12.0, horizon)
    return e, rec


def test_programmed_request_is_identical_and_only_its_cut_differs(tmp_path):
    c, _ = play("C_H", tmp_path)
    d, _ = play("D_H", tmp_path)
    assert c["steps"][0]["requests"] == d["steps"][0]["requests"]  # the reviewer sees the same first round
    assert all(abs(s["requests"][0] - F.PROG_REQUEST) < 1e-12 for s in c["steps"] + d["steps"])
    assert all(abs(s["cut"][0] - 1.0) < 1e-12 for s in d["steps"])
    assert all(abs(s["cut"][0] - s["allowance"][0]) < 1e-12 for s in c["steps"])
    assert not any(s["audited"][0] for s in c["steps"] + d["steps"])


def test_visible_adds_only_the_neighbour_line(tmp_path):
    _, h = play("D_H", tmp_path)
    _, v = play("D_V", tmp_path)
    assert not any("neighbour cut" in p for p, _ in h.prompts)
    first_round = [p for p, _ in v.prompts if p.startswith("Round 1 of")]
    assert first_round and not any("neighbour cut" in p for p in first_round)  # nothing to show before round 1
    later = [p for p, _ in v.prompts if "Send your request" in p and not p.startswith("Round 1 of")]
    assert later and all("Last round, your left neighbour cut" in p for p in later)
    hp = [p for p, _ in h.prompts if "Send your request" in p and p.startswith("Round 2 of")]
    vp = [p for p, _ in v.prompts if "Send your request" in p and p.startswith("Round 2 of")]
    for a, b in zip(hp, vp):
        stripped = b.split("\nLast round, your left neighbour")[0] + "\nSend" + b.split("\nSend", 1)[1]
        assert stripped == a


def test_seeds_and_audits_are_shared_across_cells(tmp_path):
    runs = {cell: play(cell, tmp_path) for cell in F.CELLS}
    audits = {cell: [s["audited"] for s in e["steps"]] for cell, (e, _) in runs.items()}
    assert len({json.dumps(a) for a in audits.values()}) == 1
    seeds = {cell: [s for _, s in r.prompts][:6] for cell, (_, r) in runs.items()}
    assert len({json.dumps(s) for s in seeds.values()}) == 1


def test_defector_damages_its_neighbours(tmp_path):
    c, _ = play("C_H", tmp_path, horizon=3)
    d, _ = play("D_H", tmp_path, horizon=3)
    hc, hd = c["steps"][-1]["health_after"], d["steps"][-1]["health_after"]
    assert hd[1] < hc[1] and hd[5] < hc[5]


def test_pilot_rule():
    assert F.pilot_choice(0.10) == 0.0 and F.pilot_choice(0.5) == 12.0 and F.pilot_choice(0.9) == 24.0
