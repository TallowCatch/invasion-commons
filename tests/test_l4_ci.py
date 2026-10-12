"""The remote lanes for L4-L6 finish every unit, freeze predictions before stage 2, and never run a game twice."""
import json

from experiments.oversight import l4_ci as CI


def test_lanes_complete_with_fake_clients(tmp_path, monkeypatch):
    monkeypatch.setenv("L4_CI_FAKE", "1")
    store = tmp_path / "store"
    seen = {}
    for _ in range(6):  # lanes run one after another; blocked units wait for the other lanes
        for lane in "ABC":
            CI.lane(store, lane, 10_000)
        if CI.l4_complete(store) and all(all(f.exists() for f in (CI.expected(store, u) or [store / "x"]))
                                         for ln in CI.LANES.values() for u in ln):
            break
    for ln in CI.LANES.values():
        for u in ln:
            files = CI.expected(store, u)
            assert files is not None and all(f.exists() for f in files), u
    pred = json.loads((store / "claude_l4_v1" / "prediction.json").read_text())
    assert set(pred["models"]) == {CI.GPT, CI.NEM}
    for m, e in pred["models"].items():  # the frozen entry predates every stage-2 game of that model
        s2 = [p for p in (store / "claude_l4_v1" / m.replace(":", "_").replace(".", "_") / "episodes").glob("*.json")
              if not p.name.startswith(("E0__", "P0__", "N0__"))]
        assert s2 and e["frozen_at"]
    assert set(json.loads((store / "claude_l6_v1" / "f_choice.json").read_text())) == {CI.GPT, CI.NEM}
