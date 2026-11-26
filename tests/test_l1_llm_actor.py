import json
import re

import numpy as np

from experiments.oversight import run_l1_llm_pilot as R
from fishery_sim import llm_actor as L


def test_parse_decision_rejects_out_of_range_and_garbage():
    assert L.parse_decision('{"catch": 0.4}', "catch", 0, 0.5) == (0.4, None)
    assert L.parse_decision('{"catch": 0.9}', "catch", 0, 0.5)[1] == "range"
    assert L.parse_decision("nope", "catch", 0, 0.5)[1] == "parse"


def test_comprehension_scoring():
    text, ans = L.comprehension_prompt("fine_high", 1 / 6, 6.0)
    assert "17%" in text and ans == {"q1": "C", "q2": "C", "q3": "A"}
    assert L.parse_comprehension(json.dumps(ans), ans) == 3
    assert L.parse_comprehension("{}", ans) == 0


def test_rules_text_is_neutral_and_permission_adds_one_sentence():
    silent = L.rules_text("fine_high", "silent", 20, 1 / 6, 6.0)
    explicit = L.rules_text("fine_high", "explicit", 20, 1 / 6, 6.0)
    for word in ("cheat", "honest", "test", "experiment", "lie"):
        for text in (silent, explicit):
            assert not re.search(rf"\b{word}", text.lower())
    assert explicit.startswith(silent.rsplit("\n", 1)[0]) and len(explicit) > len(silent)


def test_episode_with_fake_client_overtakes_and_fallbacks_are_counted(tmp_path):
    used = dict(tokens=0)
    with open(tmp_path / "calls.jsonl", "w") as log:
        e = R.episode(L.FakeClient(over=0.5, broken_every=5), 0, "fine_high", "silent", 4, log, used)
    A = e["llm_agents"]
    assert len(A) == 4 and len(e["steps"]) == 4
    over = [s["taken"][i] > s["allowance"][i] + 0.01 for s in e["steps"] for i in A]
    assert np.mean(over) > 0.5          # the fake actor over-takes whenever the cut binds
    honest = [i for i in range(6) if i not in A]
    for s in e["steps"]:                # rule-based agents always take exactly their allowance
        for i in honest:
            assert abs(s["taken"][i] - s["allowance"][i]) < 1e-12
    assert e["reprompts"] >= 1 and e["fallbacks"] >= 0
    assert used["tokens"] > 0


def test_no_audits_means_no_fines(tmp_path):
    used = dict(tokens=0)
    with open(tmp_path / "calls.jsonl", "w") as log:
        e = R.episode(L.FakeClient(over=1.0), 1, "none", "explicit", 3, log, used)
    assert sum(e["fines"]) == 0
