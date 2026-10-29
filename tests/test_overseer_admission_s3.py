"""Admission test B for the overseer (overseer/ADMISSION.md): S3 Parts A and B re-run with the overseer as reviewer.

The re-run (experiments/oversight/run_overseer_admission.py) must give the stored S3 runs exactly: every episode
record, every adaptive search, the byte-identical output files, and the stored analysis (s3_summary.json, part "AB",
recomputed by analyze_s3.parts_ab with its own bootstrap seed). About a minute (full profile, 4,480 test episodes).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from experiments.oversight import run_overseer_admission as adm
from experiments.oversight import run_s2_compliance_deterrence as s2

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def test_s3_parts_a_b_through_overseer_equal_stored(tmp_path):
    stored = {k: ROOT / v for k, v in adm.STORED.items()}
    summary = ROOT / adm.SUMMARY
    if not all((p / "episodes.jsonl.gz").exists() for p in stored.values()) or not summary.exists():
        pytest.skip("stored S3 Part A/B runs absent")
    P = s2.profile("full")
    out = {}
    for part in "AB":
        out[part] = tmp_path / part
        episodes, searches = adm.part_ab(part, P)
        adm.write(out[part], episodes, searches, part, "full", 0.0)
        for f in ("episodes.jsonl.gz", "adaptive_search.json"):
            assert sha(out[part] / f) == sha(stored[part] / f), f"S3 Part {part} {f} differs"
    res = adm.compare(out["A"], out["B"], stored, summary)
    assert res["A_n_episodes"] == (2944, 2944) and res["B_n_episodes"] == (1536, 1536)
    for k in ("A_episodes", "A_search", "B_episodes", "B_search", "summary_AB"):
        assert res[k] == [], (k, res[k][:5])
