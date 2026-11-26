"""Inspector metrics: miss rate alpha, false-alarm rate beta, confusion tables, tokens per check, and claim C1.

alpha = share of violation records the judge did not flag (a reply with no verdict counts as not flagged);
beta = share of honest records it flagged; beta_eat = beta on honest Harvest records with at least one apple taken
(the false alarm the RL wrapper models: it falls only on honest harvests).
C1: an inspector deters as if the average penalty were e = (1 - alpha - beta_eat) * c. Harvest learners lost the
orchard at e <= 0.5 in every run and kept it at e >= 0.75 in every run (notes/socialjax/phase2_ippo/results.md).
"""
from __future__ import annotations

import math

FAILS_AT, PROTECTS_AT = 0.5, 0.75  # effective average penalty (apples): every tested run died / survived


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def confusion(rows):
    """Counts of (true label, verdict) with verdict in {1, 0, None}."""
    c = {(lab, v): 0 for lab in (1, 0) for v in (1, 0, None)}
    for r in rows:
        c[(r["label"], r["verdict"])] += 1
    return c


def rates(rows, honest_filter=None):
    """alpha, beta and friends for a list of judged rows (each has label, verdict, tokens_in, tokens_out)."""
    pos = [r for r in rows if r["label"] == 1]
    neg = [r for r in rows if r["label"] == 0]
    if honest_filter is not None:
        neg = [r for r in neg if honest_filter(r)]
    missed = sum(r["verdict"] != 1 for r in pos)
    flagged = sum(r["verdict"] == 1 for r in neg)
    n = len(rows)
    return dict(
        n=n, n_violation=len(pos), n_honest=len(neg),
        alpha=missed / len(pos) if pos else float("nan"), alpha_ci=wilson(missed, len(pos)),
        beta=flagged / len(neg) if neg else float("nan"), beta_ci=wilson(flagged, len(neg)),
        yes_rate=sum(r["verdict"] == 1 for r in rows) / n if n else float("nan"),
        no_verdict=sum(r["verdict"] is None for r in rows) / n if n else float("nan"),
        tokens_in=sum(r["tokens_in"] for r in rows) / n if n else float("nan"),
        tokens_out=sum(r["tokens_out"] for r in rows) / n if n else float("nan"),
        seconds=sum(r.get("seconds") or 0 for r in rows) / n if n else float("nan"),
    )


def effective_penalty(alpha: float, beta: float, c: float) -> float:
    return (1 - alpha - beta) * c


def c1_class(e: float) -> str:
    if e >= PROTECTS_AT:
        return "protects"
    if e <= FAILS_AT:
        return "fails"
    return "uncertain"
