"""Task 4 of notes/socialjax/locality/protocol.md: central vs M overseer groups at equal total audit budget in
Willis et al.'s commons, N = 64 and 256 (CPU, no LLM calls).

  .venv/bin/python -m experiments.scale.run_scale --out results/locality/scale.json   (about a minute)

Per seed: one game of N players drawn from the committed strategy pool, then every overseer design on the same
defection matrix (experiments/scale/groups.py says why that is exact here). K = N/8 audits per round.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import overseer.adapters.emergent_llm as E
from experiments.scale.groups import assign_groups, audit, comm_share, own_group_view, play_game

SIZES = {64: (1, 4, 16), 256: (1, 4, 16, 64)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/locality/scale.json")
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    registry = E.load_registry()
    rows, games = [], []
    t0 = time.time()
    for n, Ms in SIZES.items():
        K = n // 8
        for seed in range(a.seeds):
            D, att, stock = play_game(registry, n, 7_000 + seed)
            games.append(dict(n=n, seed=seed, defect_rate=float(D[:-1].mean()), final_stock=float(stock[-1]),
                              defect_rate_by_attitude={x: float(D[:-1, att == x].mean()) for x in sorted(set(att))}))
            for M in Ms:
                for split in ("random", "by attitude"):
                    if M == 1 and split == "by attitude":
                        continue
                    rng = np.random.default_rng(seed * 1000 + M)
                    groups = assign_groups(n, M, rng, att if split == "by attitude" else None)
                    for chooser in ("random", "history"):
                        for pooled in ((False,) if M == 1 else (False, True)):
                            r = audit(D, groups, K, chooser, np.random.default_rng(seed), pooled=pooled)
                            rows.append(dict(n=n, K=K, M=M, split=split, chooser=chooser, pooled=pooled, seed=seed,
                                             comm_share=comm_share(n, M, pooled), records_per_overseer=n // M,
                                             **r))
        print(f"N = {n}: {time.time() - t0:.0f} s", flush=True)
    view = [own_group_view(n, M) for n, Ms in SIZES.items() for M in Ms]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(rows=rows, games=games, own_group=view), indent=1))


if __name__ == "__main__":
    main()
