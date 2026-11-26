"""Slide figure: language models with audits only vs audits + memory (no fine), from the stored L2 games.
Usage: PYTHONPATH=. python3 make_memory_llm.py STORE out.png   (STORE = a checkout of the l2-results branch)"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fishery_sim import llm_actor as L

THR = 0.06 / L.MAX_CATCH
root, out = Path(sys.argv[1]), sys.argv[2]
rng = np.random.default_rng(20261019)


def stats(m, run, cell):
    eps = [json.loads(p.read_text()) for p in sorted((root / run / m / "episodes").glob(cell + "__*.json"))]
    ot = np.array([np.mean([s["taken"][i] > s["allowance"][i] + THR for s in e["steps"] for i in e["llm_agents"]]) for e in eps]) * 100
    bh = np.array([np.mean([s["msy_break"] for s in e["steps"]]) for e in eps]) * 100
    def ci(x):
        d = x[rng.integers(0, len(x), (4000, len(x)))].mean(1)
        return x.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)
    return ci(ot), ci(bh)


M = [("gpt-oss_120b-cloud", "gpt-oss-120b"), ("nemotron-3-super_cloud", "Nemotron 3 Super")]
res = {lab: {"none": stats(m, "claude_l2_v1", "E0"), "mem": stats(m, "claude_l2_em_v2", "EM")} for m, lab in M}
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Calibri", "Arial", "Helvetica", "DejaVu Sans"], "font.size": 12})
fig, axs = plt.subplots(1, 2, figsize=(8.2, 3.9), dpi=220)
GREY, NAVY = "#A9A6A1", "#062746"
titles = ["Rounds a model took more than its allowance (%)", "Rounds the lake ended below half capacity (%)"]
for k, ax in enumerate(axs):
    x, w = np.arange(2), 0.36
    for j, (cond, col, name) in enumerate([("none", GREY, "Audits only"), ("mem", NAVY, "Audits + memory")]):
        vals = [res[lab][cond][k] for _, lab in M]
        est = [v[0] for v in vals]
        lo, hi = [v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]
        bars = ax.bar(x + (j - 0.5) * w, est, w, color=col, label=name, yerr=[lo, hi], capsize=3,
                      error_kw=dict(lw=0.8, ecolor="#55524E"))
        for b, e, h in zip(bars, est, hi):
            ax.text(b.get_x() + b.get_width() / 2, e + h + 2, f"{e:.1f}", ha="center", va="bottom", fontsize=11,
                    color=NAVY if col == NAVY else "#55524E", fontweight="bold" if col == NAVY else "normal")
    ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in M]); ax.set_ylim(0, 110); ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title(titles[k], fontsize=12, color="#1C191A", pad=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#888"); ax.spines["bottom"].set_color("#888"); ax.tick_params(colors="#333")
    ax.grid(axis="y", color="#E5E3DF", lw=0.7); ax.set_axisbelow(True)
h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0), fontsize=11.5)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(out, facecolor="white")
for lab, r in res.items():
    print(lab, {c: [round(v[0], 1) for v in r[c]] for c in r})
