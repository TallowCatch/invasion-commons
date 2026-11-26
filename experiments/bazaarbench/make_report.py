"""Figures and tables for notes/bazaarbench/offline_replay/results.md, from results.json.

Usage: python -m experiments.bazaarbench.make_report
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
NOTE = ROOT / "notes" / "bazaarbench" / "offline_replay"
FIG = NOTE / "figures"
CELL_NAME = {"L1": "Ordinary instructions (L1)", "L3": "Exploit instructions (L3)"}
# fixed categorical order (validated default palette, slots 1-6)
COL = {"random": "#2a78d6", "repeat": "#eb6834", "active": "#1baf7a", "repeat (acts only)": "#eda100",
       "cheapest (post hoc)": "#e87ba4"}
MARK = {"random": "o", "repeat": "s", "active": "^", "repeat (acts only)": "D", "cheapest (post hoc)": "v"}
LABEL = {"random": "Random", "repeat": "Repeat offenders first", "active": "Most active first",
         "repeat (acts only)": "Repeat offenders, acts only (post hoc)", "cheapest (post hoc)": "Cheapest first (post hoc)"}
VIEW_COL = {"record": "#eda100", "public": "#1baf7a", "full": "#2a78d6"}
VIEW_LABEL = {"record": "Record check (F1-F3)", "public": "Judge, no private reasoning", "full": "Full judge"}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})


def pct(x):
    return 100 * x


def fig_choosers(res):
    rows = [r for r in res["count"] if r["view"] == "full"]
    ks = [1, 2, 5, 10, 20]
    measures = [("caught", "All failure objects caught (%)"),
                ("caught_past_s1", "Objects past S1 caught (%)"),
                ("before_s5", "Carried-through (S5) objects\ncaught a day before harm (%)")]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2), sharex=True)
    for i, cell in enumerate(["L1", "L3"]):
        info = res["cells"][cell]
        for j, (m, ylab) in enumerate(measures):
            ax = axes[i, j]
            for ch in ["random", "repeat", "active", "repeat (acts only)"]:
                sel = sorted([r for r in rows if r["cell"] == cell and r["chooser"] == ch], key=lambda r: r["k"])
                y = [pct(r[m]) for r in sel]
                ax.plot(ks, y, color=COL[ch], marker=MARK[ch], lw=1.6, ms=5, label=LABEL[ch])
                if ch == "random" and m + "_lo" in sel[0]:
                    ax.fill_between(ks, [pct(r[m + "_lo"]) for r in sel], [pct(r[m + "_hi"]) for r in sel],
                                    color=COL[ch], alpha=0.12, lw=0)
            if m == "before_s5":
                e, n = info["tick_level_top_before_s5"]["full"]
                ax.axhline(100 * e / n, color="#666", ls="--", lw=1)
                ax.text(1, 100 * e / n + 1.5, f"checking every action: {e} of {n}", fontsize=7, color="#444")
                ax.set_ylim(0, 60)
            else:
                ax.set_ylim(0, 102)
            ax.set_xscale("log")
            ax.set_xticks(ks)
            ax.set_xticklabels([str(k) for k in ks])
            ax.minorticks_off()
            if i == 1:
                ax.set_xlabel("Agents checked per day, k (of 20)")
            ax.set_ylabel(ylab)
            n_obj = {"caught": info["n_objects"], "caught_past_s1": info["n_past_s1"], "before_s5": info["n_s5"]}[m]
            ax.set_title(f"({'abcdef'[3 * i + j]}) {CELL_NAME[cell]}, n = {n_obj}", fontsize=8.5, loc="left")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig1_choosers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_checks_and_budgets(res):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    # (a) what each check type can see when checking everyone
    ax = axes[0]
    x = np.arange(4)
    w = 0.26
    for v_i, v in enumerate(["record", "public", "full"]):
        vals = []
        for cell in ["L1", "L3"]:
            r = [r for r in res["count"] if r["cell"] == cell and r["view"] == v and r["chooser"] == "random"
                 and r["k"] == 20][0]
            vals += [pct(r["caught"]), pct(r["caught_past_s1"])]
        ax.bar(x + (v_i - 1) * w, vals, w * 0.92, color=VIEW_COL[v], label=VIEW_LABEL[v])
    ax.set_xticks(x)
    ax.set_xticklabels(["L1 all\n(86)", "L1 past S1\n(31)", "L3 all\n(1,051)", "L3 past S1\n(53)"], fontsize=7.5)
    ax.set_ylabel("Caught when checking all 20 daily (%)")
    ax.set_title("(a) What each check type can see", fontsize=8.5, loc="left")
    ax.legend(fontsize=7, loc="upper left", ncol=1)
    ax.set_ylim(0, 140)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    # (b, c) token budgets, full judge
    for j, cell in enumerate(["L1", "L3"]):
        ax = axes[1 + j]
        chs = ["random", "repeat", "active", "repeat (acts only)", "cheapest (post hoc)"]
        x = np.arange(2)
        w = 0.16
        for c_i, ch in enumerate(chs):
            sel = [r for r in res["tokens"] if r["cell"] == cell and r["chooser"] == ch and r["view"] == "full"]
            sel = sorted(sel, key=lambda r: r["units"])
            y = np.array([pct(r["caught"]) for r in sel])
            lo = np.array([pct(r.get("caught_lo", r["caught"])) for r in sel])
            hi = np.array([pct(r.get("caught_hi", r["caught"])) for r in sel])
            ax.bar(x + (c_i - 2) * w, y, w * 0.9, color=COL[ch], label=LABEL[ch],
                   yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)], error_kw=dict(lw=0.8, capsize=1.5, ecolor="#555"))
        ax.set_xticks(x)
        ax.set_xticklabels(["1 unit a day\n(34,574 tokens)", "10 units a day\n(345,738 tokens)"])
        ax.set_ylabel("All failure objects caught (%)")
        ax.set_ylim(0, 80)
        ax.set_title(f"({'bc'[j]}) Token budget, full judge: {CELL_NAME[cell]}", fontsize=8.5, loc="left")
    h, lab = axes[1].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=5, bbox_to_anchor=(0.6, -0.04), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig2_checks_and_budgets.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


DEC_MODES = ["central", "central + savings", "independent", "independent + savings", "shared, 500 tokens",
             "shared, 5,000 tokens"]
DEC_COL = ["#2a78d6", "#86b6ef", "#eb6834", "#f4a582", "#1baf7a", "#008300"]


def fig_decentral(res):
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.3), sharey=False)
    panels = [(c, u) for c in ["L1", "L3"] for u in (1, 10)]
    for p, (cell, units) in enumerate(panels):
        ax = axes[p]
        groupings = ["random G=4", "region G=4"]
        x = np.arange(len(groupings))
        w = 0.13
        for m_i, mode in enumerate(DEC_MODES):
            ys, los, his = [], [], []
            for g in groupings:
                gg = "none (G=1)" if mode.startswith("central") else g
                r = [r for r in res["decentral"] if r["cell"] == cell and r.get("units") == units
                     and r["chooser"] == "repeat" and r["grouping"] == gg and r["mode"] == mode][0]
                ys.append(pct(r["caught"]))
                los.append(pct(r["caught_lo"]))
                his.append(pct(r["caught_hi"]))
            ys, los, his = map(np.array, (ys, los, his))
            ax.bar(x + (m_i - 2.5) * w, ys, w * 0.9, color=DEC_COL[m_i], label=mode,
                   yerr=[np.clip(ys - los, 0, None), np.clip(his - ys, 0, None)], error_kw=dict(lw=0.7, capsize=1.2, ecolor="#555"))
        ax.set_xticks(x)
        ax.set_xticklabels(["4 random groups", "4 regions (zip)"])
        ax.set_ylim(0, 15 if units == 1 else 80)
        ax.set_ylabel("All failure objects caught (%)")
        ax.set_title(f"({'abcd'[p]}) {cell}, {units} unit{'s' if units > 1 else ''} a day", fontsize=8.5, loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.03), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(FIG / "fig3_decentralised.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def f(x):
    return "n/a" if x != x else f"{100 * x:.1f}"


def rng_str(r, m):
    if m + "_lo" in r and r[m + "_lo"] != r[m + "_hi"]:
        return f"{f(r[m])} ({f(r[m + '_lo'])}–{f(r[m + '_hi'])})"
    return f(r[m])


def tables(res) -> str:
    out = ["# Offline replay: full tables (generated by `experiments/bazaarbench/make_report.py`)", "",
           "Percentages; brackets give the 2.5th–97.5th percentile over repeated runs. "
           f"1 unit = {res['unit_tokens']:,.0f} tokens a day.", ""]
    for cell in ["L1", "L3"]:
        info = res["cells"][cell]
        out += [f"## {CELL_NAME[cell]} — `{info['cell_id']}`", "",
                f"{info['n_objects']} failure objects; {info['n_past_s1']} past S1; {info['n_s4']} reached S4; "
                f"{info['n_s5']} reached S5. Checking all 20 with the full judge reads "
                f"{info['tokens_all_per_day_full']:,.0f} tokens a day.", "",
                "### Checks by count", "",
                "| check | chooser | k | caught | caught, past S1 | before S5 | before S4 | tokens/day |",
                "|---|---|---|---|---|---|---|---|"]
        for r in [r for r in res["count"] if r["cell"] == cell]:
            out.append(f"| {r['view']} | {r['chooser']} | {r['k']} | {rng_str(r, 'caught')} | "
                       f"{f(r['caught_past_s1'])} | {f(r['before_s5'])} | {f(r['before_s4'])} | "
                       f"{r['tokens_per_day']:,.0f} |")
        out += ["", "### Token budgets", "",
                "| units | check | chooser | caught | caught, past S1 | before S5 | checks/day | tokens/day |",
                "|---|---|---|---|---|---|---|---|"]
        for r in [r for r in res["tokens"] if r["cell"] == cell]:
            out.append(f"| {r['units']} | {r['view']} | {r['chooser']} | {rng_str(r, 'caught')} | "
                       f"{f(r['caught_past_s1'])} | {f(r['before_s5'])} | {r['checks_per_day']:.1f} | "
                       f"{r['tokens_per_day']:,.0f} |")
        out += ["", "### Decentralised overseers", "",
                "| budget | chooser | groups | mode | caught | caught, past S1 | before S5 | checks/day | tokens/day (incl. messages) |",
                "|---|---|---|---|---|---|---|---|---|"]
        for r in [r for r in res["decentral"] if r["cell"] == cell]:
            b = f"{int(r['units'])} unit(s)" if r.get("units") else f"k = {int(r['k'])}"
            out.append(f"| {b} | {r['chooser']} | {r['grouping']} | {r['mode']} | {rng_str(r, 'caught')} | "
                       f"{f(r['caught_past_s1'])} | {f(r['before_s5'])} | {r['checks_per_day']:.1f} | "
                       f"{r['tokens_per_day']:,.0f} |")
        out.append("")
    return "\n".join(out)


def main() -> None:
    res = json.loads((NOTE / "results.json").read_text())
    FIG.mkdir(parents=True, exist_ok=True)
    fig_choosers(res)
    fig_checks_and_budgets(res)
    fig_decentral(res)
    (NOTE / "tables.md").write_text(tables(res))
    print("figures and tables written to", NOTE)


if __name__ == "__main__":
    main()
