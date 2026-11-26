"""Tables and figures for notes/socialjax/locality/ (tasks 1-4 of its protocol).

  .venv-socialjax/bin/python -m experiments.socialjax.analyze_locality --run results/locality \
      --figs notes/socialjax/locality
Reads measure.json, formula.json, gap.json and scale.json (whichever exist); prints markdown tables, writes
figures and a small locality_numbers.json next to them.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")  # fixed categorical order
TESTBEDS = ("harvest", "harvest_large", "rooms", "rooms_door", "rooms_large", "coins", "cleanup")
NAMES = {"harvest": "Harvest (7)", "harvest_large": "Harvest, large (14)", "rooms": "Rooms, closed (7)",
         "rooms_door": "Rooms, one door (7)", "rooms_large": "Rooms, large (14)", "coins": "Coins (2)",
         "cleanup": "Clean Up (7)"}
MARKERS = {1: "o", 10: "s", 50: "^", 150: "D"}


def f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan"), int(m.sum())
    rx = np.argsort(np.argsort(x[m])).astype(float)
    ry = np.argsort(np.argsort(y[m])).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), int(m.sum())


# ------------------------------------------------------------------------------------------------ task 1
def locality_tables(rows):
    """Per testbed and layout: chi pooled over populations; zeta over all violations; xi from the
    1 greedy population (the steady violator) and pooled."""
    out = []
    keyset = []
    for r in rows:
        k = (r["testbed"], r["layout"])
        if k not in keyset:
            keyset.append(k)
    for tb, lay in keyset:
        rs = [r for r in rows if r["testbed"] == tb and r["layout"] == lay]
        chi = float(np.mean([r["chi"] for r in rs]))
        nv = sum(r["violations"] for r in rs)
        zw = [(r["zeta"], r["violations"]) for r in rs if np.isfinite(r["zeta"])]
        zeta = sum(z * n for z, n in zw) / max(1, sum(n for _, n in zw)) if zw else float("nan")
        zcf = [r.get("zeta_cf") for r in rs if r.get("zeta_cf") is not None and np.isfinite(r.get("zeta_cf"))]
        rec = dict(testbed=tb, layout=lay, chi=chi, chi_min=min(r["chi"] for r in rs),
                   chi_max=max(r["chi"] for r in rs), violations=nv, zeta=zeta,
                   zeta_static=float(np.nanmean([r.get("zeta_static", np.nan) for r in rs])),
                   zeta_cf=float(np.mean(zcf)) if zcf else float("nan"), regions=rs[0]["regions"])
        for w in (1, 10, 50):
            vals = [(r[f"Lambda_{w}_direct"], r["violations"]) for r in rs if np.isfinite(r[f"Lambda_{w}_direct"])]
            lam = sum(v * n for v, n in vals) / max(1, sum(n for _, n in vals)) if vals else float("nan")
            rec[f"xi_{w}"] = 1 - lam
            rec[f"xi_{w}_formula"] = 1 - (1 - chi) ** (w - 1) * zeta if np.isfinite(zeta) else float("nan")
            sv = [(r[f"S_{w}"], r["violations"]) for r in rs if np.isfinite(r[f"S_{w}"])]
            rec[f"S_{w}"] = sum(v * n for v, n in sv) / max(1, sum(n for _, n in sv)) if sv else float("nan")
        out.append(rec)
    return out


def print_locality(tab, rows):
    print("\n### Locality per testbed and layout (scripted bots, pooled over the 4 populations)\n")
    print("| Testbed | Regions | χ (range over populations) | violations | ζ footprint | ζ static | ζ counterfactual |"
          " S_10 (formula (1−χ)^9) | ξ_loc w = 1 | w = 10 | w = 50 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in tab:
        print(f"| {NAMES[r['testbed']]} | {r['layout']} | {f(r['chi'], 4)} ({f(r['chi_min'], 4)}–{f(r['chi_max'], 4)}) "
              f"| {r['violations']} | {f(r['zeta'])} | {f(r['zeta_static'])} | {f(r['zeta_cf'])} | "
              f"{f(r['S_10'])} ({f((1 - r['chi']) ** 9)}) | {f(r['xi_1'])} | {f(r['xi_10'])} | {f(r['xi_50'])} |")
    print("\n### Per population (χ; S_10 measured vs (1 − χ)^9)\n")
    print("| Testbed | Regions | Population | χ | violations | ζ | S_10 | (1−χ)^9 | S_50 | (1−χ)^49 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {NAMES[r['testbed']]} | {r['layout']} | {r['population']} | {f(r['chi'], 4)} | {r['violations']} | "
              f"{f(r['zeta'])} | {f(r['S_10'])} | {f(r['S_10_formula'])} | {f(r['S_50'])} | {f(r['S_50_formula'])} |")


def fig_locality(tab, path):
    import matplotlib.pyplot as plt
    labels = [f"{NAMES[r['testbed']]} · {r['layout']}" for r in tab]
    cols = [COLOURS[TESTBEDS.index(r["testbed"])] for r in tab]
    y = np.arange(len(tab))[::-1]
    fig, ax = plt.subplots(1, 3, figsize=(13, 0.32 * len(tab) + 1.6), sharey=True)
    ax[0].barh(y, [r["chi"] for r in tab], color=cols, height=0.7)
    ax[0].set_xlabel("χ: share of agent-steps that cross into another region")
    ax[0].set_title("(a) How often agents cross", loc="left", fontsize=10)
    ax[1].barh(y + 0.18, [r["zeta"] for r in tab], color=cols, height=0.34, label="footprint")
    ax[1].barh(y - 0.18, [r["zeta_cf"] if np.isfinite(r["zeta_cf"]) else 0 for r in tab], color=cols, alpha=0.45,
               height=0.34, label="counterfactual regrowth (Harvest maps)")
    ax[1].set_xlim(0, 1.02)
    ax[1].set_xlabel("ζ: share of a violation's harm inside its region")
    ax[1].set_title("(b) Where the harm lands", loc="left", fontsize=10)
    ax[1].legend(fontsize=7, loc="lower left", frameon=False)
    for j, (w, a) in enumerate(((1, 1.0), (10, 0.65), (50, 0.35))):
        ax[2].barh(y + 0.27 - 0.27 * j, [r[f"xi_{w}"] for r in tab], color=cols, alpha=a, height=0.25,
                   label=f"w = {w}")
    ax[2].set_xlim(0, 1.02)
    ax[2].set_xlabel("ξ_loc = 1 − Λ_w (0 = splits cleanly into regions)")
    ax[2].set_title("(c) Non-locality for a w-step look", loc="left", fontsize=10)
    ax[2].legend(fontsize=7, loc="lower right", frameon=False)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=8)
    for a in ax:
        a.grid(axis="x", alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ task 2
def print_formula(d):
    import collections
    a = d["abstract"]
    print("\n### Task 2, abstract model: largest |simulation − formula| over the grid\n")
    print("| Crossing pattern | cells | simple P_0 formula: worst error | Markov P_0: worst error |")
    print("|---|---|---|---|")
    by = collections.defaultdict(list)
    for r in a:
        by[r["pattern"]].append(r)
    for p, rs in by.items():
        e0 = max(rs, key=lambda r: abs(r["sim"] - r["p_0"]))
        em = max(abs(r["sim"] - r["p_markov"]) for r in rs)
        print(f"| {p} | {len(rs)} | {f(e0['sim'] - e0['p_0'])} (q = {f(e0['q'])}, χ = {e0['chi']}, w = {e0['w']}, "
              f"α = {e0['alpha']}) | {f(em, 4)} |")
    if "code" in d:
        print("\n### Task 2, real overseer code (corridor, N = 8, K = 2): catch share\n")
        print("| R | w | χ | simulated | P_C | simple P_0 | Markov P_0 (q = K/N) | Markov P_0 (measured q) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in d["code"]:
            if r["R"] == 1 or r["w"] not in (1, 5, 10, 50):
                continue
            print(f"| {r['R']} | {r['w']} | {r['chi']} | {f(r['sim'])} | {f(r['p_c'])} | {f(r['p_0'])} | "
                  f"{f(r['p_markov'])} | {f(r['p_markov_qeff'])} |")
        c = [r for r in d["code"] if r["R"] > 1]
        for R in (2, 4):
            rs = [r for r in c if r["R"] == R]
            print(f"\nR = {R}: worst |sim − Markov(measured q)| = {max(abs(r['sim'] - r['p_markov_qeff']) for r in rs):.3f}; "
                  f"worst |sim − simple P_0| = {max(abs(r['sim'] - r['p_0']) for r in rs):.3f}")


def fig_formula(d, path):
    import matplotlib.pyplot as plt
    a = d["abstract"]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    q = 1 / 7
    for k, (pattern, title) in enumerate((("drift", "(a) Never comes back (the formula's assumption)"),
                                          ("two", "(b) Two regions: comes back"))):
        for j, chi in enumerate((0.05, 0.2, 0.5)):
            rs = sorted([r for r in a if r["pattern"] == pattern and np.isclose(r["q"], q) and r["chi"] == chi
                         and r["alpha"] == 0], key=lambda r: r["w"])
            w = [r["w"] for r in rs]
            ax[k].plot(w, [r["p_0"] for r in rs], color=COLOURS[j], lw=1.5, ls="--")
            ax[k].plot(w, [r["p_markov"] for r in rs], color=COLOURS[j], lw=1.5)
            ax[k].scatter(w, [r["sim"] for r in rs], color=COLOURS[j], s=22, zorder=3, label=f"χ = {chi}")
        rs = sorted([r for r in a if r["pattern"] == pattern and np.isclose(r["q"], q) and r["chi"] == 0
                     and r["alpha"] == 0], key=lambda r: r["w"])
        ax[k].plot([r["w"] for r in rs], [r["p_c"] for r in rs], color="#52514e", lw=1.2, label="central P_C")
        ax[k].set_xscale("log")
        ax[k].set_xlabel("w: steps of record a check reads")
        ax[k].set_ylabel("chance a violation is caught")
        ax[k].set_title(title, loc="left", fontsize=10)
        ax[k].set_ylim(0, 1.03)
    ax[0].legend(fontsize=7, frameon=False, title="dots: simulation\ndashed: simple P_0\nsolid: exact (Markov)",
                 title_fontsize=7)
    if "code" in d:
        c = [r for r in d["code"] if r["R"] > 1]
        for j, R in enumerate((2, 4)):
            rs = [r for r in c if r["R"] == R]
            ax[2].scatter([r["p_0"] for r in rs], [r["sim"] for r in rs], marker="x", color=COLOURS[j], s=22,
                          label=f"R = {R}: simple P_0")
            ax[2].scatter([r["p_markov_qeff"] for r in rs], [r["sim"] for r in rs], color=COLOURS[j], s=18,
                          label=f"R = {R}: Markov, measured q")
        ax[2].plot([0, 1], [0, 1], color="#52514e", lw=0.8)
        ax[2].set_xlabel("predicted catch chance")
        ax[2].set_ylabel("simulated (real overseer code)")
        ax[2].set_title("(c) The real overseer code, corridor game", loc="left", fontsize=10)
        ax[2].legend(fontsize=7, frameon=False)
    for x in ax:
        x.grid(alpha=0.25)
        x.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ task 3
def gap_points(gap_rows, loc_rows):
    """One point per (testbed, layout, w, population, records): Delta catch share and Delta eta, with xi_loc."""
    import collections
    groups = collections.defaultdict(list)
    for r in gap_rows:
        groups[(r["testbed"], r["condition"], r["population"])].append(r)

    def pooled(rs):
        v = sum(r["violations"] for r in rs)
        return sum(r["detected"] for r in rs) / v if v else float("nan"), float(np.mean([r["welfare"] for r in rs]))

    def boot(rs_c, rs_d, n=400):
        seeds = sorted({r["seed"] for r in rs_c})
        cm = {r["seed"]: r for r in rs_c}
        dm = {r["seed"]: r for r in rs_d}
        rng = np.random.default_rng(0)
        out = []
        for _ in range(n):
            s = rng.choice(seeds, len(seeds))
            vc = sum(cm[x]["violations"] for x in s)
            vd = sum(dm[x]["violations"] for x in s)
            if vc and vd:
                out.append(sum(cm[x]["detected"] for x in s) / vc - sum(dm[x]["detected"] for x in s) / vd)
        return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (np.nan, np.nan)

    loc = {(r["testbed"], r["layout"], r["population"]): r for r in loc_rows}
    pts = []
    for (tb, cond, pop), rs in groups.items():
        parts = cond.split(" ")
        w = int(parts[0][2:])
        if parts[1] in ("none", "all", "central"):
            continue
        rec = parts[-1]
        layout = " ".join(parts[1:-1])
        c = groups[(tb, f"w={w} central", pop)]
        none = groups[(tb, f"w={w} none", pop)]
        allc = groups[(tb, f"w={w} all", pop)]
        sc, wc = pooled(c)
        sd, wd = pooled(rs)
        _, w0 = pooled(none)
        _, w1 = pooled(allc)
        eta = (lambda W: (W - w0) / (w1 - w0) if abs(w1 - w0) > 1e-9 else float("nan"))
        lr = loc.get((tb, layout, pop))
        if lr is None:
            xi = s_ev = float("nan")
        elif w in (1, 10, 50):
            xi = lr[f"xi_{w}_direct"]
            s_ev = lr[f"S_{w}"]
        else:
            z = lr["zeta"] if np.isfinite(lr["zeta"]) else 1.0
            s_ev = (1 - lr["chi"]) ** (w - 1)
            xi = 1 - s_ev * z
        lo, hi = boot(c, rs)
        q = rs[0]["config"]["k"] / len(rs[0]["taken_per_agent"])
        chi = lr["chi"] if lr else np.nan
        from overseer.socialjax.locality import p_central, p_independent
        p_gap = p_central(q, w) - p_independent(q, chi, w) if np.isfinite(chi) else np.nan
        pts.append(dict(formula_gap=p_gap, q=q,testbed=tb, layout=layout, w=w, population=pop, records=rec, central=sc, dec=sd,
                        delta=sc - sd, delta_lo=lo, delta_hi=hi, eta_central=eta(wc), eta_dec=eta(wd),
                        delta_eta=eta(wc) - eta(wd), xi=xi, evidence_lost=1 - s_ev if np.isfinite(s_ev) else np.nan,
                        chi=lr["chi"] if lr else np.nan, zeta=lr["zeta"] if lr else np.nan,
                        violations_central=sum(r["violations"] for r in c),
                        violations_dec=sum(r["violations"] for r in rs),
                        unused=float(np.mean([r.get("checks", 0) for r in rs]))))
    return pts


def print_gap(pts):
    print("\n### Task 3: decentralisation gap Δ_dec (share of violations caught: central − regional)\n")
    print("| Testbed | Regions | w | Population | records | central | regional | Δ_dec (95% CI) | Δη | ξ_loc |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for p in sorted(pts, key=lambda p: (TESTBEDS.index(p["testbed"]), p["layout"], p["population"], p["w"],
                                        p["records"])):
        print(f"| {NAMES[p['testbed']]} | {p['layout']} | {p['w']} | {p['population']} | {p['records']} | "
              f"{f(p['central'])} | {f(p['dec'])} | {f(p['delta'])} ({f(p['delta_lo'])}, {f(p['delta_hi'])}) | "
              f"{f(p['delta_eta'])} | {f(p['xi'])} |")
    print("\n### Does ξ_loc order the gap? (Spearman ρ, points with ≥ 20 violations under both overseers)\n")
    print("| Records | Populations | predictor | ρ | points |")
    print("|---|---|---|---|---|")
    for rec in ("local", "handover"):
        for popset, keep in (("1 greedy, all sneaky", lambda p: "migrator" not in p["population"]),
                             ("all migrator", lambda p: "migrator" in p["population"])):
            sel = [p for p in pts if p["records"] == rec and keep(p) and p["violations_central"] >= 20
                   and p["violations_dec"] >= 20]
            for pred in ("xi", "evidence_lost"):
                rho, n = spearman([p[pred] for p in sel], [p["delta"] for p in sel])
                name = "ξ_loc = 1 − Λ_w" if pred == "xi" else "1 − S_w (evidence leaves, no ζ)"
                print(f"| {rec} | {popset} | {name} | {f(rho, 2)} | {n} |")
    comp = evidence_component(pts)
    print("\n### The part of the gap that comes from records staying local (regional with hand-over − regional "
          "with local records), against what predicts it\n")
    print("| Populations | predictor | ρ | points |")
    print("|---|---|---|---|")
    for popset, keep in (("1 greedy, all sneaky", lambda p: "migrator" not in p["population"]),
                         ("all migrator", lambda p: "migrator" in p["population"])):
        sel = [p for p in comp if keep(p)]
        for pred, name in (("xi", "ξ_loc = 1 − Λ_w"), ("evidence_lost", "1 − S_w"),
                           ("formula_gap", "P_C − P_0 (formula, q = K/N, measured χ)")):
            rho, n = spearman([p[pred] for p in sel], [p["component"] for p in sel])
            print(f"| {popset} | {name} | {f(rho, 2)} | {n} |")
    print("\n| Testbed | Regions | w | Population | χ | ξ_loc | P_C − P_0 | lost by local records |")
    print("|---|---|---|---|---|---|---|---|")
    for p in sorted(comp, key=lambda p: (TESTBEDS.index(p["testbed"]), p["layout"], p["population"], p["w"])):
        if "migrator" in p["population"]:
            continue
        print(f"| {NAMES[p['testbed']]} | {p['layout']} | {p['w']} | {p['population']} | {f(p['chi'], 4)} | "
              f"{f(p['xi'])} | {f(p['formula_gap'])} | {f(p['component'])} |")


def evidence_component(pts):
    """Per (testbed, layout, w, population): catch share with hand-over minus with local records."""
    idx = {(p["testbed"], p["layout"], p["w"], p["population"], p["records"]): p for p in pts}
    out = []
    for p in pts:
        if p["records"] != "local":
            continue
        h = idx.get((p["testbed"], p["layout"], p["w"], p["population"], "handover"))
        if h is None or p["violations_dec"] < 20 or h["violations_dec"] < 20:
            continue
        out.append(dict(p, component=h["dec"] - p["dec"]))
    return out


def fig_gap(pts, path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
    ax = [axs[0, 0], axs[0, 1], axs[1, 1]]
    panels = (("local", lambda p: "migrator" not in p["population"], "(a) Records stay local (κ = 0)"),
              ("handover", lambda p: "migrator" not in p["population"], "(b) Records travel (existing code)"),
              ("local", lambda p: "migrator" in p["population"], "(d) Migrators, records stay local"))
    comp = [p for p in evidence_component(pts) if "migrator" not in p["population"]]
    c3 = axs[1, 0]
    for p in comp:
        c = COLOURS[TESTBEDS.index(p["testbed"])]
        c3.plot(p["formula_gap"], p["component"], MARKERS[p["w"]], color=c, ms=5,
                mfc=c if "greedy" in p["population"] else "white")
    c3.plot([0, 0.2], [0, 0.2], color="#52514e", lw=0.8)
    rho, n = spearman([p["formula_gap"] for p in comp], [p["component"] for p in comp])
    c3.set_title(f"(c) Catches lost because records stay local\nSpearman ρ = {f(rho, 2)} over {n} points",
                 loc="left", fontsize=10)
    c3.set_xlabel("predicted P_C − P_0 (q = K/N, measured χ)")
    c3.set_ylabel("share caught: hand-over − local records")
    c3.grid(alpha=0.25)
    c3.spines[["top", "right"]].set_visible(False)
    for k, (rec, keep, title) in enumerate(panels):
        sel = [p for p in pts if p["records"] == rec and keep(p) and p["violations_central"] >= 20
               and p["violations_dec"] >= 20]
        for p in sel:
            c = COLOURS[TESTBEDS.index(p["testbed"])]
            ax[k].errorbar(p["xi"], p["delta"], yerr=[[p["delta"] - p["delta_lo"]], [p["delta_hi"] - p["delta"]]],
                           fmt=MARKERS[p["w"]], color=c, ms=5, elinewidth=0.6, alpha=0.85,
                           mfc=c if "greedy" in p["population"] or "migrator" in p["population"] else "white")
        rho, n = spearman([p["xi"] for p in sel], [p["delta"] for p in sel])
        ax[k].set_title(f"{title}\nSpearman ρ = {f(rho, 2)} over {n} points", loc="left", fontsize=10)
        ax[k].axhline(0, color="#52514e", lw=0.8)
        ax[k].set_xlabel("ξ_loc measured beforehand (task 1)")
        ax[k].set_xlim(-0.03, 1.03)
        ax[k].grid(alpha=0.25)
        ax[k].spines[["top", "right"]].set_visible(False)
    for a in ax:
        a.set_ylabel("Δ_dec: share caught, central − regional")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOURS[TESTBEDS.index(t)], marker="o", ls="", label=NAMES[t]) for t in TESTBEDS]
    h += [Line2D([], [], color="#52514e", marker=m, ls="", label=f"w = {w}") for w, m in MARKERS.items()]
    h += [Line2D([], [], color="#52514e", marker="o", mfc="white", ls="", label="hollow: all sneaky")]
    fig.legend(handles=h, loc="lower center", ncol=6, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ task 4
def scale_table(d):
    import collections
    rows = d["rows"]
    central = {(r["n"], r["chooser"], r["seed"]): r for r in rows if r["M"] == 1}
    g = collections.defaultdict(list)
    for r in rows:
        if r["M"] == 1:
            continue
        c = central[(r["n"], r["chooser"], r["seed"])]
        g[(r["n"], r["chooser"], r["split"], r["M"], r["pooled"])].append(
            (c["caught_share"] - r["caught_share"], r["caught_share"], c["caught_share"], r["comm_share"]))
    out = []
    for k, v in sorted(g.items()):
        dlt = np.array([x[0] for x in v])
        out.append(dict(n=k[0], chooser=k[1], split=k[2], M=k[3], pooled=k[4], delta=float(dlt.mean()),
                        se=float(dlt.std(ddof=1) / np.sqrt(len(dlt))), dec=float(np.mean([x[1] for x in v])),
                        central=float(np.mean([x[2] for x in v])), comm=v[0][3], seeds=len(v)))
    return out


def print_scale(tab, d):
    print("\n### Task 4: Willis et al.'s commons, K = N/8 audits per round, 50 games per N\n")
    print("| N | Chooser | Groups formed | M | Budget | central | groups | Δ_dec (± 1.96 SE) | numbers between overseers per round, % of central's N records |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in tab:
        print(f"| {r['n']} | {r['chooser']} | {r['split']} | {r['M']} | {'pooled' if r['pooled'] else 'fixed K/M'} | "
              f"{f(r['central'])} | {f(r['dec'])} | {r['delta']:+.3f} (± {1.96 * r['se']:.3f}) | "
              f"{100 * r['comm']:.1f}% |")
    print("\n### Version (b): what deterring one group is worth (every other group complying; always-D vs always-C)\n")
    print("| N | M | group size | to the whole commons | to that group itself | to the other groups |")
    print("|---|---|---|---|---|---|")
    for r in d["own_group"]:
        print(f"| {r['n']} | {r['M']} | {r['group_size']} | {r['total_gain']:+.1f} | {r['own_gain']:+.1f} | "
              f"{r['harm_to_others']:+.1f} |")
    gm = d["games"]
    for n in sorted({g["n"] for g in gm}):
        gs = [g for g in gm if g["n"] == n]
        print(f"\nN = {n}: defection rate {np.mean([g['defect_rate'] for g in gs]):.3f}, final stock "
              f"{np.mean([g['final_stock'] for g in gs]):.1f} of {4 * n}")


def fig_scale(tab, d, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw=dict(width_ratios=(1.3, 1)))
    sel = [r for r in tab if r["split"] == "random" and not r["pooled"]]
    sel += [r for r in tab if r["split"] == "by attitude" and not r["pooled"]]
    labels, vals, errs, cols = [], [], [], []
    for r in sel:
        labels.append(f"N={r['n']}, M={r['M']}, {r['chooser']} audits, "
                      f"{'random groups' if r['split'] == 'random' else 'groups by attitude'}")
        vals.append(r["delta"])
        errs.append(1.96 * r["se"])
        cols.append(COLOURS[0] if r["chooser"] == "random" else COLOURS[1])
    y = np.arange(len(sel))[::-1]
    ax[0].barh(y, vals, xerr=errs, color=cols, height=0.7, error_kw=dict(lw=0.7))
    ax[0].axvline(0, color="#52514e", lw=0.8)
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(labels, fontsize=7)
    ax[0].set_xlabel("Δ_dec: share of defections caught, central − M independent groups")
    ax[0].set_title("(a) Fixed groups (χ = 0) lose nothing at equal budget", loc="left", fontsize=10)
    og = [r for r in d["own_group"] if r["M"] > 1]
    x = np.arange(len(og))
    ax[1].bar(x - 0.2, [r["total_gain"] / r["n"] for r in og], width=0.4, color=COLOURS[2], label="whole commons")
    ax[1].bar(x + 0.2, [r["own_gain"] / r["n"] for r in og], width=0.4, color=COLOURS[3], label="that group only")
    ax[1].axhline(0, color="#52514e", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"N={r['n']}\nM={r['M']}" for r in og], fontsize=7)
    ax[1].set_ylabel("welfare gained per player of the commons")
    ax[1].set_title("(b) Deterring one group: who gains", loc="left", fontsize=10)
    ax[1].legend(fontsize=7, frameon=False)
    for a in ax:
        a.grid(alpha=0.25)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="results/locality")
    ap.add_argument("--figs", default="notes/socialjax/locality")
    a = ap.parse_args()
    run, figs = Path(a.run), Path(a.figs)
    figs.mkdir(parents=True, exist_ok=True)
    numbers = {}
    loc_rows = []
    if (run / "measure.json").exists():
        loc_rows = json.loads((run / "measure.json").read_text())["rows"]
        tab = locality_tables(loc_rows)
        print_locality(tab, loc_rows)
        fig_locality(tab, figs / "fig_locality.png")
        numbers["locality"] = tab
    if (run / "formula.json").exists():
        d = json.loads((run / "formula.json").read_text())
        print_formula(d)
        fig_formula(d, figs / "fig_formula.png")
    if (run / "gap.json").exists() and loc_rows:
        pts = gap_points(json.loads((run / "gap.json").read_text())["rows"], loc_rows)
        print_gap(pts)
        fig_gap(pts, figs / "fig_gap.png")
        numbers["gap"] = pts
    if (run / "scale.json").exists():
        d = json.loads((run / "scale.json").read_text())
        tab = scale_table(d)
        print_scale(tab, d)
        fig_scale(tab, d, figs / "fig_scale.png")
        numbers["scale"] = tab
        numbers["own_group"] = d["own_group"]
    (figs / "locality_numbers.json").write_text(json.dumps(numbers, indent=1, default=float))


if __name__ == "__main__":
    main()
