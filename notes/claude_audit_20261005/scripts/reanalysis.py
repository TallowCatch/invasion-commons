"""[Claude audit 2026-10-05] Post-hoc exploratory re-analysis of budgeted_reviewer_confirmation_v1.
Read-only on saved data; no new episodes are simulated."""
import gzip, json, collections
import numpy as np
from scipy.stats import norm
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.oversight_protocol import harvest_nominal_next, harvest_safe

RUN = "results/runs/budgeted_reviewer_confirmation_v1"
fc = json.load(gzip.open(f"{RUN}/frozen_cases.json.gz"))
lab = {l["case_id"]: l for l in json.load(gzip.open(f"{RUN}/labels.json.gz"))}
out = {}

# ---------------- Harvest: margin sensitivity at full inspection -------------
H = [c for c in fc if c["game"] == "harvest" and c["pre_global_safe"] == 1]
cfg = HarvestCommonsConfig(**{k: v for k, v in H[0]["config"].items() if k != "seed"})
sigma, n = cfg.weather_noise_std, cfg.n_agents
margins = {
    "current_bonferroni_per_patch": sigma * norm.isf(0.05 / n),
    "mean_only_5pct": sigma / np.sqrt(n) * norm.isf(0.05),
    "zero": 0.0,
}
out["harvest_margins"] = margins
rng = np.random.default_rng(7)
res = collections.defaultdict(collections.Counter)
precise = []
for c in H:
    L = lab[c["case_id"]]["reference_label"]
    h = np.array(c["state"], float); p = np.array(c["proposals"], float)
    nom = harvest_nominal_next(cfg, h, p)
    for name, m in margins.items():
        ok = harvest_safe(cfg, np.clip(nom - m, 0, cfg.patch_max))
        res[name][(L, "approve" if ok else "restrict")] += 1
    # precise risk with 4000 draws (exploratory, independent seed)
    w = rng.normal(0, sigma, size=(4000, n))
    fut = np.clip(nom + w, 0, cfg.patch_max)
    safe = (fut.mean(1) >= cfg.global_min_mean_patch_health) & ((fut < cfg.local_patch_failure_threshold).mean(1) < cfg.failure_fraction_threshold)
    precise.append((L, 1 - safe.mean(), float(nom.mean())))
out["harvest_joint_k6_by_margin"] = {k: {f"{a}|{b}": v for (a, b), v in d.items()} for k, d in res.items()}
P = np.array([r[1] for r in precise]); Ls = np.array([r[0] for r in precise])
out["harvest_precise_risk"] = {
    L: dict(n=int((Ls == L).sum()), median=float(np.median(P[Ls == L])),
            frac_below_0p05=float((P[Ls == L] < 0.05).mean()),
            frac_above_0p5=float((P[Ls == L] > 0.5).mean()),
            q10=float(np.quantile(P[Ls == L], .1)), q90=float(np.quantile(P[Ls == L], .9)))
    for L in ["safe", "risky", "unresolved"]}
# how many cases were from already-unsafe states (excluded)
HA = [c for c in fc if c["game"] == "harvest"]
out["harvest_cases_total_vs_initially_safe"] = [len(HA), len(H)]

# ---------------- Fishery -----------------------------------------------------
F = [c for c in fc if c["game"] == "fishery"]
FS = [c for c in F if c["pre_global_safe"] == 1]
out["fishery_cases_total_vs_initially_safe"] = [len(F), len(FS)]
fc0 = F[0]["config"]; r, K, M, thr = fc0["regen_rate"], fc0["stock_max"], fc0["max_harvest_per_agent"], fc0["collapse_threshold"]
def proj(S, d):
    y = max(0.0, S - d); return min(K, y + max(0, r * y * (1 - y / K)))
out["fishery_params"] = dict(r=r, K=K, max_harvest=M, collapse_threshold=thr, n=fc0["n_agents"],
                             stock_init=fc0["stock_init"], MSY_total_per_step=r * K / 4, MSY_residual_stock=K / 2)
risky = [c for c in FS if lab[c["case_id"]]["reference_label"] == "risky"]
by_ctx = collections.defaultdict(list)
for c in F: by_ctx[c["context"]].append(c)
last_step = {k: max(x["step"] for x in v) for k, v in by_ctx.items()}
out["fishery_risky"] = dict(
    n=len(risky), contexts=len({c["context"] for c in risky}),
    steps=collections.Counter(c["step"] for c in risky).most_common(8),
    stock_quantiles=[float(np.quantile([c["state"]["stock"] for c in risky], q)) for q in (0, .25, .5, .75, 1)],
    steps_before_last_case=collections.Counter(last_step[c["context"]] - c["step"] for c in risky).most_common(5))
safe = [c for c in FS if lab[c["case_id"]]["reference_label"] == "safe"]
nxt = np.array([proj(c["state"]["stock"], M * sum(c["proposals"])) for c in safe])
out["fishery_safe_requests_next_stock"] = dict(
    n=len(safe), below_50_MSY_level=int((nxt < 50).sum()), below_30=int((nxt < 30).sum()),
    below_20=int((nxt < 20).sum()), quantiles=[float(np.quantile(nxt, q)) for q in (0, .1, .5, .9, 1)])
out["fishery_frozen_case_steps"] = dict(mean_cases_per_context=len(F) / 64,
    stock_quantiles=[float(np.quantile([c["state"]["stock"] for c in F], q)) for q in (0, .25, .5, .75, 1)])

# closed-loop states visited vs frozen states (Fishery, budget 6)
def closed_states(mode, k, game="fishery", key="stock"):
    vals = []
    for ctx in range(64):
        for w in ([0] if game == "fishery" else [0, 1]):
            pg, rg = ("mix4", "deterministic") if game == "fishery" else ("mix2", "slow_regen")
            b = json.load(gzip.open(f"{RUN}/blocks/{game}__{pg}__{rg}__{ctx}__{w}__{mode}__{k}.json.gz"))
            for t in b["trace"]:
                vals.append(t.get("mean_patch_health_before", None))
    return vals
b = json.load(gzip.open(f"{RUN}/blocks/fishery__mix4__deterministic__0__0__joint__6.json.gz"))
out["trace_keys"] = sorted(b["trace"][0].keys())
for mode, k in [("joint", 6), ("local_bounded", 6), ("joint", 0)]:
    s, scales, real = [], [], []
    for ctx in range(64):
        bb = json.load(gzip.open(f"{RUN}/blocks/fishery__mix4__deterministic__{ctx}__0__{mode}__{k}.json.gz"))
        for t in bb["trace"]:
            s.append(t["mean_patch_health_after"]); real.append(t["realized_harvest"])
            req = np.array(json.loads(t["requested_fracs_json"])); al = np.array(json.loads(t["allowed_fracs_json"]))
            scales.append(float(al.sum() / req.sum()) if req.sum() > 0 else 1.0)
    out[f"fishery_closed_{mode}_{k}"] = dict(stock_after_quantiles=[float(np.quantile(s, q)) for q in (.1, .5, .9)],
        mean_scale=float(np.mean(scales)), frac_steps_restricted=float(np.mean(np.array(scales) < 1 - 1e-9)),
        mean_realized=float(np.mean(real)))
import os
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "reanalysis.json"), "w"), indent=1, default=str)
print(json.dumps(out, indent=1, default=str))
