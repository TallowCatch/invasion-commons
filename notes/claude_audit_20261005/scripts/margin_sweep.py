import gzip, json, collections, numpy as np
from scipy.stats import norm as _norm
import fishery_sim.budgeted_oversight as bo
from fishery_sim.harvest import HarvestCommonsConfig
from fishery_sim.oversight_protocol import MonitorSettings
RUN="results/runs/budgeted_reviewer_confirmation_v1"
fc=[c for c in json.load(gzip.open(f"{RUN}/frozen_cases.json.gz")) if c["game"]=="harvest" and c["pre_global_safe"]==1]
lab={l["case_id"]:l["reference_label"] for l in json.load(gzip.open(f"{RUN}/labels.json.gz"))}
class Shim:
    def __init__(s,z): s.z=z
    def isf(s,x): return s.z
s=MonitorSettings()
for name,z in [("current",_norm.isf(0.05/6)),("mean_only",_norm.isf(0.05)/np.sqrt(6))]:
    bo.norm=Shim(z)
    for k in (0,3,6):
        for mode in ("joint","local_bounded","local_optimistic"):
            c_=collections.Counter(); kept=[]
            for c in fc:
                cfg=HarvestCommonsConfig(**c["config"])
                vis=bo.mask_requests(c["proposals"],k,c["case_id"])
                d=bo.decide_budgeted_harvest(cfg,c["state"],vis,mode,s)
                L=lab[c["case_id"]]
                c_[(L,d.verdict)]+=1
                if L=="safe": kept.append(d.scale)
            print(name,k,mode,"safe_restricted=%d/%d"%(c_[("safe","reject")]+c_[("safe","abstain")],sum(v for (a,b),v in c_.items() if a=="safe")),
                  "risky_approved=%d/%d"%(c_[("risky","approve")],sum(v for (a,b),v in c_.items() if a=="risky")),
                  "safe_retained=%.3f"%np.mean(kept))
