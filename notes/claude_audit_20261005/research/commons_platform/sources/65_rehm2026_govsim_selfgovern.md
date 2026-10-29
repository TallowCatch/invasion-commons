# Can Agents Adapt Their Institutions And Sacrifice Themselves During Collapse? Executable Self-Governance in a GovSim Commons (GovSim-SelfGovern)

- **Authors:** Gregory B. Rehm (Meta)
- **Year:** 2026
- **Venue / identifier:** arXiv:2609.22600 (v2, 7 Oct 2026), cs.MA
- **URL:** https://arxiv.org/abs/2609.22600
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full HTML text read)

## Verbatim quotes

1. Resource model (inherits GovSim):
   > "Five agents share a renewable fishery for at most T=12 rounds."
   Eq. 1: S_{t+1} = min(C_{t+1}, 2 max[0, S_t − Σ_i r_it]).
   > "Thus, when the lake is full at C_{t} , aggregate harvest C_{t}/2 leaves enough stock to regenerate to capacity. Each agent requires a seven-unit effective catch per round."
2. Settings chosen to make three feasibility regimes, by arithmetic:
   > "G1 (Standard): C_{t}=100 . Sustainable yield is 50 against aggregate demand of 35."
   > "G3 (Fatal): capacity declines to 56. Sustainable yield reaches 28, below five-agent demand."
   > "G3 is deliberately infeasible through harvest alone."
3. Metric change and its reason:
   > "We replace GovSim's individual over-usage with a group-level analogue,"
   > "In our runs individual over-usage never exceeds 1.08% of harvest events, so it does not discriminate between arms."
4. Headline result (abstract):
   > "When the fishery is abundant, self-rule helps, raising the chance that the community survives without anyone starving from 52.5% to 75.0%."
5. Release:
   > "The release includes the game configurations, matched seed list, outcome scripts, per-run categorical audit, and diagnostic-assay manifests."

## Relevance
The newest GovSim derivative keeps GovSim's doubling-to-capacity rule untouched and sets new parameters (subsistence need 7, capacity floors 65 and 56) by simple supply-versus-demand arithmetic so that each condition is feasible, tight, or infeasible. It also shows that a metric must be checked for whether it separates conditions (GovSim's per-agent over-usage was near 0). Its description of GovSim's over-usage threshold appears to conflict with GovSim's code (see 61).
