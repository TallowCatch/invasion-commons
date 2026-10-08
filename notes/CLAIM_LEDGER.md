# Claim Ledger

Paper: `Scalable Oversight Under Capability Pressure in Multi-Agent Commons`

| Claim | Status | Evidence or Support | Action |
| --- | --- | --- | --- |
| Shared-resource systems can fail through repeated interaction. | Supported by literature. | Commons literature and sequential social dilemmas. | Keep. |
| Sequential commons are useful substrates for studying cooperation, depletion, and mixed incentives. | Supported by literature. | Leibo et al.; Melting Pot 2.0; SocialJax. | Keep. |
| Scalable oversight should measure capability gaps between actor and overseer. | Supported by literature. | Bowman et al.; Engels et al. | Keep. |
| The paper's central failure mode is local-pass/global-fail. | Supported by benchmark design and experiment. | Defined in formulation; observed in Stage A metrics and extracted trace. | Keep. |
| Local checks can pass while aggregate commons state becomes unsafe. | Supported by our experiment. | Main local-pass/global-fail metric, extracted trace, and full threshold sweep. | Keep, with benchmark scope. |
| Actor capability is measured through mutation/search entrant generation. | Supported only as benchmark design. | Search candidates and evaluation horizon define the ladder. | Keep as operational proxy; do not call it full AI capability. |
| Stronger actor settings generate stronger entrants. | Supported by our experiment. | Entrant payoff and requested/aggressive harvest increase. | Keep with validation table. |
| Stronger actor settings always increase unsafe rate. | Should be removed. | Unsafe rate decreases in the validation table. | Do not claim. Explain non-monotonicity. |
| Overseer capability is measured through detection recall, delay, target capacity, and cost. | Supported only as benchmark design. | Internal experimental design inspired by scalable oversight. | Keep as operational definition; needs more literature if expanded. |
| Hybrid oversight ranks first in 13 of 18 decision cells. | Supported by our experiment. | Stage A ranking table. | Keep. |
| Hybrid oversight is universally best. | Should be removed. | Moderate-coupling setting has local and global winners. | Do not claim. |
| Architectures with a global signal preserve patch health better than no/local oversight in the main pilot. | Supported by our experiment. | Stage A condition means and capability-gap figure. | Keep, benchmark-scoped. |
| Local oversight is insufficient in this setting. | Supported by our experiment with nuance. | Local improves unsafe rate modestly but remains close to no oversight in several metrics. | Keep with mechanism explanation. |
| Global signal and hybrid oversight reduce welfare. | Supported by our experiment. | Welfare and burden columns in condition table. | Keep and explain mechanism. |
| High-coupling settings make hybrid more valuable. | Supported by our experiment. | Hybrid wins 9/9 high-coupling cells. | Keep, but avoid causal overclaim beyond benchmark parameters. |
| Winner maps prove architecture superiority. | Should be softened. | Winner maps hide metric trade-offs and uncertainty. | Present as ranking summary only. |
| The extracted episode trace proves local-pass/global-fail robustness. | Should be softened. | One trace is explanatory case evidence. | Keep as example; rely on aggregate metric for quantitative claim. |
| Episode-level threshold sensitivity supports the case trace. | Supported by our experiment. | Threshold table. | Keep as explanatory trace validation. |
| Full threshold robustness is established for local safety margin and global patch-health threshold variation. | Supported by our experiment. | Full 9,000-row threshold sweep over 25 threshold pairs, all actor settings, all overseer settings, both stress settings, and all four architectures. | Keep, but state that failed-patch threshold and environment parameters are fixed. |
| Qwen 2.5 and Llama 3.2 can generate valid structured Harvest strategies in this pipeline. | Supported by our experiment and model citations. | LLM bank validity table. | Keep. |
| Exploitative LLM-generated strategies reproduce oversight pressure. | Supported by our experiment with limits. | LLM bridge outcomes: failure under no/local oversight and protection under global/hybrid. | Keep as pilot claim. |
| The LLM bridge proves broad LLM-agent behavior. | Should be removed. | Two small local models and offline structured strategies only. | Do not claim. |
| Live LLM agents are evaluated. | Should be removed. | Current bridge uses offline structured strategies. | State explicitly that live action selection is outside the pilot. |
| The benchmark is directly policy-prescriptive. | Should be removed. | No field calibration or external validation. | Frame as controlled benchmark. |
| The work contributes to scalable oversight by adding a multi-agent compositional failure setting. | Supported by literature and benchmark design. | Connects scalable oversight capability-gap framing to shared-resource multi-agent dynamics. | Keep, cautiously. |
| More model families are needed for a full LLM bridge. | Needs experiment. | Current bridge is limited to two small local models. | Add to limitations/TODO. |
