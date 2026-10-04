# Paper v5 Publication Audit

Paper audited: `paper/paper_v5_scalable_oversight_commons/main.tex`

Current title: `Scalable Oversight Under Capability Pressure in Multi-Agent Commons`

## Overall Assessment

The paper now has a coherent spine: agents act in a shared-resource system, actor capability increases through stronger strategy generation, overseer capability is limited through detection/capacity/delay/cost, and oversight architectures are compared through safety, resource health, welfare, and burden. This is a credible benchmark direction.

The main remaining weakness is not writing style. The largest publication risk is evidence depth. The main pilot is strong enough to motivate the framework, but several claims still rely on internally defined design choices: the capability ladder, safety thresholds, stress settings, and the LLM bridge. These are defensible if framed as benchmark design choices. They are weaker if framed as general scalable-oversight conclusions.

## 1. Full Paper Audit

| Section | Problem | Why it matters | Recommended fix |
| --- | --- | --- | --- |
| Abstract | Strong but dense. It includes benchmark design, results, and LLM bridge in one long paragraph. | A reviewer may miss the central contribution because too many elements arrive at once. | Keep as is for now, but for submission split mentally into: problem, method, result, scope. If page style allows, shorten the LLM sentence. |
| Introduction | The motivation is clear. The jump from commons to AI shared infrastructure is plausible but lightly cited. | Readers may ask whether shared compute/information channels are examples or a formal target domain. | Add one citation to multi-agent AI evaluation or AI-agent population work if this framing is expanded. |
| Introduction | “Global signal” is now better than “centralized control,” but the implementation still functions as a system-level intervention. | Edward’s concern is valid: deployed agent populations may not have a literal central controller. | Continue describing it as a benchmark abstraction. Avoid claiming it maps directly to real deployment governance. |
| Related Work | Good coverage of commons, SSDs, LLM-agent populations, and scalable oversight. Still short on mechanism-design / institutional-design literature. | The paper compares oversight architectures, so a reviewer may expect mechanism design, institution design, or reward/incentive intervention references. | Add 2-3 sources on mechanism design or institutional interventions in multi-agent systems if targeting AI/ML venues. |
| Problem Formulation | Mathematical formulation is adequate and readable. Capability rank is too coarse. | The paper’s core claim depends on capability being measurable. A rank alone can look arbitrary. | Keep rank, but always pair it with empirical validation table. Consider adding continuous search-budget variables in appendix. |
| Problem Formulation | Local/global safety thresholds are defined but not theoretically justified. | A reviewer may ask why mean patch health threshold 10.0 and sustainable margin 0.05 are chosen. | The full threshold sweep now addresses local-margin and global patch-health sensitivity. Remaining work is to test other global safety definitions. |
| Method | Baselines are clear: none, local, global, hybrid. | This is a strength. | Keep. Add exact ranking rule if not already available in appendix. |
| Method | Stress settings are called moderate/high coupling. Their exact values now appear in the paper. | This prevents the settings from reading as loose labels. | Keep the table. If moved to appendix for space, retain a compact main-text pointer. |
| Results | The results are described clearly, but some interpretation is shallow. | A reviewer wants “why did this happen,” not only “which condition won.” | Add one paragraph connecting global-signal performance to realized harvest prevention/capping and burden. |
| Results | The local-pass/global-fail case is useful, but it is one extracted episode. | A single case helps explanation but is not sufficient evidence of robustness. | Keep as qualitative example and rely on the full threshold robustness sweep for quantitative support. |
| Capability check | Good that actor capability validation exists. However, unsafe rate decreases as actor capability rises. | This can undermine the phrase “stronger actors” if readers expect stronger means more damaging. | Emphasize that capability means search/evaluation capacity and entrant payoff/action pressure, not monotonic collapse. |
| LLM bridge | Controlled strategy-bank design is methodologically sound. | It avoids live LLM unreliability and aligns with Willis et al. | Keep structured strategies. Do not overstate the bridge as a broad LLM-agent result. |
| LLM bridge | Only two 3B local models and 32+32 strategies per model. | Fine for a bridge; weak for a central LLM claim. | If LLM bridge becomes a main result, add one larger/third model and larger banks. |
| Discussion | Generally clear. | Some claims remain broad. | Keep benchmark-scoped language. Avoid “real-world policy” claims. |
| Limitations | Honest but brief. | A reviewer will want threats to validity. | Add threats: internal strategy representation, threshold dependence, small model bridge, no live-agent decision-making, synthetic commons substrate. |

## 2. Literature Grounding Audit

| Claim or Method Choice | Current support | Status | Required source or action |
| --- | --- | --- | --- |
| Commons governance depends on rules, monitoring, sanctions, participation, institutional fit. | Ostrom 1990. | Supported by citation. | Adequate. |
| Sequential social dilemmas are suitable for studying temporally extended cooperation/depletion. | Leibo et al. 2017; Melting Pot 2.0. | Supported by citation. | Adequate. |
| Complete strategies generated by LLMs can be inspected and evaluated in populations. | Willis et al. 2025/2026. | Supported by citation. | Adequate. |
| Scalable oversight should study actor/overseer capability gaps. | Bowman et al. 2022; Engels et al. 2025. | Supported by citation. | Adequate. |
| Qwen 2.5 and Llama 3.2 model choices. | Added Qwen2.5 technical report and Meta Llama 3.2 source. | Supported by citation. | Adequate for model identification. |
| Actor capability as mutation/search candidate count/evaluation horizon. | Internal benchmark design. | Needs stronger justification. | Add citation to search/evolutionary strategy generation or mark as benchmark operationalization. |
| Overseer capability as detection recall, delay, targeting capacity, and cost. | Internal benchmark design, inspired by scalable oversight but not directly sourced. | Needs citation or explicit benchmark-design framing. | Source type: oversight/evaluation framework paper, monitoring/enforcement literature, or mechanism design literature. |
| Local-pass/global-fail as compositional safety failure. | Internal metric. | Needs positioning, not necessarily citation. | Source type: compositional safety, local-vs-global specification, multi-agent safety, or systems safety literature. |
| Moderate/high coupling stress settings. | Internal parameter presets. | Needs reproducibility detail. | Add appendix table. Literature is optional if framed as stress tests, not field archetypes. |
| “Architectures with a global signal provide stronger ecological protection.” | Main pilot and LLM bridge results. | Supported by our results. | Keep benchmark-scoped. |
| “Hybrid oversight is usually strongest on patch health.” | Main pilot and LLM bridge results. | Supported by our results. | Keep conditional. |
| “LLM-generated exploitative strategies recreate the pressure.” | LLM bridge results. | Supported by our results but limited. | Keep as pilot claim; avoid broad model-family claims. |
| Field-policy relevance. | Not directly tested. | Should be softened. | Keep only as controlled benchmark relevance. |

## 3. Evaluation Quality Audit

### Baselines

The current baselines are fair for the paper’s main question: no oversight, local oversight, global signal, and hybrid oversight. This is the right comparison set because the paper is about architecture, not a new learning algorithm.

The missing baseline is a purely random or fixed non-adaptive entrant condition in the scalable-oversight pilot. The actor-capability ladder begins with mutation, which is reasonable, but a reviewer may ask what happens without deliberate capability pressure. If already available in earlier work, it should be referenced or placed in appendix.

### Metrics

The metrics are appropriate: global unsafe rate, local-pass/global-fail rate, patch health, welfare, and burden. They cover safety, ecology/resource state, performance, and governance cost.

The weakness is that the paper needs clearer prioritization. If hybrid has better patch health but lower welfare, the ranking rule should be explicit. A reviewer should not have to infer whether safety dominates welfare or whether the paper is reporting Pareto trade-offs.

### Missing Ablations

Required for stronger publication version:

1. Remove or weaken one overseer-friction component at a time: recall only, delay only, capacity only, cost only.
2. Search-budget ablation: candidate count and horizon varied separately.
3. Stress-setting ablation: show which exact Harvest parameter creates the moderate/high coupling difference.
4. Additional global-safety sensitivity: failed-patch threshold or alternative aggregate-state predicates.
5. LLM bank-size sensitivity: 16, 32, 64 strategies per attitude if feasible.
6. Model-family check: add a third model or larger model only if the LLM bridge becomes central.

### Interpretation Depth

The current interpretation is acceptable but not deep enough for a strong submission. It should explain mechanisms, not only winners. The paper needs to answer:

Why does local oversight fail under some settings?

Why does the global signal raise patch health but lower welfare?

Why does hybrid help most in high coupling?

Why does actor capability raise entrant payoff and aggression metrics while no-oversight unsafe rate decreases?

### Failure Cases

The episode trace is a good start. It should be described as a qualitative case, not as proof. A stronger version should include a small table of the most common local-pass/global-fail contexts: condition, capability gap, stress setting, mean requested harvest, mean patch health, failed-patch fraction.

### Stronger Evaluation Plan

Required baselines:

- No oversight.
- Local oversight.
- Global signal.
- Hybrid oversight.
- Fixed non-adaptive entrants or random entrants as a low-pressure baseline.
- Search-generated entrants as capability pressure.
- LLM-generated structured strategies as a strategy-source bridge.

Required metrics:

- Global unsafe rate.
- Local-pass/global-fail rate.
- Mean patch health.
- Garden failure rate.
- Welfare.
- Governance burden.
- Missed target rate.
- Delayed intervention count.
- Requested versus realized harvest.
- Entrant payoff and entrant aggression metrics.

Required ablations:

- Actor capability: mutation, search candidates, search horizon.
- Overseer capability: recall, delay, capacity, budget cost.
- Safety thresholds: local margin and global patch-health threshold.
- Stress settings: moderate versus high coupling, with exact parameter table.
- LLM bridge: model identity, attitude, bank size, exploitative share.

Qualitative examples:

- One local-pass/global-fail trace, already included.
- One global-signal success case where unsafe state is avoided.
- One welfare-cost case where ecological protection is achieved at high burden.

Threats to validity:

- Structured strategies may miss behaviors available to richer agents.
- Search-generated capability is a proxy for actor capability, not a full model of AI capability.
- Thresholds are benchmark parameters.
- The LLM bridge uses small open-weight models.
- LLMs generate offline strategies, not live actions.
- Harvest is a controlled commons benchmark, not a field-calibrated policy model.

## 4. Figure and Diagram Quality Audit

| Figure | Publication-quality? | Weakness | Recommended fix |
| --- | --- | --- | --- |
| Figure 1 evidence chain | Mostly yes. | Still slightly diagrammatic, but clear. | Keep. For final submission, reduce text density if page space is tight. |
| Figure 2 method schematic | Yes. | Good conceptual figure. Could be more visually sophisticated, but not default-looking. | Keep. Add appendix detail for exact parameters rather than crowding figure. |
| Figure 3 capability gap | Yes. | Strongest quantitative figure. Shows the paper’s main question directly. | Keep as main result figure. Consider adding confidence bands if available. |
| Figure 4 winner map | Acceptable as supporting figure. | Winner maps hide effect size and uncertainty. | Keep as summary only. Do not make it the main evidence figure. |
| Figure 5 case trace | Yes. | Good qualitative evidence. The shaded region explains the failure mode. | Keep. Caption should stress it is an example trace. |
| Figure 6 LLM bridge | Good pilot figure. | Lines are clear, but the result is averaged across stress settings. | Keep for bridge. If LLM becomes central, split by stress setting or add confidence intervals. |

Implementation status:

- Figures are exported as vector PDFs and PNGs.
- Typography is consistent and serif-based.
- Colors are consistent across conditions.
- Legends are readable.
- Figures now answer distinct questions.

Remaining figure weakness:

- No uncertainty bands in the main line plots.
- No exact parameter table for stress settings in the paper.
- Winner map lacks effect magnitude.

## 5. Claim Discipline

| Claim | Classification | Action |
| --- | --- | --- |
| Shared-resource systems can fail through interaction. | Supported by citation. | Keep. |
| Commons logic applies to multi-agent AI shared environments. | Needs citation or soft framing. | Keep as motivation, not as established fact. |
| Actor capability is operationalized through search-generated entrants. | Supported by our method. | Keep, but call it an operational proxy. |
| Overseer capability is detection, delay, capacity, and cost. | Supported by our method; needs broader grounding. | Add source or present as benchmark design. |
| Hybrid wins 13/18 decision cells. | Supported by results. | Keep. |
| Local-pass/global-fail is a compositional safety metric. | Supported by metric definition. | Keep, but cite related safety/compositionality work if expanded. |
| Model-generated exploitative strategies create resource failure under no oversight. | Supported by results. | Keep, limited to the tested models and settings. |
| Architectures with a global signal keep failure near zero. | Supported by results. | Keep, benchmark-scoped. |
| The benchmark is relevant to scalable oversight. | Supported by conceptual link and citations. | Keep with cautious wording. |
| The benchmark informs real-world policy. | Needs external validation. | Do not claim directly. |
| The LLM bridge generalizes to LLM-agent deployment. | Needs experiment. | Avoid broad version; say it tests model-generated structured strategies. |

## 6. Edited Sections Completed

Changes already made in `main.tex`:

- Clearer title.
- More direct abstract.
- Stronger introduction and research question.
- Expanded related work.
- Mathematical formulation.
- Main results section with numerical tables.
- Actor capability validation.
- Threshold sensitivity note.
- LLM bridge results.
- More cautious discussion and limitations.
- Added citations for Qwen2.5 and Llama 3.2.

## 7. Manual TODOs Before Submission

High priority:

1. Add or expand an overseer-limit ablation if the venue expects mechanism-level validation.
2. Keep the stress-setting parameter table in the paper or appendix.
3. Add one mechanism-design or institutional-intervention reference set if submitting to an AI venue.
4. Add one additional LLM model or larger model if the LLM bridge becomes central.
5. Add per-step trace logging if future threshold definitions need to be recomputed offline.

Medium priority:

1. Add a third LLM or larger open-weight model if the LLM bridge is meant to be more than a pilot.
2. Add an LLM bank-size sensitivity check.
3. Add a no-capability-pressure baseline if not already covered in earlier appendices.
4. Add qualitative success and cost cases alongside the failure trace.

Low priority:

1. Make Figure 1 more visually polished if submitting to a venue where diagram aesthetics matter heavily.
2. Convert the paper to the exact target venue template.
3. Move compile artifacts into `.gitignore` if they should not be tracked.

## Bottom Line

The paper is now much more coherent than the earlier versions. It can be defended as a controlled benchmark paper. It is not yet fully insulated against reviewer criticism. The most likely reviewer objections are:

1. The capability definition is too operational and may be arbitrary.
2. Threshold sensitivity is too narrow.
3. LLM bridge uses only small local models.
4. Stress settings need exact parameter disclosure.
5. The work needs deeper mechanism analysis beyond winner counts.

These are fixable. The paper should not be reframed again. The next work should strengthen evidence around the current spine.
