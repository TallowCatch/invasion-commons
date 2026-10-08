# Research assessment: local compliance and global resource risk

Audit date: 13 September 2026. This is an assessment of the current working tree, not only the last commit. It supersedes earlier next-step recommendations for planning purposes; it does not replace the manuscript or alter the archived results.

## Decision

There is a substantial implemented research project here. It contains environments, population evolution, oversight mechanisms, held-out evaluation, learned-policy checks, model-generated policy banks, remote execution, and recoverable experimental outputs. The immediate problem is the alignment between these components and the claims made about them.

The strongest demonstrated finding is that **local request compliance and global resource safety can disagree**, and that the implemented intervention packages change resource outcomes. The weakest headline is a measured, one-dimensional capability gap between actors and an overseer. The existing ranks do not establish that measurement. The current evidence also does not isolate why hybrid performs differently, or establish that approved joint actions cause the onset of global failure.

My recommendation is to develop the first PhD contribution as a **benchmark and evaluation protocol for local compliance versus global trajectory risk under strategic populations and limited intervention**. Keep scalable oversight as the motivating research direction, with a precise account of which parts of that problem this implementation studies. Validate the mechanisms before enlarging the experiment matrix or adding models. Publication readiness cannot be guaranteed by polishing or additional run counts.

**Working title:** Oversight in Sequential Commons: Local Compliance and Global Resource Risk.

**One-sentence thesis:** This benchmark tests when local action checks fail to indicate global resource safety, and how information, intervention limits, and strategy selection affect that mismatch in sequential multi-agent commons.

The last sentence is the proposed paper question, not a claim that all its mechanisms have already been established.

## What was inspected

The principal manuscript is `paper/paper_v5_scalable_oversight_commons/main.tex`, alongside its PDF, references, generated tables, and figures. The later supervisor-facing account is `notes/yali_problem_statement_method_sketch.tex` and its renamed PDF. I also inspected current direction/validation/ablation notes, relevant history, presentation assets, and the feedback supplied in this conversation.

The technical inspection followed `fishery_sim/harvest.py`, `harvest_evolution.py`, `harvest_llm_population.py`, the Stage A presets/runners, summaries, case extraction, threshold replay, and reduced overseer ablation. Numerical checks used the actual Stage A run CSV, recovered full threshold matrix, extracted trace, and both v3 model banks. Historical Fishery/PPO outputs were inspected as supporting work, not independently retrained or exhaustively revalidated.

The reproducible numerical audit is `experiments/audit_research_evidence.py`. Its input hashes and outputs are in `notes/research_review/evidence/`. Existing uncommitted work was preserved. No model generation or large simulation sweep was launched.

## What has actually been built

| Component | What it does | What it contributes now |
| --- | --- | --- |
| Fishery | A simpler shared renewable stock with policy interventions and changing strategy populations. | Establishes the experimental approach and earlier intervention studies. Supporting evidence, with a different protocol and outcome scale. |
| Harvest | A small numerical patch/ring commons with extraction, regeneration, weather, neighbour spillovers, messages, and optional payoff transfers. | Main controllable test environment. This is the project's implementation, not a drop-in implementation of Melting Pot or SocialJax. |
| Structured strategies | Parameters specify extraction by resource level, messaging, reciprocity, transfers, and responses to announced caps. | Inspectable policy artifacts that can be mutated, searched, generated, and replayed. |
| Population turnover | Policies are scored and replaced between generations; each evaluation episode resets the environment. | A population-selection process. The strategies are not undergoing gradient training during each episode. |
| Oversight packages | No mechanism; communication/reciprocity; global cap mechanism; communication plus neighbourhood-targeted cap mechanism. | Distinguishable intervention packages, with important confounds discussed below. |
| Stage A | 360 run rows, 72 architecture cells, 18 actor/overseer/stress contexts; five runs per cell. | A completed descriptive comparison, not merely a planned benchmark. |
| Threshold replay | 9,000 run rows covering 25 threshold pairs; completeness/default agreement checked. | Sensitivity of the safety labels on repeated underlying trajectories. |
| PPO work | Earlier learned-policy training and evaluations in Fishery and a narrower Harvest setting. | Evidence that the project has used learned policies. These are not training curves for the Stage A strategy-generation ladder. |
| Model-generated policies | Qwen 2.5 3B and Llama 3.2 3B each supplied 32 cooperative and 32 exploitative structured policies. | A functioning offline model-to-policy evaluation interface, with a substantial prompt-anchor confound. |
| Reduced ablation | 28 run jobs: one stress setting, high actor, two architectures, seven oversight presets, two runs. | Preliminary evidence about the implemented limitations; not a full factorial mechanism validation. |

The main experiment uses six agents, 15 generations, mutation or search-generated entrants, and separate training/test seeds. Test evaluation includes four held-out regimes. Selection and episode interaction are different time scales and should be drawn separately in the method diagram.

Earlier work belongs in the paper only when it answers a specific question. A thesis or repository history can encompass everything; a first paper needs a smaller coherent claim. Fishery and PPO are appropriate supporting sections/appendices unless a directly matched experiment makes them essential to the main argument.

## Highest-priority findings

### 1. The capability-gap graph does not identify a measured capability gap

The actor presets change the generation process: mutation, search over six candidates with a 30-step evaluation, and search over twelve with a 60-step evaluation. The overseer presets bundle target dropout, delay, target capacity, and cost. Subtracting their assigned ranks creates an ordinal configuration index. There is no calibration placing search ability and intervention capacity on the same unit scale.

The same gap value combines different treatments. At gap zero, hybrid's mean unsafe rates are:

| Stress setting | Low actor / weak overseer | Medium actor / limited overseer | High actor / strong overseer |
| --- | ---: | ---: | ---: |
| Moderate | 4.62% | 6.05% | 0.00% |
| High | 9.15% | 6.06% | approximately 0.00% |

This makes the group's dimensionality objection substantive. It cannot be resolved by explaining the existing subtraction more confidently. Plot actor-generation settings and individual oversight limits separately. A useful common scale would require performance calibration on a common operational task, and may still be inappropriate across roles.

The replacement diagnostic figure in `evidence/actor_overseer_separated.pdf` keeps the factors separate and shows actual run values. It changes no experiments. It still displays bundled overseer presets, so it is descriptive rather than a causal decomposition.

### 2. None and local-only have no system-level governor

`_make_condition_setup` in `fishery_sim/harvest_evolution.py:1079` returns `governor=None` for both conditions. Local-only retains communication and optional reciprocity/transfers. The local safety predicate is a diagnostic logged in every condition, not an active local checking agent that rejects unsafe requests.

For both conditions, every audited outcome is exactly identical across overseer presets at a fixed scenario, actor setting, and run. Their apparent gap-axis slopes are consequently changes in actor composition along the axis, not effects of an increasingly weak overseer. Repeated copies across inactive presets must not count as additional independent replicates in uncertainty estimates.

Use labels such as "local coordination" for the implemented baseline. If the paper wants to compare local safety enforcement against global safety enforcement, add an actual local-filter comparator with explicitly stated information and intervention authority.

### 3. The highlighted case demonstrates persistent mismatch, not failure onset

The extracted 80-step case contains 33 all-local-pass steps, 70 globally unsafe steps, 29 local-pass/global-fail steps, and six steps where at least one local check fails while the global state is safe. All 29 highlighted steps start with mean patch health already below 10 and follow an unsafe state. None is an observed safe-to-unsafe transition with all local checks passing.

This is useful evidence that acceptable current requests need not imply recovery or current system safety. It does not establish that those approved requests caused the original failure. Nor does it establish that every action earlier in the episode passed the check. Weather, previous extraction, and accumulated damage remain possible contributors.

Keep the example with a corrected caption. Across the benchmark, distinguish unsafe-state occupancy, onset from a safe state, persistence, and recovery. Evaluate the local check as a diagnostic with approval coverage and all four local/global outcomes. A strict check can reduce LPGF merely by approving almost nothing.

There is also a construct-validity risk: an action-only cutoff that ignores current resource state is expected to disagree with a state-safety predicate. Calling its parameter "sustainable" does not prove sustainability under every regeneration, spillover, and weather setting. The paper needs a defensible state-aware local comparator, alongside the simple cutoff, to show what information is genuinely missing rather than only exposing a weak check chosen by the designer.

An additional provenance check is needed: case extraction sorts strategy history by fitness, while the environment's ring neighbourhoods depend on agent ordering. Verify the regenerated population against original per-agent indices before describing it as an exact replay of the original spatial episode. This is a replay-provenance risk, not a confirmed mismatch in the exported case.

### 4. Hybrid changes more than one mechanism

Global-only disables messages/transfers and uses a global targeting scope. Hybrid enables the local channels and changes to neighbourhood targeting with neighbour expansion. Thus the observed comparison combines information exchange, voluntary policy responses, and a different enforced-target rule.

Moreover, populations evolve separately under the governance conditions. Comparing them measures the total outcome of an architecture plus the population it selects. That is a legitimate experiment, but it does not isolate the effect of applying two architectures to identical policies.

The manuscript's account that "local response and a global signal reinforce each other" is a hypothesis. Test it on frozen populations by crossing communication on/off with uniform/neighbourhood targeting, keeping intervention authority and limits matched. Report the evolutionary comparison separately as an adaptation experiment.

The stress presets also bundle regeneration, externalities, population mix, and other settings. Moderate/high stress are appropriate descriptive labels. A claim that coupling specifically explains hybrid's advantage needs a coupling-only manipulation; scenario labels alone do not identify that cause.

### 5. The implemented overseer is more specific than the paper describes

`GovernmentAgent` receives exact aggregate resource information and previous raw requests. The recall setting randomly drops intended targets during enforcement; it does not currently make the governor's resource observation less accurate. Planned caps are communicated to policies before enforcement dropout/capacity limits. Policies can voluntarily respond to a cap even when it will not be enforced.

These distinctions matter for scalable oversight. The current experiment primarily limits intervention execution. It does not directly measure a less capable reasoner verifying a more capable actor's decisions. A signal-only versus enforced-cap control is especially informative here.

The cost parameter is a price per targeted agent, not a finite budget that can run out. It is subtracted from agent payoff and also logged as intervention burden (`fishery_sim/harvest.py:655`). Do not subtract it from net welfare a second time, and correct formulations that claim it never affects welfare. With six agents, the implemented floor rule makes target share 0.33 allow one target, not two; the current rule even permits one at a nominal zero share. Report actual target counts, and test endpoint semantics before new experiments.

### 6. Full threshold completeness is real; replication claims need correction

The 25 threshold settings leave patch health, welfare, and burden exactly unchanged within every underlying run. These thresholds alter the diagnostic labels, not the policies or resource dynamics. Therefore 375/450 patch-health wins represent 15/18 original contexts counted 25 times, not 450 independent ecological tests.

The full sweep is valuable for assessing whether the mismatch diagnostic depends on a single arbitrary cutoff. It does not supply additional independent trajectory evidence or demonstrate intervention robustness to changed dynamics. Do not rerun it again. Preserve primitive step quantities so future thresholds can be recomputed offline.

### 7. Ranking and welfare interpretations need repair

The Stage A summarizer sorts first by unsafe rate, then LPGF, garden failure, patch health, and welfare. Burden is not a ranking key (`experiments/summarize_harvest_invasion.py:238`). Consequently, a local winner is not necessarily winning by sacrificing safety for welfare. In moderate stress, high actor/weak overseer, local has unsafe rate 2.63%, compared with hybrid 6.48% and global-only 9.12%.

The threshold figures use a different, patch-health-first comparison. Both rankings can be reported if their rules are explicit; their winner counts should not be combined into one scientific conclusion. Prefer scenario-specific paired effect sizes and resource/safety/burden trade-offs.

Episode rates use the executed duration, which can be shorter than the nominal horizon after collapse. The welfare table uses mean net payoff per executed step; burden is an episode total. This can obscure the difference between brief, lucrative extraction followed by collapse and sustained return. Report total return, duration/survival, and consistently normalized cost. A fixed-horizon absorbing-failure convention would require an explicit definition, not silent replacement of historical metrics.

### 8. Actor validation is encouraging but not independent calibration

The reported entrant payoff increases from 188.3 to 211.4 and 214.4. However, it comes from training evaluations in each generator's own evolving populations. The search procedure selects candidates on an internal seed and short horizon, with a collapse penalty. This establishes a budget manipulation and suggestive performance differences, not a calibrated ability ordering under common opponents.

Validate generators using identical parent/opponent populations and unseen evaluation seeds. Separate candidate count from search horizon. Measure the entrant's own held-out payoff and adverse effects on others separately. Better payoff need not imply more collapse: preserving the resource can improve the optimization objective. If more search is not reliably better on that objective, call the treatments generation budgets rather than verified capability levels.

### 9. The model-policy result needs a non-model baseline before another model

The v3 prompt supplies numerical anchors for all fourteen strategy fields, drawn from deliberately different cooperative/exploitative ranges. It asks the model to stay close to them. Of 64 Qwen strategies, 35 exactly equal all fourteen supplied anchors; 74.1% of individual field values equal their anchors. For Llama the corresponding counts are two strategies and 3.35% of fields.

This does not invalidate the generated banks. It limits attribution: much of the policy distinction is supplied by the experiment designer, and the present experiment does not show what the language model contributes beyond that sampler. Exact numerical deduplication also does not prove distinct trajectories.

The next comparator is the anchor sampler itself, using the same nonces and evaluation populations, without an LLM call. Report validation/clamping as transformations, behavioural diversity on common observations, and paired outcomes. Stronger hardware, a third model, or live-agent prompts cannot substitute for this control.

### 10. Analysis and release defects could invalidate a larger follow-up

The ablation wrapper's summary groups omit scenario (`experiments/run_overseer_limit_ablation.py:91`). The proposed full two-scenario command would pool stress settings. The stress-regime analysis labels aggregate `test` results as "nominal", although `test` pools four held-out regimes. It also selects final-generation evidence while the main study averages generations. Fix and test these definitions before reusing those outputs.

The local paper-input check passes, but it verifies file presence, not clean-clone reproduction. Results are ignored, many current scripts/assets are untracked, and the root README still foregrounds the early Fishery project. A release needs pinned dependencies, a versioned dataset manifest, checksums, trace schema, and a bounded end-to-end example. Existing data should be preserved, not deleted in the name of cleanup.

## What the existing aggregate numbers support

These are descriptive equal-weight means over the existing run rows, not population-generalization confidence estimates. For none/local, inactive-overseer duplicates leave the means unchanged but must be removed for inference.

| Implemented condition | Unsafe steps | LPGF steps | Mean patch health | Mean net payoff / executed step | Mean burden / episode |
| --- | ---: | ---: | ---: | ---: | ---: |
| None | 19.41% | 2.06% | 11.42 | 14.75 | 0.00 |
| Local coordination | 13.92% | 1.18% | 11.62 | 14.68 | 0.00 |
| Global cap package | 6.13% | 2.35% | 14.22 | 12.34 | 2.10 |
| Hybrid package | 4.10% | 1.24% | 14.33 | 12.52 | 2.05 |

The packages with caps have healthier resources and lower average unsafe occupancy. Hybrid has the lowest average unsafe rate. Global-only has the highest average LPGF; hybrid does not have lower average LPGF than local coordination. LPGF and resource protection are different outcomes. These results justify investigating when local compliance is informative, rather than claiming hybrid solves compositional safety.

The reduced ablation suggests delay deserves priority: isolated delay creates nonzero unsafe rates in its tested slice, whereas isolated target dropout/capacity/cost do not. Two runs in one slice, with cap announcements still available to policies, cannot establish a universal hierarchy of oversight limitations.

## A formulation that matches the implementation

Model a finite-horizon, general-sum, partially observed stochastic game among resource users, coupled to a fixed stateful oversight policy. At step t, the augmented state must include patch levels, previous requests, relevant credit/message memory, the governor's trend and delay state, and the failure streak. Otherwise patch levels plus a few aggregate statistics need not be Markov.

Each policy has parameters phi_i sampled by a strategy generator G_b. The generation budget b includes the mutation/search procedure, candidate count K, internal horizon H, and search seeds. Policies emit messages and harvest/transfer decisions from their permitted observations. These include the local patch, neighbour mean, last received credit, and any announced cap; neighbour resource information is supplied even when the messaging channel is disabled. The governor computes a planned signal/target action using its own information history; policies can respond to the announced cap; enforcement then modifies requests subject to its execution settings. The simulator applies extraction, transfers, regeneration, spillovers, and weather.

Let p_t be requested harvest fractions and a_t the realised actions after intervention. These are different variables. Let A_t = product_i 1[p_i,t <= alpha + epsilon], and G_t denote the global resource predicate at state t. Existing LPGF is the executed-step average of A_t(1-G_(t+1)). It is an occupancy diagnostic.

The global predicate in code requires mean patch health at least tau and failed-patch fraction strictly below rho. Preserve that strict inequality in the formulation; a less-than-or-equal version changes the boundary condition.

Add, with explicit denominators:

- Approval coverage: mean(A_t).
- Unsafe occupancy conditional on approval: sum A_t(1-G_(t+1)) / sum A_t, unavailable when no requests jointly pass.
- Approved onset: sum G_t A_t(1-G_(t+1)); report opportunities sum G_t A_t as well as the count/rate.
- Persistence: sum (1-G_t) A_t(1-G_(t+1)); also report recovery and all four local/global outcome categories.
- Resource health, total/net return, survival duration, intervention incidence, and burden in stated units.

These are proposed diagnostics, not results already available for every episode. Onset alone is not causal proof: safe-start, no-weather/no-coupling controls and explicit joint-action reasoning are still needed.

Describe the evaluation as E(b, q, d, k, beta, m, theta), retaining its separate factors. The current governor has no learned utility-maximizing policy and no strategic equilibrium has been solved. It is legitimate to study fixed institutional mechanisms in a game, provided the paper distinguishes stakeholders' actions/objectives from the experimenter's choice of mechanism.

## Literature positioning and novelty

Primary sources were consulted online. Their role is to bound the claim, not decorate the introduction.

| Lineage | Relevant established result | Implication for this paper |
| --- | --- | --- |
| Sequential social dilemmas: [Leibo et al. 2017](https://arxiv.org/abs/1702.03037), [Melting Pot 2.0](https://arxiv.org/abs/2211.13746), [SocialJax](https://arxiv.org/html/2503.14576v3) | Repeated multi-agent environments already test cooperation, resource externalities, and policy/population evaluation. | Novelty cannot be the existence of a renewable multi-agent commons. Explain why a small inspectable implementation supports diagnostic and intervention tests. |
| Scalable oversight: [Bowman et al.](https://arxiv.org/abs/2211.03540), [A Benchmark for Scalable Oversight Protocols](https://arxiv.org/abs/2504.03731), [Scaling Laws for Scalable Oversight](https://arxiv.org/html/2504.18530v3) | These operationalize oversight tasks/protocols and empirical actor/overseer performance. Engels et al. estimate role-specific ability from game outcomes rather than arbitrary preset labels. | Motivate the supervision problem; do not claim the current rank subtraction inherits their measurement validity. Define what the monitor observes, decides, and can enforce. |
| Joint safety: [Safe Multi-Agent Reinforcement Learning via Shielding](https://arxiv.org/abs/2101.11196) | Centralized and factored shields already reason about multi-agent safety and intervene on actions. | Local/global safety composition is not a newly discovered general problem. Position the contribution around sequential resource diagnostics, adaptive populations, and bounded mechanisms. A model-aware safety baseline is missing. |
| Resource-use LLM societies: [GovSim / Cooperate or Collapse](https://arxiv.org/abs/2404.16698) | LLM resource-use societies have already been tested for sustainable cooperation, communication, and interventions. | This is a close omitted comparator, not a distant footnote. Explain the benefit of reusable structured policies and controlled intervention-factor experiments. Do not claim to originate LLM commons governance. |
| Model-generated strategies: [Willis et al., larger LLM strategy populations](https://arxiv.org/abs/2602.16662) | Generating algorithms offline and studying their spread through populations already has a direct precedent. | The policy-bank interface is an extension/compatibility result; a new scientific result requires a controlled question beyond successful JSON generation. |
| Institutional mechanisms: [Incentivising Monitoring in Open Normative Systems](https://ojs.aaai.org/index.php/AAAI/article/view/10610), [Governing multi-agent systems](https://link.springer.com/article/10.1007/BF03192407) | Monitoring incentives, testimony, norms, and sanctions are established MAS questions. | State whether credits fund monitoring, reward restraint, or transfer payoff. The implemented transfer rule does not establish incentive-compatible monitoring. Ostrom supplies institutional motivation, not numerical parameter calibration. |
| Measurement validity: [NIST AI 800-3](https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.800-3.pdf) | Performance on fixed benchmark items and generalization beyond them require different statistical claims. | Separate independent runs, repeated thresholds, sampled fixed banks, and heterogeneous stress settings. Uncertainty bars do not repair a poorly defined construct. |

The plausible contribution is the **combination of a specified sequential-resource evaluation protocol, transparent strategy pressure, intervention-factor comparisons, and a useful compliance-risk diagnostic with known failure cases**. At present it is a benchmark formulation and substantial pilot. The stronger mechanism contribution remains to be demonstrated. A novelty claim stronger than this requires direct comparisons and a clearer distinction from joint-action shielding and GovSim.

Bibliographic repair: the current Sudhir reference uses incorrect author names. The source lists Abhimanyu Pallavi Sudhir, Jackson Kaunismaa, and Arjun Panickssery. The exact HDO paper/proofs referred to in earlier notes were not verified from a readable full text in this pass; do not use it to support a precise theorem or comparison until the source is confirmed.

## Paper structure and figure decisions

| Section/asset | Decision | Reason |
| --- | --- | --- |
| Abstract/introduction | Lead with compliance-risk measurement and limited intervention; bound scalable-oversight claims. | Reader needs the unresolved problem before rank ladders or environment names. |
| Method | Separate environment, strategy generator, fixed mechanism, diagnostic, and sampling unit. | Current labels conflate local coordination with checking and execution limits with information limits. |
| Figures 1-2 | Keep one compact method diagram in main text; move historical evidence chain to appendix. | Show information and action timing, not the repository chronology. |
| Figure 3 | Replace scalar-gap headline with separate actor/overseer factors. | Dimensionality and duplicate-baseline objections cannot be fixed cosmetically. |
| Figure 4 | Appendix or replace with paired effects. | Lexicographic winners conceal objective order and small margins. |
| Figure 5 | Keep as explicitly selected persistent-mismatch example after provenance check. | Useful explanation; not proof of all-safe failure onset. |
| Figure 6 | Keep as secondary policy-source result with prompt-anchor disclosure and direct-sampler control. | Current model attribution is weak even though the interface works. |
| Figure 7 | Emphasize diagnostic sensitivity, not repeated health winner counts. | Threshold relabelling does not change resource dynamics. |
| Results/discussion | Distinguish observations, mechanisms known from code, and mechanisms requiring controls. | Plausible explanations are not automatically experimentally established. |

The main figures already use vector exports, coherent typography, and uncertainty conventions. The decisive remaining improvement is inferential clarity, not switching plotting software. Training curves are appropriate for actual PPO training; use turnover trajectories and generator-budget/quality curves for Stage A.

## Recommendation for the next meeting

Present the implemented benchmark and the corrected uncertainty about its claims. Ask whether the first paper should establish the compliance-risk measurement and intervention mechanisms, with stronger-agent supervision as the next extension. Bring the separated-factor plot and one concrete onset/persistence distinction. The priority is a matched, frozen-policy mechanism test and independent generator calibration, followed by the corrected overseer-limit ablation. Do not buy compute or generate another model bank yet.

See `FEEDBACK_AND_RESPONSE.md` for the meeting-feedback map and `EXECUTION_PLAN.md` for ordered work, existing scripts, runtime assumptions, and decision gates.
