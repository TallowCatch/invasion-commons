# Publication direction and execution plan

17 September 2026. Planning only: no simulations, model calls, training, or remote jobs were started for this pass.

## 1. Decision

Develop a benchmark paper about **the information and effort needed to oversee interacting resource users**. Finish the fair monitor comparison in Harvest, then test the protocol in three distinct resource-game structures. Do not start five unrelated environments at once.

The proposed research question is:

> When can inexpensive local checks keep a shared resource safe, and when is information about other agents' actions worth its monitoring and intervention cost?

The scalable-oversight component must measure a monitor's success on a defined supervision task as the task becomes harder or its resources become limited. Increasing the agent count alone measures system scaling. Neither that nor subtracting preset ranks establishes that a weaker verifier can supervise a stronger actor. The paper should state which of these claims it actually tests.

Working title for the planned study: **Scalable Oversight in Sequential Commons: Evaluating Local Checks and Shared Information**. Keep the current title, *Evaluating Oversight in Sequential Commons*, until the resource-limited monitoring experiment exists. Titles should follow evidence.

The intended contribution has three parts:

1. A common evaluation protocol with explicit monitor observations, decisions, resource limits, and safety outcomes.
2. Controlled evidence about when failures come from an inadequate rule, missing interaction information, or uncertain resource dynamics.
3. A reproducible comparison across different commons structures, including an established external environment.

The general fact that individually acceptable actions need not ensure joint safety already has precedents. Novelty must come from what the protocol reveals and enables others to test, not from naming that fact LPGF.

## 2. Publication targets

These are assessments of fit, not predictions of acceptance. Requirements were checked against official pages on 17 September 2026.

| Venue | Recommended submission form | Fit and readiness |
| --- | --- | --- |
| **AAMAS** | Main-track full research paper; eight content pages plus references, double-blind LaTeX. EMAS is a natural area for an evaluation framework; COINE fits if institutional mechanisms become central. | Best conference audience. Requires a coherent technical comparison and useful evidence, not a repository history. [Instructions](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/). |
| **Autonomous Agents and Multi-Agent Systems (JAAMAS)** | Manuscript / Regular Paper, with the required 1-2 page information sheet explaining claim, evidence, nearest work, and prior publication. | Best journal fit and the most realistic schedule for the proposed expansion. [Guidelines](https://link.springer.com/journal/10458/submission-guidelines), [scope](https://link.springer.com/journal/10458/aims-and-scope). |
| **COINE workshop** | Research paper for feedback and presentation. The past 2026 call offered a 10-page short or 16-page full LNCS paper, excluding references. | Suitable for a bounded mechanism study. A future call must be checked; those are historical lengths, not confirmed 2027 rules. The series has archival post-proceedings, so check publication overlap before choosing this route. [2026 call](https://coin-workshop.github.io/coine-2026-paphos/call_for_papers.html). |
| **TMLR** | Regular anonymized research submission using its template and OpenReview. | Conditional fit if evaluating learned policies or supervision methods is a substantive contribution. Do not add superficial RL merely to fit this venue. [Author guide](https://jmlr.org/tmlr/author-guide.html), [criteria](https://jmlr.org/tmlr/acceptance-criteria.html). |
| **NeurIPS Evaluations & Datasets** | Benchmark/evaluation research paper with an executable, accessible artifact. | A later, more demanding target for a broadly useful release. The 2026 deadline was 6 May; a future call and format must be verified. [2026 call](https://neurips.cc/Conferences/2026/CallForEvaluationsDatasets). |

**Recommendation:** write for the AAMAS research audience; plan the expanded study on a JAAMAS timetable. Do not let an imminent deadline determine the experiments. A concise eight-page scientific narrative can be the core, with the fuller journal treatment and reproducibility material developed around it.

AAMAS 2027 lists OpenReview author registration on **17 September 2026**, abstract submission on **1 October**, and paper submission on **8 October**, all Anywhere on Earth. Every author needs an OpenReview account. Discuss the deadline with Yali now if retaining this option. Three weeks is not a responsible default schedule for building and validating a five-case suite. [Official call](https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/).

AAMAS 2027 also considers main-track submissions for **Findings** unless authors opt out. This is a full-length archival outcome of the same review process, not a separate short-paper submission. JAAMAS also has an AAMAS presentation route for eligible accepted original articles, subject to the conference's rules. Neither route guarantees acceptance or a particular presentation slot.

Do not submit overlapping work to two archival venues simultaneously. TMLR specifically excludes expanded versions of already archival conference papers. AAMAS requires disclosure when AI assistance contributes to hypotheses or experimental design, including tool/version and prompts; preserve the relevant records and have the human authors verify the design and citations. [AAMAS policies](https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/), [TMLR policies](https://jmlr.org/tmlr/editorial-policies.html).

## 3. What already exists, and what the feedback still requires

The current paper and [completed-checks report](RESULTS_AND_DIRECTION.md) are the starting evidence. The previous broad architecture sweep remains exploratory evidence. It should not dominate the next paper's structure.

| Feedback or problem | Work already done | What remains |
| --- | --- | --- |
| Yali: identify the problem before listing equations | The introduction now asks what local checks miss; the code distinguishes requests, permitted actions, and resource outcomes. | Organize introduction, method, results, and discussion around the same three questions below. |
| Yali: formulate the game and every stakeholder's decisions | User observations/actions/payoffs and the fixed stateful governor are explicit. | Specify the new monitor's observation budget, decision rule, and action authority. Do not present it as a trained strategic player or claim an equilibrium result. |
| Yali: establish the relationship to existing research | Social dilemmas, oversight, shielding, commons institutions, and model-generated policies are cited. | Compare the actual closest methods, especially factored shielding. A citation list alone cannot establish novelty. |
| Group: justify capability measurement and the single-dimensional gap | The headline rank-gap graph was removed. Held-out search tests found a partial, setting-dependent benefit. | Retain separate actor-search and monitor-resource axes. Measure performance on a common held-out task; no new weighted capability score. |
| Group: explain no-overseer curves and misleading replication | Unused overseer copies and aggregation issues were identified; repaired summaries use appropriate source-run units. | Enforce the same rules in the multi-game runner and tests. |
| Group: include local rejection with global safety | All four local/global outcomes and approval coverage are recorded. | Report false approval, unnecessary rejection, and failure onset rather than highlighting only LPGF. |
| Ed, as reported: distinguish guidance from imposed control | Announcement-only, communication-only, and enforcement controls now exist. | Keep their authority explicit. Fixed agents responding to announcements do not establish learned or persistent norms. |
| Audit: the original example only showed persistent damage | New checks include safe-to-unsafe transitions and clean-prefix cases. | Confirm on fresh populations under a protocol fixed before viewing results. |
| Audit: the supposedly stronger reference had a different objective | This discrepancy is documented in the paper. | Match the safety objective, intervention choices, and uncertainty assumptions before interpreting an information advantage. This is the first experiment. |
| User: Fishery/Harvest and model experiments feel disconnected | Their distinct roles and the limitations of the saved-model experiment are now stated. | Use a common evaluation contract and one external environment; keep exploratory model-policy work secondary. |
| Publication: reproducibility | Scripts, resumable blocks, tests, and local manifests exist. | A clean-machine release must work without private/ignored result files. |

The latest checks are substantive work: 10,880 evaluation episodes plus 576 short selection evaluations. They also overturned parts of the earlier story. In the high-stress, slow-regrowth check, a fixed local cutoff has 5.33% unsafe steps, a state-based local filter 0.125%, and the joint reference 8.30%. The last two use different safety targets. These results support better controls, not a claim that joint information is inherently worse or local oversight universally fails.

The two existing model banks are secondary. Direct numerical prompt templates reproduce the broad collapse/protection pattern. Adding another model cannot, by itself, repair that contribution.

## 4. Three game structures before five cases

| Core game | Why it earns a place | Work required |
| --- | --- | --- |
| **Fishery: one shared renewable stock** | Tests aggregate extraction without patch-level spatial complications. Provides a simple case where the resource accounting can be checked directly. | Adapt the current environment to the common logs, safety contract, and monitor interface; rerun the new comparisons. Old Fishery results are not already comparable. |
| **Repository Harvest: interacting local patches** | Tests spillovers, incomplete knowledge of neighbours' requests, and noisy regrowth. This is the main mechanism-development environment. | Finish matched comparisons and reuse the validated population/logging pipeline. |
| **Clean Up from an established implementation, preferably SocialJax** | Adds maintenance: sustaining production requires agents to do useful cleaning work, not merely extract less. Tests whether the protocol survives a different action structure. | Pin and inspect an external implementation, validate policies, define resource viability, and build an adapter. This is a feasibility task, not a drop-in import already working here. |

The external suite already supplies several social dilemmas and learning baselines. Reuse that contribution rather than recreating it. [SocialJax paper](https://arxiv.org/abs/2503.14576), [official implementation](https://github.com/cooperativex/SocialJax).

Two optional additions would make **five evaluation cases**: SocialJax Harvest Open and Harvest Closed. Open tests transfer from our numerical simulator to an established spatial implementation; Closed provides a boundary case with resource access partitioned. They are related variants, not two independent new game families. Add them only after the third game's adapter works and only if these specific transfer/boundary questions remain useful.

The current irrigation/forest settings are parameter combinations inside one Harvest implementation. Calling them separate games would exaggerate breadth. Keep their internal IDs for reproducibility, but explain their changed parameters. Likewise, our Harvest and SocialJax Harvest are different implementations.

Do not add Coins or Prisoner's Dilemma solely to reach five. General cooperation games need not instantiate the renewable-resource safety problem. Expanding the task family changes the paper's claim and requires justification.

Clean Up also exposes a real design constraint: restricting extraction alone might be unable to restore maintenance. The adapter must declare which actions a monitor may restrict, whether it can request cleaning, and whether users obey. Record infeasibility when permitted interventions cannot protect the resource. Do not silently give that monitor power to force cooperation.

## 5. The paper's three questions and the tests that answer them

### Q1. What makes an individual check miss resource failure?

First fix the Harvest comparison. Use the same requested policies, initial states, paired disturbances, intervention choices, and safety target. Vary information and the treatment of uncertainty.

Use a common per-patch protection target for the first information-only test, applied by both the local and joint monitor. This removes the current mismatch between protecting every patch and protecting only the mean. Continue reporting the original mean-health/failed-patch outcome as a separate system metric. Document when the per-patch requirement implies that outcome; the stricter requirement has its own cost and must not be presented as equivalent by definition.

Give monitors the same finite set of candidate restrictions and the same objective: retain as much requested activity as possible while meeting that target. A shared grid of joint extraction scales is a tractable starting point. The local mechanism can aggregate each patch's approval through a veto/minimum allowed scale; the joint mechanism examines the same candidate actions. Both therefore have the same restriction authority in this experiment. Describe the shared decision explicitly rather than claiming complete decentralization.

Required initial controls:

- No intervention and the existing fixed-cutoff filter.
- A state-based local monitor with an explicit estimate for missing neighbours' actions.
- A conservative local monitor using a stated bound on those missing actions.
- A joint-request monitor using the same safety target.
- Local and joint versions with a stated allowance for weather uncertainty, compared with their nominal predictions.

Unknown actions must not silently be set to zero. An uncertainty allowance must be chosen on development data or from the stated disturbance model, never tuned to improve the final plot. Joint state access still does not reveal future noise. Do not call this reference an oracle or claim a formal safety guarantee unless one is proved under the exact assumptions.

Report infeasible states where even the strongest allowed restriction cannot meet the target. Use fresh populations for confirmation. Preserve ordinary regrowth and the exploratory slow-regrowth condition; no retrospective deletion of the condition where local checks work.

**Deliverable:** a mechanism comparison, one verified onset example, and paired effect sizes with population-level uncertainty. A useful conclusion can be that good local checks suffice under the tested conditions.

### Q2. How much monitoring is needed as the supervision task grows?

After Q1, give oversight an explicit task: judge the risk of a proposed action or proposed action sequence before the outcome is observed. Use the same held-out proposals for all monitors. Evaluate judgments separately from the safety produced by later restrictions.

Start with one-step risk prediction and a fixed short lookahead as a targeted temporal check. Sequential damage is also measured in complete episodes. Do not vary horizon, model size, search, delay, recall, and cost simultaneously.

The first resource limit should be **access to current action information**, recorded as the number of requests/reports read per decision. Keep the environment model and intervention authority fixed. Compare local information, selectively shared information, and full joint information. Full information is a costly reference, not a cost-free default.

Accounting must include all local monitors and their messages. A distributed system using one monitor per agent cannot be labelled cheaper than one global monitor while its total work is ignored. Log request inspections, communication volume, model calls, and wall time separately. They are different quantities, not terms in a fabricated capability score. The evaluation logger may see full state; the monitor may not access it outside its declared interface.

Begin with small/medium/large populations (provisionally 6/12/24 in our numerical games) and three inspection levels. Check supported sizes in the external environment before fixing them. Maintain per-agent resource supply and interaction degree where possible. If map geometry or resource scaling changes difficulty, document that factor rather than calling the effect purely population size.

For actor pressure, use nested search budgets on the same policy class, objective, candidates, and selection horizon. Test selected policies on unseen populations/seeds. Report payoff or validated task performance and environmental harm separately: a better-performing actor need not be more destructive. No increase in held-out ability means no demonstrated actor-capability ordering.

Required outputs: false approvals, unnecessary rejections, approval coverage, probability calibration when the monitor predicts risk, approved failure onset, resource health, return, and monitoring/restriction burden. Stochastic outcomes require multiple evaluation disturbances; one surprising transition is not enough to establish a badly calibrated monitor.

**Deliverable:** safety and decision quality versus actual monitoring resources, with actor settings and game size shown separately. This supplies a measured limited-oversight claim. A broad stronger-AI/weaker-AI claim remains outside scope without corresponding evidence.

### Q3. Which findings survive a change of commons structure?

Run the same evaluation contract in the three core games. Keep game-native actions and safety definitions explicit. Do not apply Harvest's numeric cutoff or its fourteen strategy fields to every environment.

The common contract records: observation available to each stakeholder; action proposed and action executed; monitor judgment; intervention authority; safety before/after; payoff; and resources used by the monitor. A game adapter defines these fields, the local rule, the global viability criterion, and termination.

Select thresholds using documented resource dynamics and development tests, retain nearby-value sensitivity, and report them as benchmark design choices. Compare methods within each game. Show raw outcomes and paired changes; do not average fish stock, patch health, and cleaning returns into an unexplained score.

For the external environment, establish competent resource-use and maintenance policies before the main run. Existing checkpoints, if available and licensed, may be evaluated; otherwise validate scripted policies first and separately estimate a small trained-policy comparison. Random movement is not adequate evidence that an oversight method works on strategic agents. Frozen trained policies would strengthen the result beyond the current small hand-specified policy class, but a large multi-algorithm training contest is unnecessary.

**Deliverable:** the same questions answered across distinct structures, including where a method stops working. Three deliberately selected games support bounded transfer evidence, not universal generalization.

## 6. Literature work with a purpose

| Closest line of work | What to use it for | Difference that must be demonstrated |
| --- | --- | --- |
| [Leibo et al.](https://arxiv.org/abs/1702.03037), [Melting Pot 2.0](https://arxiv.org/abs/2211.13746), SocialJax | Define the game family, established implementations, partner variation, and sensible policy baselines. | A reusable oversight task and diagnostic protocol, rather than another claim to invent a commons simulator. |
| [Safe Multi-Agent Reinforcement Learning via Shielding](https://arxiv.org/abs/2101.11196) | Inspect centralized and factored shields, their observation assumptions, safety specifications, and guarantees. | An empirical limited-information/uncertainty comparison must add something beyond relabelling shields. Reproduce an appropriate reference or precisely document why a guarantee's assumptions do not hold. |
| [Measuring Progress on Scalable Oversight](https://arxiv.org/abs/2211.03540), [A Benchmark for Scalable Oversight Protocols](https://arxiv.org/abs/2504.03731), [Scaling Laws for Scalable Oversight](https://arxiv.org/abs/2504.18530) | Define an actual supervision task, judge success, and measure task-specific actor/monitor performance. | Joint and sequential resource consequences with explicit supervision resources. Do not infer comparability of our ordinal settings from these papers. |
| [GovSim](https://arxiv.org/abs/2404.16698) and the strategy-generation papers already cited in the manuscript | Bound claims about model-produced policies, communication, and sustainability. | Our current saved-policy interface is supporting evidence; the model itself has not been shown essential. |
| [Agarwal et al., statistical evaluation of RL](https://arxiv.org/abs/2108.13264) | Use uncertainty-aware comparisons and retain the correct independent run unit. | Apply appropriate statistics to our own sampling structure; do not count timesteps or repeated threshold labels as fresh replications. |

Before claiming novelty, read the relevant full methods and code, not only abstracts. Produce one closest-work comparison with task, observations, authority, uncertainty, and released evaluation support. The papers above establish relevant precedents, not that our final protocol is unprecedented. Ostrom supports the institutional motivation; it does not supply the simulator's numerical constants.

## 7. Execution order and compute gates

The time allowances below are planning allowances for implementation and review, not measured experiment runtimes. The expanded study is roughly a 4-6 week effort if the external adapter and useful policies are available; new RL training or major incompatibilities can extend it. Do not promise submission in that period before profiling.

| Phase | Action and output | Existing logs sufficient? | Gate |
| --- | --- | --- | --- |
| 0: 1-2 working days | Write the short common-comparison protocol and closest-work table; fix the target venue with Yali. | Yes for evidence inventory, not for the new comparison. | Everyone can state the question, controls, and allowed claims in plain language. |
| 1: 3-5 days | Implement matched objectives, observation restrictions, uncertainty controls, and unit tests in Harvest. Run a small fresh-population pilot. | New simulations required. | No unequal safety targets or hidden full-state access; correct handling of infeasibility. |
| 2: 3-5 days, may overlap phase 1 | Add Fishery adapter; audit and smoke-test an external Clean Up adapter and useful policies. | New simulations required. | Valid game-native actions, meaningful safe/unsafe outcomes, reproducible policies. No unbounded training job. |
| 3: 2-4 days | Implement budget accounting and held-out monitor judgments; time all methods and sizes. | Existing logs can test analysis code; new observations/decisions are required for the new claim. | Produce a measured compute/storage estimate and a fixed confirmatory manifest. |
| 4: bounded batch | Run the preregistered matched comparisons across three games; use new population and evaluation seeds. | New simulations required. | Sample size and contrasts fixed from pilot variance and a stated useful precision, not significance hunting. |
| 5: 4-7 days | Analyze by research question, package the release, write the paper and supervisor note, rebuild figures. | Reuse phase 4 output; no automatic expansion. | Claims match evidence and clean-machine reproduction works. |

Pilot envelope for phase 1: one game, two existing regrowth settings, eight monitor/control configurations, four independent fresh population contexts, and eight disturbances per context: **512 evaluation episodes**, before targeted noise-removal checks. Use the same contexts/disturbances across methods. This is development evidence, not the final uncertainty sample. The eight configurations are no intervention, fixed cutoff, nominal local, uncertainty-aware local, conservative local, uncertainty-aware conservative local, nominal joint, and uncertainty-aware joint. Fix missing-action assumptions and uncertainty parameters in the protocol.

For the confirmatory study, a planning envelope of three games x two stress settings x two policy-source settings x eight selected configurations x twenty independent population contexts x eight disturbances is **15,360 episodes**. This is an initial size illustration, not a power calculation or an instruction to launch it. Population-size/inspection-budget curves should be a targeted block using the relevant monitors, not every possible historical package. Final counts depend on phase 3 measurements. Never call twenty random seeds twenty independently trained policies if they all reuse one checkpoint.

Measured historical costs: the existing simple-filter screen took about 101 seconds for 3,520 episodes; all four completed checks used about 275 seconds of simulation and 65 MB of raw output. These do not predict the cost of new lookahead monitors or external grid games. Separate JAX compilation, simulation, analysis, and any policy training in the estimate.

After implementation, time a small representative block and estimate `episodes * measured seconds per episode`, plus generation/model-query overhead. Report per-game CPU/GPU requirements, wall time at a stated worker count, and artifact bytes. Use a checkpointed 15-CPU-minute first profiling allowance; if it is exceeded, revise the estimate before expansion. Do not buy cloud compute or launch a large remote sweep on an extrapolation from the old filters.

Use short background processes with explicit logs and completion manifests. Shard only after the small merge test passes. Cap concurrency, resume completed blocks, and aggregate independently of simulation. The user requested a plan here: no background work has been scheduled.

## 8. Implementation map

| Reuse | Required extension |
| --- | --- |
| `fishery_sim/harvest.py`; `experiments/validate_harvest_mechanisms.py` | Shared safety objective/candidate restrictions, observation-limited monitors, explicit disturbance allowance, fresh-context runner. |
| `experiments/analyze_harvest_validation.py` | Matched-target contrasts, monitor decision quality, budget curves, population-level intervals, per-game reporting. |
| `fishery_sim/env.py` | Fishery adapter with the same log contract; preserve old behaviour and result scales. |
| `tests/test_harvest_validation.py` | No hidden-information leakage, common objectives, budget accounting, infeasible states, deterministic/noise controls, and historical regression tests. |
| Existing calibration code | Separate candidate-count/horizon effects; keep held-out selection checks and save generation cost. |
| Existing workflow/merge patterns | New small block manifests and merge/completeness tests, only if profiling calls for remote execution. |
| Existing paper-input checks and manifests | Release inputs, checksums, exact commands, licenses, missing-data failures, and a clean-machine smoke run. |

New components needed: a small game/monitor interface, a pinned external-environment adapter, a monitor-budget runner, and its tests. Proposed locations are `fishery_sim/oversight_protocol.py`, `experiments/run_matched_oversight.py`, `experiments/analyze_matched_oversight.py`, and `tests/test_oversight_protocol.py`; these are planned files, not commands that already work. Do not refactor the entire repository to create them.

Preserve existing archived outputs and unrelated dirty files. New results should go into a fresh versioned directory; never modify old manifests to make a changed method appear resumable.

## 9. What the finished paper should look like

The useful lesson from the linked task, **Copy session knowledge base**, is its structure: a few research questions, experiments that directly answer them, and negative transfer findings retained rather than disguised. Its forest results are unrelated evidence; only that organization transfers here.

Use the same three questions in the introduction, experimental sections, and discussion. Avoid a chronological account of Fishery, then Harvest, then LLMs, then successive audits.

Suggested eight-page core, with references and detailed supplements separate:

- Problem, motivation, questions, and contributions: about one page.
- Closest work and the precise difference: about three quarters of a page.
- Game/monitor formulation, three structures, and shared protocol: about one and three quarter pages.
- Baselines, paired design, budgets, and uncertainty: about one page.
- Three result sections answering Q1-Q3: about two and a half pages.
- Discussion, limitations, and conclusion: about one page.

These are writing budgets, not an instruction to compress important assumptions into tiny text. The journal version can provide the additional detail needed for a self-contained account.

Main visual evidence:

1. One compact diagram showing the three resource structures and exactly what each monitor can see/change.
2. A matched comparison showing information, uncertainty, safety, and retained productive activity; include effect sizes, not just winners.
3. Decision quality and resource safety against measured inspection budget, with separate game-size panels.
4. A cross-game summary of paired changes, plus one explanatory onset trace if space permits.

Each figure answers a question. Use TikZ for diagrams and vector plotting for measured data. Training curves belong only where a policy really was trained and learning quality matters to the claim. Earlier winner maps, broad preset sweeps, and saved-model policies belong in supporting material unless they answer a main question independently.

### Stop adding experiments when these conditions are met

- Matched comparisons explain what can and cannot be attributed to local versus shared information.
- Limited oversight has a measured task, budget, and held-out success criterion.
- The protocol works across three justified game structures, or the paper explicitly narrows its cross-game claim if external integration fails.
- Useful policies, fresh contexts, uncertainty estimates, failure cases, and safety/return costs are reported.
- The nearest-method comparison is defensible and all numerical results have reproducible inputs.
- The conclusions remain valid if local monitors win, extra search does not help, or hybrid is not best.

If the matched test shows that a reasonable local rule solves the tested problem cheaply, preserve that finding. A systematic account of when shared oversight is unnecessary can be useful. If the external case fails, do not replace it with renamed Harvest presets to preserve a game count. Either solve the integration problem within the agreed budget or submit a narrower mechanism paper.

No live LLM agents, additional model banks, learned norms, new capability-gap score, or exhaustive old-friction grid are required for this paper. Those are separate research questions.

## 10. Supervisor decision and next implementation request

Suggested message:

> I want to focus the first paper on what information a monitor needs to keep a shared resource safe. The recent controls show that a good local rule can work well, so I am first making the local and joint comparisons fair. I propose testing the same protocol in a shared-stock game, an interacting-patch game, and an established Clean Up environment. The main results would show decision quality and resource outcomes against actual monitoring budgets. Does that give us an appropriate first benchmark paper for JAAMAS or AAMAS, with the saved-model experiments kept as supporting work?

The main choice to confirm with Yali is whether this bounded resource-limited-oversight contribution is sufficient, or whether she expects a trained weak verifier evaluating demonstrably stronger learned actors. The latter is a larger study and should not be implied by changing the title.

Recommended next request:

> Implement phases 0 and 1 of PUBLICATION_EXECUTION_PLAN.md: write the matched-monitor protocol, add the common target and observation restrictions, test them, and run the bounded 512-episode development pilot. Profile the new monitors and report results and a compute estimate before any larger batch. Preserve historical outputs. Do not add models or launch external-environment training yet.
