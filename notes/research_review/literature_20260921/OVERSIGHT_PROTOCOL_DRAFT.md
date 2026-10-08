# Proposed Protocol: Oversight in Sequential Commons

21 September 2026. **Design proposal, not an implemented or completed result.**
This specifies the next study after the [literature decision](LITERATURE_AND_GAME_DECISION.md).
It refines, rather than launches, the 17 September execution plan. No new
simulation, training, remote job, or model call was run for this review.

## 1. Question and Scope

> When resource users affect one another, how much information and checking
> effort does a monitor need to identify harmful decisions without blocking
> useful activity?

The experiment separates three possible causes of failure: a poor checking
rule, missing information about other users, and uncertain future dynamics.
Actor search remains a source of pressure, but its budget and measured success
are separate from monitor resources. More actor search is not assumed to cause
more damage.

Begin with Harvest, then transfer the same questions to Fishery and Cleanup.
Fishery isolates shared-stock accounting. Harvest adds spatial effects. Cleanup
adds costly maintenance, so reducing extraction alone may be inadequate.
These are three game structures, not three unrelated research projects or a
claim of three statistically independent samples from all possible games.

The immediate study concerns computational monitors and structured policies.
A later learned-verifier study would be needed for a strong claim about weaker
language models supervising stronger language models.

## 2. Players, Observations and Decisions

For each game family/configuration \(g\), define a finite-horizon, general-sum,
partially observed stochastic game:

\[
\mathcal M_g=(N_g,S_g,\{A_i,O_i,r_i\}_{i\in N_g},P_g,T_g).
\]

Resource user \(i\) receives an observation and proposes action
\(a_{i,t}\sim\pi_i(\cdot\mid h_{i,t};\phi_i)\). The parameters
\(\phi_i\) are produced by a declared strategy-generation process, with search
budget \(b_A\). They are not the budget itself. Selection objective, opponents,
candidate count, search horizon and held-out performance must be logged.

A monitor receives permitted evidence:

\[
e_t=Q_m(h_t,\mathbf a_t;b_V),\qquad
v_t\sim\mu_m(\cdot\mid e_t),
\quad v_t\in\{\mathrm{approve},\mathrm{reject},\mathrm{abstain}\}.
\]

Here \(m\) specifies evidence access: local, local plus a shared summary, or
joint information. The query budget \(b_V\) records actual inspections and
model-evaluation work; it is not a label such as "weak". The monitor never
receives the evaluator's hidden state or future random disturbances unless
explicitly designated as a full-information reference.

For intervention experiments, a separate executor applies an allowed action
\(u_t\) from a game-specific menu:

\[
\widetilde{\mathbf a}_t=I_g(\mathbf a_t,u_t),\qquad
s_{t+1}\sim P_g(\cdot\mid s_t,\widetilde{\mathbf a}_t).
\]

The game state includes any persistent intervention queue, ledger or resource
memory needed to make this transition definition valid. The same authority and
intervention menu are available to all monitors in a matched comparison.
Observation pooling does not, by itself, imply centralized political authority.
A fixed monitor is a controller with specified decisions, not an optimizing
stakeholder for which an equilibrium has been derived. Strategic incentives for
peer monitors require a separate payoff model and experiment.

## 3. One Safety Target Per Comparison

Define a game-native safe set before inspecting comparative results:

\[
K_g=\{s:c_{g,j}(s)\leq 0\text{ for every declared constraint }j\}.
\]

All compared monitors are evaluated against the same \(K_g\). In Harvest, do
not compare a filter protecting every patch with one protecting only mean
health and failure share, then attribute the difference to information access.
Either target can be studied, but they are separate experiments.

The initial development comparison should use the existing global target for
every method. A stricter per-patch target can be a sensitivity check. Local
methods must state how they bound unobserved contributions to the global
target; they cannot silently replace that target with their own.

For a proposed, **unmodified** joint action, define conditional short-horizon
risk under fixed continuation policies:

\[
R_{g,H}^{\pi^{\mathrm{cont}}}(s,\mathbf a)=
\Pr\!\left(\exists k\in\{1,\ldots,H\}:s_{t+k}\notin K_g
\mid s_t=s,\mathbf a_t=\mathbf a,\pi^{\mathrm{cont}}\right).
\]

Use \(H=1\) first. A predeclared \(H=5\) development check can test whether
one-step verdicts miss delayed consequences. These values are design choices,
not literature-derived constants. Continuation policies must be fixed across
monitor comparisons. This definition is not a guarantee about all future
policies or an infinite horizon.

Reference risk comes from exact calculations in tractable cases or independent,
higher-budget simulator rollouts. Record its uncertainty. If classifying risk
against a threshold \(\delta\), mark cases whose reference interval crosses
\(\delta\) as unresolved. Report their share; do not silently discard them.
Thirty-two rollouts with no failures do not certify rare-event safety.
Continuous risk-estimation error can also be reported, with reference sampling
error acknowledged.

Evaluate failure onset from states in \(K_g\). Already-unsafe states form a
separate recovery/persistence task. This prevents a damaged resource's repeated
unsafe steps from being described as repeated new failures of composition.

### Why shared evidence could help

Consider a one-step withdrawal budget of 1.5. A user requesting 0.8 cannot tell
from that request alone whether another user requests 0.2 or 1.0. The resulting
joint actions differ in safety. A conservative equal quota of 0.75 can protect
the total without seeing the other request, but can reject useful withdrawals.
Sharing the sum can relax that restriction. This is an explanatory example,
not a new theorem or a claim that all local rules are inadequate. The real
games add state-dependent renewal, spatial effects and uncertainty.

## 4. Fair Baselines and Controls

The first comparison uses frozen proposals and matched states, so differences
in judgment are not mixed with differences in evolved populations. Closed-loop
episodes follow, to test what the decisions do to resources and returns.

| Baseline | Purpose and requirement |
| --- | --- |
| No intervention | Resource and return reference. Do not replicate inactive monitor settings as independent runs. |
| Fixed local cutoff | Historical simple baseline only; never the sole representative of local oversight. |
| State-aware local check | Uses permitted own-state information and an explicit assumption about hidden requests. |
| Conservative local check | Bounds missing requests using the known action set; tests the safety/productivity cost of missing information. |
| Local checks plus a shared summary | Tests whether a small message about joint demand is enough; price the communication and its computation. |
| Joint-information check | Uses all permitted current requests, with the same safety target, model, decision objective and total work allowance. |
| Higher-budget full-state reference | Estimates what is attainable with additional evidence/work. Price it separately; do not call it infallible. |

Compare nominal dynamics with explicitly uncertainty-aware variants for the
core local and joint methods. In unbounded-noise models, a finite disturbance
margin gives a probabilistic approximation, not absolute safety.

For intervention selection, use a fixed feasible menu and a common objective:
retain useful requested activity subject to the chosen risk criterion. If no
menu action meets it, log infeasibility. Count the work needed to evaluate menu
actions. At equal total budgets, a collection of local monitors must share the
budget; do not give each of six monitors the full joint monitor's allowance.

Only the evaluator may use full state to score a local method. Missing-neighbour
actions cannot be set to zero without calling that an optimistic assumption.
Include an analytic quota/contract reference where a valid one exists. The
shielding literature already shows that carefully coordinated local rules can
be safe under specified assumptions; our numerical methods inherit no such
guarantee merely by being called shields.

For Cleanup, extraction clipping cannot force agents to perform maintenance.
An adapter must define valid authority and competent maintenance policies.
Freezing all users must not count as unqualified success: retained production,
return and maintenance are evaluated alongside safety.

## 5. Measurement Without a Scalar Capability Gap

**Actor resources:** candidate count, selection horizon, simulator calls and
available information. Change one at a time initially. Use nested candidate
sets and the same selection objective/opponents when comparing search budgets.
Validate improvement on fresh contexts using individual return or a declared
task-success measure. Useful competence and harmfulness are distinct.

**Monitor resources:** inspected observations/requests, simulator transition
calls, communication volume, response delay and wall-clock time. These are
separate axes. Wall time is implementation/hardware dependent. Intervention
capacity and cost are also separate from judgment accuracy.

Plot outcomes against actual work, with actor budget and population size as
separate panels or conditions. Do not subtract ranks or invent a weighted sum
to restore the old gap graph. A task-specific performance scale would need its
own validation, as the scalable-oversight literature emphasizes.

| Quantity | Meaning |
| --- | --- |
| Risky-proposal approval rate | Harmful proposals missed by the monitor, using the declared risk target and resolved reference labels. |
| Safe-proposal rejection rate | Useful opportunities restricted unnecessarily; not every rejection has an equal economic cost. |
| Approval and abstention coverage | Whether apparent safety comes from refusing to decide or rejecting almost everything. |
| Probability calibration/error | Where a monitor supplies risk estimates; report reference-label uncertainty. |
| Unsafe occupancy, onset and duration | How often the system is unsafe, when it first leaves safety, and how long damage persists. |
| Game-native resource health and productive activity | Stock/patch state or maintenance/emission outcomes; do not average unlike physical units across games. |
| Individual and population return | Who benefits or loses, both total and per-step. |
| Checking and intervention cost | Work performed, requests changed and direct costs, with no double subtraction of costs already included in rewards. |

Retain local-pass/global-fail as a diagnostic, not the sole success criterion.
With \(A_t\) denoting all local approvals and \(G_t=\mathbf1[s_t\in K_g]\),
report both \(A_t(1-G_{t+1})\) and onset \(G_tA_t(1-G_{t+1})\). Specify whether
approvals refer to original proposals or executed actions. In a stochastic
game, one bad outcome is not by itself proof that a probabilistic judgment was
miscalibrated or that the action caused the failure.

Use paired initial states, policy contexts and disturbances where appropriate.
Counterfactual rollouts with common random disturbances can isolate the effect
of an intervention. After trajectories diverge, ensuing policy responses are
part of its closed-loop effect. Independent population/selection seeds, not
timesteps, threshold relabels or repeated deterministic seeds, define the main
uncertainty unit. Weather trials are nested within those contexts.

## 6. Execution Order and Compute Gates

| Phase | Work and reason | Compute decision |
| --- | --- | --- |
| 0. Record current evidence | Rebuild the existing audit/analysis; preserve historical results and identify required inputs. | Offline file analysis only; do not regenerate missing results silently. |
| 1. Repair Harvest comparison | Common target, explicit information restrictions, nominal/conservative and uncertainty controls. Unit-test known small states first. | The earlier 512-episode development design is an upper bound: 8 controls x 2 regrowth settings x 4 fresh population contexts x 8 weather seeds. Not a confirmatory sample. |
| 2. Add budgeted judgments | Freeze and serialize states/proposals; evaluate verdicts before outcomes; separate checking budget from execution authority. | Small profiling case: 64 states x 4 proposals x 32 reference rollouts x 5 steps = 40,960 reference transition calls, plus monitor work. Not enough to certify low rare-event probabilities. |
| 3. Transfer to Fishery and Cleanup | Check shared-stock accounting and maintenance under the same protocol. Pin external code and validate policies. | Smoke tests and small paired pilots first. No multi-game training sweep until policy availability and cost are known. |
| 4. Confirm the useful comparisons | Freeze protocol, primary contrasts, contexts and precision targets; evaluate held-out populations. | Choose sample sizes from pilot variation and a meaningful effect/precision target, not from whether hybrid wins. Shard only after profiling and merge tests. |
| 5. Optional breadth | RICE-N for delayed harm; Mushrooms only if a defensible viability target adds a distinct test. | Separate scope decision. Five tasks are not an acceptance requirement. |

For Phase 1, freeze the exact eight controls before execution: no intervention,
fixed cutoff, nominal local, uncertainty-aware local, conservative local,
uncertainty-aware conservative local, nominal joint and uncertainty-aware
joint. Shared-summary messages enter Phase 2 after the basic comparison works.
If a method cannot operate with its declared evidence, fix or exclude it before
running the matrix; do not grant hidden state access for convenience.

Before Phases 1-3, cap the first timing probe at 15 CPU minutes and record time,
peak memory and output size. The previous 10,880-episode checks took about 275
seconds on this machine, but those cheap filters do not predict the runtime of
rollout-based verification or an external JAX game. Estimate the new run from
measured transition cost and the total charged rollout/monitor calls; report
startup and compilation separately. No defensible hour estimate is available
until the new implementation is profiled.

### What can be reused, and what is missing

Existing scripts:

- `experiments/audit_research_evidence.py`: evidence, duplicate and input audit.
- `experiments/validate_harvest_mechanisms.py`: frozen-policy controls and
  held-out search calibration; current safety objectives need matching.
- `experiments/analyze_harvest_validation.py`: completed-check summaries.
- `experiments/check_harvest_policy_sources.py`: numerical-template control for
  saved model-generated policies.
- Existing shard/manifest/merge infrastructure: useful execution patterns,
  not a reason to reuse the old oversized experimental matrix.

Planned, not yet existing, additions are the common monitor/game interface
`fishery_sim/oversight_protocol.py`, a matched runner
`experiments/run_matched_oversight.py`, its analyzer, and observation-leakage,
budget-accounting, common-target and replay tests. Keep the names consistent
with the 17 September plan rather than building a second parallel framework.

Implement a snapshot adapter containing complete simulator state, controller
memory and reproducible disturbance handling. Existing aggregate CSVs cannot
reconstruct arbitrary counterfactual futures. Old logs can support descriptive
tables and some relabelling, but new monitor decisions, unseen policies and
new game dynamics require new simulation.

The first safe command, after activating the repo's existing validation
environment, is an **offline audit**, not an experiment:

```bash
python -m experiments.audit_research_evidence \
  --output-dir notes/research_review/literature_20260921/evidence_refresh
```

This command already exists; it requires the archived local inputs. It was not
run during this literature pass. Do not present commands for the planned
runner as working commands before implementing and testing it.

## 7. Literature Rationale and Remaining Reading

- [Scaling Laws for Scalable Oversight](https://arxiv.org/abs/2504.18530)
  motivates task-specific ability measurement and cautions about dependence
  across sequential oversight steps. It does not validate our budget settings.
- [Safe MARL via Shielding](https://arxiv.org/abs/2101.11196) and
  [Contract-Based Compositional Shielding](https://arxiv.org/abs/2606.14130)
  require serious local and coordinated safety baselines, with assumptions
  explicit. We propose empirical approximations, not reproduced guarantees.
- [Institutional Monitoring and Ledgers](https://www.mdpi.com/2297-8747/31/3/69)
  already studies monitoring/sanction/review in Harvest/Cleanup; actual review
  capacity and richer monitored rules remain a more specific opening.
- [When Local Monitors Miss Compositional Harm](https://arxiv.org/abs/2607.11751)
  directly occupies the broad composition claim and demonstrates the value of
  a stronger local baseline in its controlled example. It is a preprint.
- [SocialJax](https://arxiv.org/abs/2503.14576) supports established environments,
  game-specific metrics and cooperation/defection mixture checks.
- [GovSim](https://arxiv.org/abs/2404.16698) supports task-grounded capability
  probes and population disturbances, but its three scenarios share dynamics.

Before a final novelty statement, read the full methods of Pretorius et al.
(networked common-pool information structures) and Alechina et al. (monitoring
incentives), then inspect the chosen external environment's pinned code,
licence, checkpoints and safety-state access. These are recorded in the reading
log as outstanding, not quietly assumed complete. Numerical safety thresholds,
rollout counts and the protocol above are our proposed choices.

## 8. What Would Make This a Useful First Paper?

The paper should explain when extra shared evidence improves decisions, when a
good local rule is enough, and what either approach costs in checking work and
productive activity. A negative result for shared oversight can still answer
that question. Releasing interchangeable monitors, transparent evidence
budgets, held-out tasks and reproducible inputs makes the benchmark useful to
others; adding five names to a table does not.

Confirm three scope choices with Yali: the defined judgment task plus limited
intervention; the three-game core before delayed-harm expansion; and separate
actor/monitor measurements rather than a scalar gap. The LLM banks can remain
a secondary saved-policy interface until they add difficulty or behavioural
coverage beyond numerical templates. No additional model is needed for these
questions.
