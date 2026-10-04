# Small validation run: decisions made before inspecting results

This pass keeps Harvest as the main controlled environment. Fishery remains a separate, simpler supporting study. Neither environment name implies field validation of fisheries, agriculture, or infrastructure. The general question is how individual resource use and limited intervention affect a shared system over time.

## Why these checks

The benchmark family comes from [sequential social dilemmas](https://arxiv.org/abs/1702.03037): cooperation depends on policies and repeated interaction. [Multi-agent shielding](https://arxiv.org/abs/2101.11196) motivates comparing a local decision rule with a rule that sees joint actions. Our simple filters are original benchmark controls, not reproductions of that paper's formally guaranteed shields. [Scalable oversight scaling work](https://arxiv.org/abs/2504.18530) measures performance on oversight tasks; our previous preset-rank subtraction does not inherit that justification. [GovSim](https://arxiv.org/abs/2404.16698) already studies resource-use LLM societies, so successful model-policy generation alone is not the new scientific result.

## Check 1: keep the policies fixed and change the intervention

Use final-generation no-governor policies from five Stage A runs for each of two generation processes and two stress settings. Restore the original six-agent order from previous-generation selection and verify every position against the per-agent archive. Use eight new weather seeds per population, shared across all mechanisms. These are exploratory comparisons conditional on twenty existing populations, not twenty independent populations per condition.

Compare no intervention, communication alone, communication on/off crossed with uniform/neighbourhood enforcement, a fixed local cutoff, a local state-based filter, an announcement without enforcement, and a joint-action reference. Disable credit transfers throughout. Cap packages share q=0.7, delay=1, target fraction=0.5 using floor rounding. Monetary intervention price is zero for every condition; compare prevented extraction, intervention incidence, and returns. The local filters/reference have full filtering authority and are diagnostic comparators, not matched-cost alternatives to the constrained governor.

The local state filter chooses the largest allowed harvest whose own-patch, zero-weather regrowth prediction reaches health 10. It cannot see neighbours' current actions and omits their spillover losses. It has access to local state, but does not use neighbour mean as a substitute for unobserved actions. The joint reference uses all current requests and patch states, models spillovers, and scales joint harvest down until predicted global safety passes. Both are one-step, zero-weather predictors. Neither claims safety under random weather. If even zero harvest cannot meet the prediction, they choose zero and the episode remains in the analysis.

For no intervention and the three filters, also remove weather, spillovers, or both. No thresholds or stress parameters will be adjusted to obtain a positive result. The base test uses each scenario's stated configuration, not Stage A's mixture of four held-out regimes. It must be reported separately.

Expected count: 3,520 episodes, including the joint reference and controls. Save pre/post resource states, requests, allowed fractions, actual harvest, local-pass flags, failure onset/persistence, returns, duration, cap announcements, and executed targets. Original-request LPGF and allowed-action compliance are different diagnostics and must not be substituted for one another. A clean-prefix event requires all allowed actions to have passed the fixed cutoff up to and including the onset step.

Report means within a source population first, then uncertainty across the five source runs in each setting/process. Do not count weather seeds, repeated mechanisms, or archived inactive-overseer copies as independent evolutionary runs. Use paired differences. The confirmatory sample size is a later decision; this run cannot guarantee a publishable or significant effect.

A numerical buffer of 1e-8 resource units is added to both predictive filters' target, before their runs, to avoid floating-point rounding at the exact boundary. It does not change the logged safety definition. The calibration runner at launch is archived with its outputs; this buffer changes only the filters, which calibration does not call.

## Check 2: does extra search help on unseen episodes?

For each stress setting, generate twelve fresh balanced parent/opponent populations. Rotate the entrant position over the six positions. Create twelve mutations once and use nested candidate subsets of size 1, 6, and 12. Select using either a 30- or 60-step internal evaluation. Test on sixteen separate shared seeds at horizon 80, without oversight or communication.

The objective is entrant total payoff minus the existing 25-point collapse penalty. Report raw entrant payoff, other-agent return, global safety and health too. Candidate count and search horizon stay separate. Better objective performance supports a within-task search-quality claim, not a calibrated actor-overseer gap. No monotonic increase in resource damage is required or assumed.

Expected held-out count: 2,304 episodes, plus 576 short candidate evaluations. Report paired differences across the twelve parent contexts, preserving weather-seed pairing. K=1 is an explicit no-selection control, not a request to the existing injector that clamps K to two.

## Check 3: what does the model add beyond the supplied policy numbers?

Compare each saved model bank with policies made directly from its prompt anchors, using the same policy indices and positions. Use both stress settings, exploitative shares 0/0.5/1, no oversight and the common uniform-cap mechanism, eight sampled populations and four weather seeds. This gives 1,536 episodes with no model calls. Conditions follow the new matched-test setup, including disabled credit transfers, rather than silently claiming an exact repeat of the old bank evaluation. Differences are conditional on the two fixed banks; sampled populations do not establish model-family generalization.

## Targeted follow-up after the base-setting screen

The base-setting screen had very few unsafe episodes. Existing Stage A logs, separated by held-out regime, showed the largest unsafe rate under the already-defined slow-regrowth condition (28.31% without oversight, averaged across original generation/treatment cells). Repeat the same frozen-policy comparison and controls under that exact existing regime. Keep the same policies, seed schedule, cutoff, and predictor definitions. This adds 3,520 episodes. It is an exploratory, evidence-guided follow-up, not an independent confirmation or a newly tuned stress setting. Preserve the base-setting result and report both. The earlier runners are archived alongside their completed outputs before adding this CLI option.

## Compute and integrity

Run these as resumable terminal processes on CPU. There are no model/API calls or paid remote resources. Each completed block is written atomically; manifests pin settings and source hashes and reject mixed-version resumes. Check counts and tests before interpreting output. Preserve old runs unchanged. Make a single results-and-next-step note when the jobs finish, rather than adding more speculative plans.
