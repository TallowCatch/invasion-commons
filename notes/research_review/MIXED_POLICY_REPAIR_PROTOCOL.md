# Mixed-Policy Coverage Repair

23 September 2026. Prospective **development repair** after the held-out
`useful` and `stress` policy pilot showed class separation in Harvest. The
earlier 160 episodes and their outputs are preserved. This follow-up is not
an untouched confirmation and cannot be pooled with the first pilot as if it
were one preplanned sample.

## Question and fixed policy source

Can policy populations with heterogeneous extraction behavior yield both safe
and risky original requests from initially safe states during full episodes?
If so, do strong local and joint monitors differ in harmful approvals, safe
rejections, resource outcomes, retained extraction, return or checking work?
No monitor win is required. Missing either class remains a reported failure
of the coverage gate, not permission to pick different seeds.

Use six agents. Two fixed composition strata are `mix2` (two aggressive, four
cooperative) and `mix4` (four aggressive, two cooperative). Select the agent
slots by one seeded permutation per population context. Harvest uses the
existing cooperative/adversarial structured strategy generators. Fishery uses
the previously declared useful/stress threshold-policy fraction ranges and
the same seeded slot assignment. Composition is not a scalar ability measure.
The policy generators are not fitted to the observed monitor results.

## Matrix, endpoints and resources

Four new population contexts per stratum, independent from both prior pilots.
Harvest uses base and 0.85-regrowth settings, two weather streams per context,
80 steps maximum, and the same four methods: no intervention, weather-aware
local, weather-aware conservative local, and weather-aware joint. Fishery uses
the same four applicable deterministic methods. Total: 128 Harvest and 32
Fishery episodes. All methods share the target, transition model, five-scale
intervention menu and five-candidate limit. They see paired policy parameters
and disturbances within each context/weather setting.

Every original proposal on a no-intervention trajectory is frozen before
monitor comparison. Fishery labels use exact one-step transitions. Harvest
labels use the same 128-draw, 0.05-risk, Wilson95 reference; unresolved cases
are retained. The primary engineering gate is at least one safe and one risky
resolved proposal from initially safe states in each game/stratum/regime.
This is only minimum diagnostic coverage, not statistical power. Report raw
counts and decisions separately by stratum, game and starting safety.

Closed-loop endpoints: unsafe occupancy, onset, resource health, return,
realized extraction and per-step component checks. Frozen decisions: harmful
approvals, safe rejections, abstentions and distinct inspection/communication
cost proxies. Pair closed-loop outcomes by population context after averaging
the two nested weather streams. Four contexts yield descriptive differences
and ranges only, without p-values or population confidence intervals. Do not
pool game units or treat timesteps, weather trials or repeated policies as
independent contexts.

One worker, 160 episodes maximum, 900 wall/CPU seconds, 250 MB output cap, no
training, cloud or model calls. One fixed run after tests, with one engineering
retry only for a code defect in a new directory. Record source, protocol,
config and result hashes. Stop after this coverage/interpretation check. A
separate confirmatory design would require more contexts and a precision target;
reviewer-resource scaling and Cleanup policy admission remain separate gates.

## Engineering amendment before the one permitted retry

The first fixed-design run completed, but an audit found one identical Harvest
state/proposal pair in two weather traces with different Monte Carlo reference
labels (safe versus unresolved). The reference RNG seed was keyed by trace ID.
The retry will key reference draws by physical case content, excluding the
episode weather seed. This does not select, add, remove or alter policy contexts,
proposals, monitors or outcomes; it makes duplicate reference cases receive
the same risk estimate. The first run stays untouched at
`results/runs/mixed_policy_repair_v1/`. Also correct the generated record's
protocol filename. Retry once in a new directory, then stop regardless of
which monitor benefits. The original source snapshots and output remain for
audit. The correction is an engineering amendment, not a fresh confirmation.
