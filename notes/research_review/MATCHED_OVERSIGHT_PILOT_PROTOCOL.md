# Matched Oversight Development Pilot

21 September 2026. Frozen before the 512-episode development pilot. The initial
12-step smoke test checks software plumbing only. This is a local prospective
protocol, not a publicly registered or confirmatory study.

## Question and Predictions

Does joint-request information reduce unnecessary restriction relative to
pooled local predictions, when both methods use the same safety target,
weather model and action-reduction menu?

1. With known transitions, joint predictions should avoid the extra restriction
   induced by worst-case assumptions about hidden neighbours. The test can fail
   to show a useful return improvement if the coarse menu or chosen states make
   this information irrelevant.
2. Ignoring weather can produce unsafe approvals near the safety boundary.
   Allowing for weather should reduce them, potentially restricting more useful
   extraction. This is not a prediction that welfare must always fall.
3. A shared-stock accounting game should require less detailed information than
   a spatially coupled resource game: total demand is sufficient for Fishery's
   joint one-step prediction. No analogous aggregate sufficiency is assumed for
   spatial Harvest or Cleanup.

These are directional development predictions. No p-value threshold, winner,
minimum significant effect, or result-dependent stopping rule is used. The aim
is to validate the measurement and identify a useful confirmatory contrast.

## What Is Held Fixed

Harvest uses `forest_co_management` internally, with its base and existing
0.85-regrowth settings. All monitors use the same global mean-health and
failed-patch-fraction predicate, not a different per-patch threshold. Both
settings keep that target fixed. Policies, starting states and weather streams
are paired. Communication, side payments and legacy governors are disabled.

Every monitor chooses one uniform request multiplier from `[1, .75, .5, .25, 0]`.
The first predicted-safe option is selected. The finite menu is shared; its
resolution is a limitation, not a separately tuned advantage. An unsafe zero
action is recorded as infeasible. A budget-exhausted fallback is never certified
as safe.

**Local-report scope:** each Harvest reviewer observes only its own health and
request and predicts its own next patch. A public combiner applies the global
predicate to these reports. Local nominal reports omit neighbour extraction;
local conservative reports bound both neighbours' extraction using the common
scale and maximum possible request. Neither accesses actual neighbour actions.
Joint predictions use actual joint requests. This is an information-abstraction
comparison with pooled reports, not fully decentralized autonomous control.

The weather allowance is `sigma * Normal.isf(delta/n)` with `delta=.05`, a
one-step Gaussian union bound. It bounds weather only; it does not repair omitted
neighbour damage in the optimistic local model. Conservative and joint models
also account for those effects. The prediction bound assumes the simulator's
known noise model and monotone regeneration (`0 <= rate <= 1`).

Fishery uses the existing `FisheryEnv`, with six users, stock 70/100, regrowth
0.7, maximum withdrawal 6, danger threshold 10 and the existing collapse
patience. These are declared pilot choices, not externally calibrated values.
Its local conservative baseline is an equal-share sufficient contract. The
joint method uses total withdrawals. No weather variants or repeated weather
seeds are invented for this deterministic game.

## Units, Matrix and Outcomes

- Harvest: eight methods x two regrowth settings x four new policy contexts x
  eight weather seeds = **512 episodes**, maximum 80 steps each.
- Fishery: four applicable methods x four new policy contexts = **16 episodes**.
- The independently generated policy context is the uncertainty unit. Weather
  trials are nested; states and agents do not add independent replication.
- Actor policies are sampled threshold policies, not searched or learned during
  this pilot. This does not validate a capability ordering or a capability gap.
- Frozen decisions use no-intervention states at steps 0, 5, 15, 30, 50 and 70
  when present, with requests multiplied by 1, .5, .25 and 0 to include less
  demanding proposals. Every monitor sees identical proposals and reference risk.
  Already-unsafe states are reported separately from initially safe states.
- Harvest proposal risk uses 128 independent one-step weather samples and 95%
  Wilson intervals. A reference interval crossing .05 is unresolved, not safe.
  Fishery labels use its exact deterministic transition. Neither is a long-run
  safety certificate.
- Primary outputs: risky proposals accepted, safe proposals rejected, unresolved
  share and coverage; unsafe occupancy/onset, return, resource health, prevented
  harvest, checks performed, and infeasibility.
- Context means and paired contrasts use exploratory 95% t intervals. Four
  contexts are insufficient for strong tail/generalization claims. No
  multiplicity-adjusted significance claims will be made.

Primary contrasts are joint nominal minus local nominal; joint uncertainty-aware
minus conservative uncertainty-aware local; and uncertainty-aware conservative
local minus conservative local. Joint nominal minus conservative local is also
reported for the deterministic transfer. Return and safety are separate outcomes.

Model component evaluations and transmitted scalars are explicit cost proxies.
They are not measured general intelligence, monetary costs, or hardware-normalized
FLOPs. The candidate budget limits action-repair search; a later experiment is
required to vary judgment precision or evidence access itself.

## Execution and Failure Rules

Run unit tests, then the small smoke profile. Check summaries, figures, exact
prediction/native-transition agreement, manifest mismatch handling and local
MLflow logging before the pilot. Run only one local worker. Cap a pilot
invocation at 900 seconds between atomic episode blocks. No cloud, GPUs, LLMs or
paid services. The maximum work is 528 episodes, not an open-ended sweep.

On timeout, retain completed blocks and report partial status. Resume only the
same protocol/code; code changes require a new output directory. Corrupt blocks
fail analysis rather than being excluded silently. Do not change conditions to
manufacture failures or a hybrid winner. No new large batch follows automatically.

Cleanup integration is a separate adapter/feasibility gate in this turn, using
its real native implementation. It must demonstrate reproducible transitions
and useful policies before any cross-monitor outcome claims. RICE-N is deferred
until delayed consequences become the specific unanswered question.

## Commands

```sh
.venv/bin/python -m pytest -q tests/test_oversight_protocol.py
.venv/bin/python -m experiments.run_matched_oversight --profile smoke --output-dir results/runs/matched_oversight_v1_smoke --mlflow
.venv/bin/python -m experiments.run_matched_oversight --profile smoke --output-dir results/runs/matched_oversight_v1_smoke_b --mlflow
.venv/bin/python -m experiments.run_matched_oversight --profile pilot --output-dir results/runs/matched_oversight_v1_pilot --mlflow
```

The environment installs MLflow into a repo-local virtual environment, using
existing system scientific packages without changing them. Actual runtime
versions are recorded in the manifest. Tracking is explicitly local SQLite and
local artifacts, regardless of inherited MLflow environment settings.

## Development Amendment Before Pilot

The first plumbing smoke used only original requests. Added the four fixed
proposal scales above before running the development pilot, so safe-rejection
measurement is not restricted to whatever requests happen to appear on an
unregulated trajectory. These are counterfactual proposals, not four independent
populations. The initial smoke is preserved in its own directory; the amended
smoke uses `_smoke_b`. This does not change the closed-loop episode matrix.

## Post-Run Engineering Note, 22 September 2026

The declared 528-episode pilot is complete. The runner was subsequently fixed
so resuming an already completed run verifies its inventory/checksums without
recomputing frozen judgments or overwriting the original timing record. New
runs also checksum the frozen-judgment file. A regression test covers both
idempotence and corruption detection. No simulation or scientific rule changed.

The original pilot's data, source snapshots, manifest and timing were preserved;
all 528 episode checksums were reverified. The original frozen judgments have
a recorded row count, not the newly added checksum. Since runner source hashes
changed, the historical command/directory above now correctly fails the
manifest guard with current source. Use the analysis-only command for existing
results; a future run must use a fresh directory and a new protocol. See
`MATCHED_OVERSIGHT_IMPLEMENTATION_CLOSEOUT.md` for the interpretation and next gate.
