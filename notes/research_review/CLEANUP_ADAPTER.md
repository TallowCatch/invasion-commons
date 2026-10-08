# Bounded External Cleanup Sidecar

21 September 2026. Integration and mechanics validation only, **not a
policy-ready benchmark or a local-versus-joint oversight result**. This sidecar
does not import or modify the main oversight protocol or any Harvest file.

## Provenance and Installation

The [primary paper, arXiv:2503.14576v3](https://arxiv.org/html/2503.14576v3)
links directly to [cooperativex/SocialJax](https://github.com/cooperativex/SocialJax).
The paper was read through its full-text ending sentinel
`END-OF-PAPER:6bc677c77ff5`; source code, not prose alone, determines this adapter's
semantics. No paper training result is reproduced here.

- Pinned commit: `9df972d07657c7e2d4cd4fec1ed1f437099cb128`, 21 August 2026,
  commit subject `fix agents numbers`.
- [Pinned license](https://github.com/cooperativex/SocialJax/blob/9df972d07657c7e2d4cd4fec1ed1f437099cb128/LICENSE):
  Apache License 2.0. The external checkout retains its license; no upstream
  simulator code, images, or checkpoints are vendored into this repository.
- [Pinned Cleanup source](https://github.com/cooperativex/SocialJax/blob/9df972d07657c7e2d4cd4fec1ed1f437099cb128/socialjax/environments/cleanup/clean_up.py),
  SHA-256 `16bc84c3eecd9fd98f73b781f6f7fc4dc9497e1551b47d646e80e1477a8ae73a`.
  License SHA-256: `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`.
- The current revision includes movement/respawn fixes. Upstream's README says
  its `socialjax_v1.1.0` release reproduces the paper's training curves; later
  agent-overlap fixes can change dynamics. This sidecar deliberately pins the
  inspected newer revision, not a paper-baseline reproduction.

The upstream Poetry project is named `socialjaxproject` but lacks a matching
package declaration. A direct pinned VCS `pip install --no-deps` was attempted
and failed with `No file/folder found for package socialjaxproject`. Do not
silently substitute the unrelated PyPI package or patch upstream. Use its
documented checkout/import approach. The dependency URL and revision are also
recorded in `requirements-cleanup.txt`; its active entries install CPU runtime
dependencies, not the broken upstream wheel.

From the repository root, using Python 3.10-3.12:

```bash
python3.12 -m venv /tmp/commons-cleanup-env
/tmp/commons-cleanup-env/bin/python -m pip install -r requirements-cleanup.txt
git clone https://github.com/cooperativex/SocialJax.git /tmp/commons-cleanup-socialjax
git -C /tmp/commons-cleanup-socialjax checkout --detach 9df972d07657c7e2d4cd4fec1ed1f437099cb128
export SOCIALJAX_SOURCE=/tmp/commons-cleanup-socialjax
export JAX_PLATFORMS=cpu
/tmp/commons-cleanup-env/bin/python -m pytest -q tests/test_cleanup_oversight.py
/tmp/commons-cleanup-env/bin/python -m experiments.smoke_cleanup_oversight \
  --steps 300 --seed 17 --cpu-seconds 180 \
  --output-dir /tmp/commons-cleanup-smoke
```

Reuse an existing checkout rather than cloning over it. No global packages,
CUDA extras, training frameworks, paid services, or model calls are required.
The adapter checks the source HEAD and rejects modifications/untracked files
under `socialjax`, verifies the imported module path and action enum, and fails
with `CleanupUnavailable` if source/dependencies are missing. Importing the
sidecar itself requires only the standard library. Native tests skip only when
`SOCIALJAX_SOURCE` is unset; explicitly opting in makes dependency failures test
failures. The smoke exits 2 with a machine-readable unavailable result.

## Exact Contract and Target

`CleanupAdapter(CleanupConfig(horizon=300, minimum_spawn_fraction=0.5))` exposes:

```python
state = adapter.reset(seed)
next_state, native_rewards, done = adapter.transition(state, actions, seed)
is_productive_capacity_state = adapter.safe(state)
snapshot_bytes = adapter.snapshot(state)
restored_state = adapter.restore(snapshot_bytes)
```

Seven agents, the unmodified native default map, nine native actions, individual
rewards (`shared_rewards=False`), no shaping. Native physics is always
`Clean_up.reset` / `Clean_up.step_env`, never a local reimplementation. Action
order is native agent order 0 through 6. Seeds are explicit unsigned 32-bit
integers; there is no hidden RNG stream. The adapter is host-side, not a
JIT/vmap-compatible training wrapper.

**Horizon:** upstream `step_env` resets the state and zeroes rewards at
`num_inner_steps` (source lines 1278-1303), including when bypassing the base
class step wrapper. The adapter sets that native limit to `horizon+1`, stops at
`horizon`, and returns the last physical state/reward. `done` means this explicit
finite-horizon cutoff, not ecological failure. Stepping a terminal state raises;
reset must be explicit. This changes episode bookkeeping only, not the
pre-reset native transition equations.

**Target:** let `d` count `Items.dirt` in the native
`potential_dirt_and_dirt_label`, and let
`R = len(potential_dirt_and_dirt_locs) + len(env.RIVER)`. Do not count just visible
grid dirt, because agents can obscure cells. The pollution fraction is `f=d/R`.
The source's growth rule (lines 1037-1051) implies effective spawn probability
for an eligible empty orchard cell:

```text
p(f) = 0.05 * max(0, min((f - 0.4) / (0.0 - 0.4), 1))
K    = { s : p(f(s)) >= 0.025 } = { s : f(s) <= 0.2 }
```

The half-maximum margin is our declared design choice, not a literature-derived
safety guarantee. The source only upper-clips its interpolation; negative
thresholds also yield zero actual spawning under its `uniform < threshold`
test. The diagnostic formula lower-clips for this reason. `safe` says there is
productive *capacity*, not that apples were collected, that the state is
invariant, or that future policies will maintain it. The map's orchard is
nonempty, but current apple count is not a condition of membership in `K`.
Report harvesting and maintenance beside `safe`, so idle capacity cannot count
as productive policy competence. A different margin is a different contract.

Apple regrowth uses pollution at the start of a step, before new pollution,
movement, and cleaning. Thus cleaning affects subsequent spawning, not the
already-sampled apples in that step. The native initial map contains dirt even
though the paper describes an initially uncontaminated river. Initial states
outside `K` are recovery cases; safe-to-unsafe onsets are counted separately
from unsafe occupancy. New dirt spawning is gated by `inner_t > 50`, not
`>= 50`. Noop does not suspend that process.

**Rewards and diagnostics:** pinned individual mode pays seven native units
per apple (source lines 1251-1257), rather than the prose's unit reward. The
adapter returns those rewards unchanged. Smoke apple counts are
`sum(native_rewards)/7`; neither `state.apples` (static potential coordinates)
nor cumulative int8 inventories are used as harvest counters. Upstream
`cleaned_water` / `waste_cleared` uses a pre-step clean-cell stock, not actual
cleaning events. Our `preexisting_dirt_removed` counts coordinate-aligned dirty
cells before a transition that are no longer dirty after it. It is a lower
bound on total removal: dirt spawned and removed in the same step is missed.
Sorting of the upstream location/label arrays is respected.

## Authority, Evidence and Replay

`replace_with_noop(proposals, rejected_agents)` is the complete common
intervention menu. Each output equals the original proposal or native
`Actions.stay` (6). Rejection cannot generate `zap_clean` (8), move an agent,
edit resource state, change rewards, or freeze environmental dynamics. A noop
is not an assumed safety action. Monitors may suppress requested maintenance,
and may have no feasible remedy when no agent proposes maintenance.

`local_evidence` returns precisely one native observation tensor plus own agent
ID and own proposal. It does not include other proposals, a global pollution
summary, positions, future seeds, or the simulator state. This is a principled
evidence boundary, **not an implemented local safety rule**. The native tensor
semantics, including any native quirks, are retained without augmentation.
`full_state_evidence` is explicitly privileged evaluator/reference evidence;
it is not equivalent to pooling local observations. Only evaluator code calls
`metrics`/`safe` on hidden state. No local-versus-joint performance comparison,
conservative hidden-state bound, shared-summary protocol, risk estimator, or
monitor selection algorithm is claimed in this sidecar.

Snapshots are canonical JSON bytes containing every native dataclass field,
the native observations, dtype/shape/base64 array bytes, exact configuration,
source revision and key runtime versions/settings. They use no pickle.
Restoring rejects schema/configuration/runtime mismatches. Fresh adapters may
perform one native reset to obtain the array schema. Replay also requires the
explicit future seed sequence and policy proposals. The smoke controller is
stateless; future recurrent policy memory would need its own serialized record.
Repeated snapshots are byte-identical on the checked runtime. Cross-platform
floating-point/JAX reproducibility is not promised merely by recording a SHA.
Snapshots are experiment artifacts, not a cryptographically authenticated
format for adversarial inputs.

## Bounded Evidence and Policy Gate

The smoke uses paired native reset states and transition seeds for two
mechanics probes: noop, and a deterministic full-state script assigning four
agents to greedy cleaning poses and three to nearest-apple harvesting. The
script uses breadth-first paths and native beam geometry, no simulator
lookahead or training. Its privileged proposals are not monitor interventions,
and it is not a decentralized or learned baseline. Noop is only a negative
mechanics control, never a benchmark policy.

Predeclared mechanism gate: at least 200 episode steps, all final 100 states in
`K`, positive harvesting in that tail, and positive pre-existing dirt removal
in that tail. Reset/step/snapshot parity is checked separately. Passing this
small gate establishes observed mechanics and scripted activity, **not policy
competence for matched oversight**. Every report hard-codes `policy_ready=false`.

Still required before any benchmark admission:

- Frozen observation-limited maintenance and harvesting policies, validated
  together from native resets and on held-out policy contexts/seeds.
- Competent productive free-riders and damaging proposals, with individual
  and population return, maintained capacity, and maintenance measured.
- Matched evidence/authority/work budgets, defensible bounds on hidden
  pollution and actions, and independent risk-label uncertainty.
- Controller memory snapshots for any non-stateless policies and reproducible
  continuation policies for multistep risk estimates.

The inspected tree includes `checkpoints/clean_up_seed30_reward_individual.pkl`,
but it has not been deserialized, trusted as a competent maintenance policy, or
evaluated. No compatible, validated cooperative policy pair was established.
The fixed Cleanup policy supplied upstream is random. No training or sweep is
authorized by this work.

The CLI caps a probe at 500 steps per controller and at 900 CPU seconds, with a
default 180-second CPU limit and wall alarm. CPU accounting includes worker
threads; compilation is timed separately from the warm rollout. The total
local runtime-trial budget for this implementation is 900 CPU seconds, not 900
seconds per retry. Report JSON includes measured CPU/wall time, peak RSS,
transition/reset counts, initial/final snapshot hashes, and full per-step
traces. Runtime probes use `/tmp`; completed outputs can be preserved under
the repository's ignored `results/runs/` directory, without rerunning them.

### Recorded Run

Completed on 22 September 2026, using the pinned isolated CPU environment.

- Native and contract tests: **27 passed**, in 6.69 seconds.
- Smoke: 300 steps per controller, seed 17, 603 native transition calls including
  parity checks; **10.04 seconds wall time / 11.13 seconds CPU time**. Peak RSS
  was 324,222,976 bytes. No training or sweep was run.
- Noop: 300/300 states outside the productive-capacity target, zero apples
  harvested. The initial state was already outside the target, so this is
  persistent failure to recover, not 300 newly caused failures.
- Scripted full-state controller: reached the target at step 19; all final 100
  states passed. It harvested 655 apples in total, including 240 in the final
  100 steps, with 51 pre-existing dirty cells removed in that tail.
- Reset, step, snapshot, native-transition parity and input-immutability checks
  passed. The declared mechanics gate passed; **policy_ready remains false**.
- Upstream emits a JAX int32-to-int16 scatter FutureWarning. It does not fail
  the pinned runtime, but upgrading JAX requires rerunning native tests.

Original outputs: `/tmp/commons-cleanup-smoke/`. Preserved byte-for-byte under
`results/runs/cleanup_oversight_smoke_20260922/`, with adapter, smoke script and
requirements snapshots in `source/`. `report.json` contains the exact timing,
individual returns, traces, runtime contract and snapshot hashes. This result
establishes working mechanics, not a comparison between oversight methods.
