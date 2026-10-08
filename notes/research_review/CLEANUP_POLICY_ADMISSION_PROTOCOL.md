# Cleanup Policy Admission Protocol

Frozen 23 September 2026, before running the new observation-limited policies.
This is a local development admission gate, not a monitor comparison, public
preregistration, training experiment, or paper result. The existing privileged
Cleanup mechanics smoke is prior evidence only. No new literature search is
needed: the inspected pinned native source and `CLEANUP_ADAPTER.md` define the
game, action and observation semantics.

## Question and policy contract

Can a fixed, observation-limited Cleanup population both maintain productive
capacity and harvest, while a maintenance-free variant remains economically
active but damages capacity? Failure of either part blocks Cleanup admission.
The policy interface receives exactly one native per-agent observation tensor,
the agent ID, and explicit serializable per-agent memory. It may use fixed
native action and observation-channel semantics, but no `CleanupState`, joint
observation, global dirt metric, other proposals, future seed or evaluator
feedback. There is no inter-agent communication. The controller proposes one
of nine native actions; it does not exercise monitor authority. If later
monitored, the sole intervention menu remains keep or replace with native stay;
no action can be forced to clean. The evaluator alone reads hidden state and
native rewards after each transition. Test equal-observation/memory decisions
under changed hidden state and other observations/proposals.

Two frozen variants use the same deterministic heuristic and constants:
`productive`: agents 0-3 are cleaners, 4-6 harvesters; `free_rider`: all seven
are harvesters and never propose clean. The latter is not assumed competent or
damaging; both properties must be measured. No learned weights, search,
full-state navigation, reward shaping, intervention or policy adaptation.

## Bounded execution

One attempt only. Native reset seeds 17 and 43 are the two independent episode
contexts; the variants are paired within each. For reset seed `s`, transition
step `t` (zero-based) uses `(s + 10000 + t) mod 2**32` in both variants. There
are 2 variants x 2 contexts x 180 steps = at most 720 native transitions,
one process, CPU backend only. The final 60 steps are the tail. Tests run first.
The native smoke has a 120 CPU-second and 120 wall-second cap including JAX
startup/compilation, 5 MB output cap, no retries or extra contexts. If the
environment, schema, source pin or runtime cap fails, record failure and stop.
No cloud, training, package installation or main-paper edits.

## Metrics and decision

Report each whole episode and tail separately, not pooled pseudo-replicates:
native individual and total return (seven units per apple), apples harvested
(`sum(rewards)/7`), proposed clean actions, coordinate-aligned preexisting dirt
removed, safe steps, safe-to-unsafe onsets, first safe step, final native dirt
fraction and effective spawn probability. Initial unsafe states are recovery
cases. The productive-capacity target is the existing adapter's `K`: effective
apple-spawn probability >= 0.025 (native dirt fraction <= 0.2). The removal
metric can miss dirt spawned and removed in one step. A clean proposal does not
prove actual maintenance. Track elapsed wall/CPU time and transition count.

Admission requires, **in each** of the two contexts: productive variant reaches
`K`, all final 60 states remain in `K`, tail apples harvested > 0, and tail
preexisting dirt removed > 0; free-rider variant harvests > 0 apples over the
episode, proposes zero clean actions, has strictly fewer safe tail steps than
productive, and has no more preexisting dirt removal than productive. Any
missing run, noninterference violation or criterion failure means `not admitted`.
No outcome-dependent threshold/role/seed adjustment or favorable-subset
selection. A successful two-context gate permits only designing a later,
separately authorized monitor study; it supplies no uncertainty interval or
general competence claim. Report all negative conditions and stop after this
bounded smoke regardless of outcome.
