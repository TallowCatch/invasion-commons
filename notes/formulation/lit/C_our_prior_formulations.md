# Our own earlier formulations (June to November 2026)

Read on 23 Nov 2026 for Phase 1. Ideas from these drafts should not be lost when we write the new definition.

## June and July 2026: the overseer as an oversight *architecture*, with a capability gap

Sources:
- `notes/private/yali_formulations/scalable_oversight_formulation_yali.tex` (17 Jun);
- `scalable_oversight_commons_problem_method_note.tex` (24 Jul).

Both were written for Yali.

**The problem as stated then.** In a shared environment, every agent's action can pass a local check while the agents' *joint* effect still makes the shared system unsafe. Oversight can fail at the system level even when it passes at the level of single actions.

**The model.**
- **Agents.** N agents. The state holds each agent's local resource and system-level quantities (mean patch health, the fraction of failed patches). Each agent requests a harvest fraction.
- **Oversight architecture** m ∈ {none, local, global signal, hybrid}. The architecture changes the transition and the rewards.
- **Overseer capability** as a tuple o = (detection recall q, enforcement delay d, targeting capacity κ, intervention cost β).
- **Actor capability** c, set by how strong the strategy generator is (mutation vs search).
- **The capability gap** Δ = rank(c) − rank(o). Positive Δ means the actors are stronger than the overseer. This followed Bowman et al. (2022) and Engels et al. (2025).
- **Local and global safety.**
  - A local predicate: one action is within the sustainable fraction plus a margin.
  - A global predicate: mean patch health is above a threshold and the failed fraction is below a threshold.
- **The key failure event: local-pass / global-fail (LPGF).** Every agent's action passes locally, yet the next state is globally unsafe. It is reported as a rate over the episode.
- **Evaluation.**
  - E(m, c, o) = welfare, the unsafe rate, LPGF, final resource health, and the oversight burden.
  - All of these are tracked as the gap Δ grows.

**Evidence at the time.** A Harvest-Commons pilot: 72 condition rows, 18 cells. Hybrid oversight ranked first in 13 of 18 cells, and protection cost welfare and burden. A threshold sweep (9,000 runs) supported it. An LLM bridge with Qwen and Llama 3B generated strategies offline.

**What was strong.**
- It is agent-centric. The overseer is defined by what it can see (local vs global information), how fast it acts (delay), how many it can target (capacity), and how good its detection is (recall).
- The capability gap Δ is explicit, which is the scalable-oversight question in the literature's own terms.
- LPGF names a failure that only exists in *multi*-agent systems: harm that no single-agent check can see. This links directly to:
  - Makins et al. 2026, where attacks split across agents evade per-agent monitors;
  - Clean Up, where pollution builds up slowly;
  - the gap between our rule (local apple density) and the real harm (the orchard dying).

**What was weak.**
- The environment was our own stylised commons, not SocialJax.
- The agents were generated strategies, not learners.
- Overseer capability bundled four knobs into one ladder, and was never separated.
- Δ was a difference of ranks, not a measured quantity.

## October to November 2026: the budgeted inspector

Sources:
- `notes/phd_plan/2026-10-27_calibrated_overseer_plan.md`;
- `2026-11-21_general_formulation.md`;
- `2026-11-23_decentralised_overseer_formulation.md`.

**The model.**
- Each step the overseer inspects at most k of N agents, reading the last w steps of each record.
- Detection has a miss rate α and a false-alarm rate β. A detected agent receives a sanction F.
- Regional overseers (M of them) cover parts of the map, with optional messages between them.
- Claims C1–C4: when breaking the rule stops paying, the cost to compliant agents, when targeting helps, and violations a check can't see.
- A locality number for when splitting the overseer works.

**What was strong.**
- It runs on SocialJax with learning agents.
- It is cheap to vary k, α, β, M and the inspection rule.
- It produced clean empirical regularities (tracker findings 1–25).

**What drifted.**
- The language became economics: "deterrence", "expected fine", Becker.
- The June ideas dropped out: the capability gap Δ, the local/global distinction, LPGF. The overseer's capability shrank to two numbers (α, β).

## What the new definition should keep from each

| From June/July | From Oct/Nov |
|---|---|
| The overseer described by its observation (local vs global information), its delay, its capacity and its detection quality | The budget of k inspections out of N, and the inspection window w |
| An explicit capability gap between overseer and agents | Detection quality measured, not assumed (the LLM ladder; patterned misses) |
| The local-pass/global-fail failure, the multi-agent part of the problem | Several overseers sharing a budget, and what they communicate |
| Several architectures compared (none, local, global, hybrid) | Learning agents in SocialJax as the test population |
| Evaluation as a vector: welfare, unsafe rate, burden | Bounds: no overseer, an all-seeing overseer, worst-case agents |

Open question for Phase 2: the June framing put the hard part in *what the overseer can see* (individual actions vs the system's state). The November framing put it in *how many agents it can inspect*. The new formulation should probably have both. An overseer has an observation budget (whom and how much it can inspect) and an observation scope (individual records vs system-level signals), and harm can live at either level.
