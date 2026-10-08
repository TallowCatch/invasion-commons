# Harvest Scalable-Oversight Pilot Direction

## One-Sentence Spine

This work evaluates oversight architectures in a shared-resource multi-agent benchmark as actor capability rises relative to overseer capability.

## Main Question

The pilot asks how local oversight, a global oversight signal, and hybrid oversight behave when stronger agents act in a shared-resource system and the overseer has limited detection, limited capacity, and delayed response.

The immediate aim is empirical. The pilot should show whether safety outcomes change as the actor becomes stronger than the overseer, and whether local checks can miss failures that only appear at the system level.

## Role of Fishery and Harvest

Fishery is the simpler foundation. It is useful for testing central interventions such as quotas, sanctions, and closures under strategic turnover.

Harvest is the main pilot substrate. It has local interactions, shared ecological dynamics, and multiple oversight architectures. That makes it the right environment for the scalable-oversight pilot.

The next paper should foreground Harvest. Fishery can remain supporting evidence for the general commons benchmark line.

## Oversight Language

The public-facing terms for this pilot are:

- `bottom_up_only` becomes local oversight.
- `top_down_only` becomes global signal.
- `hybrid` becomes hybrid oversight.
- `none` remains no oversight.

The internal names stay in the code and CSVs for reproducibility. Figures and paper text should use the public-facing terms.

## Stress Settings

The previous literature-backed scenario presets are still useful, but they should not dominate the scalable-oversight story.

- `community_irrigation` is displayed as moderate-coupling commons.
- `forest_co_management` is displayed as high-coupling commons.

This keeps the figure interpretation simple: the pilot compares oversight under different levels of interaction stress.

## Actor Capability

Actor capability is operationalized through how strong the entering strategies are.

- Low actor capability uses ordinary mutation entrants.
- Medium actor capability uses search-generated entrants with a smaller search budget.
- High actor capability uses search-generated entrants with a larger search budget and longer internal evaluation.

A search-generated strategy is not a live LLM agent. It is a structured Harvest strategy selected from many candidate variations because it performs well. This makes the pressure stronger while keeping the experiment inspectable and reproducible.

## Overseer Capability

Overseer capability is operationalized through detection and intervention limits.

- Strong overseer has perfect detection, no delay, full targeting capacity, and no budget cost.
- Limited overseer has imperfect detection, one-round delay, limited targeting capacity, and a small budget cost.
- Weak overseer has lower detection, longer delay, tighter capacity, and higher budget cost.

The capability gap is the actor rank minus the overseer rank. Positive values mean the actor is stronger than the overseer under this pilot definition.

## Local and Global Safety

The local safety predicate checks whether each agent's requested harvest is locally acceptable.

The global safety predicate checks whether the overall Harvest system remains safe after the joint action. It uses mean patch health and failed-patch fraction.

The key new failure mode is local-pass/global-fail. This occurs when local action checks pass, but the aggregate system becomes unsafe. This connects the commons benchmark to scalable oversight because locally acceptable steps can compose into system-level failure.

## Why LLMs Are Deferred

LLM-generated strategies are the next extension, not the first scalable-oversight pilot.

The first pilot needs a clean capability variable. Search budget gives that immediately: actor capability can be increased while the strategy format stays fixed. Once this pilot separates, LLMs can be added as another strategy source through a structured strategy bank.

This avoids turning the first pilot into a model leaderboard or a live-prompting study.

## Role of GitHub Actions

GitHub Actions is only remote compute infrastructure. It is used to run fixed-seed matrix sweeps without overloading a laptop.

The paper should describe this as scripted, reproducible experiment sweeps. It should not present GitHub Actions as a scientific method.

## First Pilot Must Show

The pilot is useful if at least one safety metric changes with actor capability, overseer capability, or their gap.

The strongest outcome would show that global unsafe rate or local-pass/global-fail rate increases when actor capability rises or overseer capability falls, and that oversight architectures differ in those high-gap settings.

Hybrid oversight does not need to win everywhere. The claim is about identifying where architectures hold up, where they weaken, and what cost they impose.

## Immediate Outputs

The first Stage A run should produce:

- a capability-gap dashboard for global unsafe rate and local-pass/global-fail rate;
- a winner map as supporting evidence;
- one concrete episode trace showing the local-pass/global-fail mechanism when such a case exists;
- pairwise contrasts across local oversight, global signal, and hybrid oversight;
- welfare and governance-burden summaries alongside safety metrics.
