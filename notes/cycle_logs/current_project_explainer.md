# Current project explainer

## Mini version of what was done
- Strengthened the Fishery Commons study so it now shows not only that governance helps, but which central governance signals hold up under stronger strategic pressure.
- Added more mechanism logging, more partner mixes, more pressure settings, and stronger non-LLM injectors for Fishery Commons.
- Built an initial orchard-style second substrate, then reframed it as Harvest Commons because that fit the sequential-commons benchmark story better.
- Ran early fixed-composition Harvest comparisons to see whether top-down, bottom-up, and hybrid governance could separate at all.
- Realized that Harvest needed the same invasion logic as Fishery Commons, so upgraded it to a proper population-turnover study with injectors, held-out regimes, and high-power sweeps.
- Added remote GitHub Actions workflows so the heavy Harvest invasion matrix could run off the laptop, then pulled the merged artifacts back into `results/`.
- Updated the paper so Fishery Commons is the first study, Harvest Commons is the second study, and the old fixed-composition Harvest work is treated as pilot evidence rather than the main result.

## What the project is now, in one sentence
You are building a two-study benchmark for governance in sequential commons, where the core question is whether cooperation stays stable when new exploitative strategies keep entering the population.

## Why there are now two studies
The project now has two linked studies because one study is not enough to answer the full governance question.

The first study uses Fishery Commons. Its job is to answer a narrow, controlled question:
> Which central governance signals work best under adversarial strategy injection?

The second study uses Harvest Commons. Its job is to answer a broader downstream question:
> Once we move beyond fixed central enforcement, how should governance be organized under the same kind of strategic pressure?

So the paper now has a clear structure:
1. First study: signal design in Fishery Commons.
2. Second study: governance architecture in Harvest Commons.

## What changed from the orchard idea
The orchard idea was not thrown away. It was the starting point for the second substrate.

At first, the goal was to move beyond fish by creating another shared-resource task where agents interact locally, can communicate with neighbors, and can share credits or benefits. That first version used an orchard-style picture: apples, local patches, and decentralized interaction.

That basic idea was fine, but the framing was weak. It sounded like a new toy environment rather than a clear second commons substrate. After looking at the benchmark logic from sequential social dilemma work, the better framing was to treat it as **Harvest Commons** instead.

That change mattered because Harvest Commons now sounds like what it really is:
- a second renewable commons task,
- with local resource patches,
- local externalities,
- communication,
- and governance choices.

So the orchard idea became the basis of Harvest Commons rather than being discarded outright.

## The big scientific idea
The scientific idea is still the same as before, but it is now stronger and more complete.

You are not asking:
- can cooperation appear once?
- can a model act nicely in one setup?

You are asking:
- if new strategies keep entering,
- if some of them are more selfish or exploitative,
- and if the environment shifts,
- can governance still keep the commons stable?

That is why the project is about **invasion pressure** and not just cooperation rates.

## The core terms in plain English

### Strategy
A strategy is a rule for what an agent does.

In Fishery Commons, the strategy decides how much fish to take when stock looks low, medium, or high.

In Harvest Commons, the strategy is a bit richer. It decides:
- how much to take from a local patch,
- how cautious to be,
- whether to ask neighbors for restraint or credit,
- whether to offer help to neighbors,
- and how closely to comply with government caps.

### Population
A population is the set of strategies currently active in one generation.

### Generation
A generation is one full loop:
1. evaluate the population,
2. rank strategies by fitness,
3. keep stronger ones,
4. replace weaker ones with new injected strategies.

### Injection
Injection is the replacement step where new strategies enter the population.

### Mutation
Mutation means taking a parent strategy and changing its numeric parameters slightly.

### Adversarial heuristic injection
This is a more direct attacker generator. Instead of making a nearby child, it samples strategies from aggressive ranges on purpose.

### Random injection
This samples fresh strategies without trying to imitate or optimize against the current population.

### Invasion pressure
Invasion pressure means the population is under repeated strategic turnover. The system never gets to settle into one fixed set of behaviors.

### Governance
Governance means the rules that change behavior in the environment.

In Fishery Commons this includes things like monitoring, sanctions, adaptive quotas, and temporary closures.

In Harvest Commons this includes architecture choices such as:
- top-down only,
- bottom-up only,
- hybrid.

## What the first study does now
The first study lives mostly in:
- [fishery_sim/evolution.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/evolution.py)
- [fishery_sim/env.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/env.py)
- [experiments/run_study1b.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_study1b.py)

It now has two layers.

### Layer 1: matched baseline
This is the older `paper_v1` logic.
It compares:
- no governance,
- monitoring,
- monitoring with sanctions,

under:
- mutation injection,
- live LLM JSON injection.

That is the part that gives the clean result that monitoring with sanctions reduces collapse relative to no governance for both injector types.

### Layer 2: stronger Fishery Commons sweep
This is the newer Study 1b layer.
It compares stronger top-down signals:
- no governance,
- monitoring with sanctions,
- adaptive quota,
- temporary closure.

It also varies:
- partner mix,
- adversarial pressure,
- injector family.

That is the part that gives the stronger result:
- adaptive quotas are the strongest overall central signal,
- temporary closures remain a strong fallback in hostile populations,
- monitoring with sanctions helps but does not suppress aggressive extraction as strongly as the more forceful top-down policies.

## What the second study does now
The second study now lives mainly in:
- [fishery_sim/harvest.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest.py)
- [fishery_sim/harvest_benchmarks.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_benchmarks.py)
- [fishery_sim/harvest_evolution.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_evolution.py)
- [experiments/run_harvest_invasion.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_invasion.py)
- [experiments/run_harvest_invasion_matrix.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_invasion_matrix.py)

This is the important correction: the second study is no longer just a fixed set of hand-written agent types. It now has the same population-turnover idea as the first study.

That means Harvest Commons now has:
1. a strategy population,
2. fitness-based ranking,
3. replacement,
4. injectors,
5. held-out test regimes,
6. high-power sweeps.

That is what makes it a real second study instead of just a side demo.

## How Harvest Commons works
Harvest Commons is built around local patches rather than one global stock.

Each agent mainly interacts with a local patch and nearby agents.
The environment can support:
- local communication,
- local credit transfers,
- and government caps targeted at aggressive neighborhoods.

That means the second study can ask a different kind of governance question:
- is pure top-down control enough?
- does local bottom-up coordination help?
- does hybrid governance work better?

## Why government is treated differently in the second study
In Fishery Commons, governance is mostly an environment-level rule set.

In Harvest Commons, government is closer to an agent-like controller. It has explicit behavior:
- it observes the local state,
- it decides where to intervene,
- it applies targeted caps.

That matters because one of the ideas behind the second study is that governance itself may be part of the agent system, not just an invisible background rule.

## What the injectors are doing in Harvest Commons
This was the missing part before, and it is now fixed.

Harvest Commons now has:
- `random` injection,
- `mutation` injection,
- `adversarial_heuristic` injection.

So the second study now asks a real invasion question too:
- when Harvest strategies keep evolving,
- and new exploitative strategies keep entering,
- which governance architecture holds up best?

That makes the second study fit the main paper much better.

## What the remote GitHub Actions work was for
The Harvest invasion matrix was large enough that local runs were taking too long and tying up the machine.

So the repo now includes:
- [.github/workflows/harvest-invasion-matrix.yml](/Users/ameerfiras/invasion-commons/invasion-commons/.github/workflows/harvest-invasion-matrix.yml)
- [experiments/merge_harvest_invasion_outputs.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/merge_harvest_invasion_outputs.py)

That workflow shards the Harvest matrix across GitHub Actions jobs, merges the CSV outputs, and lets you pull the bundled artifacts back into the repo.

That change is not part of the science itself, but it matters because it made the large Stage B and Stage C Harvest invasion runs practical.

## What the current key results are

### First study: Fishery Commons
Safe result:
- monitoring with sanctions improves robustness relative to no governance under both mutation and live LLM injection.

Stronger Study 1b result:
- adaptive quotas are the strongest overall top-down signal in the medium fishery tier,
- temporary closures remain competitive in more hostile partner mixes,
- mechanism logging shows why sanctions are weaker than stronger top-down interventions.

### Second study: Harvest Commons
Pilot result:
- hybrid governance won most Stage B pilot cells.

High-power result:
- hybrid governance wins most Stage C decision-critical cells,
- it usually improves patch health and reduces neighborhood overharvest relative to top-down-only control,
- the welfare effect is mixed and depends on social mix and task difficulty.

That is an important point: the second study is not saying hybrid governance is always best on every metric. It is saying that under real invasion pressure, hybrid governance usually buys better local ecological control, but not always at zero cost.

## What the paper now says
The paper in:
- [paper/paper_v2/main.pdf](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v2/main.pdf)

now has this structure:
1. Introduction and framing of governance under adversarial strategy injection.
2. First study in Fishery Commons about central governance signals.
3. Second study in Harvest Commons about governance architecture.
4. Discussion of how the two studies fit together.

That is the right structure because the two studies are related, but they do not ask exactly the same question.

## The clean way to explain the whole project now
If you need one tight explanation, use this:

> “The project now has two linked studies. The first study uses Fishery Commons to ask which central governance signals stay effective when new exploitative strategies keep entering the population. The second study moves to Harvest Commons and asks what happens when governance is not only a central rule, but can also include local coordination and hybrid control. Both studies now use population turnover, injectors, train and held-out test conditions, and uncertainty estimates, so the full paper is really about governance under adversarial strategy injection in sequential commons.”

## What is safe to claim
You can safely claim:
- the project moved from simple emergence to stability under invasion pressure,
- Fishery Commons identifies which top-down signals work best,
- Harvest Commons now uses the same invasion logic and shows that governance architecture matters,
- hybrid governance often improves local ecological control in the second substrate.

You should avoid claiming:
- that hybrid governance universally dominates,
- that Harvest already has live LLM injection,
- that the current substrates are realistic enough to represent full real-world systems without further extension.

## What is next
The next cycle is not about rebuilding everything again. It is about deciding which of these extensions comes next:
1. live LLM injection in Harvest Commons,
2. a stronger non-LLM or learning baseline in Harvest Commons,
3. richer realism in either substrate,
4. or freezing the current paper and moving to the next paper-level question.
