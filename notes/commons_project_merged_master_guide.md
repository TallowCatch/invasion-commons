# Commons Governance Project: Merged Master Guide

This is the merged version of:

- the newer interview guide in this repo
- the older supervisor notes from the earlier fishery-focused stage

The purpose of this file is to give you one document that does three jobs at once:

1. It explains the full project line from the early fishery work to the current Harvest paper.
2. It explains the mechanics in plain English, including the low-level things people often ask about.
3. It gives you ready answers for supervision meetings, interviews, and paper questions.

This is not written like a paper. It is written to help you think and speak clearly.

## 1. The shortest accurate summary

The project studies governance in shared-resource systems under strategic pressure. It begins with a simple Fishery Commons environment that asks which central governance signals work best when stronger or more exploitative strategies keep entering the system over time. It then moves to Harvest Commons, a richer environment with local patches, communication, side-payments, and multiple governance architectures. The strongest result from the earlier Harvest work is that architectures with central intervention are much more robust ecologically than local-only governance, and hybrid governance often performs best. The current paper extends that line by adding literature-backed scenario archetypes and explicit oversight frictions. That extension shows that the ranking between hybrid and top-down-only governance depends on the setting.

## 2. The one-minute spoken version

If you need a short oral explanation:

I built a benchmark for governance under changing strategy populations in shared-resource systems. In the simpler Fishery Commons environment, I compare central interventions like sanctions, quotas, and closures under repeated strategic turnover. In the richer Harvest Commons environment, I compare governance architectures such as local-only, central-only, and hybrid governance. The main result so far is that central intervention matters a lot for ecological robustness, hybrid governance is often strong, but its advantage over central-only governance is not universal. Once I add more realistic scenario structure and imperfect oversight, the architecture ranking becomes setting-dependent.

## 3. What the project is actually about

At the core, this is a governance benchmark project.

It is not mainly:

- a pure RL project
- a pure LLM project
- a direct policy simulation project

Those things matter, but they are not the center.

The center is this question:

How do we compare governance designs in multi-agent systems where the strategy population changes over time and stronger exploitative behaviours can keep entering?

That is why the project uses:

- commons environments
- repeated strategic turnover
- multiple governance conditions
- held-out evaluation
- architecture comparison
- pressure ladders with stronger attacker families
- welfare-incidence analysis

## 4. How the older notes and newer notes fit together

The older notes were written when the project was more fishery-centered.

At that stage, the big contribution was:

- define invasion pressure clearly
- show how population turnover works
- explain mutation and LLM injection
- show that governance should be evaluated as a defence against repeated pressure, not only as a one-shot improvement

The newer notes are more Harvest-centered.

At that stage, the big contribution became:

- move from central signal design to governance architecture
- compare bottom-up-only, top-down-only, and hybrid governance
- add capability ladders
- add welfare-incidence analysis
- add institutional-friction scenarios

So the old notes are still useful because they explain the mechanics very clearly.
The new notes are still useful because they explain where the project has actually ended up.

This file combines both.

## 5. The project line across papers

### Paper v2

This was the two-study paper.

Study 1:

- Environment: Fishery Commons
- Question: which central governance signals work best under repeated adversarial strategy injection?
- Main governed conditions included sanctions, adaptive quotas, and temporary closures
- Main result: adaptive quotas emerged as the strongest central signal in the medium tier

Study 2:

- Environment: Harvest Commons
- Question: does hybrid governance outperform top-down-only governance under stronger search-based pressure?
- Main result: hybrid ranked first in the confirmatory architecture matrix with better patch health and a moderate welfare cost

### Paper v3

This was the Harvest-led follow-up.

The framing became:

- less "does governance help?"
- more "which governance architectures remain robust as pressure rises?"

This version added:

- full architecture restoration
- stronger confirmatory comparison
- welfare-incidence analysis
- capability-ladder analysis

### Paper v4

This is the current institutional-friction extension.

It keeps the same research line but adds:

- literature-backed scenario archetypes
- imperfect detection
- delayed enforcement
- limited targeting capacity
- governance budget cost

The result is more nuanced and more realistic:

- hybrid is still often strong
- but hybrid does not rank first everywhere
- once scenario structure and implementation constraints are added, the architecture ranking depends on the setting

## 6. The basic problem in plain English

Imagine a group of agents sharing a renewable resource.

Each agent benefits from taking more for itself.
But if too many agents do that, the shared system deteriorates.

That is the commons problem.

The extra twist in this project is that the strategy population changes over time.

So the benchmark does not ask:

- "Do cooperative agents cooperate once?"

It asks:

- "What happens when stronger or more exploitative strategies keep entering?"
- "Which governance arrangements still work under that pressure?"

That is what adversarial strategy injection and later capability escalation mean in this project.

## 7. The main environments

### Fishery Commons

This is the simpler environment.

You can think of it as:

- one shared stock
- multiple agents
- each agent decides how much to harvest
- the stock regenerates
- sustained overuse can collapse the system

Why it matters:

- very clean
- easy to interpret
- ideal for comparing central signals

Why it is limited:

- no local neighborhoods
- no richer local interaction
- governance is mostly signal design, not architecture design

### Harvest Commons

This is the richer environment.

You can think of it as:

- many local renewable patches
- local overuse matters
- neighboring agents matter
- communication can be enabled
- side-payments or credits can be enabled
- governance can be local-only, central-only, or hybrid

Why it matters:

- it lets governance be architectural, not just top-down signaling
- it is better suited to institutional design questions
- it is the main substrate for the later papers

## 8. Core concepts and terms

### Strategy

A fixed rule for how an agent behaves.

In the earlier fishery setting, it is a threshold policy over stock.
In Harvest, it is a richer threshold policy over patch state, restraint, credit behaviour, reciprocity, and cap compliance.

### Population

The set of strategies currently active in one generation.

### Generation

One full loop:

1. evaluate current strategies
2. rank them
3. keep stronger ones
4. replace weaker ones with new child strategies

### Injection

The entry of new strategies into the population during replacement.

### Invasion pressure

Repeated entry of potentially stronger or more exploitative strategies across generations.

### Governance

The rules or interventions imposed by the environment, such as sanctions, caps, quotas, or local coordination structures.

### Held-out test regimes

Conditions not used for selection, used to test robustness rather than just fit to one setting.

### Collapse

A state where the resource remains degraded badly enough, for long enough, that the system is treated as failed.

## 9. The governance conditions in Harvest

These are the core governance conditions you need to explain clearly.

### `none`

No governance.

Agents act only according to their strategy.

### `bottom_up_only`

Local-only governance.

In plain English:

- local coordination exists
- local restraint can matter
- but there is no strong central capping authority

### `top_down_only`

Central-only governance.

In plain English:

- there is a central authority that monitors and caps aggressive extraction
- but there is no extra local cooperative layer on top

### `hybrid`

Multi-scale governance.

In plain English:

- central intervention exists
- local coordination also exists
- both levels are combined

## 10. What adversarial strategy injection means

This does not mean someone is hacking the code.

It means:

- the strategy population is not fixed
- weaker strategies are removed over generations
- new strategies are introduced
- those new strategies can become more exploitative or more adaptive as pressure rises

That is why this is a robustness benchmark rather than a static population test.

## 11. The injector families

In the main Harvest capability ladder, attacker families are ordered from weaker to stronger:

1. `random`
2. `mutation`
3. `adversarial_heuristic`
4. `search_mutation`

### `random`

Pure random candidate generation.

### `mutation`

Take a parent strategy and perturb its parameters.

### `adversarial_heuristic`

A stronger hand-designed attacker that pushes strategies in a more exploitative direction.

### `search_mutation`

Generate multiple mutated candidates, evaluate them, and keep the strongest one.

That is why it is the strongest non-LLM attacker in the main evidence chain.

## 12. What capability escalation means

Capability escalation is the cleaner institutional framing of the same idea.

In this project:

- weaker injectors correspond to lower strategic pressure
- stronger injectors correspond to higher strategic pressure

So capability escalation means that the newly introduced strategies become more effective, more exploitative, or more adaptive over time.

## 13. How the evolutionary protocol works

The population evolves over generations.

At each generation:

1. current strategies are evaluated
2. fitness is computed
3. strong strategies survive
4. a fraction of the population is replaced
5. new strategies are generated by the injector
6. the next generation is evaluated

This creates sustained pressure on the governance condition.

## 14. Fishery mechanics in plain English

These are the older low-level explanations, updated into the current project language.

### Fishery time-step dynamics

One fishery time step works like this:

1. start with current stock
2. each agent requests a harvest amount
3. requests are clipped by per-agent maximum
4. if total requested harvest exceeds available stock, requests are scaled down proportionally
5. remaining stock is computed
6. stock regrows using a logistic rule
7. the new stock is bounded above by stock maximum

### Fishery collapse logic

Collapse does not happen from one bad step.

Instead:

1. if stock is below the collapse threshold, a below-threshold counter increases
2. if stock recovers, that counter resets
3. if the counter reaches collapse patience, collapse occurs

So sustained overuse causes failure.

### Fishery governance mechanics

When governance is active in the fishery setting:

1. a quota is computed from current stock and the quota rule
2. agents can be audited
3. violators can be forced down to quota
4. sanctions can be applied
5. repeated violators face stronger penalties

This changes real extraction behaviour, not just reward accounting.

## 15. How a Fishery strategy works

In the simple fishery setting, a strategy is a five-parameter threshold policy:

- low stock threshold
- high stock threshold
- low harvest fraction
- mid harvest fraction
- high harvest fraction

Plain-English behaviour:

1. if stock is low, harvest lightly
2. if stock is medium, harvest moderately
3. if stock is high, harvest more aggressively

That is useful because it is interpretable and easy to compare across generations.

## 16. What mutation means in Fishery and Harvest

Mutation is not gradient learning.

It is controlled perturbation.

In plain English:

1. take a parent strategy
2. change the numerical parameters slightly
3. clip them into legal bounds
4. create a child strategy

Under stronger pressure, the mutation process can be biased toward more aggressive behaviour.

## 17. Why the old fishery notes still matter

The old notes are still important because they define the scientific backbone of the whole project:

- governance should be tested under repeated strategic turnover
- train/test separation matters
- stronger results come from held-out robustness, not just in-sample performance
- mechanism clarity matters

Those principles carried forward into Harvest.

## 18. What Harvest adds beyond Fishery

Harvest adds:

- local resource patches instead of one global stock
- richer local interaction
- local aggression and neighborhood overharvest
- communication
- side-payments or credits
- a real architecture comparison between local-only, central-only, and hybrid governance

This is why Harvest is the main environment in the later work.

## 19. The Harvest strategy representation

Harvest strategies are also structured and interpretable.

The internal JSON schema includes:

- `rationale`
- `low_patch_threshold`
- `high_patch_threshold`
- `low_harvest_frac`
- `mid_harvest_frac`
- `high_harvest_frac`
- `restraint_low`
- `restraint_high`
- `credit_request_low`
- `credit_request_high`
- `credit_offer_threshold`
- `credit_offer_amount`
- `neighbor_reciprocity_weight`
- `credit_response_weight`
- `cap_compliance_margin`

This comes from [harvest_evolution.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_evolution.py).

## 20. What JSON means in this project

JSON here is just a structured text format for a strategy.

It is used so that:

- the model does not write arbitrary code
- the output is easy to parse
- required keys can be checked
- values can be clamped into legal ranges
- the strategy stays interpretable

## 21. Example Fishery JSON policy

This is the kind of fishery policy JSON the project uses:

```json
{
  "rationale": "Harvest conservatively when stock is low and push harder when stock is healthy.",
  "low_stock_threshold": 40.0,
  "high_stock_threshold": 130.0,
  "low_harvest_frac": 0.08,
  "mid_harvest_frac": 0.45,
  "high_harvest_frac": 0.75
}
```

Meaning:

- low stock -> light harvest
- medium stock -> moderate harvest
- high stock -> aggressive harvest

## 22. Example Harvest JSON policy

This is the kind of Harvest policy JSON the project uses:

```json
{
  "rationale": "Protect weak patches, request help when stressed, and comply with caps under pressure.",
  "low_patch_threshold": 4.0,
  "high_patch_threshold": 12.0,
  "low_harvest_frac": 0.10,
  "mid_harvest_frac": 0.40,
  "high_harvest_frac": 0.75,
  "restraint_low": 0.85,
  "restraint_high": 0.20,
  "credit_request_low": 0.70,
  "credit_request_high": 0.05,
  "credit_offer_threshold": 10.0,
  "credit_offer_amount": 0.35,
  "neighbor_reciprocity_weight": 0.60,
  "credit_response_weight": 0.55,
  "cap_compliance_margin": 0.08
}
```

Meaning:

- weak local patch -> act carefully
- strong local patch -> harvest more
- if I am under pressure -> request support
- if I am doing well -> offer some support
- account for reciprocity
- comply with caps with some safety margin

## 23. What happens if the JSON is bad

The project does not assume model output is perfect.

The pipeline:

1. extracts the first JSON object it can find
2. checks required keys
3. converts values to the right types
4. clamps values into legal ranges
5. logs parse status and parse errors
6. if needed, falls back to mutation

That is why the code tracks:

- direct JSON fraction
- repaired JSON fraction
- effective LLM fraction
- unrepaired fallback fraction

## 24. Why LLM generation is controlled rather than free-form

The project does not put a language model directly in the action loop.

Instead, the model proposes a full strategy once, as structured JSON.

Why this is better:

- safer
- more reproducible
- easier to debug
- easier to analyze
- easier to compare with mutation-based injectors

## 25. Why Harvest LLM is not in the main evidence chain

Harvest has an LLM path, but it was not promoted into the core evidence chain because reliability was not yet strong enough.

The project used an evidence gate:

- effective LLM fraction at least 0.90
- unrepaired fallback fraction at most 0.05

The Harvest LLM path did not meet that gate strongly enough in the targeted cells.

That is why it stayed diagnostic rather than central.

## 26. The main Harvest architecture results before the newest extension

### Architecture Stage B

Purpose:

- restore the full architecture spectrum
- include `none`, `bottom_up_only`, `top_down_only`, and `hybrid`

Main value:

- `bottom_up_only` is distinct from `none`
- but it remains ecologically weaker than the architectures with central intervention

### Architecture Stage C

Purpose:

- high-power confirmatory architecture matrix
- strong attacker
- decision-critical architecture comparison

Main result:

- hybrid ranks first in all eight decision cells under the main ranking rule
- hybrid improves patch health relative to top-down-only in all eight cells
- hybrid is non-worse on neighborhood overharvest in six of eight cells
- the mean welfare delta relative to top-down-only is `-0.1363`

This is the strongest clean baseline architecture result in the project.

### Capability ladder

Purpose:

- test when architectures break as attacker strength rises

Main result:

- hybrid ranks first in 25 of 32 ladder cells
- top-down-only ranks first in 7 of 32 ladder cells
- both architectures with central intervention remain ecologically stronger than bottom-up-only governance

Interpretation:

- the main robust split is between “has central intervention” and “does not”
- the finer split between hybrid and top-down-only is more sensitive and more contested

### Welfare incidence

Purpose:

- explain who actually pays the welfare cost of governance

Main interpretation:

- the big welfare penalty appears when moving away from local-only governance toward ecologically stronger governance
- the extra cost of moving from top-down-only to hybrid is much smaller

## 27. The ranking rule in Harvest

Architectures are ranked within each experimental cell using:

1. lower garden failure
2. then higher patch health
3. then higher welfare

This matters because:

- a condition can look good on welfare while being weak ecologically
- the project gives priority to avoiding system failure

## 28. What garden failure means

In the current Harvest implementation, a garden-failure event occurs when:

- more than half of the local patches remain below four resource units
- for five consecutive steps

In plain English:

the local resource system stays degraded badly enough, for long enough, that it counts as a meaningful failure event rather than a short fluctuation.

## 29. Welfare incidence in plain English

Earlier architecture results showed that governance can improve ecology but impose a welfare cost.

The next question was:

Who actually pays that cost?

That is what welfare incidence answers.

It avoids reporting only an average number and instead asks how the cost is distributed.

### Aggression-based incidence

Agents are grouped into:

- low aggression
- middle aggression
- high aggression

Then the analysis compares:

- welfare
- prevented harvest
- realized harvest
- targeted fraction

Purpose:

- see whether governance mainly penalizes the most exploitative agents or broadly taxes everyone

### Targeting-based incidence

Agents are grouped into:

- targeted
- untargeted

Then the analysis compares:

- welfare
- prevented harvest
- realized harvest
- local patch health

Purpose:

- see whether welfare loss comes from targeted restraint or wider system inefficiency

## 30. The institutional-friction extension

This is the current paper's main new contribution.

It adds:

1. literature-backed scenario archetypes
2. explicit oversight frictions

### The three scenario archetypes

These are benchmark archetypes, not calibrated field replicas.

#### Regulated fishery

Interpretation:

- centralized quota-setting
- monitoring
- enforcement under ecological uncertainty

#### Community irrigation

Interpretation:

- local user monitoring
- local rule enforcement
- self-governing local management

#### Forest co-management

Interpretation:

- mixed local stewardship
- higher-level intervention
- polycentric governance

## 31. Scenario preset values in the current paper

From [harvest_benchmarks.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_benchmarks.py):

### Regulated fishery

- tier: `medium_h1`
- partner mix: `balanced`
- regen rate: `0.70`
- neighbor externality: `0.10`
- communication: off
- credits: off

### Community irrigation

- tier: `medium_h1`
- partner mix: `balanced`
- regen rate: `0.64`
- neighbor externality: `0.15`
- communication: on
- credits: off

### Forest co-management

- tier: `hard_h1`
- partner mix: `adversarial_heavy`
- regen rate: `0.56`
- neighbor externality: `0.22`
- communication: on
- credits: on

## 32. Oversight frictions

The current extension compares two oversight regimes.

### Ideal

- detection recall = `1.0`
- enforcement delay rounds = `0`
- max target share = `1.0`
- governance budget cost = `0.0`

Meaning:

governance sees everything and can act immediately without cost or capacity limits.

### Constrained

- detection recall = `0.7`
- enforcement delay rounds = `1`
- max target share = `0.5`
- governance budget cost = `0.02`

Meaning:

governance is imperfect, delayed, capacity-limited, and not free.

## 33. Main findings from the institutional-friction extension

### Community irrigation

Hybrid ranks first at both pressure levels under both ideal and constrained oversight.

Interpretation:

in this more locally managed setting, the combined local-plus-central arrangement remains strongest.

### Forest co-management

Hybrid ranks first under ideal oversight.
Top-down-only ranks first under constrained oversight.

Interpretation:

once oversight becomes imperfect and capacity-limited, the earlier hybrid advantage is not preserved automatically.

### Regulated fishery

Hybrid ranks first at lower pressure.
Top-down-only ranks first at higher pressure under both regimes.

Interpretation:

in a more regulator-facing setting, stronger central control becomes more competitive as pressure rises.

## 34. The main conclusion of the current paper

Governance architecture can be benchmarked under strategic pressure, and central intervention is consistently important for ecological robustness. Hybrid governance is often strong and can outperform central-only control, but that advantage is not universal. Once scenario structure and oversight frictions are added, the ranking between hybrid and top-down-only governance becomes setting-dependent.

## 35. What the current paper does not claim

The paper does not claim:

- a universal winner across all institutions
- direct field-policy prescription
- that the scenarios are calibrated replicas of real fisheries, irrigation systems, or forests
- that the LLM path is already the main evidence chain

That restraint is part of what makes the paper stronger.

## 36. What is still pending

### Done

- Fishery benchmark and signal-design study
- Harvest architecture benchmark
- full architecture restoration
- confirmatory high-power matrix
- capability ladder
- welfare incidence
- literature-backed scenario presets
- oversight-friction regimes
- institutional-friction Module A
- current `paper_v4_institutional_commons` draft

### Not yet done as main evidence

- Harvest Module B: LLM-population governance map
- Harvest Module C: LLM-population turnover pilot
- full paper-grade strategy-bank cycle as part of the main narrative

## 37. What Module B and Module C would do

### Module B: LLM-population governance map

This would:

- load a Harvest strategy bank
- sample populations with varying exploitative share
- compare conditions such as `none`, `top_down_only`, and `hybrid`
- test whether the governance story still holds when populations come from LLM-generated strategies

### Module C: LLM-population turnover pilot

This would:

- treat `(model, attitude)` as a gene
- let better-performing genes survive
- repopulate the next generation from survivor-weighted strategy banks
- track whether exploitative types spread
- test whether governance can slow or prevent that spread

These modules are the clean bridge to the LLM-population direction without abandoning Harvest.

## 38. Why this is still relevant to the real world

This project does not directly tell a ministry exactly what policy to use next week.

That would be too strong.

What it does do is:

- compare governance designs in a controlled way
- make ecological-versus-welfare trade-offs explicit
- show how architecture rankings change with stronger strategic pressure
- show how implementation frictions can change those rankings

That is a meaningful real-world bridge, even though it is still a benchmark.

## 39. Why testbeds are still useful

A benchmark like this is useful because:

- field settings are messy
- real institutional changes are hard to isolate causally
- governance problems involve many interacting trade-offs
- it is often impossible to vary one mechanism cleanly in the real world

The benchmark is useful if it is framed correctly:

good claim:

- this is a controlled testbed for comparing institutional designs under strategic pressure

bad claim:

- this proves exactly what a real fishery or forest authority should do

## 40. What each main figure means

### Confirmatory Harvest architecture figure

This shows the baseline strong architecture result before the new frictions.

How to explain it:

- the figure shows paired deltas across matched cells
- positive patch-health delta means hybrid protects the ecology better than top-down-only in that cell
- the welfare delta shows the cost of that extra ecological control

### Capability ladder figure

This shows the first attacker rung at which an architecture loses its advantage.

How to explain it:

- later break means stronger robustness
- early break means fragility
- the main lesson is that the split between central intervention and local-only governance is robust, while the split between hybrid and top-down-only is more sensitive

### Institutional-friction winner map

This shows:

- which architecture ranks first in each scenario, oversight regime, and pressure cell
- the hybrid-minus-top-down patch-health delta inside each cell

How to explain it:

- the winner label tells you the overall best architecture under the ranking rule
- the patch-health delta tells you whether hybrid still has an ecological edge even when it is not the overall winner

## 41. Repo map for important files

### Paper

- [main.tex](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v4_institutional_commons/main.tex)
- [introduction.tex](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v4_institutional_commons/sections/introduction.tex)
- [methods.tex](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v4_institutional_commons/sections/methods.tex)
- [results.tex](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v4_institutional_commons/sections/results.tex)
- [discussion.tex](/Users/ameerfiras/invasion-commons/invasion-commons/paper/paper_v4_institutional_commons/sections/discussion.tex)

### Benchmark presets and scenarios

- [harvest_benchmarks.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_benchmarks.py)
- [harvest_invasion_presets.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/harvest_invasion_presets.py)

### Main experiment runners

- [run_harvest_invasion.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_invasion.py)
- [run_harvest_invasion_matrix.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_invasion_matrix.py)
- [summarize_harvest_invasion.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/summarize_harvest_invasion.py)

### Plot scripts

- [plot_harvest_capability_ladder_publication.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/plot_harvest_capability_ladder_publication.py)
- [plot_institutional_friction_winner_map.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/plot_institutional_friction_winner_map.py)

### LLM strategy-bank and bridge code

- [harvest_llm_population.py](/Users/ameerfiras/invasion-commons/invasion-commons/fishery_sim/harvest_llm_population.py)
- [run_harvest_llm_governance_map.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_llm_governance_map.py)
- [run_harvest_llm_turnover.py](/Users/ameerfiras/invasion-commons/invasion-commons/experiments/run_harvest_llm_turnover.py)

## 42. What you can safely claim

You can safely claim:

1. You built a benchmark for governance under repeated strategic turnover.
2. You compared governance architectures under matched settings.
3. You used held-out evaluation rather than only reporting in-sample performance.
4. You organized attackers into a capability ladder.
5. You measured who bears the welfare cost through incidence analysis.
6. You showed that adding scenario structure and oversight frictions changes the architecture ranking.

You should avoid claiming:

1. that one architecture always wins
2. that the scenarios are calibrated field models
3. that the current paper already proves the LLM-population bridge
4. that the benchmark directly yields a specific real-world policy

## 43. Likely interview and supervision questions with answers

### What is your project in one sentence?

It is a benchmark for comparing governance designs in renewable commons when stronger or more exploitative strategies can keep entering the population over time.

### What is the scientific novelty?

The novelty is the protocol: repeated strategic turnover, held-out robustness evaluation, architecture comparison, capability ladders, incidence analysis, and now scenario and oversight-friction extensions.

### Why is this better than standard cooperation studies?

Because standard studies often ask whether cooperation emerges once. This project asks whether governance remains effective under repeated strategic disruption.

### Why two environments?

Fishery is the cleaner signal-design environment. Harvest is the richer architecture-design environment.

### What is the strongest result so far?

The strongest baseline result is the confirmatory Harvest architecture matrix, where hybrid ranks first in all eight decision cells under the main ranking rule.

### What is the most important new result in the current paper?

That architecture rankings become scenario-dependent once oversight frictions and literature-backed scenario structure are added.

### So does hybrid win?

Hybrid often wins and remains a strong architecture, but its advantage over top-down-only governance is not universal once the newer frictions and scenarios are introduced.

### What is JSON in your project?

A structured text format used to define strategies safely and interpretably instead of letting a model write arbitrary code.

### Why use JSON instead of Python code generated by the model?

Because JSON is easier to validate, easier to clamp into legal ranges, easier to debug, safer, and more reproducible.

### What does a generation mean?

Evaluate current strategies, rank them, keep the best, inject replacements, and then start the next cycle.

### What is injection?

The population entry step where newly created child strategies replace weaker strategies.

### What is mutation?

A child-creation operator that changes the numeric parameters of a parent strategy.

### What is the difference between mutation and injection?

Mutation creates the child. Injection is the step where the child enters the population.

### What is capability escalation?

It is the increase in attacker strength as you move from weaker injectors like random or mutation to stronger ones like search over mutations.

### What is welfare incidence?

It is the analysis of who bears the welfare cost of governance instead of only reporting one average welfare number.

### Why does that matter?

Because a welfare cost has different meaning depending on whether it mainly restrains exploitative agents or broadly burdens everyone.

### Are your scenarios real-world calibrated?

No. They are literature-backed benchmark archetypes, not calibrated field replicas.

### Then how is this real-world relevant?

It is relevant because it compares governance designs under explicit strategic pressure and implementation constraints that resemble real institutional problems, even though it is still a benchmark rather than a field simulator.

### Why is this publishable?

Because it is a coherent benchmark contribution with a cumulative evidence chain, not just a toy example or a vague opinion piece.

### What is the biggest limitation?

The current paper is strongest on the non-LLM benchmark side. The LLM-population bridge is prepared in code but not yet part of the main completed evidence chain.

### What would you do next?

Run the Harvest strategy-bank experiments, starting with the governance map first and the turnover pilot second.

## 44. Very short supervisor-ready script

If you need a concise spoken version:

I started with a simpler fishery benchmark to ask which central governance signals remain effective when new exploitative strategies keep entering a renewable commons over time. That let me establish the invasion-pressure protocol clearly. I then moved to a richer Harvest environment where the question becomes architectural rather than only top-down: local-only, central-only, and hybrid governance can be compared directly. The strongest baseline result is that architectures with central intervention are much more robust ecologically than local-only governance, and hybrid often performs best. In the newest extension, I added literature-backed scenario archetypes and explicit oversight frictions. That made the result more realistic and more nuanced, because the ranking between hybrid and top-down-only governance now depends on the setting.

## 45. Final advice on how to talk about the work

The best way to talk about this project is:

- start with the problem, not the filenames
- say "shared-resource systems under strategic pressure"
- say "governance architecture"
- say "benchmark" and "controlled comparison"
- be clear that the project is about institutional robustness, not just cooperation appearing once
- be honest that the benchmark is not a direct policy simulator
- be equally clear that it is still a serious empirical contribution

The cleanest final summary is:

This is a benchmark for institutional robustness in stylized commons. It shows that governance architecture matters under strategic pressure, central intervention is consistently important for ecological robustness, and the ranking between hybrid and central-only governance becomes setting-dependent once more realistic institutional frictions are introduced.

