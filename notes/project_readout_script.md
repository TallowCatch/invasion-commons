# Project Readout Script

This file is meant to be easy to read out loud.

It is written for:

- supervisor meetings
- interviews
- viva-style questions
- any situation where someone asks what the project is, what has been done, and what comes next

I wrote this in a direct way on purpose.

## 1. The shortest version

My project studies governance in shared-resource systems under strategic pressure. I started with a simpler Fishery Commons environment to compare central interventions such as sanctions, quotas, and closures when stronger strategies keep entering over time. I then moved to Harvest Commons, which is a richer environment with local patches, communication, credits, and multiple governance architectures. The strongest baseline result is that central intervention matters a lot for ecological robustness, and hybrid governance is often strong. In the newest extension, I added literature-backed scenario archetypes and oversight frictions, and that made the ranking between hybrid and top-down-only governance depend on the setting.

## 2. The one-minute version I can say quickly

I built a benchmark for governance under changing strategy populations in shared-resource systems. In Fishery Commons, I ask which central governance signals work best when exploitative strategies keep entering. In Harvest Commons, I ask which governance architecture is more robust: no governance, local-only governance, central-only governance, or hybrid governance. Earlier Harvest results showed that hybrid governance often performs best under strong search-based pressure. The current paper adds more realistic scenario structure and imperfect oversight. That extension shows a more nuanced result: hybrid stays strong in some settings, but top-down-only becomes stronger in others once implementation constraints are introduced.

## 3. What Edward pushed me to do

Edward's push was mainly about institutional framing and architecture.

The key points were:

- treat the existing two-study paper as finished rather than endlessly rewriting it
- make Harvest the main study
- frame the question around institutional robustness under rising strategic pressure
- restore the full architecture space instead of relying on a narrow pairwise comparison
- add a stronger architecture matrix
- add welfare-incidence analysis so the cost of governance is not just an average number
- add a capability ladder so the project shows where architectures break as pressure increases
- keep Harvest LLM off the main evidence chain unless it is reliable enough

## 4. How I addressed Edward's feedback

I addressed that feedback in the following way.

First, I stopped treating `paper_v2` as the place to keep adding everything. I froze it as the earlier two-study baseline.

Second, I moved the center of gravity to Harvest. That changed the main question from “does governance help?” to “which governance architectures remain robust as pressure rises?”

Third, I ran the architecture restoration pilot so the full Harvest architecture space was back in play:

- `none`
- `bottom_up_only`
- `top_down_only`
- `hybrid`

Fourth, I ran the high-power architecture confirmation. That gave the strongest baseline architecture result in the project.

Fifth, I added the capability ladder. That let me show when architectures lose their advantage as attacker strength increases.

Sixth, I added welfare-incidence analysis. That let me answer who actually bears the cost of governance.

Seventh, I kept the Harvest LLM path out of the main evidence chain because the reliability gate was not strong enough.

## 5. What Yali pushed me to think about

Yali's push was mainly about outward direction and relevance.

The key points I took from that were:

- make the work point toward something that has a real-world interpretation
- make the direction look less like a closed toy world
- connect more clearly to collective behaviour and LLM-population work
- keep a path toward something that could matter for real deployment settings

## 6. How I addressed Yali's feedback

I addressed that feedback without abandoning the commons line.

First, I introduced literature-backed scenario archetypes:

- regulated fishery
- community irrigation
- forest co-management

These are grounded in commons-governance literature, so the benchmark settings now map to recognizable institutional forms.

Second, I introduced explicit oversight frictions:

- imperfect detection
- delayed enforcement
- targeting limits
- governance budget cost

That makes governance less idealized and more interpretable.

Third, I kept the code path that bridges toward LLM-population work:

- structured JSON strategy generation
- Harvest strategy-bank utilities
- governance-map pipeline for LLM-generated populations
- turnover pilot for LLM-generated banks

That bridge exists, but it is not yet the main evidence chain in the current paper.

## 7. What the project is now

The project is now a benchmark for governance architecture in renewable commons under strategic pressure.

The current contribution is strongest when I describe it like this:

I compare governance architectures in stylized but interpretable shared-resource systems where stronger and more exploitative strategies can keep entering the population over time, and I test how those architecture rankings change once scenario structure and oversight constraints are introduced.

## 8. The paper line in order

### Earlier paper

The earlier paper had two studies.

Study 1 was Fishery Commons.
Study 2 was Harvest Commons.

The Fishery study answered which central signals were strongest.
The Harvest study answered whether hybrid could beat top-down-only under stronger pressure.

### Current line

The later work made Harvest the main study.

That work added:

- full architecture restoration
- a stronger confirmatory matrix
- capability-ladder analysis
- welfare-incidence analysis
- literature-backed scenarios
- oversight frictions

## 9. What I have done since the last major meeting or checkpoint

There are really two layers of “since last meeting” progress in this project.

### First layer: the older fishery-to-paper strengthening work

I added stronger supporting evidence to the earlier project line:

- learned-policy PPO validation in Fishery Commons
- learned-policy PPO validation in Harvest Commons
- stronger figure polish and cleaner paper presentation

Those RL results were supporting evidence. They helped show that the earlier paper’s conclusions were not tied only to one hand-written strategy family.

### Second layer: the main Harvest follow-up work

This is the more important progress for the current line.

I did the following:

1. restored the full Harvest architecture space
2. ran the architecture Stage B pilot
3. ran the Stage C high-power confirmatory architecture matrix
4. ran the capability ladder
5. added welfare-incidence analysis
6. added literature-backed scenario presets
7. added oversight-friction regimes
8. ran the institutional-friction Module A extension
9. wrote and iteratively revised the current paper
10. cleaned and rebuilt the figures into reproducible plotting scripts

## 10. The environments

### Fishery Commons

Fishery Commons is the simpler environment.

It has:

- one shared stock
- multiple agents
- per-agent harvest decisions
- regeneration
- collapse dynamics

Why it matters:

- it is simple enough to interpret clearly
- it is very good for signal design
- it gives clean comparisons between top-down governance signals

### Harvest Commons

Harvest Commons is the richer environment.

It has:

- local renewable patches
- local ecological deterioration
- communication
- side-payments or credits
- multiple governance architectures

Why it matters:

- it supports architecture comparison rather than only central signal comparison
- it is the main environment for the current line

## 11. The core governance conditions

### `none`

No governance.

This is the baseline.

### `bottom_up_only`

Local-only governance.

This captures decentralized coordination without strong central intervention.

### `top_down_only`

Central-only governance.

This captures direct central oversight and capping without the extra local governance layer.

### `hybrid`

Combined local and central governance.

This is the multi-scale condition.

## 12. What “adversarial strategy injection” means

This phrase describes the main stress mechanism in the project.

It means:

- the population is not fixed
- weaker strategies are removed over generations
- new strategies are introduced
- stronger injectors can generate more exploitative or more adaptive strategies

So the question is about whether governance stays effective while the population keeps changing.

## 13. The attacker families in the capability ladder

The non-LLM attacker ladder goes from weaker to stronger:

1. `random`
2. `mutation`
3. `adversarial_heuristic`
4. `search_mutation`

This matters because it lets me show not only who wins, but when an architecture breaks.

## 14. The main baseline results I can say clearly

### Fishery result

In Fishery Commons, adaptive quotas emerged as the strongest central signal in the medium tier.

The broader point is that the benchmark can rank central interventions under repeated strategic turnover.

### Harvest Stage B result

The architecture restoration pilot showed that `bottom_up_only` is empirically distinct from `none`, but still ecologically weaker than architectures with central intervention.

### Harvest Stage C result

The high-power confirmatory matrix showed:

- hybrid governance ranked first in all eight decision cells
- hybrid improved patch health relative to top-down-only in all eight cells
- hybrid was non-worse on neighborhood overharvest in six of eight cells
- the mean welfare delta for hybrid relative to top-down-only was `-0.1363`

That is the strongest baseline architecture result in the project.

### Capability-ladder result

The capability ladder showed:

- hybrid ranked first in 25 of 32 ladder cells
- top-down-only ranked first in 7 of 32
- both architectures with central intervention remained ecologically stronger than bottom-up-only governance

### Welfare-incidence result

The welfare-incidence analysis showed:

- the big welfare penalty appears when moving away from local-only governance toward ecologically stronger governance
- the extra cost of moving from top-down-only to hybrid is much smaller

## 15. What the current paper adds on top of that

The current paper adds:

- three literature-backed scenario archetypes
- two oversight regimes
- a new architecture comparison under those scenarios and regimes

The three scenarios are:

- regulated fishery
- community irrigation
- forest co-management

The two oversight regimes are:

### Ideal

- full detection
- no delay
- full targeting capacity
- no governance budget cost

### Constrained

- detection recall `0.7`
- one-round delay
- max target share `0.5`
- budget cost `0.02`

## 16. Main findings from the current paper

### Community irrigation

Hybrid ranks first at both pressure levels under both ideal and constrained oversight.

### Forest co-management

Hybrid ranks first under ideal oversight.
Top-down-only ranks first under constrained oversight.

### Regulated fishery

Hybrid ranks first at lower pressure.
Top-down-only ranks first at higher pressure under both oversight regimes.

## 17. The main conclusion I should say now

The main conclusion is:

governance architecture matters under strategic pressure, central intervention is consistently important for ecological robustness, and the ranking between hybrid and top-down-only governance depends on the institutional setting and the practical limits of oversight.

## 18. What I mean when I say the result is “setting-dependent”

I mean that the best-ranked architecture changes across:

- scenario type
- pressure level
- oversight regime

So the result is no longer one flat ranking across all cases.

That is a stronger and more realistic contribution.

## 19. What the paper is saying about real-world relevance

The paper uses benchmark archetypes rather than calibrated field cases.

That means:

- the settings are grounded in real institutional forms from the literature
- the numerical values are still benchmark abstractions
- the contribution is controlled institutional comparison, not direct policy prescription

That is the right level of claim for this project.

## 20. What I should say if someone asks whether this can apply in real life

I should say:

The project does not directly prescribe a specific fishery or irrigation policy. It gives a controlled way to compare governance designs under strategic pressure and imperfect oversight. That is useful because real institutional settings face the same kinds of issues: strategic noncompliance, incomplete monitoring, capacity limits, delays, and trade-offs between system protection and user burden.

## 21. What I should say if someone asks whether this is just a toy environment

I should say:

It is a benchmark environment. The value of the benchmark is that it isolates governance structure, strategic pressure, and implementation frictions in a way that is hard to do in the field. The contribution comes from clean comparative evidence and disciplined interpretation.

## 22. What JSON means in this project

JSON is a structured text format used to define strategies.

Instead of asking a model to write code, the project asks it to output named fields.

The code then:

1. parses the object
2. checks that required keys exist
3. converts values to the right type
4. clamps values into legal ranges
5. converts the result into an executable strategy

That makes the LLM path safer and easier to interpret.

## 23. What a Fishery JSON policy looks like

Example:

```json
{
  "rationale": "Harvest lightly when stock is low and more aggressively when stock is high.",
  "low_stock_threshold": 40.0,
  "high_stock_threshold": 130.0,
  "low_harvest_frac": 0.08,
  "mid_harvest_frac": 0.45,
  "high_harvest_frac": 0.75
}
```

This means:

- low stock -> light harvest
- medium stock -> moderate harvest
- high stock -> aggressive harvest

## 24. What a Harvest JSON policy looks like

Example:

```json
{
  "rationale": "Protect weak patches, request support when needed, and comply with caps.",
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

This means:

- weak local patch -> more restraint
- strong local patch -> more extraction
- if stressed -> ask for help
- if comfortable -> offer some help
- respond to neighbors and to credits
- comply with central caps with a margin

## 25. What happens if the model outputs bad JSON

The project handles that explicitly.

The pipeline:

1. tries to parse the first JSON object it can find
2. checks required keys
3. repairs some malformed outputs when possible
4. clamps values into safe ranges
5. logs the parse status
6. falls back to mutation if the strategy still cannot be used

That is why the project tracks things like:

- direct JSON fraction
- repaired JSON fraction
- effective LLM fraction
- unrepaired fallback fraction

## 26. Why Harvest LLM is not the main evidence chain

The Harvest LLM path exists and matters for the future direction, but it did not clear the reliability gate strongly enough for main evidence.

The gate was:

- effective LLM fraction at least `0.90`
- unrepaired fallback fraction at most `0.05`

That standard was not met reliably enough in the targeted Harvest cells.

## 27. The strategy-bank direction

The codebase already includes the next bridge to LLM-population work.

That bridge uses strategy banks:

- model-labeled
- provider-labeled
- attitude-labeled
- deduplicated by policy signature

The two main attitudes in the current bank utilities are:

- cooperative
- exploitative

This makes it possible to sample whole populations from model-generated strategies in a controlled way.

## 28. What Module B would do

Module B is the LLM-population governance map.

It would:

- load a Harvest strategy bank
- sample populations at different exploitative shares
- compare `none`, `top_down_only`, and `hybrid`
- test whether the architecture story survives when the population is built from model-generated strategies

## 29. What Module C would do

Module C is the LLM-population turnover pilot.

It would:

- treat `(model, attitude)` as a gene
- evaluate populations over generations
- let better-performing genes survive
- repopulate from survivor-weighted banks
- track whether exploitative types spread
- test whether governance slows or prevents that spread

## 30. What is already in place for that future direction

The code already has:

- strategy-bank prompt generation
- bank loading and sampling
- governance-map runner
- turnover runner

So the future direction is not speculative. The infrastructure is already there.

## 31. What is still left to do from here

The most sensible next steps are:

1. decide whether the current paper is being finalized and submitted as the main benchmark paper
2. if yes, treat Module B as the next major empirical bridge
3. only run Module C after Module B is stable and interpretable
4. if needed, add broader sensitivity analysis within each scenario family

## 32. What I should say about forward direction

The next step is to build the LLM-population bridge inside Harvest rather than leaving it as infrastructure only. The cleanest order is to run the governance-map module first, because it directly tests whether the current architecture conclusions survive when the population is sampled from model-generated strategy banks. After that, the turnover pilot becomes the right next step, because it asks whether governance can stop exploitative strategy families from spreading over generations.

## 33. What I should say about the role of the older fishery work

The older fishery work still matters because it established the core invasion-pressure protocol in the simplest setting. It showed that central interventions can be ranked under repeated strategic turnover, and it gave the project a clean methodological foundation before the move into richer architecture questions in Harvest.

## 34. What I should say if someone asks why I moved from Fishery to Harvest

I moved from Fishery to Harvest because Fishery is ideal for comparing central signals, while Harvest is the right environment for comparing governance architectures. Once the question becomes local-only versus central-only versus hybrid governance, Harvest is the more informative substrate.

## 35. What I should say if someone asks where the RL work fits

The RL work is supporting validation rather than the main evidence chain. It helped show that the qualitative ranking from the earlier papers was not tied only to hand-designed threshold strategies. The strongest claims in the project still come from the invasion benchmarks.

## 36. What each main figure means

### Baseline Harvest architecture figure

This figure shows the strong baseline result from the confirmatory Harvest architecture matrix. It shows how hybrid compares to top-down-only across matched cells, especially on patch health, welfare, and control-related metrics.

### Capability ladder figure

This figure shows the first attacker rung at which a given architecture contrast loses its advantage. It tells me where an architecture breaks as pressure rises.

### Institutional-friction winner map

This figure shows which architecture ranks first in each scenario, oversight regime, and pressure cell. The annotation inside each cell gives the hybrid-minus-top-down patch-health difference.

## 37. What I should say if someone asks for the single clean contribution

I built a benchmark for institutional robustness in stylized commons. It shows that governance architecture matters under strategic pressure, that central intervention is consistently important for ecological robustness, and that the ranking between hybrid and central-only governance depends on scenario structure and oversight constraints.

## 38. Quick answers for common questions

### What is the project in one sentence?

It is a governance benchmark for renewable commons under changing strategy populations.

### What is the strongest result?

The strongest baseline result is the confirmatory Harvest matrix where hybrid ranks first in all eight decision cells.

### What is the strongest new result?

The strongest new result is that the ranking between hybrid and top-down-only governance becomes scenario-dependent once oversight frictions are introduced.

### What is the most important limitation?

The current paper is strongest on the non-LLM benchmark side. The LLM-population bridge is prepared but not yet part of the main finished evidence chain.

### What is the next step?

Run the Harvest strategy-bank governance-map module, then the turnover pilot.

### Why does this matter?

It gives controlled evidence about governance design under strategic pressure and makes the ecological-versus-welfare trade-off explicit.

## 39. Short script I can read near the end of a meeting

Since the earlier fishery-centered stage, the project has become a stronger and more institutionally focused benchmark. I used Fishery Commons to establish that central interventions can be ranked under repeated strategic turnover. I then moved to Harvest Commons to compare governance architectures directly. That work showed that architectures with central intervention are much more robust ecologically than local-only governance, and that hybrid governance is often strong. In the current extension, I added literature-backed scenario archetypes and explicit oversight frictions. The newer result is more nuanced and more useful: the ranking between hybrid and top-down-only governance depends on the scenario and on how effectively governance can actually be implemented. From here, the clean next step is to connect that benchmark more directly to LLM-generated strategy populations using the strategy-bank and turnover infrastructure that is already in place.

