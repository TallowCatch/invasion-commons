# Research Direction

## Central thesis

The paper studies scalable oversight in sequential commons under capability pressure. It asks how local, global-signal, and hybrid oversight architectures behave when agents become more strategically pressuring and overseers have bounded detection, delay, targeting capacity, and cost.

The clean thesis is that scalable oversight in multi-agent systems can fail compositionally. Each agent can pass a local action check, while the aggregate shared-resource state becomes unsafe after the joint action and repeated resource dynamics. The benchmark measures that failure through local-pass/global-fail rate and compares which oversight architectures reduce unsafe aggregate outcomes.

## Why this is primarily a scalable-oversight benchmark paper

The core design varies actor capability and overseer capability separately. Actor capability is operationalized through the strength of strategy generation: mutation entrants at the low level and search-generated entrants with larger candidate sets and longer internal evaluation horizons at higher levels. Overseer capability is operationalized through detection recall, intervention delay, targeting capacity, and intervention cost.

That makes the main contribution a benchmark formulation and evaluation protocol. The paper does not claim to solve oversight in deployment. It gives a controlled way to ask what happens when the actor side becomes more capable than the overseer side in an interacting resource system.

## Why the sequential commons substrate matters

The commons substrate makes aggregate failure measurable. Agents request harvest from local resource patches, the resource evolves over time, and repeated extraction can reduce system health. That allows the paper to report global unsafe rate, mean patch health, garden failure, welfare, and oversight burden in the same experiment.

This substrate also makes the local/global distinction concrete. A local check can approve each individual request, while the joint trajectory can still leave the resource below a global safety threshold. That is the key reason Harvest is useful for the scalable-oversight question.

## Why local-pass/global-fail is the key diagnostic

Local-pass/global-fail is the paper's most important diagnostic because it measures a specific oversight failure rather than only a bad outcome. It records steps where every local action passes the local predicate, but the next aggregate resource state is unsafe.

This matters because an overseer can appear successful if it only checks individual actions. In a sequential commons, the dangerous behavior can be distributed across agents and time. The diagnostic shows when local approval is insufficient for global safety.

## Why actor/overseer capability separation is the core design choice

The project is strongest when it separates the actor side from the overseer side. Actor capability controls how strong the entering strategies are. Overseer capability controls how well the oversight system detects, responds, targets, and absorbs intervention cost. The capability gap is then an experimental variable rather than a vague description.

The current results already validate part of this design: stronger actor settings produce higher-payoff or more strategically pressuring entrants, while weaker overseer settings change unsafe rate, local-pass/global-fail rate, patch health, and burden. The next step is to separate the overseer limitations more carefully.

## Why the LLM bridge is useful but secondary

The LLM bridge is useful because it shows that model-generated structured strategies can enter the same Harvest pipeline. Local open-weight models generate complete strategies offline, the strategies are validated and deduplicated, and the simulator evaluates them under the same oversight architectures.

The bridge should stay secondary for now. It does not test live LLM agents, frontier models, tool use, memory, or prompt-sensitive deployment behavior. Its value is to show that the benchmark can evaluate model-produced strategy artifacts without changing the core experimental substrate.

## Next experiment

The next technical question is which overseer limitation drives failure: detection recall, delay, targeting capacity, or intervention cost. The current overseer capability axis bundles these, so a fuller ablation is the most direct way to strengthen the benchmark paper.

The best next experiment is therefore a fuller overseer-limit ablation. It should compare strong overseer, recall-only limitation, delay-only limitation, capacity-only limitation, cost-only limitation, bundled limited overseer, and bundled weak overseer. The reduced version should focus on high-coupling commons and high actor capability. The full version should extend across both stress settings and all actor-capability levels.

## What should not be pursued yet

Do not expand into live LLM agents yet. Live agents would introduce prompt design, action validity, context management, memory, tool use, safety-policy interactions, and reproducibility problems. That is a separate paper-sized direction.

Do not add a third LLM model before the core benchmark is stronger. A third model would strengthen the bridge, but it would not answer the main reviewer question about the bundled overseer capability axis.

Do not claim that hybrid oversight solves compositional failure. The current evidence supports a narrower claim: global-signal and hybrid oversight preserve resource health and reduce global unsafe rate, but local-pass/global-fail remains nonzero and threshold-dependent.

Do not claim field-policy calibration. The paper is a controlled benchmark, not a direct policy simulator.
