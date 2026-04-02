# Harvest Capability Ladder Stage B Closeout

## What was run

- Preset: `capability_ladder_stageB`
- Conditions: `bottom_up_only`, `top_down_only`, `hybrid`
- Tiers: `medium_h1`, `hard_h1`
- Partner mixes: `balanced`, `adversarial_heavy`
- Pressures: `0.3`, `0.5`
- Injectors: `random`, `mutation`, `adversarial_heuristic`, `search_mutation`
- Runs per cell: `3`

## Main readout

- `hybrid` ranks first in `25 / 32` capability-ladder cells.
- `top_down_only` ranks first in `7 / 32` cells.
- No local-only condition ranks first.

By injector rung:

- `random`: `hybrid` wins `8 / 8`
- `mutation`: `hybrid` wins `6 / 8`, `top_down_only` wins `2 / 8`
- `adversarial_heuristic`: `hybrid` wins `4 / 8`, `top_down_only` wins `4 / 8`
- `search_mutation`: `hybrid` wins `7 / 8`, `top_down_only` wins `1 / 8`

## First-break interpretation

### Governed vs local-only

- For `hybrid - bottom_up_only`, the first ecological break is `none` in all `8 / 8` decision cells.
- For `top_down_only - bottom_up_only`, the first ecological break is also `none` in all `8 / 8` cells.

So the ecological split between architectures with central intervention and local-only governance survives the full non-LLM attacker ladder.

### Hybrid vs top-down-only

- Ecological break:
  - `none` in `3 / 8`
  - `mutation` in `2 / 8`
  - `adversarial_heuristic` in `2 / 8`
  - `search_mutation` in `1 / 8`
- Control break:
  - `none` in `3 / 8`
  - `mutation` in `1 / 8`
  - `adversarial_heuristic` in `3 / 8`
  - `search_mutation` in `1 / 8`
- Costly-robustness flag:
  - `random` in `6 / 8`
  - `search_mutation` in `1 / 8`
  - `none` in `1 / 8`

So the contested part of the ordering is not governed vs local-only. It is the finer `hybrid` vs `top_down_only` margin.

## Substantive conclusion

The capability ladder strengthens the Harvest architecture story in a useful way.

- The benchmark now shows a stable ecological advantage for architectures with central intervention over local-only governance.
- It also shows that the `hybrid` advantage over `top_down_only` is real but conditional: it often survives into stronger attacker rungs, but it is the part of the ordering most sensitive to attacker family and welfare cost.

That is enough to make the Harvest-led paper empirically complete under the planned gate. The remaining work is manuscript integration, not another required experiment.
