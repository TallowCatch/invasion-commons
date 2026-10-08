# Institutional Commons Module A Closeout

## Scope

Module A completed the first real readout for the literature-backed institutional commons extension. The completed runs cover three scenario archetypes:

- `regulated_fishery`
- `community_irrigation`
- `forest_co_management`

Each scenario was evaluated under:

- `top_down_only`, `bottom_up_only`, and `hybrid`
- `ideal` and `constrained` oversight regimes
- `search_mutation`
- two pressure levels per scenario

## Main Result

The result is strong enough for the paper, but it is not a universal `hybrid` win.

The clean interpretation is:

> governance architecture rankings become scenario-dependent once realistic oversight frictions are introduced.

## Scenario-Level Readout

### Community irrigation

This is the strongest case for multi-scale governance.

- `hybrid` ranks first in all four tested cells
- `hybrid - top_down_only` is positive on patch health in every cell
- the constrained regime does not overturn the ranking

### Forest co-management

This is the clearest friction-sensitive case.

- under `ideal`, `hybrid` ranks first in both cells
- under `constrained`, `top_down_only` ranks first in both cells

This means the friction layer is substantively changing the institutional conclusion rather than acting as a cosmetic robustness check.

### Regulated fishery

This scenario produces a pressure-sensitive split.

- at lower pressure, `hybrid` ranks first
- at higher pressure, `top_down_only` ranks first
- this split appears in both oversight regimes

## Cross-Scenario Summary

For the paired `hybrid - top_down_only` comparison:

- patch health is positive in `5/6` cells under `ideal`
- patch health is positive in `5/6` cells under `constrained`

But ranking tells the more decision-relevant story:

- under `ideal`, `hybrid` ranks first in `5/6` cells
- under `constrained`, `hybrid` ranks first in `3/6` cells

So ecological gains often remain, but they do not always survive once welfare and control tradeoffs are included.

The decentralized baseline remains ecologically weakest:

- `top_down_only` beats `bottom_up_only` on patch health in all evaluated cells
- `hybrid` also beats `bottom_up_only` on patch health in all evaluated cells

`bottom_up_only` continues to provide the highest-welfare but least robust baseline, preserving the central welfare-versus-robustness tradeoff from the earlier Harvest work.

## Paper Implication

The paper should not claim that hybrid governance universally dominates.

It should claim that:

- institutional rankings differ across literature-backed commons archetypes
- multi-scale governance is especially strong in irrigation-like settings
- central-only governance can regain the advantage under tighter oversight constraints or higher-pressure regulator-facing settings
- the benchmark is useful because it identifies where each architecture remains robust

## Immediate Next Step

Do not treat Modules B and C as required before writing.

The right sequence after Module A is:

1. consolidate the Module A artifacts in the repo
2. update `paper_v4` around the non-universal result
3. decide whether LLM strategy-bank evidence is needed as support, rather than assuming it must carry the main claim
