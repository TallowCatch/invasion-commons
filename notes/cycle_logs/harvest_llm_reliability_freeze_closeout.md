# Harvest LLM Reliability Freeze Closeout

## Outcome

The Harvest live `llm_json` path is now technically real but remains frozen as capability-only evidence. It should not be promoted into the main Study 2 evidence chain for the current paper.

## What changed in this cycle

- Tightened the Harvest JSON prompt around a single valid object, explicit field meanings, and numeric ranges.
- Lowered local evidence-run temperature to `0.2`.
- Added deterministic repair before mutation fallback for:
  - fenced JSON
  - trailing prose
  - single-quoted JSON
  - numeric strings
  - missing outer object
- Extended integrity reporting with:
  - `repaired_json_fraction`
  - `effective_llm_fraction`
  - `unrepaired_fallback_fraction`
  - grouped parse-error counts
- Added narrow local shard presets for:
  - Phase A reliability
  - Phase B narrow pilot
  - Phase C narrow confirmatory

## Phase A result

Phase A passed after the repair and integrity-accounting fixes.

- All targeted reliability cells reached:
  - `effective_llm_fraction = 1.0`
  - `unrepaired_fallback_fraction = 0.0`

This was enough to justify running the narrow matched pilot.

## Phase B result

Phase B looked scientifically promising but failed the evidence gate on integrity.

- Mean `effective_llm_fraction = 0.7311`
- Mean `unrepaired_fallback_fraction = 0.2689`
- `usable_for_evidence = false` in all `8 / 8` LLM cells
- Dominant parse failure mode: `missing_keys`

The paired direction was still broadly favorable to `hybrid`:

- patch health positive in `3 / 4` LLM comparison cells
- neighborhood overharvest non-worse in `3 / 4`
- mean welfare delta remained moderate rather than catastrophic

That makes the path scientifically interesting, but still not evidence-grade.

## Decision

- Do not run Harvest LLM Phase C in the current paper cycle.
- Do not patch the paper to treat Harvest LLM as main or supporting quantitative evidence.
- Keep the main Study 2 claim anchored in the `search_mutation` / search-over-mutations result.
- Keep Harvest LLM described only as a capability and diagnostics path in notes and limitations.

## Best next move

If Harvest LLM is revisited, the highest-value next step is not a larger matrix. It is a narrow model-selection or output-format cycle aimed at schema completeness:

1. try a stronger local model, or
2. simplify the output format away from free-form JSON object generation,

then rerun Phase A and Phase B only.
