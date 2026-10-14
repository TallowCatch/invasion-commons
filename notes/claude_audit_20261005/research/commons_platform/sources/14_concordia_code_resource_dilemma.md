# Concordia source: `examples/resource_dilemma` (Pasture, Irrigation, Network, Fishery; standard vs election)

- **Authors:** Google DeepMind / Concordia contributors (follows Faulkner et al. 2026 and Piatti et al. 2024)
- **Year:** 2026; repository state 13 Oct 2026
- **Identifier:** github.com/google-deepmind/concordia, commit `c18c968a653202b1291ae3c3322082588c5440f7`. Files: `examples/resource_dilemma/README.md`, `scenarios/fishery.py`, `gamemaster/resource_components.py`
- **URL:** https://github.com/google-deepmind/concordia/tree/main/examples/resource_dilemma
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026

## Verbatim lines

1. README:
   > "Each scenario supports two governance modes: "standard" (unregulated free-for-all) and "election" (agents elect leaders and vote on harvest policies)."
2. `scenarios/fishery.py` lines 36–37 and 49:
   ```python
   DEFAULT_NUM_CYCLES = 6
   DEFAULT_CAPACITY = 100
   ```
   > "and the lake has a carrying capacity of 120 tonnes."
3. `scenarios/fishery.py` (leaders' policies in election mode):
   > "a) A recommended catch limit (max tonnes per fisher per cycle) b) A penalty schedule for violators"
4. `gamemaster/resource_components.py` line 683 (resource dynamics are plain Python):
   ```python
   s.resource_level = min(s.resource_level * 2.0, s.carrying_capacity)
   ```
5. `gamemaster/resource_components.py` (harvest resolution subtracts the parsed total):
   ```python
   self._sim_state.resource_level = max(
       0.0, self._sim_state.resource_level - total_harvest
   ```

## Relevance
This is the nearest existing Concordia counterpart to Ameer's Fishery. It is GovSim-style (stock doubles, capped at capacity) with an elected-leader institution. The "penalty schedule" exists only in prompt text. A grep of the GM code for penalty, violation, enforcement and sanction finds no code that detects violations or applies fines, so there are no audits or enforced sanctions. Adding a GM component that audits and fines would be new and would fit Concordia's design.
