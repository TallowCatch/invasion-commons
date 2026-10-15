# Melting Pot source: Lua component library and the "concepts" design doc

- **Authors:** DeepMind Technologies Limited
- **Year:** 2021–2026; repository state 13 Oct 2026
- **Identifier:** github.com/google-deepmind/meltingpot, commit `bc456f694a08dc6c4019fa9ca2fee1caaa2f4928`. Files: `meltingpot/lua/levels/commons_harvest/components.lua` (DensityRegrow), `meltingpot/lua/modules/component_library.lua` (StochasticIntervalEpisodeEnding), `meltingpot/lua/modules/avatar_library.lua` (Avatar, Zapper), `meltingpot/utils/substrates/specs.py`, `docs/concepts.md`, `docs/extending.md`
- **URL:** https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/lua
- **Source type:** primary-code + docs
- **Date accessed:** 14 Oct 2026

## Verbatim lines

1. `docs/concepts.md`:
   > "Every object in Melting Pot is a GameObject, this includes the avatars, walls, spawn points, purely logical objects, etc. Everything that does anything is a GameObject."
2. `docs/concepts.md`:
   > "Substrates created with DMLab2D consist of a grid with coordinates `(x, y, layer)`. You can have as many layers as you want."
3. `commons_harvest/components.lua` lines 122–124 (DensityRegrow picks the rate by neighbour count):
   ```lua
   -- If more nearby than probabilities declared then use the last declared
   -- probability in the table (normally the high probability).
   local idx = math.min(idx, #self._config.regrowthProbabilities)
   ```
4. `component_library.lua` lines 891–893 (why episodes end at random):
   > "In game theory it is common to consider repeated interactions that end with some probability after each stage. This assumption is needed in order to avoid trivial backwards induction solutions"
5. `avatar_library.lua` lines 364–367 (how reward is given: components call this):
   ```lua
   function Avatar:addReward(amount)
     local function _addReward(amount)
       self._playerVolatileVariables.reward = (
             self._playerVolatileVariables.reward + amount)
   ```
6. `utils/substrates/specs.py` line 39:
   ```python
   'RGB': dm_env.specs.Array(shape=(88, 88, 3), dtype=np.uint8, name='RGB'),
   ```

## Relevance
This shows how a substrate is built. An ASCII map assigns a prefab to each character. A prefab is a GameObject made of Lua components (StateManager, Transform, Appearance, plus game logic such as Edible, DensityRegrow, Zapper). Reward is whatever components add through `Avatar:addReward`. The only parameter with a written rationale is stochastic episode ending, which avoids backward induction. An overseer (audits, fines) would be one more scene-level Lua component that reads events and calls `addReward` with negative amounts. That needs Lua work inside a C++/Lua engine.
