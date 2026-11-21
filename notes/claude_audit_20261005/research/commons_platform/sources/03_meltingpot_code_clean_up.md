# Melting Pot source: `clean_up.py` config and `lua/levels/clean_up/components.lua`

- **Authors:** DeepMind Technologies Limited
- **Year:** 2022; repository state 13 Oct 2026
- **Identifier:** github.com/google-deepmind/meltingpot, `meltingpot/configs/substrates/clean_up.py` and `meltingpot/lua/levels/clean_up/components.lua`, commit `bc456f694a08dc6c4019fa9ca2fee1caaa2f4928`
- **URL:** https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/clean_up.py
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026

## Verbatim code / docstring lines

1. `clean_up.py` lines 398–403:
   ```python
   "component": "AppleGrow",
   "kwargs": {
       "maxAppleGrowthRate": 0.05,
       "thresholdDepletion": 0.4,
       "thresholdRestoration": 0.0,
   ```
2. `clean_up.py` lines 601–612:
   ```python
   "dirtSpawnProbability": 0.5,
   "delayStartOfDirtSpawning": 50,
   ...
   "minimumFramesPerEpisode": 1000,
   "intervalLength": 100,  # Set equal to unroll length.
   "probabilityTerminationPerInterval": 0.2
   ```
3. `clean_up.py` zapper and cleaner (lines 707–727):
   ```python
   "cooldownTime": 10, "beamLength": 3, "beamRadius": 1, "framesTillRespawn": 50,
   "penaltyForBeingZapped": 0, "rewardForZapping": 0, "removeHitPlayer": True,
   ...
   "component": "Cleaner", "kwargs": {"cooldownTime": 2, "beamLength": 3, "beamRadius": 1,
   ```
4. `components.lua` lines 64–76 (apple growth rule):
   ```lua
   local dirtFraction = dirtCount / (dirtCount + cleanCount)
   ...
   local interpolation = (dirtFraction - depletion) / (restoration - depletion)
   ...
   local probability = self._config.maxAppleGrowthRate * interpolation
   ```
5. `components.lua` docstring:
   > "`thresholdDepletion` (float in [0, 1]): Once dirt fraction exceeds this value then the apple regrowth rate drops to 0.0."
6. `clean_up.py` docstring:
   > "Clean Up was first described in Hughes et al. (2018)."

## Relevance
In Clean Up, each potential apple grows with probability 0.05 × (1 − dirt fraction/0.4) per step. Dirt arrives with probability 0.5 per step, starting after 50 frames. The game has 7 players, a 30 x 21 map, 9 actions (one of them clean) and an RGB view of 88 x 88. As with Commons Harvest, the only justification is a citation to the paper that introduced the game. It also emits a puppeteer signal, `NUM_OTHERS_WHO_CLEANED_THIS_STEP`, that conditional bots read (see 05).
