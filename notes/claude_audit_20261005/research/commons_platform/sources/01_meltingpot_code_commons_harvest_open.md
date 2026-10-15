# Melting Pot source: `commons_harvest__open.py` substrate config

- **Authors:** DeepMind Technologies Limited (Melting Pot team)
- **Year:** 2022 (file copyright); repository state as of 13 Oct 2026
- **Identifier:** github.com/google-deepmind/meltingpot, file `meltingpot/configs/substrates/commons_harvest__open.py`, commit `bc456f694a08dc6c4019fa9ca2fee1caaa2f4928`
- **URL:** https://github.com/google-deepmind/meltingpot/blob/main/meltingpot/configs/substrates/commons_harvest__open.py
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026 (shallow clone, read only, nothing executed)

## Verbatim code / docstring lines

1. Lines 57–58:
   ```python
   APPLE_RESPAWN_RADIUS = 2.0
   REGROWTH_PROBABILITIES = [0.0, 0.0025, 0.005, 0.025]
   ```
2. Lines 34–35 (the only stated origin of the mechanism):
   > "This mechanism was first described in Janssen et al (2010) and adapted for multi-agent reinforcement learning in Perolat et al (2017)."
3. Lines 305–310 (episode ending, in the scene object):
   ```python
   "component": "StochasticIntervalEpisodeEnding",
   "kwargs": {
       "minimumFramesPerEpisode": 1000,
       "intervalLength": 100,  # Set equal to unroll length.
       "probabilityTerminationPerInterval": 0.15
   ```
4. Lines 493–501 (zapper):
   ```python
   "component": "Zapper",
   "kwargs": {
       "cooldownTime": 2,
       "beamLength": 3,
       "beamRadius": 1,
       "framesTillRespawn": 4,
       "penaltyForBeingZapped": 0,
       "rewardForZapping": 0,
   ```
5. Lines 482–487 (egocentric view) and 560, 578:
   ```python
   "view": {"left": 5, "right": 5, "forward": 9, "backward": 1, "centered": False},
   config.default_player_roles = ("default",) * 7
   maxEpisodeLengthFrames=5000,
   ```
6. Line 377: `"rewardForEating": 1.0,` (the only reward source; apples, map 24 x 18 cells, 64 apple cells counted from the ASCII map).

## Relevance
All the settings are bare constants. There is no comment on how 0.0025/0.005/0.025, radius 2, the 4-frame zap timeout or the 0.15 stopping chance were picked. Note that the code's one-neighbour rate (0.0025) is not the 0.001 printed in both Melting Pot papers (see 07, 08); the closed and partnership files do use 0.001 (see 02). So a published paper and the shipped code disagree on a key commons parameter.
