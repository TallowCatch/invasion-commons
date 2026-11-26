# Melting Pot source: `commons_harvest__closed.py` and `commons_harvest__partnership.py`

- **Authors:** DeepMind Technologies Limited
- **Year:** 2022; repository state 13 Oct 2026
- **Identifier:** github.com/google-deepmind/meltingpot, files `meltingpot/configs/substrates/commons_harvest__closed.py` and `commons_harvest__partnership.py`, commit `bc456f694a08dc6c4019fa9ca2fee1caaa2f4928`
- **URL:** https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/configs/substrates
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026

## Verbatim code / docstring lines

1. `commons_harvest__closed.py` line 41 (same in partnership line 44):
   ```python
   REGROWTH_PROBABILITIES = [0.0, 0.001, 0.005, 0.025]
   ```
2. `commons_harvest__closed.py` zapper (lines 478–481), the same in partnership (511–514):
   ```python
   "cooldownTime": 1,
   "beamLength": 4,
   ...
   "framesTillRespawn": 100,
   ```
3. Closed docstring:
   > "In the case of _Commons Harvest: Closed, agents can learn to defend naturally enclosed regions. Once they have done that then they have an incentive to avoid overharvesting the patches within their region."
4. Closed docstring, continued:
   > "However, they usually involve significant inequality since many agents are excluded from any natural region."
5. Partnership adds an invisible tile that punishes one role (lines 238–258):
   ```python
   {"component": "RoleBasedRewardTile",
    "kwargs": {
        "avatarRoleComponent": "Role",
        "getRoleFunction": "getRole",
        "rolesToRewards": {"putative_cooperator": -10},
   ```

## Relevance
In the closed variants, the zap timeout is 25 times longer than in Open (100 frames against 4), and the one-neighbour regrowth rate is lower. The only reason given is behavioural: walls make regions defensible. The −10 tile is the training-time pseudoreward used to make "good partner" bots (see 07). It shows that Melting Pot builds bot behaviour by changing rewards at training time, not by scripting. Each variant has 7 players, a 5000-frame cap and the same stochastic ending (min 1000 frames, then a 0.15 chance per 100 frames).
