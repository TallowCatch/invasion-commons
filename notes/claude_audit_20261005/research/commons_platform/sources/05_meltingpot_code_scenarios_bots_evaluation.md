# Melting Pot source: scenarios, bots, puppeteers and evaluation library

- **Authors:** DeepMind Technologies Limited
- **Year:** 2022–2026; repository state 13 Oct 2026
- **Identifier:** github.com/google-deepmind/meltingpot, commit `bc456f694a08dc6c4019fa9ca2fee1caaa2f4928`. Files: `meltingpot/configs/scenarios/__init__.py`, `meltingpot/configs/bots/__init__.py`, `meltingpot/utils/puppeteers/puppeteer.py`, `meltingpot/utils/puppeteers/clean_up.py`, `meltingpot/utils/evaluation/evaluation.py`
- **URL:** https://github.com/google-deepmind/meltingpot/tree/main/meltingpot/configs
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026

## Verbatim lines

1. `configs/scenarios/__init__.py` (scenario dataclass doc):
   > "is_focal: indicates whether the corresponding player slot is to be filled by a focal player or a bot."
2. `configs/scenarios/__init__.py`, `commons_harvest__open_0`:
   ```python
   description=('focals are resident and visited by two bots who zap ' +
                'and harvest unsustainably'),
   ...
   is_focal=(True,) * 5 + (False,) * 2,
   bots_by_role={'default': {'commons_harvest__open__free_0', 'commons_harvest__open__free_1',},},
   ```
3. `configs/bots/__init__.py` (bots are stored neural networks):
   ```python
   commons_harvest__open__pacifist_0=saved_model(
       substrate='commons_harvest__open',
       model='pacifist_0',
   ```
4. `configs/bots/__init__.py` (a puppet bot: a goal-conditioned network driven by a scripted puppeteer):
   ```python
   clean_up__puppet_corrigible_reciprocator_0=puppet(
       ...
       puppeteer_builder=functools.partial(
           clean_up.CorrigibleReciprocator,
           cooperate_goal=_PUPPET_GOALS['clean_up_3_goals']['CLEAN'],
           defect_goal=_PUPPET_GOALS['clean_up_3_goals']['EAT'],
   ```
5. `utils/puppeteers/clean_up.py`:
   > "This puppeteer's behavior depends on the behavior of others. In particular, it tracks the total amount of others' "cleaning", and integrates this signal using a rolling window."
6. `utils/evaluation/evaluation.py` lines 102–103 (the headline metric):
   ```python
   subscribe(focal_return_subject.pipe(ops.map(np.mean)),
             on_next=data['focal_per_capita_return'].append)
   ```

## Relevance
There are scenarios for each commons substrate: Open has 2, Closed 4, Partnership 6, and Clean Up more than 20. Each one fixes how many slots are focal, against bots that are saved RL models or puppets. The library records focal per-capita return and also background per-capita return. Ameer can copy this protocol (a focal population against held-out bots, resident and visitor modes) in his own engine without using Lab2D.
