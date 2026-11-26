# Concordia source: README, engines, entity prefab, GM components

- **Authors:** Google DeepMind (Concordia team)
- **Year:** 2023–2026; repository state 13 Oct 2026, package version 2.4.0 (`setup.py`)
- **Identifier:** github.com/google-deepmind/concordia, commit `c18c968a653202b1291ae3c3322082588c5440f7`. Files: `README.md`, `concordia/environment/engines/{sequential,simultaneous,asynchronous}.py`, `concordia/prefabs/entity/basic.py`, `concordia/components/game_master/{event_resolution,inventory,payoff_matrix}.py`, `concordia/language_model/call_limit_wrapper.py`, `LICENSE`
- **URL:** https://github.com/google-deepmind/concordia
- **Source type:** primary-code
- **Date accessed:** 14 Oct 2026 (shallow clone, read only)

## Verbatim lines

1. README:
   > "The engine requests player entity actions and asks game master entities to provide observations and resolve those actions. These are roles in that call, rather than intrinsic entity classes."
2. README:
   > "Entities describe their intended actions in natural language, and the GM translates these into appropriate outcomes e.g. checking physical plausibility in simulated worlds."
3. `components/game_master/inventory.py` line 83:
   > `"""A grounded inventory tracking amounts of items in python."""`
4. `prefabs/entity/basic.py` (the default agent asks itself three LLM questions before acting):
   ```python
   pre_act_label=(f'\nQuestion: What situation is {entity_name} in right now?' '\nAnswer'),
   pre_act_label=(f'\nQuestion: What kind of person is {entity_name}?\nAnswer'),
   pre_act_label=(f'\nQuestion: What would a person like {entity_name} do in ' 'a situation like this?\nAnswer'),
   ```
5. `language_model/call_limit_wrapper.py`:
   ```python
   max_calls: int = 1200,
   ```
6. `LICENSE`: "Apache License Version 2.0, January 2004"

## Relevance
There are three engines: sequential, simultaneous and asynchronous. Agents are entities built from components, and the GM is also an entity with components such as event resolution, inventory, payoff matrix, scene tracker and terminate. A search of the whole repo found no Melting Pot or Lab2D bridge; the only match is `/lab2d/` in `.gitignore`. Cost per step is not documented. From the basic prefab, one agent action takes about 4 LLM calls (3 reasoning questions plus the act call), and the GM adds at least one resolution call per event. This is an estimate from reading the code, not a measured figure.
