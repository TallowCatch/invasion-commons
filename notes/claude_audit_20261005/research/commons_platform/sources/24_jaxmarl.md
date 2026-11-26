# 24. JaxMARL: Multi-Agent RL Environments and Algorithms in JAX

- **Authors:** Alexander Rutherford, Benjamin Ellis, Matteo Gallici, Jonathan Cook, Andrei Lupu, Garðar Ingvarsson, Timon Willi, Ravi Hammond, Akbir Khan, Christian Schroeder de Witt, Alexandra Souly, Saptarashmi Bandyopadhyay, Mikayel Samvelyan, Minqi Jiang, Robert Tjarko Lange, Shimon Whiteson, Bruno Lacerda, Nick Hawes, Tim Rocktäschel, Chris Lu, Jakob Nicolaus Foerster
- **Year / venue:** NeurIPS 2024 Datasets and Benchmarks Track (README BibTeX)
- **Identifier:** github.com/FLAIROx/JaxMARL, commit `976aeb152cb184a5095021968bba94da96eb6394` (10 Sep 2026). Cloned `--depth 1`, read only. Licence Apache-2.0.
- **Source type:** primary-code / docs
- **Accessed:** 14 Oct 2026

## Verbatim quotes

1. README environment table, STORM: "Matrix games represented as grid world scenarios"
2. README environment table, Coin Game: "Two-player grid world environment which emulates social dilemmas"
3. `jaxmarl/environments/storm/README.md`: "Agents collect resources which define their strategy during interactions and are rewarded based on a pre-specified payoff matrix."
4. OpenSocialJax related work (source 22): "JaxMARL ( Rutherford et al., 2024 ) is a general-purpose MARL library that includes Coin Game and STORM, two grid-world environments that support the study of social dilemmas."

## Contents of `jaxmarl/environments/` at this commit

coin_game, hanabi, jaxnav, mabrax, mpe, overcooked, overcooked_v2, robotarium, smax, storm, switch_riddle. No Commons Harvest, Clean Up or any renewable-resource game.

## Relevance

JaxMARL has no commons or common-pool-resource game; its social-dilemma content is Coin Game and STORM (in-the-matrix style). SocialJax copies its API (`MultiAgentEnv`, `make`, wrappers), so JaxMARL matters only as the interface standard, not as a substrate for an overseer study.
