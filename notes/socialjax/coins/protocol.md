# Coins with central and decentralised overseers: scripted-bot protocol

Written 22 Nov 2026, **before the sweep ran**. Code: `experiments/socialjax/run_decentralised_bots.py --game coins`; rule in `overseer/socialjax/games.py` (`CoinsRules`), bots in `game_bots.py` (`coins_policy`).

## The game and the rule

SocialJax Coin Game (`coin_game.py` at commit 18e2171):
- 2 players on a 16 x 11 map; the env hard-codes two players, so N = 2 here (more agents would need a rewrite of its reward code).
- Coins of each player's colour appear on empty cells at 0.05% per cell per step for each colour. Player 0 owns red coins, player 1 green.
- Payoff matrix `[[1, 1, -2], [1, 1, -2]]`: taking any coin pays the taker 1; taking the **other** player's coin also costs the owner 2. With individual rewards SocialJax multiplies by the number of players, so a coin is worth 2 to the taker and a stolen coin costs the owner 4. Results are reported in coins (reward / 2).
- So stealing gains the thief 1 coin and loses the pair 1 coin net: a social dilemma (Lerer & Peysakhovich 2017; Foerster et al. 2018).

**The rule:** don't take a coin of the other player's colour. It is checkable from one step: the wrapper reads it off the rewards and the env's `eat_own_coins` log (a test checks that the decoded thefts reproduce the rewards exactly). Taking your own coin is the honest act a false alarm could fall on.

## Setting

999 steps, window 1, perfect inspector, a caught player is frozen for 20 steps. Regions: top and bottom halves of the map (`rows` layout), R = 2.

Conditions as in Harvest (`../harvest/protocol.md`), at k = 1 and 2 total checks per step:
- none;
- central random and central history;
- R = 2: random, history (independent), history + shared suspicion, and history + shared suspicion and coverage (0.25 per message).

With 2 players, central k = 2 checks both every step. 13 conditions, 10 seeds.

Populations: 2 honest; 1 greedy + 1 honest; 2 greedy; 2 sneaky (steal once unchecked for 20 steps); 2 migrator (steal only in the half whose overseer has not checked it for 20 steps).

## Predictions (written before the runs)

Seen in development runs (3 seeds, no overseer), so these are not predictions: 2 honest players collect about 160 coins in all; with 1 greedy + 1 honest, the greedy player nets about 130 and the honest one about −19, and the greedy one steals 33–40 coins.

- **K1, two thieves are worse than one.** With 2 greedy players and no overseer, the pair's total is below the 1 greedy + 1 honest total (about 110 coins), and each greedy player nets less than the lone greedy player does.
- **K2, central random k = 1** catches about half the thefts (1 of 2 players checked); **central history k = 1** catches over 80% once the thief has been caught once; k = 2 catches all.
- **K3, splitting hurts more with 2 players.** With R = 2 and k = 1, each half gets the single check on about half the steps. When both players stand in the same half, half the budget is wasted. Prediction: decentralised random k = 1 catches 35–45% of thefts (central 50%), and 20–30% of checks go unused. At k = 2 (one check per half every step), it catches 70–85% (central 100%).
- **K4, migrators steal more under split overseers** than under one central overseer at the same budget.
- **K5, freezing a caught thief** gives the honest player more coins (it is free to collect while the thief is held), so welfare rises with the catch share.
- **K6, greed stops paying** at an average penalty of 1 coin per theft (computed from catches afterwards), because one theft gains the thief exactly 1 coin.
