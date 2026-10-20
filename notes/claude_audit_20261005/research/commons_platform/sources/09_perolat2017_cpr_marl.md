# A multi-agent reinforcement learning model of common-pool resource appropriation

- **Authors:** Julien Perolat, Joel Z. Leibo, Vinicius Zambaldi, Charles Beattie, Karl Tuyls, Thore Graepel
- **Year:** 2017
- **Venue / identifier:** NeurIPS 2017 (NIPS 30); arXiv:1707.06600
- **URL:** https://arxiv.org/abs/1707.06600
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. Original regrowth rule (Appendix A, Eq. 1): per-step respawn probability p_t(L) = "0 if L = 0; 0.01 if L = 1 or 2; 0.05 if L = 3 or 4; 0.1 if L > 4", where L counts apples "in a ball of radius 2".
2. The only design principle stated for the map (Fig. 6 caption):
   > "Map 6b is considerably larger. It was designed so that a single agent, acting alone cannot remove all the apples but with several agents harvesting simultaneously it becomes relatively easy to remove them all."
3. Link to human experiments and the deliberate departure from fines:
   > "So far, the proposed commons game is quite similar to the dynamic game used in human behavioral experiments [16, 18, 17, 14, 15]. However, it departs in one notable way."
4. > "participants were given the option of paying a fee in order to fine another participant, reducing their score. In contrast, in our commons game, agents can tag one another with a "time-out beam"."
5. > "Any agent caught in the path of the beam is removed from the game for 25 steps." and "Each episode lasted for 1,000 steps."
6. Evaluation tools:
   > "we show how behavior of groups may be characterized along four social outcome metrics called: efficiency, equality, sustainability, and peace."

## Relevance
The values here (0/0.01/0.05/0.1, a 25-step tag, 1000 steps) differ from Melting Pot's (0/0.001–0.0025/0.005/0.025, a 4- or 100-frame zap). So the parameters were re-set in each paper, and none of them gives a numeric rationale. The grounding offered is qualitative: a single agent cannot deplete the map, a group can, and this is checked with Schelling diagrams (empirical game-theoretic analysis). Perolat swapped the human-lab fine for a time-out beam on purpose, which matters because Ameer's overseer uses fines.
