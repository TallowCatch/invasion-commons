# Inequity aversion improves cooperation in intertemporal social dilemmas

- **Authors:** Edward Hughes, Joel Z. Leibo, Matthew Phillips, Karl Tuyls, Edgar Dueñez-Guzman, Antonio García Castañeda, Iain Dunning, Tina Zhu, Kevin McKee, Raphael Koster, Heather Roff, Thore Graepel
- **Year:** 2018
- **Venue / identifier:** NeurIPS 2018; arXiv:1803.08884
- **URL:** https://arxiv.org/abs/1803.08884
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. Cleanup parameters (App. A.3):
   > "In Cleanup, waste is produced uniformly in the river with probability 0.5 on each timestep, until the river is saturated with waste, which happens when the waste covers 40% of the river."
2. > "For a given saturation x of the river, apples spawn in the field with probability 0.125x."
3. Harvest parameters (App. A.3):
   > "The spawn probabilities are 0, 0.005, 0.02, 0.05 for 0, 1, 2 and ≥ 3 apples inside the radius respectively."
4. Sanctions were fines here:
   > "all agents are equipped with a fining beam which administers −1 reward to the user and −50 reward to the individual that is being fined."
5. How the games were validated as dilemmas (Sec. 2.5):
   > "In complex, spatially and temporally extended Markov games, it is not feasible to analytically determine cooperating and defecting policies. Instead, we must study the environment empirically."
6. > "In Harvest, we enforce cooperation by modifying the environment to prevent some agents from gathering apples in low-density areas. In Cleanup, we enforce free-riding by removing the ability of some agents to clean up waste."

## Relevance
This gives a third parameter set for Harvest (0/0.005/0.02/0.05, ℓ1 radius 2, 5 agents, 1000 steps), different again from Perolat and Melting Pot. The settings are validated after the fact with empirical Schelling diagrams, built by forcing some agents to cooperate or defect; they are not taken from data. It is the closest published precedent for fines in a gridworld commons: the punisher pays −1, the target −50. That gives Ameer a cited fine-to-cost ratio to compare against his own.
