# Multi-agent Reinforcement Learning in Sequential Social Dilemmas

- **Authors:** Joel Z. Leibo, Vinicius Zambaldi, Marc Lanctot, Janusz Marecki, Thore Graepel
- **Year:** 2017
- **Venue / identifier:** AAMAS 2017; arXiv:1702.03037
- **URL:** https://arxiv.org/abs/1702.03037
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full HTML text read)

## Verbatim quotes

1. Matrix-game social dilemma inequalities (after Macy & Flache):
   > "R>P Mutual cooperation is preferred to mutual defection."
   > "R>S Mutual cooperation is preferred to being exploited by a defector."
   > "2R>T+S This ensures that mutual cooperation is preferred to an equal probability of unilateral cooperation and defection."
   > "either greed : T>R Exploiting a cooperator is preferred over mutual cooperation"
   > "or fear : P>S Mutual defection is preferred over being exploited."
2. SSD definition rests on empirical game-theoretic analysis:
   > "This definition is based on a formalization of empirical game-theoretic analysis [ 20 , 21 ] ."
3. Cooperation is graded, so C/D policy sets are made by thresholding a behaviour metric:
   > "Thus we are forced to define \Pi^{C} and \Pi^{D} by thresholding a continuous social behavior metric ."
4. How the C and D policies were obtained in Gathering:
   > "We chose the set of policies \Pi^{C} that were trained in the high abundance / low conflict-cost environments (low aggression policies) and \Pi^{D} as policies trained in the low abundance and high conflict-cost environments (high aggression policies), and used these to compute empirical payoff matrices as follows."
5. Parameters are swept, not taken from data:
   > "By manipulating the rate at which apples respawn after being collected, N_{\text{apple}} , we could control the abundance of apples in the environment."

## Relevance
The founding definition of a sequential social dilemma: a Markov game is a dilemma only if policies labelled cooperative and defecting, when matched, yield an empirical payoff matrix satisfying the inequalities. Environment parameters (respawn time, tag time-out) were swept over a grid and the resulting games classified as Prisoner's Dilemma, Chicken or Stag Hunt (Fig. 6). This is the procedure Ameer can apply to Fishery/Forest/River with programmed "restrained" and "greedy" policies to show each setting is a dilemma, and of which kind.
