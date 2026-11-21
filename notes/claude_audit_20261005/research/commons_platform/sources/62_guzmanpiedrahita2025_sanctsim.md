# Corrupted by Reasoning: Reasoning Language Models Become Free-Riders in Public Goods Games (SanctSim)

- **Authors:** David Guzman Piedrahita, Yongjin Yang, Mrinmaya Sachan, Giorgia Ramponi, Bernhard Schölkopf, Zhijing Jin
- **Year:** 2025
- **Venue / identifier:** arXiv:2506.23276 (v2, 24 Jul 2025); code https://github.com/davidguzmanp/SanctSim
- **URL:** https://arxiv.org/abs/2506.23276
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full HTML text read)

## Verbatim quotes

1. Where the game comes from:
   > "Inspired by the experimental paradigm of Gurerk et al. (2006) , we model a public goods dilemma in which individual contributions benefit the group, but rewards are distributed equally regardless of contribution."
   > "This dual-institution approach, originally explored in human experiments ( Gurerk et al., 2006 ; Sutter et al., 2010 ) , enables us to observe not only contribution patterns but also institutional preferences."
2. Exact parameters:
   > "Our game consists of 15 rounds of the public goods game with institutional choice, simulated using 7 LLM agents."
   > "In each round, all agents receive a 20-token endowment and decide how many tokens to contribute to their institution's common pool. Contributions are multiplied by a factor of 1.6 before being equally distributed among members of that institution."
   Sanctioning: endowment s = 20 tokens; reward costs 1 token for +1; punishment costs 1 token for −3.
3. How the dilemma is checked (analytically, by payoffs):
   > "This setup creates a payoff range where full cooperation yields 52 tokens per agent (the social optimum), while universal free-riding yields only 40 tokens (the Nash equilibrium)—a 12-token gap representing the potential gains from cooperation."
4. Sensitivity analysis:
   > "These findings suggest that the identified behavioral archetypes are not artifacts of a specific parameterization but reflect more fundamental strategic tendencies of the models."
   (Ablations on multiplier α, punishment cost/effect, endowment; single runs.)
5. Runs: 5 independent runs per model, except o1-preview and o3-mini-high (single run, "due to budget constraints"); temperature 1.0 for non-reasoning models.

## Relevance
SanctSim is the clearest LLM example of taking a human-lab paradigm as the template (Gürerk, Irlenbusch & Rockenbach 2006, Science, institutional choice between a sanctioning and a sanction-free institution), with a 20-token endowment, 1.6 multiplier and 1:3 punishment technology, and it compares LLM results with human results (its RQ2 and Table 1). Whether every number matches Gürerk et al. exactly was not checked against the 2006 paper in this run. The dilemma is established analytically (Nash 40 vs optimum 52), and robustness is shown with a small parameter sweep. This is the pattern Ameer can follow for audits/fines: borrow the human-lab numbers and report a sweep.
