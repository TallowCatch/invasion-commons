# CoopEval: Benchmarking Cooperation-Sustaining Mechanisms and LLM Agents in Social Dilemmas

- **Authors:** Emanuel Tewolde, Xiao Zhang, David Guzman Piedrahita, Vincent Conitzer, Zhijing Jin
- **Year:** 2026
- **Venue / identifier:** arXiv:2604.15267
- **URL:** https://arxiv.org/abs/2604.15267
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full HTML text read)

## Verbatim quotes

1. Scope:
   > "Across four social dilemmas testing distinct components of robust cooperation, we evaluate four families of mechanisms: (1) repeating the game for many rounds, (2) reputation systems, (3) third-party mediators to delegate decision making to, and (4) contract agreements for outcome-conditional payments between players."
2. Base game parameterisation:
   > "In the Prisoner's Dilemma ( Table 1 ), cooperating costs 1 unit to a player in order to generate 2 units for the other player."
3. Mechanism parameters chosen by proof, not by data:
   > "In our main experiments with Repetition and Reputation , we include information on action outcomes from the past k=3 rounds, and set the continuation probability to \delta=0.8 . As proven in Appendix D , these settings are comfortably sufficient to sustain cooperation in our social dilemmas."
4. What counts as a mechanism:
   > "Our mechanisms are firmly grounded in game theory."
   (Theorem 1: each mechanism lets a Pareto-improving profile be sustained in subgame perfect equilibrium.)
5. Sample size and metrics:
   > "Each combination is repeated three times. This sums to 8586 decisions per LLM model, or >50.000 in total."
   Metrics: mean cross-play payoff, "Fitness" after 1000 steps of discrete replicator dynamics, and deviation ratings (Marris et al. 2025).
6. Agents' goal:
   > "each LLM is instructed to maximize its own (total) points from the mechanism-modified game."

## Relevance
CoopEval validates its games the textbook way: small normal-form games whose dilemma property is provable, and mechanism parameters (δ, k) set so a theorem guarantees cooperation is an equilibrium. It also evaluates populations of mixed LLMs with cross-play and evolutionary (replicator) metrics. For Ameer, this is the model for showing an overseer is a "mechanism" in a formal sense: show that, under the audit probability and fine chosen, compliance is an equilibrium for a selfish agent, then measure LLMs against that prediction.
