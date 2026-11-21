# 25. The AI Economist: Taxation policy design via two-level deep multiagent reinforcement learning

- **Authors:** Stephan Zheng, Alexander Trott, Sunil Srinivasa, David C. Parkes, Richard Socher
- **Year / venue:** 2022, *Science Advances* 8(18): eabk2607, 4 May 2022. doi:10.1126/sciadv.abk2607
- **URL:** https://pmc.ncbi.nlm.nih.gov/articles/PMC9067926/
- **Source type:** academic
- **Accessed:** 14 Oct 2026 (PMC HTML downloaded, text extracted)

## Verbatim quotes

1. Timing: "Each simulation simulates 10 tax years. Each tax year lasts 100 time steps (so that H = 1000), with the agents acting each time step, and the planner setting and changing tax rates at the start of each tax year."
2. How parameters were chosen: "We tuned the environment hyperparameters to roughly align the income distribution under U.S. Federal taxes (in the four-agent case) with income frequencies reported under those taxes in 2018."
3. Also: "All of our experiments use a fixed set of build-skills, which, along with labor costs, are roughly calibrated so that (i) agents need to be strategic in how they choose to earn income and (ii) the shape of the resulting income distribution roughly matches that of the 2018 U.S. economy with trained optimal agent behaviors."
4. Training: "In phase one, agent policies are trained from scratch in a free-market (no-tax) environment for 50 million steps. In phase two, agents continue to learn in the presence of taxes for another 1 billion steps."
5. Stabilisation: "We empirically found that the planner fails to improve its objective if it starts learning prematurely, i.e., when the entropy coefficient is annealed too quickly."
6. Baselines and result: "In the Open-Quadrant layout with 4 (10) agents ( Fig. 3 ), AI-driven taxes improve the utilitarian objective by more than 8% (2%) and the product of equality and productivity by more than 12% (8.6%) over the Saez tax."

## Other facts (paraphrased)

- Four planners compared: free market (0%), U.S. Federal 2018 schedule, Saez formula (elasticity chosen by grid search; offline regression gave about 0.87 but about 3 was best), AI Economist.
- Evaluation: end-of-training metrics over 10 seeds (5 for one-step / Split-World), standard errors, significance stars; equality = 1 − Gini of coin; productivity = total coin.
- Agents observe a local spatial view, endowments, skill, market state, tax rates; CNN + MLP + LSTM policies.

## Relevance

The closest published "overseer in a 2D grid": a non-embodied planner that acts once per 100-step period, observes incomes, and changes everyone's payoff. It shows the standard evaluation for a planner (fixed-rule baselines including a theory-derived one, two welfare objectives, seeds with SE, behavioural analysis such as tax gaming). Its settings were calibrated to an external target (2018 US income distribution), unlike SSD substrates. Its world has no commons dilemma (see source 26).
