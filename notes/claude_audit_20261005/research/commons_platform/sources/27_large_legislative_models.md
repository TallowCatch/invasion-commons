# 27. Large Legislative Models: Towards Efficient AI Policymaking in Economic Simulations

- **Authors:** Henry Gasztowtt, Benjamin Smith, Vincent Zhu, Qinxun Bai, Edwin Zhang (Oxford, UCSB, Horizon Robotics, OpenAI/Harvard, Humanity Unleashed)
- **Year / venue:** 2024, arXiv:2410.08345 (submitted 10 Oct 2024). No journal or conference listed on the arXiv page; I found no peer-reviewed venue.
- **URL:** https://arxiv.org/html/2410.08345v1 ; code github.com/hegasz/large-legislative-models (MIT per GitHub API; last push 10 Oct 2024; file `large_legislative_models/environment/meltingpot_utils.py` shows it runs on Melting Pot)
- **Source type:** academic (preprint)
- **Accessed:** 14 Oct 2026

## Verbatim quotes

1. Method: "we propose a novel method in which we instead utilize pre-trained Large Language Models (LLMs), as sample-efficient policymakers in socially complex multi-agent reinforcement learning (MARL) scenarios."
2. Commons Harvest overseer: "In Harvest, the principal outputs a three-tiered tax rate to be applied to the reward signal of agents collecting apples, where the applicable tax rate is determined by the number of apples an agent has collected in the past 10 timesteps. Episodes are terminated after 1000 timesteps."
3. Followers are RL, not LLMs: "in Harvest and Clean Up we use pre-trained follower agents reset each principal action as implementations of oracles"
4. Cost limit: "Because of the high cost of LLM APIs for frontier models, we terminate LLM runs after several dozen steps of convergence – totaling 90 generated actions on Harvest, 60 on Clean Up, and 100 on CER."
5. Ablation of the AI Economist: "AI Economist was outperformed or matched by a much simpler $\epsilon$ -greedy bandit algorithm on two of our three environments"
6. Weakness: "we found in our experiments that LLMs can demonstrate a tendency to trap themselves in local minima, choosing to make small adjustments to past outputs rather than attempting a more dramatic shift."

## Other facts (paraphrased)

- Models: GPT-4o mini and Gemini-1.5 flash (plus legacy Gemini-1.0 pro in an ablation). 10 seeds per method.
- Baselines: AI Economist, MetaGrad, UCB, Thompson sampling, ε-greedy.
- Taxes are redistributed evenly; principal payoff is a social-welfare measure; one principal action per episode (several convergence episodes of the RL followers in between).

## Relevance

This is the only work I found that puts an **LLM overseer (principal) on a 2D Melting Pot commons** (Commons Harvest Open, Clean Up). The overseer sets a harvest-rate-dependent tax once per episode on RL agents; there are no audits, no imperfect monitoring and no LLM agents. It is direct prior art for "LLM overseer in a grid commons" and a template for its evaluation (LLM principal vs bandits vs AI Economist, sample efficiency).
