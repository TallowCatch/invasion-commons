# 20. SocialJax: An Evaluation Suite for Multi-agent Reinforcement Learning in Sequential Social Dilemmas

- **Authors:** Zihao Guo, Shuqing Shi, Richard Willis, Tristan Tomilin, Joel Z. Leibo, Yali Du
- **Year / venue:** ICLR 2026 (proceedings vol. 2026, pp. 52757–52780, as cited by OpenSocialJax); arXiv:2503.14576 (v3, 17 Mar 2026)
- **URL:** https://arxiv.org/html/2503.14576v3
- **Source type:** academic
- **Accessed:** 14 Oct 2026 (HTML downloaded and text extracted, math from alttext)

## Verbatim quotes

1. Schelling procedure (Sec. 4, "Schelling Diagram Analysis"): "The cooperative policies are sampled from the agents that used a common reward, while the defector policies originate from agents trained with independent rewards. We evaluate the environments over 30 episodes and compute the average rewards for the cooperative and defecting agents."
2. Dilemma conditions (Sec. 3.1): "Mutual cooperation is greater than mutual defection: $R_{c}(N)>R_{d}(0).$" and "Mutual cooperation is greater than being exploited by defectors: $R_{c}(N)>R_{c}(0).$" and "Fear: $R_{d}(i)>R_{c}(i)$ for sufficiently small $i$ ;  Greed: $R_{d}(i)>R_{c}(i)$ for sufficiently large $i$ ."
3. Axes (Sec. 3.1): "The lines $R_{c}(l+1)$ and $R_{d}(l)$ depict the average reward for an agent choosing to either cooperate or defect, as a function of the number of co-players that are cooperating (see Figure 3 )."
4. Harvest result: "In Harvest: Open, we observe that it only takes a single defector agent to have a severely negative impact on the returns of other players. This is because defector agents typically harvest the last apple in a patch, permanently depleting it."
5. Metric for Harvest (Sec. 3.4): "we measure the number of apples remaining on the map to assess whether agents are capable of preserving the long-term collective interest."
6. Origin of the games: "SocialJax environments are mainly derived from Melting Pot 2.0 ( Agapiou et al., 2022 ) . However, the layout of SocialJax environments is not the same as Melting Pot, but rather more similar to MiniGrid"

## Relevance

The SSD test is behavioural, not analytic: train two RL populations (common reward = "cooperator", individual reward = "defector"), mix them at test time, and plot mean return per role against the number of cooperators. The paper gives no parameter-choice rationale beyond "derived from Melting Pot 2.0"; the README adds that failing to find a dilemma does not prove there is none. This is the procedure an overseer study on SocialJax would be expected to reproduce for the un-overseen game.
