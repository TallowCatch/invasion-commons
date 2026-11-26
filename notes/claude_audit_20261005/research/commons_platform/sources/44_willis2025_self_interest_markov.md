# [44] Quantifying the Self-Interest Level of Markov Social Dilemmas

- **Authors:** Richard Willis, Yali Du, Joel Z. Leibo, Michael Luck
- **Year:** 2025
- **Venue / identifiers:** IJCAI 2025, DOI 10.24963/ijcai.2025/33 (via OpenAlex; OpenAlex lists the fourth author as "G. Flucke", which looks like an indexing error, since the arXiv PDF lists Michael Luck). arXiv:2501.16138v2 [cs.GT] (v1 27 Jan 2025, v2 17 May 2025), CC BY-SA 4.0
- **URL:** https://arxiv.org/abs/2501.16138
- **Code:** https://github.com/willis-richard/meltingpot (a fork, Apache-2.0, described as "Code for 'Quantifying the Self-Interest Level of Markov Social Dilemmas' (IJCAI 2025)"; HEAD `2c9e94b7e98b8c207c2532defb3ccf7a7d87e556`; not cloned)
- **Source type:** conference paper (arXiv HTML full text read)
- **Date accessed:** 14 Oct 2026

## Verbatim quotes

> "We extend the concept of self-interest level from normal-form games to Markov games, providing a quantitative measure of the minimum reward exchange required to align individual and collective interests. We demonstrate our method on three environments from the Melting Pot suite, representing either common-pool resources or public goods." (Abstract)

> "Our approach begins by training independent agents to maximise their individual rewards, which commonly results in suboptimal collective outcomes in social dilemmas. We then introduce reward exchange and progressively increase the amount shared between agents, identifying the critical threshold where these initially selfish policies transition to collective behaviours."

Reward exchange: $R'_i = sR_{i}+\frac{1-s}{n-1}\sum_{j\neq i}R_{j}$; "The self-interest level of a Markov social dilemma, denoted $s^{*}$ , represents the maximum amount of self-interest that agents can retain while using reward exchange to resolve the dilemma".

Environments: Commons Harvest (seven agents), Clean Up (seven agents), Externality Mushrooms (five agents). Learning algorithm: PPO.

Results:
> "We therefore estimate the self-interest level to be in the range $0.29\leq s^{*}<0.33$ ." (Commons Harvest)
> "We therefore estimate the self-interest level to be in the range $0.25\leq s^{*}<0.29$ ." (Clean Up)
> "The Schelling diagram in Figure 4(c) reveals that this is not a Markov social dilemma because social welfare is maximised when one player defects" (Externality Mushrooms)
> "Note that longer training periods or a larger number of seeds might yield slightly different results. However, limited computational resources prevent an exhaustive investigation."

Future work (Conclusion):
> "We could investigate the impact on the self-interest level of systematic changes to an environment, for example, by increasing resource scarcity or introducing a sanctioning mechanism."
> "As a solution mechanism, reward exchange can be applied to problems like fishery management, where traditional quotas often fail due to persistent incentives to overfish."

## Relevance
This paper shows how Willis checks that an environment really is a social dilemma: Schelling diagrams from fixed "cooperate" and "defect" policy sets, with welfare on a second axis. It also names "introducing a sanctioning mechanism" as future work, which is the overseer question. The Commons Harvest s* of about 0.3 is a benchmark for how hard that commons is to resolve without enforcement.
