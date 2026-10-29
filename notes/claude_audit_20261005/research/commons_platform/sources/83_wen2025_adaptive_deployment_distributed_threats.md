# Adaptive Deployment of Untrusted LLMs Reduces Distributed Threats

- **Authors:** Jiaxin Wen, Vivek Hebbar, Caleb Larson, Aryan Bhatt, Ansh Radhakrishnan, Mrinank Sharma, Henry Sleight, Shi Feng, He He, Ethan Perez, Buck Shlegeris, Akbir Khan (Wen, Hebbar and Larson equal contribution)
- **Year:** 2024 (arXiv), ICLR 2025
- **Venue / identifier:** ICLR 2025 (proceedings listing found via search); arXiv:2411.17693
- **URL:** https://arxiv.org/abs/2411.17693 (HTML v1 read)
- **Source type:** Peer-reviewed conference paper
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract and introduction read from HTML v1.

## Verbatim quotes

1. "a setting where no single action is catastrophic and no single action provides overwhelming evidence of misalignment"
2. "the macro-protocol maintains an adaptive credence on the untrusted model’s alignment based on its past actions"
3. "We plot Pareto frontiers of safety (# of non-backdoored solutions) and usefulness (# of correct solutions)"
4. "reduces the number of backdoors by 80% compared to non-adaptive baselines"

## Relevance to the commons overseer

The 'distributed threat' (low-stakes) setting is the closer match to a commons: one over-take is not a catastrophe, harm accumulates, and evidence about an agent builds up over rounds. The macro-protocol (a running credence per agent that picks safer or riskier micro-protocols) is a direct model for an overseer with memory, which Ameer's reviewer memory already approximates. Safety here is a count, not a probability of zero failures.
