# [43] Resolving social dilemmas with minimal reward transfer

- **Authors:** Richard Willis, Yali Du, Joel Z. Leibo, Michael Luck
- **Year:** 2024
- **Venue / identifiers:** Autonomous Agents and Multi-Agent Systems 38(2):49, October 2024, DOI 10.1007/s10458-024-09675-4 (citation as given in [44]; DOI via OpenAlex). Preprint arXiv:2310.12928v3 (v1 19 Oct 2023, v3 31 Jul 2024; "Submitted to the Journal of Autonomous Agents and Multi-Agent Systems: Special Issue on Citizen-Centric AI Systems"). Also an AAMAS 2025 extended abstract "Resolving Social Dilemmas with Minimal Reward Transfer - Extended Abstract", DOI 10.5555/3709347.3744102 (OpenAlex). Earlier workshop version cited in [44]: Willis & Luck, "Resolving social dilemmas through reward transfer commitments", Adaptive and Learning Agents Workshop, London, May 2023 (not read).
- **URL:** https://arxiv.org/abs/2310.12928 ; open copy https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11469975/
- **Code:** https://github.com/willis-richard/reward_transfer_matrix (HEAD `71eb37d239568494facdc718ba29e7c336118a04`, no licence listed by GitHub API; not cloned)
- **Source type:** journal article (arXiv HTML full text read)
- **Date accessed:** 14 Oct 2026

## Verbatim quotes

> "we formalise social dilemmas and introduce a novel metric, the general self-interest level, to quantify the disparity between individual and group rationality in such scenarios. This metric represents the maximum proportion of their individual rewards that agents can retain while ensuring that a social welfare optimum becomes a dominant strategy." (Abstract)

> "To align individual incentives, we introduce the concept of agents committing to share proportions of their future game rewards with others, a technique that we label reward transfer."

> "We argue that a solution should require only a single agreement between participants and be scalable to an arbitrary number of agents. It should also rely solely on the incentives present in the scenario and not require agents to have private knowledge of, or previous experience with others."

> "Unfortunately, it can be difficult to detect when an opponent is shirking cooperation, and players have an incentive to do so if they can avoid punishment. Our approach uses incentives instead of punishments to support cooperation." (Section 2.3)

> "Our analytical findings reveal that superficially similar multi-player social dilemmas can exhibit dramatically different cooperative incentives. ... In some cases, when an appropriate transfer structure is in place, the difficulty of resolving social dilemmas does not necessarily increase as the number of participants rises. However, in other scenarios, achieving cooperation among many participants may require them to share almost all of their rewards."

> "This algorithm employs a linear program to compute the minimal reward transfer matrix." (Conclusion)

> "The limiting values quantified by the metrics are sufficient to ensure that cooperation is the dominant strategy." (Conclusion)

## Relevance
Willis's own institution is reward transfer, a binding contract that changes incentives without detection or punishment. He explicitly sets it against punishment because shirking is hard to detect. An audit-and-fine overseer is the opposite design: it needs detection, but no contract between agents. The self-interest level gives a ready comparator: how much reward exchange matches the deterrence of a fine F at audit rate q.
