# [47] Richard Willis's GitHub repositories and publication search (incl. Jianing Zhao)

- **Source type:** search records (GitHub API, arXiv API, OpenAlex API, web search). DBLP returned a bot-check page and was not used. Google Scholar was not queried.
- **Date accessed:** 14 Oct 2026

## GitHub (`https://api.github.com/users/willis-richard/repos`), descriptions verbatim
- `emergent_llm`: "Code for 'Evaluating Collective Behaviour of Hundreds of LLM Agents' (AAMAS SE 2026)", MIT, updated 2026-10-02
- `evollm`: "Code for 'Will Systems of LLM Agents Lead to Cooperation?' (AAMAS 2025)", MIT, updated 2026-10-08
- `meltingpot` (fork): "Code for 'Quantifying the Self-Interest Level of Markov Social Dilemmas' (IJCAI 2025) ", Apache-2.0
- `reward_transfer_matrix`: "Code for 'Resolving social dilemmas with minimal reward transfer' (JAAMAS 38) ", no licence listed
- `Melting-Pot-Contest-2023` (fork), Apache-2.0; `connect4`: "A reinforcement learning agent taught entirely through self-play to a decent strength at connect4"; `chex-feedstock` (fork); `muff2n.github.io`

evollm README (raw, HEAD `b7622b6`): "This project is being extended to multi-player games: public goods game, collective risk dilemma and a common pool resource / fisheries game, in my other repo [emergent_llm]".

## arXiv author query `au:Willis_Richard` (KCL-relevant hits)
2602.16662 [41]; 2501.16173 [42]; 2310.12928 [43]; 2501.16138 [44]; 2312.05162 [45]; 2503.14576 [46]. One further hit, 2305.02039 (hand-gesture classification; co-authors Smith, Thiagarajan, Makris, Torlak), looks like a different Richard Willis and is excluded.

## OpenAlex
Confirms the venues: IJCAI 2025 (10.24963/ijcai.2025/33) for [44]; AAMAS 2025 extended abstracts 10.5555/3709347.3744012 ([42]) and 10.5555/3709347.3744102 (reward transfer EA, [43]); JAAMAS 2024, 10.1007/s10458-024-09675-4 ([43]).

## Jianing Zhao
- The only paper I can verify for the KCL Jianing Zhao (jianing.1.zhao@kcl.ac.uk) is [41].
- arXiv `au:Zhao_Jianing AND cat:cs.MA` also returns 2507.13814 (CodeEdu; co-authors include Jiannong Cao) and 2403.17704 (multi-agent LTL planning; co-authors Bowen Ye, Shaoyuan Li, Xiang Yin). These co-author groups are outside KCL, and nothing links them to the same person, so I treat them as **unverified, probably a different Jianing Zhao**.
- A web search ("Jianing Zhao King's College London LLM agents multi-agent Yali Du") found no profile page.

## Relevance
The complete list of Willis's verifiable work is short and coherent. Two lines run through it: (a) reward-transfer institutions and self-interest levels, in normal-form and Melting Pot games; (b) LLM-generated fixed strategies under evolutionary selection, first in the IPD and then in n-player PGG, CRD and CPR. Neither line has tested an external audit-and-fine overseer.
