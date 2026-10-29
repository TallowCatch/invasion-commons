# [42] Will Systems of LLM Agents Cooperate: An Investigation into a Social Dilemma

- **Authors:** Richard Willis, Yali Du, Joel Z. Leibo, Michael Luck
- **Year:** 2025
- **Venue / identifiers:** AAMAS 2025 extended abstract (Detroit), proceedings PDF https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2025/pdfs/p2786.pdf (HTTP 200 checked 14 Oct 2026); DOI 10.5555/3709347.3744012 (via OpenAlex, title there "Will Systems of LLM Agents Lead to Cooperation: An Investigation into a Social Dilemma"); full version arXiv:2501.16173v1 (27 Jan 2025), CC BY-SA 4.0
- **URL:** https://arxiv.org/abs/2501.16173
- **Code:** https://github.com/willis-richard/evollm, MIT, HEAD `b7622b6976d28479dce4201932b413935ef86211` (via `git ls-remote`, not cloned)
- **Source type:** conference extended abstract + arXiv full paper (full text read, HTML)
- **Date accessed:** 14 Oct 2026

## Verbatim quotes

> "Unlike previous research where LLMs output individual actions, we prompt state-of-the-art LLMs to generate complete strategies for iterated Prisoner's Dilemma. Using evolutionary game theory, we simulate populations of agents with different strategic dispositions (aggressive, cooperative, or neutral) and observe their evolutionary dynamics." (Abstract)

> "Each match consists of 1000 rounds of Prisoner’s Dilemma (Table 1)" — payoffs (C,C)=3,3; (C,D)=0,5; (D,C)=5,0; (D,D)=1,1. "Some matches use noise, in which case there is, independently for each player, a 10% chance that their action choice is replaced with the alternative action. Our implementation uses the Axelrod Python library"

> "We experiment with three different prompt styles" — Default; Refine ("we use Self-Refine ... to ask the LLM to provide and incorporate self-feedback"); Prose (a non-game-theoretic scenario).

> "We select the LLMs ChatGPT-4o and Claude 3.5 Sonnet as they are popular frontier models. For each LLM and prompt style, we create 25 strategies for the three attitudes. We then use ChatGPT-4o to convert the natural language strategies for all LLMs into Python functions."

> "These generated strategies are manually checked for safety before their performance is assessed in all-play-all IPD tournaments." / "Where an algorithm has a bug, we manually fix this if the model’s intention is clear. Otherwise, we delete the strategy and generate a new one."

> "Claude 3.5 Sonnet would frequently refuse to write an aggressive strategy in such a situation." (academic-science prose scenario; replaced with commercial engineering)

Evolution: "Specifically, we use a Moran process Moran (1958)" — fitness from all-play-all; "A player is chosen to be cloned proportionally to their fitness. They replace a uniformly randomly selected player." "Terminate when all players have the same attitude." 100 Moran processes, population 12, initial ratios 1:1:1 and 4:1:1 (A:C:N).

Findings:
> "For both LLMs, the neutral and cooperative strategies perform well, whilst the aggressive strategies perform poorly." (vs 11 human-written strategies, Beaufils tournament)
> "Compared to the Default prompt, a Refine prompt improves the performance of aggressive strategies without negatively impacting neutral and cooperative strategies."
> "The addition of noise generally leads to a marked increase in the likelihood of converging to aggressive equilibria. We posit this is due to the fact that noise can mask aggression"
> "Across most scenarios, aggressive strategies tend to be disadvantaged, leading to a lower likelihood of systems converging to aggressive equilibria."

## Relevance
This is the method's origin: natural-language strategy, then code, then evolutionary selection over attitude-genotypes. Its noise finding (noise "can mask aggression") is directly relevant to an audit overseer, since imperfect detection is another form of noise about who defected. It is a two-player game with identities visible, so Tit-for-Tat-style enforcement is possible there, but not in the anonymous n-player CPR of [41].
