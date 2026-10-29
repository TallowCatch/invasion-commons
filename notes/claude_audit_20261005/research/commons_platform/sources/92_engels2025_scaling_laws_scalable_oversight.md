# Scaling Laws For Scalable Oversight

- **Authors:** Joshua Engels, David D. Baek, Subhash Kantamneni, Max Tegmark
- **Year:** 2025
- **Venue / identifier:** NeurIPS 2025 (Spotlight, per the arXiv abstract page); arXiv:2504.18530
- **URL:** https://arxiv.org/abs/2504.18530 (full text at https://arxiv.org/html/2504.18530v3)
- **Source type:** Peer-reviewed conference paper
- **Date accessed:** 14 Oct 2026
- **Depth read:** Abstract plus full-text search (framework, Mafia section, Elo proxy). Note: the summariser tool flagged embedded text in the page and refused to quote; inspection shows these were the paper's appendix prompts for the game players, not instructions to me. Quotes below were checked against the downloaded text (the extractor splits 'oversight -specific'; the hyphenated form is the paper's).

## Verbatim quotes

1. "the players have oversight-specific Elo scores that are a piecewise-linear function of their general intelligence, with two plateaus corresponding to task incompetence and task saturation"
2. "four oversight games: Mafia, Debate, Backdoor Code and Wargames"
3. "A Mafia game has five villagers (Guards) and one Mafia (Houdini)"
4. "We use Chatbot Arena Elo scores as a proxy for general intelligence in these experiments"
5. "NSO success rates at a general Elo gap of 400 are 13.5% for Mafia, 51.7% for Debate, 10.0% for Backdoor Code, and 9.4% for Wargames"

## Relevance to the commons overseer

The protocol for measuring how oversight degrades with the capability gap: play every Guard model against every Houdini model, fit Guard-Elo and Houdini-Elo from win rates, and regress each on a general-capability Elo (Chatbot Arena) with a 'double ReLU'. Mafia is already a multi-agent game with one hidden defector among five overseers, which is structurally close to one planted over-taker among honest harvesters. A commons version would cross overseer model x agent model and fit the same curves.
