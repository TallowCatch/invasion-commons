# ICML 2027 literature review: the reading list in Zotero

Built 28 October 2026. **278 papers** were saved to Zotero, in **My Library**, all tagged `ICML27`. The full list is in `reading_list.csv`, which you can open in Excel. `resolve.py` and `build.py` rebuild it from scratch.

## Where the papers came from

| Tag | Count | What it is |
|---|---|---|
| `from: project notes` | 156 | Every paper the project had already cited or noted: the 58-paper AI review, the 93-source platform study, the literature ledger, the novelty notes and the paper draft's bibliography |
| `added: MARL & oversight (28 Oct)` | 50 | New core multi-agent RL work: opponent shaping, reciprocity, social preferences, incentive and mechanism design for learning agents, Stackelberg MARL, mediators, costly-observation RL, security games, PPO/IPPO/MAPPO, JaxMARL |
| `added: arXiv sweep 2025-26` | 72 | Recent arXiv papers (sorted newest first) from 15 searches, screened by title: sequential social dilemmas, sanctions in MARL, regulators and planners for MARL, AI control with many agents, audit budgets, LLM norm enforcement, LLM commons |

**Topic tags** (`topic: …`) were assigned automatically from title keywords, so expect some mislabels; a paper can carry several topics. **`★ core for gameplan`** (54 papers) marks what the SocialJax plan builds on directly.

## How each did, and how it was checked

- Every arXiv paper's title, authors, date and abstract come from the arXiv API.
- DOIs were checked on Crossref.
- Papers with neither were found by title on OpenAlex, and kept only on an exact title match (score ≥ 0.9).
- Two items were entered by hand:
  - Bell & Madisetti (ICML 2026, from the conference page only);
  - Konuganti 2026 (from the draft's `refs.bib`; it isn't indexed anywhere I checked).

## What was left out on purpose

- **Grant-only items:** FrugalGPT, RouteLLM and the Qwen2.5 report.
- **Three DOIs in the notes that point to unrelated papers:** a 1980 microcircuits paper, a 1984 expert-system paper and a toxicology review.
- **Act 1 fisheries and irrigation governance papers** from the old paper bibliographies (v2–v4). These are Ostrom-style case studies, not ML.
- **Four new-title searches that matched the wrong paper:** "Gifting in MARL", "Learning to Incentivize Information Acquisition", a Stackelberg least-squares paper, and the Konuganti search, which is why it was entered by hand.

## In Zotero

- **To see everything:** click the `ICML27` tag in the tag selector at the bottom left.
- **To see what to read first:** also click `★ core for gameplan`.
- **To put them in a collection:** create one, select all items under the `ICML27` tag, and drag them in.
