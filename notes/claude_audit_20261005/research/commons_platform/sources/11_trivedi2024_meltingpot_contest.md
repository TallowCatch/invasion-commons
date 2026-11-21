# Melting Pot Contest: Charting the Future of Generalized Cooperative Intelligence

- **Authors:** Rakshit S. Trivedi, Akbir Khan, Jesse Clifton, Lewis Hammond, Edgar A. Duéñez-Guzmán, John P. Agapiou, Jayd Matyas, Sasha Vezhnevets, Dipam Chakraborty, Yue Zhao, Marko Tesic, Barna Pásztor, Yunke Ao, Omar G. Younis, Jiawei Huang, Benjamin Swain, Haoyuan Qin, Mian Deng, Ziwei Deng, Utku Erdoğanaras, Natasha Jaques, Jakob N. Foerster, Vincent Conitzer, Jose Hernandez-Orallo, Dylan Hadfield-Menell, Joel Z. Leibo
- **Year:** 2024
- **Venue / identifier:** NeurIPS 2024 Datasets and Benchmarks Track; DOI 10.52202/079017-0517; OpenReview slqbOc67W8 (no arXiv version found)
- **URL:** https://proceedings.neurips.cc/paper_files/paper/2024/file/1d3ea22480873b389a3365d711eb1e91-Paper-Datasets_and_Benchmarks_Track.pdf
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted)

## Verbatim quotes

1. Scoring (Sec. 3.1):
   > "Scores were normalized using the best and worst scores from the baseline solutions reported in the Melting Pot 2.0 paper [2]. A score above 1.0 indicates the submission outperformed the best baseline."
2. Held-out set:
   > "we generated a total of 51 held-out scenarios across the 4 substrates and we used these held-out set to evaluate agents in the next two phases."
3. Gaming the focal metric:
   > "teams with high per capita focal returns often do not perform well in terms of the background population's per capita return."
4. Lesson (i):
   > "Rule-based hardcoded policies showed more overfitting to the focal objective than RL solutions."
5. Lessons (ii)–(iii):
   > "Using multiple social welfare metrics for evaluating submissions would have better prevented anti-social behaviors"
6. Successor:
   > "The new contest will involve LLM-based agents, and will be based on Concordia [33]."

## Relevance
The contest used four substrates: Prisoner's Dilemma in the Matrix Arena (8 players), Clean Up (7), Allelopathic Harvest (16) and Territory Rooms (9). It ran three phases: 22 public validation scenarios, then 12 sampled held-out scenarios, then all 51 held-out ones. Ranking was by mean normalised focal per-capita return. Its own post-mortem is the strongest published criticism of that metric. Teams scored highly by coordinating among their own agents against the background agents, for example killing the only background agent and then sharing the territory. The organisers say they should have used several welfare metrics and ranked per substrate. For Ameer, this argues for reporting background or overall welfare next to any focal score.
