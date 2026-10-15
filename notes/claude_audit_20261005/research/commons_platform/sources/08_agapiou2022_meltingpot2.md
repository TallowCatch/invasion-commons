# Melting Pot 2.0

- **Authors:** John P. Agapiou, Alexander Sasha Vezhnevets, Edgar A. Duéñez-Guzmán, Jayd Matyas, Yiran Mao, Peter Sunehag, Raphael Köster, Udari Madhushani, Kavya Kopparapu, Ramona Comanescu, DJ Strouse, Michael B. Johanson, Sukhdeep Singh, Julia Haas, Igor Mordatch, Dean Mobbs, Joel Z. Leibo
- **Year:** 2022 (v6 revised 31 Oct 2023)
- **Venue / identifier:** arXiv:2211.13746 (tech report)
- **URL:** https://arxiv.org/abs/2211.13746
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted; spacing restored)

## Verbatim quotes

1. Headline metric:
   > "We assess the performance of the focal population on the test scenario by the focal per-capita return, the mean return obtained across the focal players (i.e. excluding the background players)."
2. Modes:
   > "When focal players outnumber background players we say the test scenario is in resident mode."
3. Normalisation:
   > "Each exploiter is a self-interested ACB or OPRE MAPLA trained directly on a single test scenario, using their individual return as the reward signal without further augmentation."
4. Normalisation failures:
   > "Below is the list of scenarios where the exploiter performed as no better than a random agent: 1. Clean Up: scenario 1; ... 3. Commons Harvest Closed: scenario 3;"
5. Metric blind spot (Commons Harvest Closed section):
   > "it wouldn't really be rewarded by our current overall scoring metric of per capita return in the focal population since it is insensitive to the distribution of rewards over players."
6. How a Commons Harvest Open bot was made:
   > "augmented with a pseudoreward term that strongly discouraged them from using the zap action (a negative reward whenever they select that action). So as a result they ended up learning pacifist policies that never zap."

## Relevance
This is the main description of the protocol, including secondary metrics. The report suggests background per-capita return for "negative externalities" and, following Perolat, the inequality of background returns. It also reports that agent hyper-parameters were tuned, but says nothing about how substrate parameters were set. It repeats the 0.001 one-neighbour regrowth figure, while the shipped Open code uses 0.0025 (see 01). The normalisation is admitted to fail in several commons scenarios.
