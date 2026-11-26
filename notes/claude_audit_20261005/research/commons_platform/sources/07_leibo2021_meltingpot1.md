# Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot

- **Authors:** Joel Z. Leibo, Edgar Duéñez-Guzmán, Alexander Sasha Vezhnevets, John P. Agapiou, Peter Sunehag, Raphael Koster, Jayd Matyas, Charles Beattie, Igor Mordatch, Thore Graepel
- **Year:** 2021
- **Venue / identifier:** ICML 2021 (PMLR 139); arXiv:2107.06857
- **URL:** https://arxiv.org/abs/2107.06857
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text extracted with pypdf; spacing lost in extraction was restored in the quotes)

## Verbatim quotes

1. Stated goal (Sec. 2):
   > "generalize to interact effectively with unfamiliar individuals not seen during training (who may well be human), and 3. pass a universalization test: answering positively to the question "what if everyone behaved like that?"."
2. Universalization mode:
   > "we sample from f once and use this policy repeatedly. So the joint policy consists of N copies of the same policy"
3. Bot pipeline (Sec. 6):
   > "Our approach to creating background populations involves three steps: (1) specification, (2) training, and (3) quality control"
4. Quality control:
   > "Bot quality control is done by running 10–30 episodes where candidate bots interact with other fixed bots."
5. Commons Harvest regrowth as printed (Appendix):
   > "regrowth probabilities are 0.025 when there are three or more apples in the neighborhood, 0.005 if there are exactly two apples, 0.001 if there is exactly one, and 0 if there are no apples"
6. How the Partnership test was made (App. E.13):
   > "We gave bots negative rewards during training for crossing invisible tiles located along the vertical midline of each room."

## Relevance
Melting Pot 1.0 sets out the evaluation protocol: focal population, held-out bots, resident and visitor modes, universalization, and min-max normalisation between a random agent and an exploiter. It also sets out the bot-building process: specify, train with pseudorewards, then check quality over 10–30 episodes. Substrate parameters are stated, not justified; the paper says only that the mechanism comes from Perolat 2017 and Janssen 2010. In v1, Commons Harvest Open was a 16-player game; v2 uses 7.
