# Predicting Behavior in New Behavioral Experiments: Outcomes of a Modeling Competition

- **Authors:** Marco A. Janssen, Nathan Rollins
- **Year:** 2012
- **Venue / identifier:** Journal of Artificial Societies and Social Simulation 15(3): 5
- **URL:** https://www.jasss.org/15/3/5.html (PDF https://www.jasss.org/15/3/5/5.pdf)
- **Source type:** academic
- **Date accessed:** 14 Oct 2026 (full PDF text read)

## Verbatim quotes

1. Grid and group:
   > "Groups consist of five participants who share a 29 × 29 grid of cells (the borders do not wrap around)."
2. Initial stock and payment:
   > "At the start of the experiment, 25% of the grid space is filled with tokens, thus 210 tokens are randomly allocated on the board."
   > "Each token harvested is worth a certain amount of money (in these sessions, 2 cents each.)"
3. Regrowth rule (neighbour-dependent, per second):
   > "Every second, empty cells have a possibility of generating a new token."
   > "p t = P · n t / N where n t is the number of neighboring cells containing a token, N is the total number of neighboring cells (N = 8), and P = 0.01."
   > "At least one adjacent cell must contain a token for a new token generation to occur."
4. Why this rule:
   > "By designing the environment in this manner, the experiment captures a key characteristic of many spatially-dependent renewable resources."
5. Decision periods and the optimal pattern:
   > "The experiments consisted of multiple 4-minute decision periods in which different treatments were used."
   > "An optimal strategy of harvesting to maximize regrowth and harvesting profits would involve maintaining a checkerboard pattern of tokens."

## Relevance
Gives the exact human-lab parameters of the Janssen et al. (2010) foraging commons: 5 players, 29 × 29 non-wrapping grid, 25% initial density (210 tokens), 4-minute periods, regrowth p = 0.01 × (occupied neighbours / 8) per second, $0.02 per token. Commons Harvest's "no neighbours, no regrowth" rule and local-density regrowth are inherited from here; Perolat, Hughes and Melting Pot then changed the probabilities without stating why (09, 10, 01). It also shows the human data were used as a benchmark for agent models (a modelling competition), which is a validation route for agent behaviour in a 2D commons.
