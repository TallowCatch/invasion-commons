# FishBase resilience categories as prior ranges for the intrinsic growth rate r (via CMSY studies)

- **Primary source used:** Wang, Wang, Liang, Zhang, Xian (2020), "Assessment of 12 Fish Species in the Northwest Pacific Using the CMSY and BSM Methods", Frontiers in Marine Science 7:616, DOI 10.3389/fmars.2020.00616 (Table 2, which reproduces the FishBase ranges; citing Froese & Pauly 2019, FishBase).
- **Method source (abstract only):** Froese, Demirel, Coro, Kleisner, Winker (2017), "Estimating fisheries reference points from catch and resilience", Fish and Fisheries 18: 506–526, DOI 10.1111/faf.12190.
- **URL:** https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2020.00616/full
- **Source type:** academic (FishBase itself, the "docs" origin, was not opened directly)
- **Date accessed:** 14 Oct 2026 (Frontiers full HTML read; Froese et al. abstract via OpenAlex)

## Verbatim quotes

1. Frontiers 2020, method:
   > "To determine priors r for the 12 species being evaluated, the resilience levels provided by FishBase ( Froese and Pauly, 2019 ) were converted to r -ranges ( Table 2 )."
2. Frontiers 2020, Table 2 (flattened): "Resilience ( r ) Prior r range ... High 0.6–1.5 ... Medium 0.2–0.8 ... Low 0.05–0.5 ... Very low 0.015–0.1" with footnote:
   > "Ranges suggested by FishBase ( Froese and Pauly, 2019 ) for population growth rate (year –1 )."
3. Froese et al. 2017 abstract:
   > "This study presents a Monte Carlo method ( CMSY ) for estimating fisheries reference points from catch, resilience and qualitative stock status information on data‐limited stocks."
   > "It also presents a Bayesian state‐space implementation of the Schaefer production model ( BSM ), fitted to catch and biomass or catch‐per‐unit‐of‐effort ( CPUE ) data."
4. Frontiers 2020 on CMSY's model:
   > "Catch-MSY was proposed as a new method to estimate the maximum sustainable yield (MSY) and related indicators (notably carrying capacity k , and intrinsic rate of population growth r , or 'resilience') using only catch data and a Monte-Carlo approach ( Martell and Froese, 2013 )."

## Relevance
Verifies the r ranges quoted in the brief exactly (high 0.6–1.5, medium 0.2–0.8, low 0.05–0.5, very low 0.015–0.1 per year), as FishBase defaults used in Schaefer-model stock assessment. Two consequences for Ameer: (a) a logistic r in these ranges is citable as "a medium/high-resilience fish stock"; (b) r is per year, so a game whose rounds are not years must state the time step. For comparison, GovSim's doubling per month (60, 61) is far faster than any FishBase category if a round is a month.
