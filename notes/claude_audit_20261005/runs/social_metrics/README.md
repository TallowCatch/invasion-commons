# Social outcome metrics, the same in all three games (8 October 2026, post hoc)

**Why.** The paper defines harm from the games themselves, not from real-world regulation. The measures copy the closest commons-game papers. Both sources were checked in full text on 8 Oct 2026:
- **GovSim** (Piatti et al. 2024, arXiv:2404.16698):
  - efficiency u = 1 − max(0, T·f(0) − Σ_t R_t)/(T·f(0)), i.e. harvest as a share of the maximum sustainable harvest, capped at 1;
  - equality e = 1 − Gini;
  - survival time and survival rate;
  - over-usage.
- **Perolat et al. 2017** (arXiv:1707.06600):
  - "The Utilitarian metric (U), also known as Efficiency, measures the sum total of all rewards obtained by all agents";
  - Equality (E) "is defined using the Gini coefficient";
  - Sustainability (S), "the average time at which the rewards are collected";
  - Peace (P), which needs tagging, so it does not apply here.

**One harm line from the models.** All three games regrow logistically, g(x) = r·x·(1 − x/K), and regrowth is fastest at x = K/2. Harm means the resource is below K/2. The maximum sustainable harvest per round is rK/4.

| Game | K | Harm line (K/2) | Max sustainable harvest per round (rK/4) | Line used until now |
| --- | ---: | ---: | ---: | --- |
| Fishery | 100 t | stock left after fishing < 50 | 0.7·100/4 = 17.5 t | the same (MSY limit) |
| Forest | 20 per plot | mean plot health < 10 | 6 plots × 0.595·20/4 = 14.28 | the same level, as a 5% chance limit |
| River | 100 | water quality < 50 | discharge whose damage equals 0.25·r·100: 1.67 per round | quality < 30 (pre-registered; still reported) |

**Files:** `social_metrics_llm.csv` (L2 and the corrected EM; L3 is added only once it is complete), `social_metrics_t1.csv` (T1, all three games), `social_metrics_info.json`. They are produced by `experiments/oversight/social_metrics.py`. The River games were re-run from their seeds to read the quality at every step, and 192 of 192 reproduced the saved outcomes exactly.

## What came out (post hoc, descriptive)

**T1, audit aiming (claim 5), with harm at half capacity in all games:**

| Game | Arm | Efficiency | Rounds below half capacity | Pre-registered harm |
| --- | --- | ---: | ---: | ---: |
| Fishery | trust / report / random | 0.99 / 1.00 / 1.00 | 98.9% / 96.2% / **7.4%** | the same |
| Forest | trust / report / random / signal | 0.59 / 0.60 / 0.60 / 0.60 | not saved per round | 2.0% / 1.4% / 1.3% / **0.5%** (a 5% chance limit at half capacity) |
| River | trust / report / random | 0.82 / 0.94 / **0.96** | 52.5% / 41.3% / **36.8%** | 21.0% / 6.0% / 2.8% (quality < 30) |

- **At half capacity, T1's conclusion is unchanged in River.** Random audits are best, aiming at the largest report is worse, and trusting everyone is worst. The gap between report-aimed and random grows from 3.2 to 4.5 points. River now spends about 37–53% of rounds below half capacity, so the stricter line shows much more harm in every arm.
- **Efficiency alone misleads in Fishery.** Cheating keeps the total harvest at the maximum sustainable rate (0.99–1.00) while the stock sits below half capacity, because the cheaters take what the honest agents lose. This is why GovSim and Perolat report a set of measures rather than one. In Forest, every arm harvests about 60% of the maximum: that is the price of the reviewer's 5% chance limit.

**L2 (LLM agents in Fishery), selected cells:**

| Model | Cell | Efficiency (capped) | Equality | Rounds at or above half capacity | Survival |
| --- | --- | ---: | ---: | ---: | ---: |
| gpt-oss | E0, no fine | 0.73 | 0.69 | 7.5% | 5/10 |
| gpt-oss | E8, fine below the gain | 1.00 | 0.75 | 16.5% | 10/10 |
| gpt-oss | **E36, fine above the gain** | **0.99** | **0.81** | **100%** | 10/10 |
| gpt-oss | EM, memory | 1.00 | 0.79 | 80.5% | 10/10 |
| Nemotron | E1, small fine | 0.50 | 0.67 | 0% | **0/10** |
| Nemotron | **E36** | **0.99** | **0.81** | **100%** | 10/10 |
| Nemotron | EM | 1.00 | 0.80 | 60.5% | 10/10 |
| Mistral | all cells | 0.97–1.00 | 0.86–0.87 | 100% | 10/10 |
| Gemma | all cells | 0.95–1.00 | 0.79–0.88 | 90–97% | 10/10 |

- **Deterrence costs no efficiency.** A fine above the gain gives 99% of the maximum sustainable harvest, with the stock at or above half capacity in every round.
- **Fines below the gain** give either short-run over-harvest (efficiency near 1, stock below half capacity in over 80% of rounds) or collapse.
- **Cheating also makes outcomes less equal:** equality is 0.67–0.75 without deterrence, against 0.81 with it.
- **Memory** keeps the stock at or above half capacity in 61–81% of rounds with no efficiency loss.
