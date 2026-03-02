# paper_v1 closeout

## 1. Research question (this cycle)
Which governance signals stabilize cooperation under adversarial strategy injection, and does this hold across tiered regime difficulty and injector type (mutation vs live LLM)?

## 2. What was implemented
- Matched matrix across:
  - tiers: easy_v1, medium_v1
  - injectors: mutation, llm_json (live Ollama qwen2.5:3b-instruct)
  - governance: none, monitoring, monitoring_sanctions
- Train/test split and fixed benchmark packs used.
- Confidence-interval summary pipeline run.
- Composite governance effect-size ranking added.
- Optional hard_v1 live smoke run added as appendix-only stress check.

## 3. Key artifacts
- Main results: results/runs/showcase/curated/paper_v1_main_results.md
- Methods: results/runs/showcase/curated/paper_v1_methods.md
- Gates: results/runs/showcase/curated/paper_v1_gates.json
- Main deltas table: results/runs/ablation/curated/paper_v1_table1_main_deltas.csv
- LLM integrity: results/runs/ablation/curated/paper_v1_table2_llm_integrity.csv
- Effect-size ranking: results/runs/ablation/curated/paper_v1_table3_effect_size_ranking.csv
- Medium ranking: results/runs/ablation/curated/paper_v1_table3_medium_effect_size_ranking.csv
- Figures A–D: results/runs/showcase/curated/paper_v1_figureA_governance_effect.png, ...B..., ...C..., ...D...
- Hard appendix note: results/runs/showcase/curated/paper_v1_appendix_hard_note.md

## 4. Decision-critical results
- medium_v1 collapse reduction (none - monitoring_sanctions):
  - mutation: 0.109896, 95% CI [0.031042, 0.189583]
  - llm_json: 0.083542, 95% CI [0.015417, 0.150938]
- Gate status:
  - primary gate: PASS
  - strong gate: PASS
  - medium saturation: FALSE

## 5. Interpretation
- Governance improves robustness under invasion pressure in medium regimes for both injectors.
- hard_v1 is saturated in smoke profile and should stay appendix-only.

## 6. Limitations
- LLM runs are slow and sensitive to runtime load.
- CIs are finite-sample; easy tier still has broader uncertainty.
- hard_v1 does not separate mechanisms reliably.

## 7. Supervisor talk track (short)
I moved the project from emergence to stability-under-invasion. I ran matched mutation vs live-LLM governance comparisons with train/test regime split. In medium_v1 high-power reruns, monitoring+sanctions reduced collapse vs none with positive confidence bounds for both injectors. This supports the claim that minimal governance signals improve invasion robustness. hard_v1 is treated as appendix stress evidence only.

## 8. Next cycle objective
Convert paper_v1 evidence into thesis/paper writing package and run only targeted follow-up checks.
