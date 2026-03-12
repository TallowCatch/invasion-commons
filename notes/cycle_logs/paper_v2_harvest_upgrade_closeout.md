# paper_v2 harvest upgrade closeout

## 1. What was implemented
- Study 1b in Fishery Commons was extended beyond the earlier matched baseline into a broader top-down sweep.
- Added Fishery Commons mechanism logging for requested harvest, realized harvest, quota clipping, sanctions, repeat-offender rate, closure activity, and stock-recovery lag.
- Added extra top-down governance variants for Fishery Commons:
  - `adaptive_quota`
  - `temporary_closure`
- Added broader Study 1b generalization dimensions:
  - partner mixes: `balanced`, `adversarial_heavy`
  - adversarial pressure: `0.3`, `0.5`
  - non-LLM injectors: `mutation`, `adversarial_heuristic`, `search_mutation`
- Built an initial second-substrate pilot as Orchard Commons, then reframed and renamed it as `Harvest Commons` so it fit the sequential-commons benchmark logic more cleanly.
- In that first Harvest pilot, added:
  - top-down only governance
  - bottom-up only governance
  - hybrid governance
  - local communication
  - credit sharing / side-payments
  - a government controller acting like an agent
- Ran fixed-composition Harvest batches and used them as a pilot to understand whether the hybrid mechanism was doing anything visible.
- After that, upgraded Harvest Commons to the same invasion logic as Fishery Commons instead of leaving it as a fixed-composition comparison.
- Added a parameterized `HarvestStrategySpec` and executable Harvest strategy agent family.
- Added a dedicated Harvest invasion engine with:
  - population evaluation
  - selection
  - replacement
  - held-out regime testing
  - garden-failure logic
- Added Harvest invasion injectors:
  - `random`
  - `mutation`
  - `adversarial_heuristic`
- Added Harvest tier presets and held-out regime packs:
  - `easy_h1`
  - `medium_h1`
  - `hard_h1`
  - held-out regimes: `noisy_weather`, `strong_externality`, `low_init`, `slow_regen`
- Added Harvest invasion summaries, rankings, paired delta outputs, bootstrap confidence intervals, figures, and notebook support.
- Added a remote sharded GitHub Actions workflow so the large Harvest invasion sweeps no longer had to run locally.
- Downloaded and re-integrated the remote Stage B and Stage C artifacts into the local `results/` tree.
- Updated the paper so the real Study 2 evidence now comes from the Harvest invasion Stage C runs, while the older fixed-composition Harvest results are now treated as pilot / precursor evidence in the appendix.

## 2. Key artifacts
- Fishery Study 1b summaries:
  - `results/runs/ablation/curated/study1b_medium_hp_mutation_summary.md`
  - `results/runs/ablation/curated/study1b_medium_hp_advheur_summary.md`
  - `results/runs/ablation/curated/study1b_medium_hp_searchmut_summary.md`
- Fishery Study 1b tables:
  - `results/runs/ablation/curated/study1b_medium_hp_mutation_table.csv`
  - `results/runs/ablation/curated/study1b_medium_hp_advheur_table.csv`
  - `results/runs/ablation/curated/study1b_medium_hp_searchmut_table.csv`
- Early Harvest pilot readouts:
  - `results/runs/showcase/curated/harvest_commons_batch5_local_behavior_readout.md`
  - `results/runs/showcase/curated/harvest_commons_batch8_neighbor_targeted_readout.md`
- Harvest fixed-composition matrix outputs:
  - `results/runs/showcase/curated/harvest_matrix_v2_full12_summary.md`
  - `results/runs/showcase/curated/harvest_matrix_v2_full12_ci_summary.md`
  - `results/runs/harvest/curated/harvest_matrix_v2_full12_runs.csv`
  - `results/runs/harvest/curated/harvest_matrix_v2_full12_ci.csv`
- Harvest invasion Stage B pilot:
  - `results/runs/showcase/curated/harvest_invasion_stageB_pilot_gh_summary.md`
  - `results/runs/showcase/curated/harvest_invasion_stageB_pilot_gh_ranking.csv`
  - `results/runs/showcase/curated/harvest_invasion_stageB_pilot_gh_ci.csv`
  - `results/runs/harvest_invasion/curated/harvest_invasion_stageB_pilot_gh_runs.csv`
- Harvest invasion Stage C high-power:
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_summary.md`
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_delta.csv`
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_ci.csv`
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_main.png`
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_mechanisms.png`
  - `results/runs/harvest_invasion/curated/harvest_invasion_stageC_highpower_gh_runs.csv`
- Harvest remote sweep infrastructure:
  - `.github/workflows/harvest-invasion-matrix.yml`
  - `experiments/merge_harvest_invasion_outputs.py`
- Harvest invasion code path:
  - `fishery_sim/harvest.py`
  - `fishery_sim/harvest_benchmarks.py`
  - `fishery_sim/harvest_evolution.py`
  - `experiments/run_harvest_invasion.py`
  - `experiments/run_harvest_invasion_matrix.py`
  - `experiments/summarize_harvest_invasion.py`
  - `experiments/plot_harvest_invasion.py`
  - `notebooks/06_harvest_invasion.ipynb`
- Paper artifacts:
  - `paper/paper_v2/main.pdf`
  - `paper/paper_v2/figures/study1b_topdown_summary.png`
  - `paper/paper_v2/figures/study2_harvest_invasion_stagec_outcomes.png`
  - `paper/paper_v2/figures/study2_harvest_invasion_stagec_mechanisms.png`

## 3. Decision-critical results
- Fishery Commons matched baseline, medium tier collapse reduction (`none - monitoring_sanctions`):
  - mutation: `0.109896`, 95% CI `[0.031042, 0.189583]`
  - live LLM JSON: `0.083542`, 95% CI `[0.015417, 0.150938]`
- Fishery Commons Study 1b top-down sweep:
  - `adaptive_quota` is the strongest overall central signal in the medium tier
  - `temporary_closure` ties it on collapse in many cells and remains competitive in adversarial-heavy mixes
- Harvest invasion Stage B pilot:
  - `hybrid` wins `18/27` cells
  - `top_down_only` wins `9/27` cells
- Harvest invasion Stage C high-power:
  - `hybrid` wins `14/16` cells
  - `top_down_only` wins `2/16` cells
  - mean paired `hybrid - top_down_only` deltas:
    - garden failure: `0.0000`
    - patch health: `+0.0228`
    - welfare: `-0.0295`
    - local aggression: `-0.0034`
    - neighborhood overharvest: `-0.0352`

## 4. Interpretation
- The first Fishery Commons study now does more than show that governance helps. It distinguishes weak top-down intervention from stronger top-down intervention and shows that adaptive quotas are the strongest overall signal once pressure, partner mix, and injector family vary.
- The earlier Orchard idea was useful as a prompt, but the better framing was a second sequential commons substrate. That is why it became Harvest Commons.
- The early Harvest fixed-composition work was useful as pilot evidence, but it was not enough because it did not use the same invasion logic as the fishery study.
- The Harvest invasion upgrade fixed that asymmetry. The second study now uses population turnover, injectors, held-out regimes, and paired deltas in the same general style as the first study.
- The current Harvest result is not “hybrid wins everywhere.” The stronger claim is that hybrid governance usually improves local ecological control under invasion pressure, while the welfare tradeoff depends on the social mix and task difficulty.
- Garden failure is not the main discriminating signal in Harvest. Patch health, local aggression, and neighborhood overharvest are more informative.

## 5. Limitations
- Live LLM injection currently appears only in Fishery Commons, not yet in Harvest Commons.
- The first Harvest fixed-composition matrix is still useful, but it should be treated as pilot evidence rather than main evidence.
- Harvest Commons is now a real invasion study, but it is still a simpler and more stylized substrate than many real decentralized systems.
- The remote GitHub Actions workflow solves the runtime problem, but it depends on a clean artifact download / merge path and is less convenient for quick iteration than local smoke runs.
- Some Harvest high-power rows still show a real ecological-control versus welfare tradeoff, so the second study should not be oversold as universal dominance of hybrid governance.

## 6. Supervisor talk track (short)
I kept the fishery result as the main identification study and extended it in two ways. First, I strengthened Fishery Commons with mechanism logging, extra top-down signals, broader partner mixes, pressure levels, and stronger non-LLM injectors. Second, I built a real second commons substrate. I first explored it as an orchard-style pilot, then reframed it as Harvest Commons, and finally upgraded it to the same invasion logic as the fishery study. That means Harvest now has evolving strategy populations, replacement, non-LLM injectors, held-out regimes, and high-power comparisons. The main Harvest result is that hybrid governance usually improves patch health and reduces local overharvest relative to top-down-only control, though the welfare effect depends on the setting.

## 7. Next cycle objective
Freeze the current two-study paper state, then decide whether the next empirical step is:
- Harvest live LLM injection,
- a stronger Study 2 baseline such as a simple best-response learner,
- or a more realistic extension of governance and communication in the second substrate.
