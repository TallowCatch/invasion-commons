# experiments/

Scripts are grouped by the work they serve. Run them from the repository root
as modules, for example:

```bash
python -m experiments.paper_v5.plot_reviewer_decisions --help
```

`configs/` stays here, and its paths (`experiments/configs/...`) are unchanged.
Scripts import each other across folders as `experiments.<folder>.<name>`.

| Folder | Scripts | What it holds |
| --- | ---: | --- |
| `paper_v5/` | 21 | Builds and checks paper v5: data export, figures, provenance-listed analyses, input checks, source packaging. |
| `common/` | 11 | Shared reviewer and Harvest libraries imported across folders (`run_matched_oversight` is the core). |
| `oversight/` | 9 | Current oversight experiments R1, S1, S1b and S2, and the progress figures (`notes/claude_audit_20261005/`). |
| `archive/harvest_2026q2/` | 35 | Earlier Harvest work: Stage A invasion matrix, threshold-replay shards, LLM strategy bridge, RL, actor-pressure pilot, decision suite, Clean Up. |
| `archive/fishery_2026q1/` | 20 | The earlier single-stock Fishery study. |
| `archive/oneoff/` | 8 | GIFs, the showcase report, talk figures, the results organiser, the LLM setup check. |

## Old path → new path (moved 5 October 2026)

Notes, protocols and results written before the move still use the old flat
paths, such as `experiments.run_matched_oversight`. They were left unedited
because they are records. Use this table to translate.

The scripts' contents changed only in their `experiments.*` paths and, where
they locate the repository root, `parents[1]` → `parents[2]`. Script hashes
recorded in earlier manifests therefore refer to the pre-move files at that
manifest's `git_head`.

| Old | New |
| --- | --- |
| `experiments.admit_cleanup_policies` | `experiments.archive.harvest_2026q2.admit_cleanup_policies` |
| `experiments.analyze_actor_pressure_pilot` | `experiments.archive.harvest_2026q2.analyze_actor_pressure_pilot` |
| `experiments.analyze_budgeted_reviewer` | `experiments.common.analyze_budgeted_reviewer` |
| `experiments.analyze_budgeted_reviewer_confirmation` | `experiments.paper_v5.analyze_budgeted_reviewer_confirmation` |
| `experiments.analyze_harvest_oversight_stageA` | `experiments.paper_v5.analyze_harvest_oversight_stageA` |
| `experiments.analyze_harvest_validation` | `experiments.paper_v5.analyze_harvest_validation` |
| `experiments.analyze_heldout_oversight` | `experiments.common.analyze_heldout_oversight` |
| `experiments.analyze_llm_bridge_uncertainty` | `experiments.paper_v5.analyze_llm_bridge_uncertainty` |
| `experiments.analyze_matched_oversight` | `experiments.common.analyze_matched_oversight` |
| `experiments.analyze_overseer_limit_ablation` | `experiments.paper_v5.analyze_overseer_limit_ablation` |
| `experiments.analyze_oversight_decision_suite` | `experiments.archive.harvest_2026q2.analyze_oversight_decision_suite` |
| `experiments.analyze_r1_repaired_reviewer` | `experiments.oversight.analyze_r1_repaired_reviewer` |
| `experiments.analyze_reviewer_longrun` | `experiments.paper_v5.analyze_reviewer_longrun` |
| `experiments.analyze_s1_reporting_audit` | `experiments.oversight.analyze_s1_reporting_audit` |
| `experiments.analyze_s1b_s2` | `experiments.oversight.analyze_s1b_s2` |
| `experiments.analyze_stagea_stress_regimes` | `experiments.paper_v5.analyze_stagea_stress_regimes` |
| `experiments.analyze_threshold_replay_grid` | `experiments.paper_v5.analyze_threshold_replay_grid` |
| `experiments.audit_overseer_limit_ablation` | `experiments.paper_v5.audit_overseer_limit_ablation` |
| `experiments.audit_research_evidence` | `experiments.paper_v5.audit_research_evidence` |
| `experiments.audit_threshold_sweep_completeness` | `experiments.paper_v5.audit_threshold_sweep_completeness` |
| `experiments.build_harvest_strategy_bank` | `experiments.archive.harvest_2026q2.build_harvest_strategy_bank` |
| `experiments.check_harvest_policy_sources` | `experiments.paper_v5.check_harvest_policy_sources` |
| `experiments.check_llm_setup` | `experiments.archive.oneoff.check_llm_setup` |
| `experiments.check_paper_inputs` | `experiments.paper_v5.check_paper_inputs` |
| `experiments.claude_oversight_common` | `experiments.oversight.claude_oversight_common` |
| `experiments.diagnose_budgeted_reviewer_retention` | `experiments.archive.harvest_2026q2.diagnose_budgeted_reviewer_retention` |
| `experiments.evaluate_fishery_rl` | `experiments.archive.fishery_2026q1.evaluate_fishery_rl` |
| `experiments.evaluate_harvest_rl` | `experiments.archive.harvest_2026q2.evaluate_harvest_rl` |
| `experiments.export_harvest_scenario_table` | `experiments.archive.harvest_2026q2.export_harvest_scenario_table` |
| `experiments.export_reviewer_paper_data` | `experiments.paper_v5.export_reviewer_paper_data` |
| `experiments.extract_harvest_oversight_case` | `experiments.common.extract_harvest_oversight_case` |
| `experiments.generate_paper_v2_artifacts` | `experiments.archive.fishery_2026q1.generate_paper_v2_artifacts` |
| `experiments.harvest_invasion_presets` | `experiments.archive.harvest_2026q2.harvest_invasion_presets` |
| `experiments.harvest_oversight_reporting` | `experiments.common.harvest_oversight_reporting` |
| `experiments.make_episode_gif` | `experiments.archive.oneoff.make_episode_gif` |
| `experiments.make_governance_comparison_gif` | `experiments.archive.oneoff.make_governance_comparison_gif` |
| `experiments.make_invasion_gif` | `experiments.archive.oneoff.make_invasion_gif` |
| `experiments.make_progress_figures` | `experiments.oversight.make_progress_figures` |
| `experiments.merge_harvest_invasion_outputs` | `experiments.archive.harvest_2026q2.merge_harvest_invasion_outputs` |
| `experiments.merge_threshold_replay_shards` | `experiments.archive.harvest_2026q2.merge_threshold_replay_shards` |
| `experiments.organize_results` | `experiments.archive.oneoff.organize_results` |
| `experiments.package_paper_sources` | `experiments.paper_v5.package_paper_sources` |
| `experiments.plot_actor_pressure_pilot` | `experiments.archive.harvest_2026q2.plot_actor_pressure_pilot` |
| `experiments.plot_extra_talk_figures` | `experiments.archive.oneoff.plot_extra_talk_figures` |
| `experiments.plot_fishery_rl_paper` | `experiments.archive.fishery_2026q1.plot_fishery_rl_paper` |
| `experiments.plot_harvest_architecture_followup` | `experiments.archive.harvest_2026q2.plot_harvest_architecture_followup` |
| `experiments.plot_harvest_capability_ladder_publication` | `experiments.archive.harvest_2026q2.plot_harvest_capability_ladder_publication` |
| `experiments.plot_harvest_commons` | `experiments.archive.harvest_2026q2.plot_harvest_commons` |
| `experiments.plot_harvest_highpower_ci` | `experiments.archive.harvest_2026q2.plot_harvest_highpower_ci` |
| `experiments.plot_harvest_invasion` | `experiments.archive.harvest_2026q2.plot_harvest_invasion` |
| `experiments.plot_harvest_invasion_paper` | `experiments.archive.harvest_2026q2.plot_harvest_invasion_paper` |
| `experiments.plot_harvest_matrix_deltas` | `experiments.archive.harvest_2026q2.plot_harvest_matrix_deltas` |
| `experiments.plot_institutional_commons_v4` | `experiments.archive.harvest_2026q2.plot_institutional_commons_v4` |
| `experiments.plot_institutional_friction_winner_map` | `experiments.archive.harvest_2026q2.plot_institutional_friction_winner_map` |
| `experiments.plot_paper_v2_polish_figures` | `experiments.archive.fishery_2026q1.plot_paper_v2_polish_figures` |
| `experiments.plot_results` | `experiments.archive.fishery_2026q1.plot_results` |
| `experiments.plot_reviewer_decisions` | `experiments.paper_v5.plot_reviewer_decisions` |
| `experiments.plot_scalable_oversight_paper_v5` | `experiments.paper_v5.plot_scalable_oversight_paper_v5` |
| `experiments.plot_study1b_summary` | `experiments.archive.fishery_2026q1.plot_study1b_summary` |
| `experiments.replay_coupled_local` | `experiments.paper_v5.replay_coupled_local` |
| `experiments.run_actor_pressure_pilot` | `experiments.archive.harvest_2026q2.run_actor_pressure_pilot` |
| `experiments.run_budgeted_reviewer` | `experiments.common.run_budgeted_reviewer` |
| `experiments.run_budgeted_reviewer_confirmation` | `experiments.paper_v5.run_budgeted_reviewer_confirmation` |
| `experiments.run_fishery_rl_baseline` | `experiments.archive.fishery_2026q1.run_fishery_rl_baseline` |
| `experiments.run_governance_ablation` | `experiments.archive.fishery_2026q1.run_governance_ablation` |
| `experiments.run_greedy_sweep` | `experiments.archive.fishery_2026q1.run_greedy_sweep` |
| `experiments.run_harvest_invasion` | `experiments.common.run_harvest_invasion` |
| `experiments.run_harvest_invasion_local_shards` | `experiments.archive.harvest_2026q2.run_harvest_invasion_local_shards` |
| `experiments.run_harvest_invasion_matrix` | `experiments.common.run_harvest_invasion_matrix` |
| `experiments.run_harvest_llm_governance_map` | `experiments.archive.harvest_2026q2.run_harvest_llm_governance_map` |
| `experiments.run_harvest_llm_turnover` | `experiments.archive.harvest_2026q2.run_harvest_llm_turnover` |
| `experiments.run_harvest_rl_baseline` | `experiments.archive.harvest_2026q2.run_harvest_rl_baseline` |
| `experiments.run_harvest_study` | `experiments.archive.harvest_2026q2.run_harvest_study` |
| `experiments.run_heldout_oversight` | `experiments.common.run_heldout_oversight` |
| `experiments.run_invasion` | `experiments.archive.fishery_2026q1.run_invasion` |
| `experiments.run_matched_oversight` | `experiments.common.run_matched_oversight` |
| `experiments.run_orchard_study` | `experiments.archive.harvest_2026q2.run_orchard_study` |
| `experiments.run_overseer_limit_ablation` | `experiments.paper_v5.run_overseer_limit_ablation` |
| `experiments.run_oversight_decision_suite` | `experiments.archive.harvest_2026q2.run_oversight_decision_suite` |
| `experiments.run_r1_repaired_reviewer` | `experiments.oversight.run_r1_repaired_reviewer` |
| `experiments.run_s1_reporting_audit` | `experiments.oversight.run_s1_reporting_audit` |
| `experiments.run_s1b_ablation_msy` | `experiments.oversight.run_s1b_ablation_msy` |
| `experiments.run_s2_compliance_deterrence` | `experiments.oversight.run_s2_compliance_deterrence` |
| `experiments.run_single` | `experiments.archive.fishery_2026q1.run_single` |
| `experiments.run_study1b` | `experiments.archive.fishery_2026q1.run_study1b` |
| `experiments.run_sweep` | `experiments.archive.fishery_2026q1.run_sweep` |
| `experiments.run_targeted_threshold_replay` | `experiments.paper_v5.run_targeted_threshold_replay` |
| `experiments.run_threshold_replay_shard_group` | `experiments.archive.harvest_2026q2.run_threshold_replay_shard_group` |
| `experiments.run_visual_governance_pair` | `experiments.archive.oneoff.run_visual_governance_pair` |
| `experiments.showcase_project` | `experiments.archive.oneoff.showcase_project` |
| `experiments.smoke_cleanup_oversight` | `experiments.archive.harvest_2026q2.smoke_cleanup_oversight` |
| `experiments.summarize_fishery_rl` | `experiments.archive.fishery_2026q1.summarize_fishery_rl` |
| `experiments.summarize_governance_injector_match` | `experiments.archive.fishery_2026q1.summarize_governance_injector_match` |
| `experiments.summarize_harvest_highpower_ci` | `experiments.archive.harvest_2026q2.summarize_harvest_highpower_ci` |
| `experiments.summarize_harvest_invasion` | `experiments.archive.harvest_2026q2.summarize_harvest_invasion` |
| `experiments.summarize_harvest_matrix` | `experiments.archive.harvest_2026q2.summarize_harvest_matrix` |
| `experiments.summarize_injector_comparison` | `experiments.archive.fishery_2026q1.summarize_injector_comparison` |
| `experiments.summarize_paper_v1` | `experiments.archive.fishery_2026q1.summarize_paper_v1` |
| `experiments.summarize_study1b` | `experiments.archive.fishery_2026q1.summarize_study1b` |
| `experiments.summarize_tiered_ablation` | `experiments.archive.fishery_2026q1.summarize_tiered_ablation` |
| `experiments.threshold_replay_shards` | `experiments.archive.harvest_2026q2.threshold_replay_shards` |
| `experiments.train_fishery_rl` | `experiments.archive.fishery_2026q1.train_fishery_rl` |
| `experiments.train_harvest_rl` | `experiments.archive.harvest_2026q2.train_harvest_rl` |
| `experiments.validate_harvest_mechanisms` | `experiments.common.validate_harvest_mechanisms` |
