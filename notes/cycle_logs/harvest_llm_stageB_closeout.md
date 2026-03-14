# harvest llm stageB closeout

## 1. What was implemented
- Added live and replay `llm_json` injection to Harvest Commons invasion so Study 2 can use the same LLM-style strategy generation path as Fishery Commons.
- Reused the existing policy-client pattern instead of inventing a separate LLM interface for Harvest.
- Added Harvest-specific JSON parsing, clamping, and mutation fallback inside the Harvest invasion engine.
- Added LLM integrity logging for Harvest:
  - `llm_json_fraction`
  - `llm_fallback_fraction`
  - provider and model metadata
- Added a resumable local shard runner for long Ollama sweeps:
  - `experiments/run_harvest_invasion_local_shards.py`
- Completed the full local Stage B Harvest LLM matrix with shard saving, merge, summary, and plots.

## 2. Key artifacts
- LLM smoke:
  - `results/runs/showcase/curated/harvest_invasion_llm_stageA_smoke_summary.md`
  - `results/runs/showcase/curated/harvest_invasion_llm_stageA_smoke_integrity.csv`
- Stage B local LLM matrix:
  - `results/runs/harvest_invasion/curated/harvest_invasion_llm_stageB_local_runs.csv`
  - `results/runs/harvest_invasion/curated/harvest_invasion_llm_stageB_local_generation_history.csv`
  - `results/runs/harvest_invasion/curated/harvest_invasion_llm_stageB_local_strategy_history.csv`
  - `results/runs/showcase/curated/harvest_invasion_llm_stageB_local_summary.md`
  - `results/runs/showcase/curated/harvest_invasion_llm_stageB_local_delta.csv`
  - `results/runs/showcase/curated/harvest_invasion_llm_stageB_local_ci.csv`
  - `results/runs/showcase/curated/harvest_invasion_llm_stageB_local_ranking.csv`
- Current non-LLM Harvest reference:
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_summary.md`
  - `results/runs/showcase/curated/harvest_invasion_stageC_highpower_gh_delta.csv`

## 3. Decision-critical results
- Harvest LLM Stage B:
  - `hybrid` wins `11/16` cells
  - `top_down_only` wins `5/16` cells
- Mean paired `hybrid - top_down_only` deltas under live LLM injection:
  - patch health: `+0.0234`
  - welfare: `-0.1181`
  - local aggression: `-0.0135`
  - neighborhood overharvest: `-0.1244`
- Non-LLM Harvest high-power reference:
  - `hybrid` wins `14/16` cells
  - mean paired deltas:
    - patch health: `+0.0228`
    - welfare: `-0.0295`
    - local aggression: `-0.0034`
    - neighborhood overharvest: `-0.0352`

## 4. Integrity readout
- The Harvest LLM path works technically.
- Stage A smoke passed parser stability and fallback checks.
- Stage B fallback is not catastrophic, but it is uneven across cells.
- Rows with fallback above the rough `10%` comfort threshold include:
  - `hard_h1 / balanced / hybrid / 0.3`
  - `medium_h1 / adversarial_heavy / hybrid / 0.3`
  - `medium_h1 / balanced / hybrid / 0.3`
  - `medium_h1 / balanced / top_down_only / 0.3`

## 5. Interpretation
- Harvest live LLM injection is now a real capability, not a missing feature.
- The ecological direction still broadly matches the non-LLM Study 2 result: hybrid often improves patch health and reduces neighborhood overharvest.
- The problem is that the LLM result is less stable and carries a larger welfare penalty, especially in the medium tier.
- That means the Harvest LLM path is good enough for smoke runs, capability demonstrations, and targeted follow-ups, but not yet strong enough for an expensive Stage C high-power claim.

## 6. Decision
- Do **not** run Harvest LLM Stage C next.
- Keep the current paper frozen.
- Treat Harvest LLM Stage B as capability and diagnostic evidence.
- Use a stronger non-LLM Harvest baseline as the next main evidence cycle.

## 7. Supervisor talk track (short)
I closed the remaining feature gap between Fishery and Harvest by adding live LLM injection to Harvest Commons and running a full Stage B local Ollama matrix. The path works and the ecological direction is broadly sensible, but the result is weaker and noisier than the non-LLM Harvest evidence, especially on welfare and in medium-tier cells. So the right next step is not a high-power Harvest LLM follow-up. It is a stronger non-LLM baseline that stays inside the same causal setup.

## 8. Next cycle objective
Run a Harvest `mutation` versus `search_mutation` evidence cycle and use that as the next Study 2 strengthening pass.
