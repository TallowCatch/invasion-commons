# Scientific Skills Installation

Date: 2026-09-21. Scope: user skills plus this report only. Available next turn.

## Installed Versus Custom

| Skill | Exact entrypoint | Status |
|---|---|---|
| experiment-planner | `/Users/ameerfiras/.codex/skills/experiment-planner/SKILL.md` | Installed with skill-installer, then locally adapted; MIT attribution retained. |
| research-evidence | `/Users/ameerfiras/.codex/skills/research-evidence/SKILL.md` | Installed with skill-installer, then locally adapted; MIT attribution and helper retained. |
| mlflow-local | `/Users/ameerfiras/.codex/skills/mlflow-local/SKILL.md` | Custom instructions, not an official MLflow skill or package installation. |
| scientific-experiment | `/Users/ameerfiras/.codex/skills/scientific-experiment/SKILL.md` | Custom general-purpose orchestration, not a downloaded skill. |

Custom record schema: `/Users/ameerfiras/.codex/skills/scientific-experiment/references/experiment-record.md`.
Custom skills have no upstream commit; they were authored locally in this task.

## Sources and Inspection

- Read `/Users/ameerfiras/.codex/skills/.system/skill-installer/SKILL.md` and `/Users/ameerfiras/.codex/skills/.system/skill-creator/SKILL.md`, plus the installer implementation and validator.
- Queried the [OpenAI curated catalog](https://github.com/openai/skills/tree/49f948faa9258a0c61caceaf225e179651397431/skills/.curated) using `list-skills.py`, then repeated at commit `49f948faa9258a0c61caceaf225e179651397431`. None of the requested names was present.
- Installed [experiment-planner](https://github.com/sidiangongyuan/codex-skills-library/tree/41f5a211b1a8d210023f51f9d56088311a3dae79/skills/experiment-planner) and [research-evidence](https://github.com/sidiangongyuan/codex-skills-library/tree/41f5a211b1a8d210023f51f9d56088311a3dae79/skills/research-evidence) from `sidiangongyuan/codex-skills-library`, commit `41f5a211b1a8d210023f51f9d56088311a3dae79`, paths `skills/experiment-planner` and `skills/research-evidence`.
- Before installation, inspected both complete SKILL.md files, every bundled reference, both UI metadata files, the MIT license, recursive file inventory/modes, and the complete `research-evidence/scripts/arxiv_query_matrix.py`. Planner contains no scripts. The evidence helper uses the standard library, prints to stdout, and contacts arXiv only with `--run`; no installer or shell execution is embedded in it.
- Inspected the [official MLflow skills tree](https://github.com/mlflow/skills/tree/0766761276a7dd378d88ac8aa7ca742c92b830fe) and [mlflow-onboarding/SKILL.md](https://github.com/mlflow/skills/blob/0766761276a7dd378d88ac8aa7ca742c92b830fe/mlflow-onboarding/SKILL.md), commit `0766761276a7dd378d88ac8aa7ca742c92b830fe`. Real official MLflow skills exist, but onboarding mixes GenAI tracing/autologging and general integration rather than enforcing this task's local-only logging boundary. None was installed; a smaller custom adapter is explicitly labeled as such.
- Consulted the [official MLflow tracking quickstart](https://mlflow.org/docs/latest/ml/tracking/quickstart/) and [Python API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html) for tracking, artifact destinations, and run-resumption behavior. These are live documentation, not pinned dependencies. No arbitrary third-party installer was executed.

Installation used the bundled helper, with all temporary download/extraction files confined to the skills directory and removed afterward:

```sh
env TMPDIR=/Users/ameerfiras/.codex/skills/.scientific-install-tmp PYTHONDONTWRITEBYTECODE=1 python3 -B /Users/ameerfiras/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py --repo sidiangongyuan/codex-skills-library --ref 41f5a211b1a8d210023f51f9d56088311a3dae79 --path skills/experiment-planner skills/research-evidence --dest /Users/ameerfiras/.codex/skills --method download
```

## Local Adaptations

- Planner: rewrote `SKILL.md` and `references/experiment-matrix.md`; prefixed `references/source-map.md` with installation provenance. Removed obligatory paper-story framing, single-seed inference defaults, absent-skill dependencies, and automatic delegation. Added preregistration, independent-unit uncertainty, fair target/authority comparisons, separate actor/reviewer resources, and bounded execution. Historical upstream source-map pins remain attribution only, not additional inspected/installed dependencies.
- Evidence: adapted `SKILL.md` and `references/tooling.md`. Removed venue-prestige defaults and the assumed author-specific environment; added primary-source/manual fallback, counterevidence, evidence depth, and claim-location checks. RefChecker, paper-search packages, credentials, and MCP servers were not installed. The existing `research-papers` skill was preserved.
- Custom orchestration routes planning, coding/tests, analysis, pilot, and continuation separately. It covers question/evidence, frozen hypotheses/baselines/matrix, implementation checks, capped local pilot, optional local MLflow, statistics/plots, justified ablation, `experiment.md`, and a stop/repair/continue/new-study decision.
- General safeguards cover same-target/same-authority local versus joint oversight, total reviewer costs separate from actor costs, no scalar-capability shortcut, no pseudoreplication, no retry-until-significant, and no mandatory novelty/winner. No repository-specific simulation assumptions are embedded.
- All manual file edits used `apply_patch`. Pre-existing skills were not edited. Simulation, paper, dependencies, and repository configuration were not modified by this task.

## Validation and Small Checks

Ran `python3 -B /Users/ameerfiras/.codex/skills/.system/skill-creator/scripts/quick_validate.py` separately against all four skill directories: **4/4 valid**. This checks skill structure, not scientific decision quality.

Ran the inspected arXiv helper with `--domain-term 'resource governance' --method-term oversight --dry-run`: exit 0, four query rows, no network query or report writes. Its installed Git blob is `99c4d7c4bd84620514537dfab4e6fbd47a913308`, matching the pinned tree; SHA-256 is `e710c40d6de7ecf219514189b2c0f347b8e770cc6b00296e03308ccbfb5884fc`. Both retained MIT licenses also match their upstream blob. Rechecked 438 captured pre-existing file hashes: unchanged. The initial hash listing was truncated, so this is not an exhaustive before/after filesystem audit.

Self-administered prompt walkthroughs used independent synthetic scenarios, not repository experiments; no additional agent/model was spawned. These are limited self-review, not independent-model behavioral validation:

| Test prompt | Observed decision |
|---|---|
| Plan only: local reviewers veto individual components; joint review revises the whole allocation; joint actors get more tokens. Is this a fair comparison? | No. Match the full target and intervention authority, hold actor budget fixed, account for total reviewer budget, and isolate information/coordination. No run or mandatory paper story. |
| Analyze existing paired-world totals: local `[10,11,9,10]`, joint `[11,13,9,11]`; each world contains 500 timesteps. | Four independent paired worlds, not 2,000 timestep replicates. Differences `[1,2,0,1]`, mean +1, sample SD 0.8165, SE 0.4082 under independent-world assumptions; exploratory/descriptive interpretation, no automatic winner or new runs. Arithmetic checked with Python's standard library. |
| Track a local pilot; MLflow is absent and an inherited tracking URI points to Databricks. | Use authorized local manifest/run files; no remote contact, credential use, global install, invented MLflow ID, or rerun for logging. |
| Fixed cap exhausted, p=0.06; keep adding seeds until p is below 0.05. | Stop the current study, retain all attempts and uncertainty, and propose only a separately preregistered follow-up. No significance-seeking retries. |
| Fix only a metric aggregation unit test. | Implementation/test scope only; no literature review, full experiment loop, or sweep. |

MLflow was absent from `/Library/Frameworks/Python.framework/Versions/3.12/bin/python3`; other project environments were not probed. No MLflow runtime logging test, service, simulation pilot, package installation, paid/cloud call, or paper/code test was performed. Future tracking must check the actual project interpreter and local artifact destination.

Use `$scientific-experiment` next turn for the orchestrated workflow, or invoke any of the three narrower skills directly.

## Parent-Task Integration, 22 September 2026

The installation-only report above describes the skill subtask. The main task
subsequently installed `mlflow-skinny==3.11.1`, SQLAlchemy and Alembic in the
repository-local `.venv`; it did not change the global scientific environment.
Actual inherited scientific-package versions are recorded in each run manifest.
`requirements-scientific.txt` specifies a fresh-install environment; the pilot
used the manifest's actual versions rather than claiming an exact lockfile match.

The completed Harvest/Fishery pilot was logged to local SQLite and local
artifacts, with run ID `1836ba58328a44baae481fc1155f834d`. No hosted MLflow server,
cloud account or external artifact destination was used. See
`MATCHED_OVERSIGHT_IMPLEMENTATION_CLOSEOUT.md` for the executed work and limits.
All four skills are now available in the active Codex skill inventory.
