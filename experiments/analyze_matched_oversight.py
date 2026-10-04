"""Context-level summaries for the bounded matched-monitor pilot."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from experiments.run_matched_oversight import read_json, digest

GROUP = ["game", "scenario", "regime", "method"]
METRICS = ["global_unsafe_rate", "total_welfare", "mean_patch_health", "mean_realized_harvest",
           "mean_prevented_harvest", "onset_count", "checks_per_step", "infeasible_rate"]
LABELS = {"none": "No intervention", "cutoff_uniform": "Uniform cutoff",
    "local_nominal": "Local reports: nominal", "local_uncertain": "Local reports: noise margin",
    "local_conservative": "Local reports: bounded neighbours",
    "local_conservative_uncertain": "Local reports: bounds + noise",
    "joint_nominal": "Joint: nominal", "joint_uncertain": "Joint: noise margin"}


def interval(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan, np.nan, 0
    mean = float(a.mean())
    if len(a) < 2:
        return mean, np.nan, np.nan, len(a)
    width = float(t.ppf(.975, len(a)-1) * a.std(ddof=1) / np.sqrt(len(a)))
    return mean, mean-width, mean+width, len(a)


def context_summary(episodes):
    # Weather trials are nested, not independent population discoveries.
    contexts = episodes.groupby(GROUP + ["context"], as_index=False)[METRICS].mean()
    rows = []
    for keys, frame in contexts.groupby(GROUP):
        row = dict(zip(GROUP, keys))
        for metric in METRICS:
            mean, low, high, n = interval(frame[metric])
            row.update({metric: mean, metric+"_lo": low, metric+"_hi": high, metric+"_n_contexts": n})
        rows.append(row)
    return contexts, pd.DataFrame(rows)


def paired_contrasts(contexts):
    comparisons = [("joint_nominal", "local_nominal"),
        ("joint_uncertain", "local_conservative_uncertain"),
        ("local_conservative_uncertain", "local_conservative"),
        ("joint_nominal", "local_conservative")]
    rows = []
    for keys, frame in contexts.groupby(["game", "scenario", "regime"]):
        for left, right in comparisons:
            a = frame[frame.method.eq(left)].set_index("context")
            b = frame[frame.method.eq(right)].set_index("context")
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            for metric in METRICS:
                mean, low, high, n = interval(a.loc[common, metric] - b.loc[common, metric])
                rows.append(dict(zip(["game", "scenario", "regime"], keys), left=left, right=right,
                    metric=metric, mean_difference=mean, ci95_low=low, ci95_high=high,
                    n_contexts=n, inference="exploratory t interval; no multiplicity adjustment"))
    return pd.DataFrame(rows)


def judgment_summary(judgments):
    sums = ["risky_resolved", "safe_resolved", "unresolved", "harmful_accepted", "safe_rejected", "approved", "abstained"]
    rows = []
    for keys, frame in judgments.groupby(GROUP + ["pre_global_safe"]):
        counts = frame[sums].sum().to_dict()
        row = dict(zip(GROUP + ["pre_global_safe"], keys), **counts)
        row.update(n_proposals=len(frame), n_contexts=frame.context.nunique(),
            harmful_accept_rate=counts["harmful_accepted"] / counts["risky_resolved"] if counts["risky_resolved"] else np.nan,
            safe_reject_rate=counts["safe_rejected"] / counts["safe_resolved"] if counts["safe_resolved"] else np.nan,
            unresolved_rate=counts["unresolved"] / len(frame), approval_coverage=counts["approved"] / len(frame),
            abstention_coverage=counts["abstained"] / len(frame),
            mean_component_evaluations=frame.component_evaluations.mean())
        rows.append(row)
    return pd.DataFrame(rows)


def plot(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 9, "axes.spines.top": False,
        "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.linewidth": .6, "savefig.dpi": 300})
    for game, game_data in summary.groupby("game"):
        regimes = list(game_data.regime.unique())
        fig, axes = plt.subplots(len(regimes), 3, figsize=(10.8, max(3, 2.8*len(regimes))), squeeze=False)
        for r, regime in enumerate(regimes):
            frame = game_data[game_data.regime.eq(regime)].copy()
            order = list(LABELS)
            frame["order"] = frame.method.map(order.index)
            frame = frame.sort_values("order")
            yy = np.arange(len(frame))
            for col, (metric, title) in enumerate([("global_unsafe_rate", "Unsafe steps (lower is better)"),
                    ("total_welfare", "Population return"), ("checks_per_step", "Model components checked / step")]):
                ax = axes[r, col]
                for pos, (_, row) in enumerate(frame.iterrows()):
                    color = "#086F73" if row.method.startswith("joint") else "#9A5B27" if row.method.startswith("local") else "#60666A"
                    mean, low, high = row[metric], row[metric+"_lo"], row[metric+"_hi"]
                    ax.plot(mean, pos, "D" if row.method.startswith("joint") else "o", color=color, ms=4)
                    if np.isfinite(low):
                        ax.plot([low, high], [pos, pos], color=color, lw=1)
                ax.set_yticks(yy, [LABELS[m] for m in frame.method] if col == 0 else [])
                ax.invert_yaxis()
                ax.set_title(title, loc="left", fontsize=9)
                ax.grid(axis="x", alpha=.13, lw=.5)
                ax.set_xlabel("" if metric != "global_unsafe_rate" else "Fraction of observed episode steps")
                if col == 0:
                    ax.set_ylabel(regime.replace("_", " "))
        fig.suptitle(f"{game.title()}: matched-target development pilot", x=.02, ha="left", fontsize=12)
        fig.text(.02, .012, "Dots: context means; bars: exploratory 95% t intervals across policy contexts, not timesteps.\n"
                 "Local Harvest reports are pooled; all methods use the same global target and scale menu.", fontsize=8)
        fig.tight_layout(rect=(0, .065, 1, .96))
        for extension in ("pdf", "svg", "png"):
            fig.savefig(output / f"{game}_matched_pilot.{extension}", bbox_inches="tight")
        plt.close(fig)


def analyze(directory):
    directory = Path(directory)
    checksums = read_json(directory / "block_checksums.json")
    completion = read_json(directory / "completion.json")
    if len(checksums) != completion["expected_blocks"]:
        raise ValueError("Incomplete pilot")
    blocks, rows = [], []
    for name, checksum in checksums.items():
        path = directory / "blocks" / f"{name}.json.gz"
        if digest(path) != checksum:
            raise ValueError(f"Corrupt block: {path}")
        block = read_json(path)
        blocks.append(block)
        row = {key: block[key] for key in GROUP + ["context", "weather"]}
        row.update(block["metrics"])
        row["checks_per_step"] = row["component_evaluations"] / row["t_end"]
        row["infeasible_rate"] = row["infeasible_steps"] / row["t_end"]
        rows.append(row)
    episodes = pd.DataFrame(rows)
    contexts, summary = context_summary(episodes)
    judgments = pd.DataFrame(read_json(directory / "frozen_judgments.json.gz"))
    # Every method must see identical proposals, including reference-label noise.
    keys = ["game", "regime", "context", "weather", "step", "proposal_scale"]
    if (judgments.groupby(keys).risk.nunique() != 1).any():
        raise ValueError("Reference labels differ between monitors")
    output = directory / "analysis"
    output.mkdir(exist_ok=True)
    episodes.to_csv(output / "episodes.csv", index=False)
    contexts.to_csv(output / "context_means.csv", index=False)
    summary.to_csv(output / "outcomes.csv", index=False)
    paired_contrasts(contexts).to_csv(output / "paired_contrasts.csv", index=False)
    judgments.to_csv(output / "frozen_judgments.csv", index=False)
    judgment_summary(judgments).to_csv(output / "decision_quality.csv", index=False)
    plot(summary, output)
    manifest = read_json(directory / "manifest.json")
    table = ["| Game / setting | Monitor | Unsafe steps | Return | Checks / step |", "| --- | --- | ---: | ---: | ---: |"]
    for row in summary.itertuples():
        table.append(f"| {row.game} / {row.regime} | {row.method} | {row.global_unsafe_rate:.3%} | {row.total_welfare:.2f} | {row.checks_per_step:.2f} |")
    text = "\n".join([
        "# Matched Oversight Development Experiment", "", "## Question",
        "When does joint action information add value over pooled local predictions, at a common safety target and authority?", "",
        "## Design", f"Profile: {manifest['protocol']['profile']}. {len(episodes)} closed-loop episodes. "
        f"{manifest['protocol']['contexts']} independently seeded policy contexts; weather trials are nested.",
        "Harvest uses one spatial configuration and two regeneration settings. Fishery uses one deterministic shared-stock configuration.",
        "Policies are freshly sampled and paired across monitors. Search capability is not manipulated in this pilot.",
        "Frozen decision comparisons use identical no-intervention snapshots, separately from closed-loop outcome comparisons.", "",
        "## Results", *table, "", "## Interpretation Boundaries",
        "This is a development pilot, not confirmatory evidence or a ranking of general actor/overseer capability.",
        "Harvest local reviewers send predicted patch-health reports to a common aggregator. These are not fully independent local decisions.",
        "Nominal local predictions ignore neighbour excess; conservative variants bound it. All variants use the same global predicate.",
        "Uncertainty-aware variants use a one-step Gaussian union-bound margin. Nominal local + noise does not bound omitted neighbour damage.",
        "Fishery local conservative uses an equal-share sufficient contract; the joint check sees total demand. Fishery has no stochastic weather axis.",
        "Reported checks count model components, not calibrated hardware cost. Candidate budget limits repair search, not forecast intelligence.",
        "Reference risk labels are exact for Fishery and Monte Carlo with Wilson intervals for Harvest. Unresolved labels are retained.",
        "The judgment corpus samples the no-intervention trajectory and four proposal scales (1, .5, .25, 0), not all possible proposals. Frames and variants within a context are dependent.",
        "Historical fixed-cutoff results are not reproduced here: cutoff_uniform uses the shared finite scale menu.",
        "## Next Decision",
        "Inspect whether shared evidence reduces unnecessary restriction and whether uncertainty assumptions explain unsafe approvals. "
        "Validate Cleanup policies before a comparative sweep. Add search pressure and true evidence/rollout budgets only after these controls pass.", "",
        "## Reproduction",
        "See manifest.json for protocol, source hashes, runtime packages and seeds; block_checksums.json validates atomic raw outputs.",
        "Rerun the same command/directory only with an identical manifest. Analyze with python -m experiments.analyze_matched_oversight --input-dir PATH.", "",
    ])
    (output / "experiment.md").write_text(text, encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    analyze(parser.parse_args().input_dir)
