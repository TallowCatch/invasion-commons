import argparse
import os

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = os.path.join("results", ".mplconfig")
if "XDG_CACHE_HOME" not in os.environ:
    os.environ["XDG_CACHE_HOME"] = os.path.join("results", ".cache")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

from fishery_sim.agents import ConditionalCooperator, ConservativeAgent, GreedyAgent, Punisher
from fishery_sim.config import load_config
from fishery_sim.env import FisheryEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a per-step episode animation GIF showing stock and agent harvest dynamics."
    )
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--output", default="results/runs/showcase/episode_stock_harvest_dynamics.gif")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def make_agents(max_h: float):
    agents = []
    for _ in range(3):
        agents.append(GreedyAgent(max_h=max_h))
    for _ in range(3):
        agents.append(ConditionalCooperator(max_h=max_h))
    agents.append(ConservativeAgent(max_h=max_h))
    agents.append(Punisher(max_h=max_h))
    return agents


def rollout_episode(args: argparse.Namespace):
    cfg = load_config(args.config)
    rng = np.random.default_rng(cfg.seed)
    agents = make_agents(cfg.max_harvest_per_agent)
    cfg.n_agents = len(agents)

    env = FisheryEnv(
        n_agents=cfg.n_agents,
        stock_init=cfg.stock_init,
        stock_max=cfg.stock_max,
        regen_rate=cfg.regen_rate,
        collapse_threshold=cfg.collapse_threshold,
        collapse_patience=cfg.collapse_patience,
        max_harvest_per_agent=cfg.max_harvest_per_agent,
        obs_noise_std=cfg.obs_noise_std,
        monitoring_prob=cfg.monitoring_prob,
        quota_fraction=cfg.quota_fraction,
        base_fine_rate=cfg.base_fine_rate,
        fine_growth=cfg.fine_growth,
        rng=rng,
    )
    env.reset()

    stock = []
    harvest_mat = []
    below_counts = []
    collapsed_step = None
    n_steps = min(cfg.horizon, args.max_steps)
    agent_labels = [f"A{i}\n{agent.name}" for i, agent in enumerate(agents)]

    for t in range(n_steps):
        obs = env.observe_stock()
        harvests = np.array([a.act(obs, t, cfg.n_agents) for a in agents], dtype=float)
        step = env.step(harvests)
        stock.append(step.stock)
        harvest_mat.append(step.harvests)
        below_counts.append(step.below_threshold_count)
        if step.collapsed:
            collapsed_step = t
            break

    return (
        np.array(stock, dtype=float),
        np.array(harvest_mat, dtype=float),
        np.array(below_counts, dtype=int),
        collapsed_step,
        cfg,
        agent_labels,
    )


def main() -> None:
    args = parse_args()
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    stock, harvest_mat, below_counts, collapsed_step, cfg, agent_labels = rollout_episode(args)
    n_frames = len(stock)
    if n_frames == 0:
        raise RuntimeError("No frames generated; check config/horizon.")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        gridspec_kw={"height_ratios": [1.2, 1.0]},
        constrained_layout=True,
    )
    ax_stock, ax_harv = axes
    x_all = np.arange(n_frames)

    def animate(frame_idx: int):
        ax_stock.clear()
        ax_harv.clear()

        x = x_all[: frame_idx + 1]
        y = stock[: frame_idx + 1]
        ax_stock.plot(x, y, color="#1f77b4", linewidth=2, label="stock")
        ax_stock.scatter([frame_idx], [stock[frame_idx]], color="#d62728", zorder=3)
        ax_stock.axhline(cfg.collapse_threshold, color="#ff7f0e", linestyle="--", linewidth=1.5, label="collapse threshold")
        ax_stock.set_ylim(0, max(cfg.stock_max * 1.05, 1.0))
        ax_stock.set_xlim(0, max(1, n_frames - 1))
        ax_stock.set_title("Stock Through Time")
        ax_stock.set_ylabel("Stock")
        ax_stock.grid(alpha=0.25)
        ax_stock.legend(loc="upper right", fontsize=8)

        cur_h = harvest_mat[frame_idx]
        ax_harv.bar(np.arange(len(cur_h)), cur_h, color="#2ca02c")
        ax_harv.set_xticks(np.arange(len(cur_h)))
        ax_harv.set_xticklabels(agent_labels, rotation=0, fontsize=8)
        ax_harv.set_ylim(0, max(cfg.max_harvest_per_agent * 1.1, 1.0))
        ax_harv.set_title("Harvest by Agent (Current Step)")
        ax_harv.set_ylabel("Harvest")
        ax_harv.grid(alpha=0.2, axis="y")

        if collapsed_step is not None and frame_idx >= collapsed_step:
            status_text = f" | collapsed at t={collapsed_step}"
        elif stock[frame_idx] <= 0.0 and below_counts[frame_idx] < cfg.collapse_patience:
            status_text = (
                f" | depleted (patience {int(below_counts[frame_idx])}/{cfg.collapse_patience})"
            )
        elif collapsed_step is not None:
            status_text = f" | pre-collapse (collapse occurs at t={collapsed_step})"
        else:
            status_text = " | no collapse in shown horizon"
        fig.suptitle(f"Fishery Episode Dynamics | t={frame_idx}{status_text}", fontsize=13, fontweight="bold")

    anim = FuncAnimation(fig, animate, frames=n_frames, interval=max(80, int(1000 / max(1, args.fps))), repeat=True)
    anim.save(args.output, writer=PillowWriter(fps=max(1, args.fps)), dpi=args.dpi)
    plt.close(fig)
    print(f"Saved GIF: {args.output}")


if __name__ == "__main__":
    main()
