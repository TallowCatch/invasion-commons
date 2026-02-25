import os
import pandas as pd
from tqdm import tqdm

from fishery_sim.config import load_config, FisheryConfig
from fishery_sim.simulation import run_episode
from fishery_sim.agents import GreedyAgent, ConservativeAgent, ConditionalCooperator, Punisher
from fishery_sim.metrics import gini


def make_agents(cfg: FisheryConfig):
    agents = (
        [GreedyAgent(max_h=cfg.max_harvest_per_agent)] * 3
        + [ConditionalCooperator(max_h=cfg.max_harvest_per_agent)] * 3
        + [ConservativeAgent(max_h=cfg.max_harvest_per_agent)] * 1
        + [Punisher(max_h=cfg.max_harvest_per_agent)] * 1
    )
    cfg.n_agents = len(agents)
    return agents


def main():
    cfg = load_config("experiments/configs/base.yaml")

    rows = []
    os.makedirs("results", exist_ok=True)

    for seed in tqdm(range(0, 200)):
        cfg.seed = seed
        agents = make_agents(cfg)
        out = run_episode(cfg, agents)
        rows.append({
            "seed": seed,
            "regen_rate": cfg.regen_rate,
            "obs_noise_std": cfg.obs_noise_std,
            "collapsed": out["collapsed"],
            "t_end": out["t_end"],
            "final_stock": out["final_stock"],
            "mean_stock": out["mean_stock"],
            "payoff_sum": float(out["payoffs"].sum()),
            "payoff_gini": gini(out["payoffs"]),
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/sweep.csv", index=False)
    print(df.groupby("collapsed")[["mean_stock", "payoff_sum", "payoff_gini"]].mean())


if __name__ == "__main__":
    main()