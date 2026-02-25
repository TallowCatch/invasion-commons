import os
import copy
import pandas as pd
from tqdm import tqdm

from fishery_sim.config import load_config
from fishery_sim.simulation import run_episode
from fishery_sim.agents import GreedyAgent, ConservativeAgent, ConditionalCooperator, Punisher


CONFIG_PATH = "experiments/configs/base.yaml"
N_SEEDS = 200


def make_population(cfg, n_greedy):
    total_agents = 8
    n_remaining = total_agents - n_greedy

    n_cond = n_remaining // 2
    n_cons = (n_remaining - n_cond) // 2
    n_pun = n_remaining - n_cond - n_cons

    agents = []

    # IMPORTANT: create fresh objects
    for _ in range(n_greedy):
        agents.append(GreedyAgent(max_h=cfg.max_harvest_per_agent))

    for _ in range(n_cond):
        agents.append(ConditionalCooperator(max_h=cfg.max_harvest_per_agent))

    for _ in range(n_cons):
        agents.append(ConservativeAgent(max_h=cfg.max_harvest_per_agent))

    for _ in range(n_pun):
        agents.append(Punisher(max_h=cfg.max_harvest_per_agent))

    return agents


def main():
    base_cfg = load_config(CONFIG_PATH)
    os.makedirs("results", exist_ok=True)

    rows = []

    print(f"\nRunning greedy composition sweep (regen={base_cfg.regen_rate})\n")

    for n_greedy in range(0, 9):

        collapsed_count = 0

        for seed in tqdm(range(N_SEEDS), leave=False):

            cfg = copy.deepcopy(base_cfg)
            cfg.seed = seed

            agents = make_population(cfg, n_greedy)

            out = run_episode(cfg, agents)

            if out["collapsed"]:
                collapsed_count += 1

        collapse_rate = collapsed_count / N_SEEDS

        rows.append({
            "regen_rate": base_cfg.regen_rate,
            "n_greedy": n_greedy,
            "collapse_rate": collapse_rate
        })

        print(f"Greedy: {n_greedy} | Collapse rate: {collapse_rate:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv("results/greedy_sweep.csv", index=False)

    print("\nSaved to results/greedy_sweep.csv")


if __name__ == "__main__":
    main()