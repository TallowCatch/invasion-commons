from fishery_sim.config import load_config
from fishery_sim.simulation import run_episode
from fishery_sim.agents import GreedyAgent, ConservativeAgent, ConditionalCooperator, Punisher


def main():
    cfg = load_config("experiments/configs/base.yaml")

    # Example population mix (edit this)
    agents = (
        [GreedyAgent(max_h=cfg.max_harvest_per_agent)] * 3
        + [ConditionalCooperator(max_h=cfg.max_harvest_per_agent)] * 3
        + [ConservativeAgent(max_h=cfg.max_harvest_per_agent)] * 1
        + [Punisher(max_h=cfg.max_harvest_per_agent)] * 1
    )
    cfg.n_agents = len(agents)

    out = run_episode(cfg, agents)
    print("collapsed:", out["collapsed"])
    print("t_end:", out["t_end"])
    print("final_stock:", out["final_stock"])
    print("payoffs:", out["payoffs"])


if __name__ == "__main__":
    main()