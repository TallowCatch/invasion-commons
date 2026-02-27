from fishery_sim.config import load_config
from fishery_sim.simulation import run_episode
from fishery_sim.agents import GreedyAgent, ConservativeAgent, ConditionalCooperator, Punisher


def main():
    cfg = load_config("experiments/configs/base.yaml")

    # Example population mix (edit this)
    agents = []
    for _ in range(3):
        agents.append(GreedyAgent(max_h=cfg.max_harvest_per_agent))
    for _ in range(3):
        agents.append(ConditionalCooperator(max_h=cfg.max_harvest_per_agent))
    agents.append(ConservativeAgent(max_h=cfg.max_harvest_per_agent))
    agents.append(Punisher(max_h=cfg.max_harvest_per_agent))
    cfg.n_agents = len(agents)

    out = run_episode(cfg, agents)
    print("collapsed:", out["collapsed"])
    print("t_end:", out["t_end"])
    print("final_stock:", out["final_stock"])
    print("payoffs:", out["payoffs"])


if __name__ == "__main__":
    main()
