import numpy as np
from .env import FisheryEnv
from .config import FisheryConfig
from .agents import BaseAgent


def run_episode(cfg: FisheryConfig, agents: list[BaseAgent]) -> dict:
    assert len(agents) == cfg.n_agents, "agents list must match n_agents"

    rng = np.random.default_rng(cfg.seed)

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

    stock_trace = []
    payoff_total = np.zeros(cfg.n_agents)
    sanction_total = 0.0
    violation_events = 0
    collapsed = False
    t_end = -1

    for t in range(cfg.horizon):
        t_end = t
        obs = env.observe_stock()
        harvests = np.array([a.act(obs, t, cfg.n_agents) for a in agents], dtype=float)
        step = env.step(harvests)

        stock_trace.append(step.stock)
        payoff_total += step.payoffs
        sanction_total += step.sanction_total
        violation_events += step.num_violations

        if step.collapsed:
            collapsed = True
            break

    return {
        "seed": cfg.seed,
        "collapsed": collapsed,
        "t_end": t_end,
        "final_stock": float(stock_trace[-1]) if stock_trace else float(env.stock),
        "mean_stock": float(np.mean(stock_trace)) if stock_trace else 0.0,
        "payoffs": payoff_total,
        "stock_trace": stock_trace,
        "sanction_total": sanction_total,
        "violation_events": violation_events,
    }
