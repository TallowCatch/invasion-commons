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
        governance_variant=cfg.governance_variant,
        adaptive_quota_min_scale=cfg.adaptive_quota_min_scale,
        adaptive_quota_sensitivity=cfg.adaptive_quota_sensitivity,
        temporary_closure_trigger=cfg.temporary_closure_trigger,
        temporary_closure_quota_fraction=cfg.temporary_closure_quota_fraction,
        rng=rng,
    )
    env.reset()

    stock_trace = []
    payoff_total = np.zeros(cfg.n_agents)
    sanction_total = 0.0
    violation_events = 0
    requested_trace = []
    realized_trace = []
    audit_rate_trace = []
    quota_trace = []
    quota_clipped_trace = []
    repeat_offender_rate_trace = []
    closure_trace = []
    collapsed = False
    t_end = -1
    low_stock_start = None
    recovery_lags = []

    for t in range(cfg.horizon):
        t_end = t
        obs = env.observe_stock()
        harvests = np.array([a.act(obs, t, cfg.n_agents) for a in agents], dtype=float)
        step = env.step(harvests)

        stock_trace.append(step.stock)
        payoff_total += step.payoffs
        sanction_total += step.sanction_total
        violation_events += step.num_violations
        requested_trace.append(step.requested_harvest_total)
        realized_trace.append(step.realized_harvest_total)
        audit_rate_trace.append(step.audit_count / max(cfg.n_agents, 1))
        quota_trace.append(step.quota)
        quota_clipped_trace.append(step.quota_clipped_total)
        repeat_offender_rate_trace.append(step.repeat_offender_count / max(cfg.n_agents, 1))
        closure_trace.append(1.0 if step.closure_active else 0.0)

        if step.stock < cfg.collapse_threshold:
            if low_stock_start is None:
                low_stock_start = t
        elif low_stock_start is not None:
            recovery_lags.append(t - low_stock_start)
            low_stock_start = None

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
        "mean_requested_harvest": float(np.mean(requested_trace)) if requested_trace else 0.0,
        "mean_realized_harvest": float(np.mean(realized_trace)) if realized_trace else 0.0,
        "mean_audit_rate": float(np.mean(audit_rate_trace)) if audit_rate_trace else 0.0,
        "mean_quota": float(np.mean(quota_trace)) if quota_trace else 0.0,
        "mean_quota_clipped_total": float(np.mean(quota_clipped_trace)) if quota_clipped_trace else 0.0,
        "mean_repeat_offender_rate": float(np.mean(repeat_offender_rate_trace)) if repeat_offender_rate_trace else 0.0,
        "closure_active_fraction": float(np.mean(closure_trace)) if closure_trace else 0.0,
        "mean_stock_recovery_lag": float(np.mean(recovery_lags)) if recovery_lags else 0.0,
    }
