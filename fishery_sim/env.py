from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    stock: float
    harvests: np.ndarray
    payoffs: np.ndarray
    collapsed: bool
    below_threshold_count: int
    sanction_total: float = 0.0
    num_violations: int = 0


class FisheryEnv:
    """
    Minimal global-stock fishery.
    """

    def __init__(
        self,
        n_agents: int,
        stock_init: float,
        stock_max: float,
        regen_rate: float,
        collapse_threshold: float,
        collapse_patience: int,
        max_harvest_per_agent: float,
        obs_noise_std: float = 0.0,
        monitoring_prob: float = 0.0,
        quota_fraction: float = 0.0,
        base_fine_rate: float = 1.0,
        fine_growth: float = 0.5,
        rng: np.random.Generator | None = None,
    ):
        self.n_agents = n_agents
        self.stock_init = float(stock_init)
        self.stock_max = float(stock_max)
        self.regen_rate = float(regen_rate)
        self.collapse_threshold = float(collapse_threshold)
        self.collapse_patience = int(collapse_patience)
        self.max_harvest_per_agent = float(max_harvest_per_agent)
        self.obs_noise_std = float(obs_noise_std)
        self.monitoring_prob = float(np.clip(monitoring_prob, 0.0, 1.0))
        self.quota_fraction = float(max(0.0, quota_fraction))
        self.base_fine_rate = float(max(0.0, base_fine_rate))
        self.fine_growth = float(max(0.0, fine_growth))

        self.rng = rng if rng is not None else np.random.default_rng()

        self.reset()

    def reset(self):
        self.stock = self.stock_init
        self.below_count = 0
        self.collapsed = False
        self.offense_counts = np.zeros(self.n_agents, dtype=int)

    # ---- Observation ----
    def observe_stock(self) -> float:
        if self.obs_noise_std > 0:
            noisy = self.stock + self.rng.normal(0.0, self.obs_noise_std)
        else:
            noisy = self.stock

        # IMPORTANT: clip observation
        return float(np.clip(noisy, 0.0, self.stock_max))

    # ---- Environment Step ----
    def step(self, harvests: np.ndarray) -> StepResult:

        if self.collapsed:
            return StepResult(
                stock=self.stock,
                harvests=harvests,
                payoffs=np.zeros(self.n_agents),
                collapsed=True,
                below_threshold_count=self.below_count,
            )

        requested = np.asarray(harvests, dtype=float)
        requested = np.clip(requested, 0.0, self.max_harvest_per_agent)
        harvests = requested.copy()
        fines = np.zeros(self.n_agents)

        sanction_total = 0.0
        num_violations = 0

        if self.monitoring_prob > 0.0 and self.quota_fraction > 0.0:
            # Quota scales with current stock and is capped by action bound.
            quota = float(np.clip(self.quota_fraction * self.stock, 0.0, self.max_harvest_per_agent))
            audited = self.rng.random(self.n_agents) < self.monitoring_prob
            violations = np.maximum(0.0, requested - quota)
            offenders = audited & (violations > 0.0)

            if np.any(offenders):
                # Enforced quota reduces extraction pressure for audited violators.
                harvests[offenders] = quota

                multipliers = 1.0 + self.fine_growth * self.offense_counts
                fines[offenders] = self.base_fine_rate * multipliers[offenders] * violations[offenders]

                self.offense_counts[offenders] += 1
                sanction_total = float(fines.sum())
                num_violations = int(offenders.sum())

            self.offense_counts[~offenders] = np.maximum(0, self.offense_counts[~offenders] - 1)
        total_harvest = harvests.sum()

        # Can't take more than exists
        if total_harvest > self.stock and total_harvest > 0:
            harvests *= self.stock / total_harvest
            total_harvest = harvests.sum()

        payoffs = np.maximum(0.0, harvests - fines)

        remaining = max(0.0, self.stock - total_harvest)

        # Logistic growth
        growth = self.regen_rate * remaining * (1.0 - remaining / self.stock_max)
        self.stock = float(np.clip(remaining + max(0.0, growth), 0.0, self.stock_max))

        # Collapse tracking
        if self.stock < self.collapse_threshold:
            self.below_count += 1
        else:
            self.below_count = 0

        if self.below_count >= self.collapse_patience:
            self.collapsed = True
            self.stock = 0.0

        return StepResult(
            stock=self.stock,
            harvests=harvests,
            payoffs=payoffs,
            collapsed=self.collapsed,
            below_threshold_count=self.below_count,
            sanction_total=sanction_total,
            num_violations=num_violations,
        )
