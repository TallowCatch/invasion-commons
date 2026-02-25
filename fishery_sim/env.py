from dataclasses import dataclass
import numpy as np


@dataclass
class StepResult:
    stock: float
    harvests: np.ndarray
    payoffs: np.ndarray
    collapsed: bool
    below_threshold_count: int


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

        self.rng = rng if rng is not None else np.random.default_rng()

        self.reset()

    def reset(self):
        self.stock = self.stock_init
        self.below_count = 0
        self.collapsed = False

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

        harvests = np.asarray(harvests, dtype=float)
        harvests = np.clip(harvests, 0.0, self.max_harvest_per_agent)

        total_harvest = harvests.sum()

        # Can't take more than exists
        if total_harvest > self.stock and total_harvest > 0:
            harvests *= self.stock / total_harvest
            total_harvest = harvests.sum()

        payoffs = harvests.copy()

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
        )