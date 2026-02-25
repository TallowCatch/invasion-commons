from dataclasses import dataclass
import numpy as np


class BaseAgent:
    name: str = "base"

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        raise NotImplementedError


@dataclass
class GreedyAgent(BaseAgent):
    name: str = "greedy"
    max_h: float = 10.0

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        return self.max_h


@dataclass
class ConservativeAgent(BaseAgent):
    name: str = "conservative"
    max_h: float = 10.0
    target_fraction: float = 0.03  # harvest small fraction of perceived stock

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        return float(np.clip(obs_stock * self.target_fraction, 0.0, self.max_h))


@dataclass
class ConditionalCooperator(BaseAgent):
    name: str = "conditional"
    max_h: float = 10.0
    safe_stock: float = 60.0

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        # Cooperate when stock is healthy, tighten when low
        if obs_stock >= self.safe_stock:
            return min(self.max_h * 0.5, self.max_h)
        return min(self.max_h * 0.1, self.max_h)


@dataclass
class Punisher(BaseAgent):
    """
    Simple proxy for punishment: if stock is low, harvest even less (self-restraint).
    Later you can implement actual sanctions / audits.
    """
    name: str = "punisher"
    max_h: float = 10.0
    low_stock: float = 40.0

    def act(self, obs_stock: float, t: int, n_agents: int) -> float:
        return self.max_h * (0.2 if obs_stock < self.low_stock else 0.6)