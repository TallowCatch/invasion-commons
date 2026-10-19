"""
R1b Pool B (blind) strategies for common_pool.

Generated with Willis et al.'s pipeline (two steps, their validation, test harness and retries).
- Provider: Ollama
- Model: gpt-oss:120b-cloud
- Game: common_pool
- Built from records/ by experiments/oversight/run_r1b_generate.py
"""

from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from numpy.typing import NDArray
import random

from emergent_llm.players import BaseStrategy
from emergent_llm.games import PublicGoodsDescription, CollectiveRiskDescription, CommonPoolDescription
from emergent_llm.common import Action, C, D, PlayerHistory



class Strategy_ALTRUISTIC_1(BaseStrategy):
    """
    Altruistic Conservation Strategy.

    - α (safety proportion) = 0.4, τ = α·capacity.
    - Cooperate (C) in the first and last round.
    - If current stock ≥ τ → cooperate.
    - If stock < τ → project the stock after an all‑cooperate round;
      if the projected stock would be positive (or zero) still cooperate.
    - Optionally modulate a cooperation probability based on the observed
      defection rate, but never actually defect.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.4
        self.tau = self.alpha * float(self.game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_idx = history.round_number
        if round_idx == 0:
            return Action.C
        if round_idx == self.game_description.n_rounds - 1:
            return Action.C
        if current_stock >= self.tau:
            return Action.C
        capacity = float(self.game_description.capacity)
        if capacity == 0.0:
            return Action.C
        stock_remain = current_stock - current_stock / 2.0
        growth = 2.0 * stock_remain * (1.0 - stock_remain / capacity)
        projected_stock = min(stock_remain + growth, capacity)
        if projected_stock > 0.0 or projected_stock == 0.0:
            if history.round_number > 0:
                coop_opps = int(history.opponent_cooperators[-1])
            else:
                coop_opps = 0
            n_players = self.game_description.n_players
            n_opponents = n_players - 1
            defection_rate = (n_opponents - coop_opps) / max(1, n_opponents)
            coop_prob = max(0.9, 1.0 - defection_rate)
            if random.random() <= coop_prob:
                return Action.C
            else:
                return Action.C
        else:
            return Action.C

class Strategy_ALTRUISTIC_2(BaseStrategy):
    """
    Altruistic Stock‑Preserving Tit‑for‑Cooperate (ASPTC).

    - Starts with cooperation.
    - In the last round cooperates unless the stock is below a critical collapse floor ε.
    - Otherwise:
        * If stock < θ (safety threshold) cooperate only when the proportion of
          cooperating opponents in the previous round meets or exceeds φ.
        * If stock ≥ θ cooperate when that proportion meets or exceeds φ;
          otherwise defect with probability equal to the shortfall (1 – proportion).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_opponents = self.n_players - 1
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta = 0.6 * self.capacity
        self.epsilon = 0.05 * self.capacity
        self.phi = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.n_rounds - 1:
            if current_stock < self.epsilon:
                return Action.D
            else:
                return Action.C
        c_prev = int(history.opponent_cooperators[-1])
        if self.n_opponents > 0:
            coop_rate = c_prev / self.n_opponents
        else:
            coop_rate = 0.0
        if current_stock < self.theta:
            if coop_rate >= self.phi:
                return Action.C
            else:
                return Action.D
        elif coop_rate >= self.phi:
            return Action.C
        else:
            p_defect = 1.0 - coop_rate
            if random.random() < p_defect:
                return Action.D
            else:
                return Action.C

class Strategy_ALTRUISTIC_3(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The agent tracks a trust level based on opponents' past cooperation,
    adapts a stock threshold, and may enter a short defect‑phase when trust
    is low.  The decision rule follows the specification provided.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.trust: float = 0.0
        self.defect_phase_remaining: int = 0
        self.alpha: float = 0.2
        self.tau_trust: float = 0.4
        self.min_stock_fraction: float = 0.25

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide action for the upcoming round."""
        t = history.round_number
        S = float(current_stock)
        if t > 0:
            recent_coop = float(history.opponent_cooperators[t - 1])
            current_trust = recent_coop / self.n
            self.trust = (1.0 - self.alpha) * self.trust + self.alpha * current_trust
        if t == 0:
            if S < 0.5 * self.capacity:
                return Action.D
            else:
                return Action.C
        if self.defect_phase_remaining > 0:
            tau_stock = self.capacity * (0.5 + 0.3 * self.trust)
            if self.trust >= self.tau_trust and S >= tau_stock:
                self.defect_phase_remaining = 0
            else:
                self.defect_phase_remaining -= 1
                return Action.D
        if S <= 0.0:
            return Action.D
        if S < self.min_stock_fraction * self.capacity:
            return Action.D
        if t == self.r - 1:
            return Action.C if S > 0.0 else Action.D
        tau_stock = self.capacity * (0.5 + 0.3 * self.trust)
        expected_consumption = self.n * (S / (2.0 * self.n))
        S_rem = S - expected_consumption
        if self.capacity == 0:
            growth = 0.0
        else:
            growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
        S_next = min(S_rem + growth, self.capacity)
        coop_cond = S >= tau_stock and self.trust >= self.tau_trust and (S_next >= self.min_stock_fraction * self.capacity)
        if coop_cond:
            return Action.C
        else:
            if self.trust < self.tau_trust:
                R_after = self.r - (t + 1)
                self.defect_phase_remaining = min(3, max(R_after, 0))
            return Action.D

class Strategy_ALTRUISTIC_4(BaseStrategy):
    """
    Altruistic Common‑Pool Strategy (ACS).

    The player cooperates in the first round and generally cooperates
    throughout the game, only defecting in the last round when the stock
    is below a low threshold or in critical stock conditions when recent
    opponent cooperation is low.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta_high = 0.75 * game_description.capacity
        self.theta_low = 0.3 * game_description.capacity
        self.alpha = 0.6

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        t = history.round_number
        if self.game_description.capacity == 0:
            return Action.D
        if current_stock == 0:
            return Action.D
        if t == 0:
            return Action.C
        if t == r - 1:
            return Action.C if current_stock >= self.theta_low else Action.D
        k = min(5, t)
        if k == 0:
            prop = 0.0
        else:
            recent = history.opponent_cooperators[-k:]
            total_coop = int(recent.sum())
            prop = total_coop / (k * (n - 1))
        if current_stock >= self.theta_high:
            return Action.C
        elif current_stock >= self.theta_low:
            return Action.C
        else:
            return Action.C if prop >= 0.5 else Action.D

class Strategy_ALTRUISTIC_5(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC) strategy for the Common Pool Resource game.
    Starts with a short cooperation seed, cooperates unconditionally when the
    resource is abundant, and adapts its willingness to cooperate based on
    observed opponent cooperation and the current stock level. In the final
    rounds it cooperates if enough stock remains, otherwise defects to avoid
    catastrophic collapse.
    """
    CoopThreshold: float = 0.6
    GraceRounds: int = 2
    EndGrace: int = 2

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.StockLow = 0.3 * game_description.capacity
        self.StockHigh = 0.7 * game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        rnd = history.round_number
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        if rnd < self.GraceRounds:
            return Action.C
        if rnd >= r - self.EndGrace:
            if current_stock >= self.StockLow:
                return Action.C
            else:
                return Action.D
        c_prev = int(history.opponent_cooperators[-1]) if rnd > 0 else 0
        coop_frac = c_prev / (n - 1) if n > 1 else 0.0
        if current_stock >= self.StockHigh:
            return Action.C
        if current_stock <= self.StockLow:
            if coop_frac >= self.CoopThreshold:
                return Action.C
            else:
                return Action.D
        denom = self.StockHigh - self.StockLow
        if denom == 0:
            dynamic_thresh = 0.0
        else:
            dynamic_thresh = self.CoopThreshold * ((self.StockHigh - current_stock) / denom)
        if coop_frac >= dynamic_thresh:
            return Action.C
        else:
            return Action.D

class Strategy_ALTRUISTIC_6(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The player cooperates whenever the projected stock after a cooperative
    round stays above a safety threshold that adapts to the observed
    cooperation level of opponents.  When the test fails, the player defects
    once as a corrective signal and then enters a short punishment phase
    (up to two rounds of unconditional defection).  The final round is always
    cooperative, and a depleted stock is treated as cooperative as well.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.punish_counter: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        R = self.n_rounds - t
        if current_stock <= 0.0:
            self.punish_counter = 0
            return Action.C
        if t == self.n_rounds - 1:
            self.punish_counter = 0
            return Action.C
        if self.punish_counter > 0:
            self.punish_counter -= 1
            return Action.D
        if t == 0:
            p_t = 1.0
        else:
            c_t = int(history.opponent_cooperators[-1])
            denom = max(1, self.n_players - 1)
            p_t = c_t / denom
        safety_factor = max(0.25, 0.5 * p_t)
        safety = self.capacity * safety_factor
        S = current_stock
        coop_consumption = S * ((p_t / 2.0 + 1.0 - p_t) / self.n_players)
        S_after_coop = S - coop_consumption
        growth = 2.0 * S_after_coop * (1.0 - S_after_coop / self.capacity)
        S_next_coop = min(S_after_coop + growth, self.capacity)
        if S_next_coop >= safety:
            return Action.C
        else:
            if R > 2:
                self.punish_counter = min(2, R - 1)
            else:
                self.punish_counter = 0
            return Action.D

class Strategy_ALTRUISTIC_7(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The strategy cooperates in the first and last round.
    In between, it monitors the recent cooperation rate of opponents
    (over the last k rounds) and the ecological stock level.
    If both are above the sustainability thresholds (α and β),
    it cooperates; otherwise it initiates a short bounded punishment
    burst of defections whose length is limited by γ.
    A punishment counter is kept across rounds to enforce the burst.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.6
        self.beta = 0.3
        self.k = 2
        self.gamma = 0.1
        self.punish_counter = 0
        total_rounds = self.game_description.n_rounds
        self.max_punish_len = max(1, int(math.ceil(self.gamma * total_rounds)))

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        total_rounds = self.game_description.n_rounds
        capacity = self.game_description.capacity
        if history.round_number == 0:
            return Action.C
        if history.round_number == total_rounds - 1:
            return Action.C
        recent_vals = history.opponent_cooperators[-self.k:] if history.round_number >= self.k else np.concatenate((np.zeros(self.k - history.round_number, dtype=int), history.opponent_cooperators))
        recent_coop = recent_vals.sum() / (self.k * n_opponents) if n_opponents > 0 else 0.0
        safety_stock = self.beta * capacity
        if self.punish_counter > 0:
            self.punish_counter -= 1
            return Action.D
        if recent_coop >= self.alpha and current_stock >= safety_stock:
            return Action.C
        else:
            remaining_rounds = total_rounds - history.round_number - 1
            self.punish_counter = min(self.max_punish_len, remaining_rounds)
            if self.punish_counter > 0:
                self.punish_counter -= 1
            return Action.D

class Strategy_ALTRUISTIC_8(BaseStrategy):
    """
    Altruistic Sustainability Tit‑for‑Tat (AST‑TFT).

    - Starts by cooperating.
    - Each round computes a sustainability target stock based on the
      remaining rounds; if current stock falls below this target the player
      defects to protect the resource.
    - Otherwise reacts to the proportion of cooperating opponents in the
      previous round (generous TFT).  Cooperation is continued when the
      observed cooperation proportion meets a tolerance of 0.6; otherwise the
      player defects, with graduated retaliation (both lead to defection in
      this implementation).
    - Handles edge cases for the final round (target\u202f=\u202f0) and zero stock.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.tolerance = 0.6
        self.mild_threshold = 0.3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            return Action.C
        t = history.round_number
        total_rounds = self.game_description.n_rounds
        capacity = self.game_description.capacity
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        remaining = total_rounds - t
        if t == total_rounds - 1:
            target = 0.0
        else:
            target = capacity * (remaining / total_rounds)
        if current_stock < target:
            return Action.D
        last_coop_count = history.opponent_cooperators[-1] if n_opponents > 0 else 0
        observed_coop = last_coop_count / n_opponents if n_opponents > 0 else 0.0
        if observed_coop >= self.tolerance:
            return Action.C
        elif observed_coop >= self.mild_threshold:
            return Action.D
        else:
            return Action.D

class Strategy_ALTRUISTIC_9(BaseStrategy):
    """
    Altruistic Adaptive Strategy (A‑Coop).

    The player starts and ends by cooperating.
    While the stock is above a sustainability factor α, cooperation is
    maintained if enough opponents cooperated in the previous round
    (fraction ≥ β) or if a forgiveness window is active.
    When the stock falls below α the player defects to preserve the
    remaining resource, staying in a defecting streak until the stock
    recovers.  A forgiveness counter γ gives a grace period after a
    successful cooperative round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha: float = 0.5
        self.beta: float = 0.6
        self.gamma: int = 3
        self.defect_streak: int = 0
        self.forgive_counter: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        total_rounds = self.game_description.n_rounds
        if current_stock == 0:
            return Action.C
        if history.round_number == 0:
            self.defect_streak = 0
            self.forgive_counter = 0
            return Action.C
        if history.round_number == total_rounds - 1:
            return Action.C
        if history.opponent_cooperators.size > 0:
            c_last = int(history.opponent_cooperators[-1])
        else:
            c_last = 0
        p = c_last / (n - 1) if n > 1 else 0.0
        stock_factor = current_stock / capacity if capacity > 0 else 0.0
        if stock_factor >= self.alpha:
            if p >= self.beta or self.forgive_counter > 0:
                action = Action.C
            else:
                action = Action.D
        elif self.defect_streak > 0:
            action = Action.D
        else:
            action = Action.D
            self.defect_streak = 1
        if action == Action.C:
            self.defect_streak = 0
            if p >= self.beta:
                self.forgive_counter = self.gamma
        if self.forgive_counter > 0:
            self.forgive_counter -= 1
        return action

class Strategy_ALTRUISTIC_10(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC) strategy.
    Cooperates when the resource is healthy and enough opponents cooperated
    in the previous round, otherwise defects, with special handling for the
    first and last round and for critically low stock levels.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.alpha = 0.5
        self.beta = 0.6
        self.gamma = 0.2

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if S == 0.0:
            return Action.D
        if t == 0:
            k = self.n - 1
        else:
            k = int(history.opponent_cooperators[-1])
        n_opponents = self.n - 1
        coop_rate = k / n_opponents if n_opponents > 0 else 0.0
        safe_stock = self.alpha * self.capacity
        stock_ratio = S / self.capacity if self.capacity != 0 else 0.0
        if t == self.r - 1:
            if stock_ratio >= self.gamma:
                return Action.C
            else:
                return Action.D
        if S >= safe_stock:
            if coop_rate >= self.beta:
                return Action.C
            else:
                return Action.D
        elif stock_ratio < self.gamma:
            return Action.D
        elif coop_rate >= self.beta:
            return Action.C
        else:
            return Action.D

class Strategy_ALTRUISTIC_11(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The player estimates opponent cooperation probability from the previous round,
    projects the resulting stock for both possible actions, and chooses the action
    that best preserves the common pool.  It cooperates aggressively when the pool
    is abundant, protects it when scarce, and respects a protective mode that
    activates after a sudden drop in cooperation combined with low stock.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.capacity = float(game_description.capacity)
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.alpha = 0.4 * self.capacity
        self.beta = 0.7 * self.capacity
        self._protective = False

    def _proj_stock(self, stock: float, action: Action, p_hat: float) -> float:
        """Expected stock after the current round if we take *action*."""
        n = self.n_players
        cap = self.capacity
        k_opp = p_hat * (n - 1)
        k = k_opp + (1 if action == Action.C else 0)
        s_rem = stock * (k / (2.0 * n))
        growth = 2.0 * s_rem * (1.0 - s_rem / cap) if cap != 0 else 0.0
        new_stock = s_rem + growth
        return new_stock if new_stock <= cap else cap

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        stock = float(current_stock)
        if t == 0:
            p_hat = 0.5
        else:
            C_prev = int(history.opponent_cooperators[-1])
            p_hat = C_prev / (self.n_players - 1) if self.n_players > 1 else 0.5
            if C_prev < (self.n_players - 1) / 2 and stock <= self.alpha:
                self._protective = True
            if self._protective and stock > self.beta:
                self._protective = False
        if self._protective:
            return Action.D
        if t == self.n_rounds - 1:
            return Action.C if stock >= self.alpha else Action.D
        if stock <= 0.0:
            return Action.D
        if stock >= self.capacity:
            return Action.C
        s_coop = self._proj_stock(stock, Action.C, p_hat)
        s_def = self._proj_stock(stock, Action.D, p_hat)
        if stock >= self.beta:
            return Action.C
        if stock <= self.alpha:
            return Action.D
        return Action.C if s_coop >= s_def else Action.D

class Strategy_ALTRUISTIC_12(BaseStrategy):
    """
    Altruistic Conditional‑Cooperate (ACC).

    Starts by cooperating.  In intermediate rounds it monitors:
        - the proportion of opponents that cooperated in the previous round,
        - the current stock level relative to capacity.

    If either the cooperation ratio drops below θ_low (0.30) or the stock
    ratio drops below θ_stock (0.40), a “defect streak” begins.
    The first round of a streak triggers a single defection, after which
    the agent enters a forgiveness window of τ = 2 rounds of cooperation.
    The streak counter resets when the observed cooperation ratio rises
    back above the threshold.  In the final round the agent cooperates
    as long as any stock remains, otherwise it defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta_low = 0.3
        self.theta_stock = 0.4
        self.tau = 2
        self.defect_streak = 0
        self.forgive_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        total_opponents = n - 1
        if t == 0:
            self.defect_streak = 0
            self.forgive_counter = 0
            return Action.C
        if t == self.game_description.n_rounds - 1:
            return Action.C if current_stock > 0 else Action.D
        coop_obs = int(history.opponent_cooperators[-1])
        coop_ratio = coop_obs / total_opponents if total_opponents else 0.0
        stock_ratio = current_stock / capacity if capacity else 0.0
        if coop_ratio < self.theta_low or stock_ratio < self.theta_stock:
            self.defect_streak += 1
        else:
            self.defect_streak = 0
        if self.defect_streak >= 1 and self.forgive_counter == 0:
            self.forgive_counter = self.tau
            return Action.D
        if self.forgive_counter > 0:
            self.forgive_counter -= 1
            return Action.C
        return Action.C

class Strategy_ALTRUISTIC_13(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The agent adjusts its cooperation probability each round based on:
      * the health of the common pool (current stock / capacity),
      * the observed cooperation level of opponents in the previous round,
      * a safety boost when the resource is critically low,
      * and an unconditional cooperation in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._safety_threshold = 0.15
        self._min_p_when_critical = 0.9

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        health = current_stock / capacity if capacity != 0 else 0.0
        if t == 0:
            coop_signal = 1.0
        else:
            coop_signal = history.opponent_cooperators[-1] / (n - 1)
        base_p = health * coop_signal
        if health < self._safety_threshold:
            base_p = max(base_p, self._min_p_when_critical)
        if t == r - 1:
            p_t = 1.0
        else:
            p_t = base_p
        if p_t < 0.0:
            p_t = 0.0
        elif p_t > 1.0:
            p_t = 1.0
        if random.random() <= p_t:
            return Action.C
        else:
            return Action.D

class Strategy_ALTRUISTIC_14(BaseStrategy):
    """
    Altruistic Adaptive Strategy (AA‑S).

    Cooperates in the first and last round.
    In intermediate rounds it cooperates when the common pool is comfortably
    above the safe level (≥\u202f0.75\u202f*\u202fcapacity) **and** at least half of the
    opponents cooperated in the previous round (≥\u202f⌊n/2⌋).  Otherwise it switches
    to a protective mode: if the stock is below the logistic midpoint
    (capacity/2) it cooperates to avoid further depletion; if the stock is at
    or above the midpoint it defects to extract a short‑term gain that allows
    logistic growth to restore the pool.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.S_mid = self.capacity / 2.0
        self.S_safe = 0.75 * self.capacity
        self.C_mid = self.n_players // 2

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0 or t == self.n_rounds - 1:
            return Action.C
        coop_prev = history.opponent_cooperators[-1] if history.opponent_cooperators.size > 0 else 0
        stock_ok = current_stock >= self.S_safe
        coop_ok = coop_prev >= self.C_mid
        if stock_ok and coop_ok:
            return Action.C
        elif current_stock < self.S_mid:
            return Action.C
        else:
            return Action.D

class Strategy_ALTRUISTIC_15(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    Cooperates when the common pool is sufficiently healthy (stock fraction ≥ Δ)
    and the observed cooperation proportion in the previous round is at least Φ.
    In the final γ rounds the safety threshold is relaxed to 5\u202f% of capacity.
    If the stock is depleted the strategy defaults to cooperation (no consumption).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.T = game_description.n_rounds
        self.Cap = float(game_description.capacity)
        self.delta = 0.2
        self.phi = 0.5
        self.gamma = max(1, math.floor(0.1 * self.T))
        self.endgame_delta = 0.05

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        last_my_action = history.my_actions[-1]
        last_opponents_coop = int(history.opponent_cooperators[-1])
        c_prev = last_opponents_coop + (1 if last_my_action else 0)
        stock_fraction = current_stock / self.Cap if self.Cap != 0 else 0.0
        stock_safe = stock_fraction >= self.delta
        coop_rate = c_prev / self.n_players
        sufficient_coop = coop_rate >= self.phi
        if t >= self.T - self.gamma:
            stock_safe = stock_fraction >= self.endgame_delta
        if stock_safe and sufficient_coop:
            return Action.C
        else:
            if current_stock == 0:
                return Action.C
            return Action.D

class Strategy_ALTRUISTIC_16(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    Cooperates when the stock is healthy (>= β of capacity) and a sufficient
    fraction (>= α) of opponents cooperated in the previous round.
    Defects otherwise, with a “gentle” defect mode when the stock is
    critically low (stock fraction < β/2). Always cooperates in the final
    round and when the pool is exhausted.
    """
    _ALPHA = 0.6
    _BETA = 0.4

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        total_rounds = self.game_description.n_rounds
        if current_stock == 0:
            return Action.C
        t = history.round_number
        if t == total_rounds - 1:
            return Action.C
        if t == 0:
            coop_frac = 1.0
        else:
            coop_cnt = int(history.opponent_cooperators[-1])
            coop_frac = coop_cnt / (n - 1) if n > 1 else 1.0
        stock_frac = current_stock / capacity if capacity != 0 else 0.0
        if stock_frac >= self._BETA and coop_frac >= self._ALPHA:
            return Action.C
        elif stock_frac < self._BETA / 2:
            return Action.D
        else:
            return Action.D

class Strategy_BENEVOLENT_1(BaseStrategy):
    """
    Benevolent Adaptive Strategy (BAS).

    Cooperates when the stock is above a horizon‑adjusted safety threshold
    and opponents have been sufficiently cooperative, otherwise defects.
    A final‑round override forces cooperation, and an additional safety
    check avoids collapsing the pool.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._s_min = 0.1 * game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        theta = capacity / 2.0 * (r - t) / r
        if t == 0:
            rho = 1.0
        else:
            c_prev = history.opponent_cooperators[-1]
            if c_prev is None or not isinstance(c_prev, (int, np.integer)):
                rho = 0.0
            else:
                rho = c_prev / (n - 1)
        if current_stock >= theta and rho >= 0.5:
            action = Action.C
        else:
            action = Action.D
        if t == r - 1:
            action = Action.C
        if action == Action.C:
            per_player_cons = current_stock / (2.0 * n)
        else:
            per_player_cons = current_stock / n
        total_consumption = n * per_player_cons
        s_remaining = max(current_stock - total_consumption, 0.0)
        growth = 2.0 * s_remaining * (1.0 - s_remaining / capacity)
        s_next = min(s_remaining + growth, capacity)
        if action == Action.C and s_next < self._s_min:
            action = Action.D
        return action

class Strategy_BENEVOLENT_2(BaseStrategy):
    """
    Benevolent Adaptive Strategy for the Common Pool Resource Game.

    The agent aims to preserve the common stock, cooperating whenever possible
    and defecting only in the final round, during an active punishment block,
    or when projected stock is higher after defection under intermediate
    stock conditions.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta_high = 0.6 * self.capacity
        self.theta_low = 0.3 * self.capacity
        self.p_tol = 0.4
        self.punish_remaining = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            Contains past actions, payoffs and opponent cooperation counts.
        current_stock : float
            The common pool stock at the start of the current round.

        Returns
        -------
        Action
            Action.C (cooperate) or Action.D (defect).
        """
        t = history.round_number
        if t > 0 and self.punish_remaining == 0:
            k_prev = int(history.opponent_cooperators[-1])
            defectors_last = self.n - 1 - k_prev
            if defectors_last / (self.n - 1) > self.p_tol:
                self.punish_remaining = min(3, self.r - t - 1)
        if t == 0:
            k = 0
        else:
            k = int(history.opponent_cooperators[-1])
        if t == self.r - 1:
            return Action.D
        if self.punish_remaining > 0:
            self.punish_remaining -= 1
            return Action.D
        S = current_stock
        if S >= self.theta_high:
            return Action.C
        if S <= self.theta_low:
            if k / (self.n - 1) <= self.p_tol:
                return Action.C
            else:
                return Action.D
        coop_consume = S / (2 * self.n)
        defect_consume = S / self.n
        S_coop = S - coop_consume - k * S / (2 * self.n) - (self.n - 1 - k) * S / self.n
        S_defect = S - defect_consume - k * S / (2 * self.n) - (self.n - 1 - k) * S / self.n
        S_coop = max(S_coop, 0.0)
        S_defect = max(S_defect, 0.0)
        G_coop = 2 * S_coop * (1 - S_coop / self.capacity)
        G_defect = 2 * S_defect * (1 - S_defect / self.capacity)
        proj_coop = min(S_coop + G_coop, self.capacity)
        proj_defect = min(S_defect + G_defect, self.capacity)
        if proj_coop >= proj_defect:
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_3(BaseStrategy):
    """
    Adaptive Benevolent Guard.

    The player watches the common‑pool stock and the recent
    cooperation level of opponents (memory length k = ⌈r/4⌉,
    or 1 for very short games).  Depending on stock thresholds
    (θ_low = 0.3·capacity, θ_high = 0.7·capacity) and the
    average opponent cooperation rate, the player either
    cooperates (C) or defects (D).  In the final round it cooperates
    if any viable stock (≥ θ_low) remains.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        if self.r <= 3:
            self.k = 1
        else:
            self.k = math.ceil(self.r / 4)
        self.theta_low = 0.3 * self.capacity
        self.theta_high = 0.7 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.C if current_stock >= self.theta_low else Action.D
        if t == 0:
            avg_coop = 1.0
        else:
            look = min(self.k, t)
            recent = history.opponent_cooperators[-look:]
            total_coop = int(recent.sum())
            denom = look * (self.n - 1)
            avg_coop = total_coop / denom if denom else 0.0
        if current_stock >= self.theta_high:
            return Action.C if avg_coop >= 0.5 else Action.D
        elif current_stock <= self.theta_low:
            return Action.C if avg_coop >= 0.75 else Action.D
        else:
            return Action.C if avg_coop >= 0.6 else Action.D

class Strategy_BENEVOLENT_4(BaseStrategy):
    """
    Benevolent Adaptive Cooperation (BAC) strategy for the Common Pool Resource Game.

    The player cooperates by default, defects only during a bounded punishment phase
    triggered when a sufficiently large fraction of opponents defect while the stock
    is not critical. The strategy also protects the pool in the final round by always
    cooperating.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.3
        self.beta = 0.6
        self.gamma = 2
        self.punish_counter = 0
        self.cooperate_last = True

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n_players = self.game_description.n_players
        capacity = self.game_description.capacity
        if t == self.game_description.n_rounds - 1:
            self.punish_counter = 0
            self.cooperate_last = True
            return Action.C
        if t == 0:
            self.punish_counter = 0
            self.cooperate_last = True
            return Action.C
        if self.punish_counter > 0:
            self.punish_counter -= 1
            self.cooperate_last = False
            return Action.D
        safe_stock = self.alpha * capacity
        stock_critical = current_stock < safe_stock
        opp_coop = int(history.opponent_cooperators[-1])
        opponents = n_players - 1
        k_t = opponents - opp_coop
        defect_rate = k_t / opponents if opponents > 0 else 0.0
        if defect_rate >= self.beta and current_stock >= safe_stock and self.cooperate_last:
            self.punish_counter = self.gamma
            self.cooperate_last = True
            return Action.C
        self.cooperate_last = True
        return Action.C

class Strategy_BENEVOLENT_5(BaseStrategy):
    """
    Sustainable‑Tit‑for‑Cooperate (STFC).

    The player cooperates while the common‑pool stock stays above a safety
    threshold and the observed fraction of defecting opponents does not exceed a
    tolerance.  The decision also checks that cooperating would keep the stock
    above the safety threshold in the next round (except in the final round).
    If the stock is exhausted the player defects by convention.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.4
        self.theta = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        total_rounds = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        safety_threshold = self.alpha * capacity
        if current_stock <= 0.0:
            return Action.D
        t = history.round_number
        if t == 0:
            c_prev = n - 1
        else:
            c_prev = int(history.opponent_cooperators[t - 1])
        d_frac = (n - 1 - c_prev) / (n - 1)

        def projected_stock(own_action: str) -> float:
            coop_cons = current_stock / (2 * n)
            defect_cons = current_stock / n
            own_cons = coop_cons if own_action == 'C' else defect_cons
            total_cons = c_prev * coop_cons + (n - 1 - c_prev) * defect_cons + own_cons
            s_rem = max(current_stock - total_cons, 0.0)
            growth = 2 * s_rem * (1 - s_rem / capacity) if capacity != 0 else 0.0
            projected = min(s_rem + growth, capacity)
            return projected
        s_next_coop = projected_stock('C')
        s_next_defect = projected_stock('D')
        cooperate = current_stock >= safety_threshold and d_frac <= self.theta and (s_next_coop >= safety_threshold)
        if t == total_rounds - 1:
            cooperate = current_stock >= safety_threshold and d_frac <= self.theta
        return Action.C if cooperate else Action.D

class Strategy_BENEVOLENT_6(BaseStrategy):
    """
    Benevolent Adaptive Strategy:
    - Cooperates on the first, last, zero‑stock, or low‑stock rounds.
    - Defects only during a bounded punishment phase (length L = 2) that is
      triggered when at least θ = 0.5 of the observed opponents defected in the
      previous round and the stock is above the safety floor S_min.
    - The punishment phase is reset if the stock falls below S_min.
    """
    _THETA = 0.5
    _L = 2

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._S_min = 0.2 * game_description.capacity
        self.punish_until = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        if t == 0:
            return Action.C
        if t == self.game_description.n_rounds - 1:
            return Action.C
        if current_stock == 0.0:
            return Action.C
        if current_stock < self._S_min:
            self.punish_until = 0
            return Action.C
        if self.punish_until > 0:
            self.punish_until -= 1
            return Action.D
        c = int(history.opponent_cooperators[-1])
        p_def = (n_opponents - c) / n_opponents if n_opponents > 0 else 0.0
        if p_def >= self._THETA:
            self.punish_until = self._L - 1
            return Action.D
        return Action.C

class Strategy_BENEVOLENT_7(BaseStrategy):
    """
    Conditional Cooperation with Tolerant Retaliation.

    - Starts by cooperating.
    - Cooperates in the final round and whenever the stock is exhausted or below 20\u202f% of capacity.
    - Otherwise looks at the last up‑to‑5 rounds: if the proportion of opponent defections
      exceeds 40\u202f% it defects for the current round, else it cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.low_thresh = 0.2 * game_description.capacity
        self.defect_tol = 0.4
        self.max_window = 5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if current_stock <= 0:
            return Action.C
        if t == self.game_description.n_rounds - 1:
            return Action.C
        if current_stock < self.low_thresh:
            return Action.C
        window = min(self.max_window, t)
        recent_coops = history.opponent_cooperators[-window:]
        n_opponents = self.game_description.n_players - 1
        recent_defects = n_opponents * window - int(recent_coops.sum())
        defect_rate = recent_defects / (window * n_opponents)
        if defect_rate > self.defect_tol:
            return Action.D
        else:
            return Action.C

class Strategy_BENEVOLENT_8(BaseStrategy):
    """
    Benevolent Adaptive Strategy.

    The player cooperates while the stock is above a decreasing
    target level and enough opponents have cooperated in the previous round.
    If cooperation among opponents drops, a short punishment phase of
    defections is triggered, followed by a single forgiveness round
    when all opponents have cooperated.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.min_coop_ratio = 0.5
        self.punish_len = 2
        self.forgive_len = 1
        self.punish_counter = 0
        self.forgive_counter = 0
        self.last_observed_coop = self.n - 1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0:
            self.last_observed_coop = int(history.opponent_cooperators[-1])
        if t == self.r - 1:
            action = Action.D
        elif current_stock == 0:
            action = Action.D
        elif self.punish_counter > 0:
            action = Action.D
            self.punish_counter -= 1
        elif self.forgive_counter > 0:
            action = Action.C
            self.forgive_counter -= 1
        else:
            target_frac = 0.8 - 0.6 * (t / (self.r - 1))
            cooperate_threshold = self.capacity * target_frac
            if current_stock >= cooperate_threshold:
                coop_ratio = self.last_observed_coop / (self.n - 1) if self.n > 1 else 1.0
                if coop_ratio >= self.min_coop_ratio:
                    action = Action.C
                else:
                    self.punish_counter = self.punish_len
                    action = Action.D
                    self.punish_counter -= 1
            else:
                action = Action.D
        if self.punish_counter == 0 and action == Action.C and (self.last_observed_coop == self.n - 1):
            self.forgive_counter = self.forgive_len
        return action

class Strategy_BENEVOLENT_9(BaseStrategy):
    """
    Benevolent Adaptive Strategy for the Common Pool Resource Game.

    - Cooperates in the first and last round.
    - Uses stock thresholds (high/low) to set a stock‑based inclination.
    - Adjusts behavior based on the recent defection rate of opponents
      (up to the previous three rounds).
    - When signals differ, gives priority to stock preservation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.highThresh = 0.75 * game_description.capacity
        self.lowThresh = 0.3 * game_description.capacity
        self.midThresh = (self.highThresh + self.lowThresh) / 2.0
        self.n = game_description.n_players
        self.r = game_description.n_rounds

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        if current_stock >= self.highThresh:
            stockIncl = Action.D
        elif current_stock <= self.lowThresh:
            stockIncl = Action.C
        else:
            stockIncl = Action.C if current_stock < self.midThresh else Action.D
        start = max(0, t - 2)
        recent_len = t - start
        if recent_len == 0:
            defRate = 0.0
        else:
            coop_sum = int(np.sum(history.opponent_cooperators[start:t]))
            total_possible = recent_len * (self.n - 1)
            defRate = (total_possible - coop_sum) / total_possible
        if defRate > 0.55:
            behaviorAdj = Action.C
        elif defRate < 0.3:
            behaviorAdj = Action.D
        else:
            behaviorAdj = stockIncl
        if stockIncl == behaviorAdj:
            chosen = stockIncl
        else:
            chosen = Action.C if stockIncl == Action.C else behaviorAdj
        return chosen

class Strategy_BENEVOLENT_10(BaseStrategy):
    """
    Sustaining Cooperation strategy for the Common Pool Resource game.

    The agent keeps an exponential moving average of the opponents' cooperation
    rate (coop_est).  Each round it computes a cooperation threshold θ based on
    the current stock, the sustainability baseline, the cooperation estimate,
    and a risk buffer.  The action rule is:

        - Play C in the final round.
        - Play D if the stock is effectively depleted (≤ ε).
        - Otherwise cooperate if θ > 0, else defect.

    Parameters α, β, γ, ε are fixed constants.
    """
    _ALPHA = 0.7
    _BETA = 0.3
    _GAMMA = 0.4
    _EPS = 0.001

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.coop_est: float = 1.0
        self._s_star: float = 0.5 * float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if t > 0:
            obs = int(history.opponent_cooperators[-1])
            coop_rate = obs / n_opponents
            self.coop_est = self._ALPHA * self.coop_est + (1.0 - self._ALPHA) * coop_rate
        else:
            self.coop_est = 1.0
        buffer = self._BETA * max(0.0, (self.game_description.capacity - current_stock) / self.game_description.capacity)
        raw_theta = (current_stock - self._s_star) / self.game_description.capacity + self._GAMMA * (self.coop_est - 0.5) - buffer
        theta = max(-1.0, min(1.0, raw_theta))
        if t == total_rounds - 1:
            return Action.C
        if current_stock <= self._EPS:
            return Action.D
        if theta > 0.0:
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_11(BaseStrategy):
    """
    Sustainable‑Threshold Conditional Cooperation (STCC).

    The player tries to keep the common pool above a target proportion of
    capacity. It cooperates when the projected stock after its move stays
    above the threshold, defects otherwise, and retaliates for a short
    period if the proportion of defecting opponents exceeds a tolerance.
    """
    _TARGET_RATIO = 0.5
    _DEFECT_TOLERANCE = 0.4
    _RETALIATION_LENGTH = 2

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._retaliation_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t == 0:
            return Action.C
        c_prev = int(history.opponent_cooperators[-1])
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            if current_stock >= self._TARGET_RATIO * capacity:
                return Action.D
            else:
                return Action.C
        if self._retaliation_counter > 0:
            self._retaliation_counter -= 1
            action = Action.D
        else:
            coop_consume = current_stock / (2.0 * n)
            defect_consume = current_stock / float(n)
            opp_coop = c_prev * (current_stock / (2.0 * n))
            opp_defect = (n - 1 - c_prev) * (current_stock / float(n))
            S_after_coop = current_stock - (coop_consume + opp_coop + opp_defect)
            S_after_coop = max(S_after_coop, 0.0)
            S_after_defect = current_stock - (defect_consume + opp_coop + opp_defect)
            S_after_defect = max(S_after_defect, 0.0)

            def grow(x: float) -> float:
                return 2.0 * x * (1.0 - x / capacity)
            S_next_coop = min(S_after_coop + grow(S_after_coop), capacity)
            S_next_defect = min(S_after_defect + grow(S_after_defect), capacity)
            if S_next_coop >= self._TARGET_RATIO * capacity:
                action = Action.C
            elif S_next_defect >= self._TARGET_RATIO * capacity:
                action = Action.D
            else:
                action = Action.C
            if n > 1:
                defect_proportion = (n - 1 - c_prev) / (n - 1)
            else:
                defect_proportion = 0.0
            if defect_proportion > self._DEFECT_TOLERANCE:
                self._retaliation_counter = self._RETALIATION_LENGTH
                action = Action.D
        return action

class Strategy_BENEVOLENT_12(BaseStrategy):
    """
    Adaptive Benevolent Cooperation (ABC).

    The player adapts its cooperation propensity based on the current
    resource stock, observed cooperation of opponents in the previous
    round, and an end‑game safeguard that preserves enough stock for the
    remaining rounds.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.prev_coop_ratio: float = 1.0
        self.high_stock_thr = self.capacity * 0.75
        self.low_stock_thr = self.capacity * 0.25
        self.endgame_thr = self.r * 0.2

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            self.prev_coop_ratio = 1.0
            return Action.C
        opponents = self.n - 1
        last_coop = history.opponent_cooperators[-1]
        observed_ratio = last_coop / opponents if opponents else 0.0
        if current_stock >= self.high_stock_thr:
            base_prop = 0.9
        elif current_stock <= self.low_stock_thr:
            base_prop = 0.2
        else:
            base_prop = 0.5
        adj_prop = base_prop * (0.5 + 0.5 * self.prev_coop_ratio)
        if t >= self.r - self.endgame_thr:
            remaining = self.r - t
            min_needed = remaining * (self.capacity / (2 * self.n))
            if current_stock < min_needed:
                adj_prop = 0.0
            else:
                adj_prop = max(adj_prop, 0.6)
        if current_stock <= 0.0:
            adj_prop = 0.0
        draw = random.random()
        action = Action.C if draw <= adj_prop else Action.D
        self.prev_coop_ratio = observed_ratio
        return action

class Strategy_BENEVOLENT_13(BaseStrategy):
    """
    Sustainable Reciprocity – a conditional cooperation strategy for the
    common‑pool resource game.

    The player cooperates (takes the minimal share) when the stock is healthy
    or merely endangered, and only defects temporarily if opponents have
    repeatedly over‑defected beyond a tolerated excess.  The retaliation is
    limited (defect_excess ≥ 2) and the strategy resets to full cooperation
    as soon as the stock recovers above the abundance threshold.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta_low = 0.25 * self.capacity
        self.theta_high = 0.75 * self.capacity
        self.delta = 0.15
        self.defect_excess = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Choose C or D for the current round based on Sustainable Reciprocity."""
        if current_stock > self.theta_high:
            self.defect_excess = 0
        t = history.round_number
        if t == self.r - 1:
            if current_stock >= self.theta_low:
                return Action.C
            else:
                return Action.D
        if t == 0:
            return Action.C
        last_my_action = bool(history.my_actions[-1])
        opp_coop_last = int(history.opponent_cooperators[-1])
        c_prev = opp_coop_last + (1 if last_my_action else 0)
        if current_stock < self.theta_low:
            return Action.C
        if current_stock > self.theta_high and self.defect_excess == 0:
            return Action.C
        defectors_last = self.n - c_prev
        tolerated_excess = self.delta * self.n
        excess = max(0.0, defectors_last - tolerated_excess)
        if excess > 0:
            self.defect_excess += 1
        else:
            self.defect_excess = max(0, self.defect_excess - 1)
        if self.defect_excess >= 2:
            return Action.D
        else:
            return Action.C

class Strategy_BENEVOLENT_14(BaseStrategy):
    """
    Sustainable‑Cooperate‑When‑Safe (SCWS).

    Cooperate when the current stock is above a safety margin that linearly
    decreases from 0.9·capacity at the start to 0.5·capacity at the final round.
    If the stock is below that margin, cooperate with a probability that grows
    with the observed cooperation rate of the opponents (threshold = 0.3 + 0.5·ρ).
    When the stock is depleted (S == 0) the strategy always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        if self.n_rounds > 1:
            margin_factor = 0.5 + 0.4 * (self.n_rounds - t - 1) / (self.n_rounds - 1)
        else:
            margin_factor = 0.5
        M = self.capacity * margin_factor
        if current_stock >= M:
            return Action.C
        if t == 0:
            c_obs = 0
        else:
            c_obs = int(history.opponent_cooperators[-1])
        max_opponents = max(self.n_players - 1, 1)
        rho = c_obs / max_opponents
        rho = min(max(rho, 0.0), 1.0)
        threshold = 0.3 + 0.5 * rho
        if random.random() < threshold:
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_15(BaseStrategy):
    """
    A sustainability‑focused strategy for the Common Pool Resource Game.
    It cooperates in the first round, uses the most recent observation of opponents'
    cooperation to forecast the next stock level under both actions, and selects the
    action that (i) keeps the projected stock above a safety threshold (θ_low) and
    (ii) maximises immediate payoff when both actions are safe. In the final round
    it cooperates if the stock is still healthy, otherwise defects to extract the
    remaining value.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.theta_low = 0.55 * self.capacity
        self.theta_high = 0.8 * self.capacity
        self.eps = 0.01 * self.capacity
        self.n_minus_one = self.n - 1
        self.two_n = 2.0 * self.n
        self.n_float = float(self.n)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_num = history.round_number
        total_rounds = self.game_description.n_rounds
        if round_num == 0:
            return Action.C
        if round_num == total_rounds - 1:
            if current_stock >= self.theta_low:
                return Action.C
            else:
                return Action.D
        c_prev = int(history.opponent_cooperators[-1])
        p_coop = c_prev / float(self.n_minus_one)
        coop_consumption = current_stock / self.two_n
        defect_consumption = current_stock / self.n_float
        exp_opp_cons = self.n_minus_one * (p_coop * coop_consumption + (1.0 - p_coop) * defect_consumption)
        total_consume_C = coop_consumption + exp_opp_cons
        s_rem_C = current_stock - total_consume_C
        if s_rem_C < 0.0:
            s_rem_C = 0.0
        growth_C = 2.0 * s_rem_C * (1.0 - s_rem_C / self.capacity) if self.capacity != 0 else 0.0
        s_next_C = min(s_rem_C + growth_C, self.capacity)
        total_consume_D = defect_consumption + exp_opp_cons
        s_rem_D = current_stock - total_consume_D
        if s_rem_D < 0.0:
            s_rem_D = 0.0
        growth_D = 2.0 * s_rem_D * (1.0 - s_rem_D / self.capacity) if self.capacity != 0 else 0.0
        s_next_D = min(s_rem_D + growth_D, self.capacity)
        coop_safe = s_next_C >= self.theta_low
        defect_safe = s_next_D >= self.theta_low
        if coop_safe and (not defect_safe):
            return Action.C
        if defect_safe and coop_safe:
            if defect_consumption > coop_consumption:
                return Action.D
            else:
                return Action.C
        if coop_safe:
            return Action.C
        if current_stock >= self.theta_high:
            return Action.D
        else:
            return Action.C

class Strategy_BENEVOLENT_16(BaseStrategy):
    """
    Benevolent Adaptive Strategy (BAS).

    Adapts cooperation based on current stock level and the observed
    number of cooperating opponents in the previous round.  The
    thresholds are computed once from the known capacity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.cap = float(game_description.capacity)
        self.high = 0.75 * self.cap
        self.medium = 0.4 * self.cap
        self.low = 0.2 * self.cap

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            c_prev = self.n - 1
        else:
            c_prev = int(history.opponent_cooperators[-1])
        if t == self.r - 1:
            return Action.D
        if current_stock == 0.0:
            return Action.C
        if self.n > 1:
            coop_frac = c_prev / (self.n - 1)
        else:
            coop_frac = 0.0
        if current_stock >= self.high:
            if coop_frac >= 0.5:
                return Action.C
            else:
                return Action.D
        elif current_stock >= self.medium:
            if random.random() < coop_frac:
                return Action.C
            else:
                return Action.D
        elif coop_frac >= 0.75:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_1(BaseStrategy):
    """
    Adaptive common‑pool strategy based on sustainable harvest thresholds
    and an exponential moving average of personal consumption.

    - Starts with cooperation to signal goodwill.
    - Tracks a weighted average of own payoffs (avg_cons) with β = 0.7.
    - Computes the maximal sustainable per‑player harvest H(S) = G(S)/n where
      G(S) = 2·S·(1‑S/capacity) is the theoretical growth if no one harvests.
    - When recent average consumption is below H(S) the agent may defect
      if at least half of the opponents cooperated in the previous round;
      otherwise it cooperates.
    - If consumption exceeds the sustainable limit the agent always
      cooperates to let the pool recover.
    - In the final round it defects only when the stock is comfortably
      above 40\u202f% of capacity; otherwise it cooperates.
    - If the stock is empty it cooperates (payoff is zero either way).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.avg_cons: float = 0.0
        self._beta: float = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t > 0:
            last_payoff = float(history.my_payoffs[-1])
            self.avg_cons = self._beta * self.avg_cons + (1.0 - self._beta) * last_payoff
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            return Action.C
        if t == r - 1:
            if current_stock > 0.4 * capacity:
                return Action.D
            else:
                return Action.C
        growth = 2.0 * current_stock * (1.0 - current_stock / capacity) if capacity != 0 else 0.0
        sustainable_harvest = growth / n if n != 0 else 0.0
        c_prev = int(history.opponent_cooperators[-1])
        coop_ratio = (c_prev + 1) / n
        if self.avg_cons <= sustainable_harvest:
            if coop_ratio >= 0.5:
                return Action.D
            else:
                return Action.C
        else:
            return Action.C

class Strategy_COLLECTIVE_2(BaseStrategy):
    """
    Adaptive Collective Sustain‑and‑Earn (ACSE) strategy.
    Cooperates in the first round, defects in the final round, and otherwise
    decides based on a weighted propensity that combines current stock health
    and the observed cooperation level of opponents in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        if current_stock <= 0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            return Action.C
        k_last = int(history.opponent_cooperators[-1])
        coop_ratio = k_last / (n - 1) if n > 1 else 0.0
        stock_factor = current_stock / capacity if capacity != 0 else 0.0
        propensity = 0.6 * (1.0 - stock_factor) + 0.4 * coop_ratio
        if propensity >= 0.5:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_3(BaseStrategy):
    """
    Adaptive Collective Stewardship:
    Cooperates when the common stock is healthy, defects when the stock is low
    or when opponents have recently defected, and retaliates for a short,
    bounded burst (k_ret rounds). The last round is always a defection unless
    the stock is depleted.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta_high = 0.6
        self.theta_low = 0.3
        self.k_ret = 2
        self.retaliation_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if self.capacity == 0:
            stock_frac = 0.0
        else:
            stock_frac = current_stock / self.capacity
        if current_stock == 0:
            return Action.C
        if t == self.n_rounds - 1:
            return Action.D
        if self.retaliation_counter > 0:
            self.retaliation_counter -= 1
            return Action.D
        p_prev = None
        if t > 0:
            opp_coops = int(history.opponent_cooperators[-1])
            p_prev = opp_coops / (self.n_players - 1) if self.n_players > 1 else 0.0
            if p_prev < self.theta_high * stock_frac:
                self.retaliation_counter = max(self.k_ret - 1, 0)
                return Action.D
        if stock_frac >= self.theta_high:
            return Action.C
        elif stock_frac <= self.theta_low:
            return Action.D
        else:
            prob_c = (stock_frac - self.theta_low) / (self.theta_high - self.theta_low)
            return Action.C if random.random() < prob_c else Action.D

class Strategy_COLLECTIVE_4(BaseStrategy):
    """
    Adaptive Sustenance Threshold (AST) strategy.
    Adjusts cooperation based on remaining stock, a sustainability
    threshold per remaining round, and a dynamically updated confidence
    threshold θ that tracks opponents' past cooperation levels.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta = 0.6

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the upcoming round."""
        if history.round_number > 0:
            last_coop = history.opponent_cooperators[-1] / (self.n - 1) if self.n > 1 else 0.0
            self.theta = 0.9 * self.theta + 0.1 * last_coop
        else:
            last_coop = 0.0
        if current_stock <= 0.0:
            return Action.D
        t = history.round_number
        R_t = self.r - t
        T_t = self.capacity / R_t
        C_max = current_stock / (2 * self.n)
        D_max = current_stock / self.n
        U_t = max(0.0, current_stock - T_t)
        if U_t >= D_max:
            chosen_action = Action.D
        elif U_t >= C_max:
            chosen_action = Action.C
        else:
            chosen_action = Action.C if last_coop > self.theta else Action.D
        if t == self.r - 2:
            opp_coop_est = int(history.opponent_cooperators[-1]) if history.round_number > 0 else 0
            projected_stock = self._simulate_stock(current_stock, chosen_action, opp_coop_est)
            final_threshold = self.capacity / (R_t - 1)
            if projected_stock < final_threshold:
                chosen_action = Action.C
        if t == self.r - 1:
            chosen_action = Action.D if current_stock >= D_max else Action.C
        return chosen_action

    def _simulate_stock(self, stock: float, my_action: Action, opp_coop: int) -> float:
        """
        Project the stock after the current round given:
        - stock: current stock level S_t
        - my_action: Action.C or Action.D for this player
        - opp_coop: estimated number of cooperating opponents this round
        """
        coop_cnt = opp_coop + (my_action == Action.C)
        def_cnt = self.n - 1 - opp_coop + (my_action == Action.D)
        total_consumption = stock * (coop_cnt / (2 * self.n) + def_cnt / self.n)
        remaining = max(0.0, stock - total_consumption)
        growth = 2 * remaining * (1 - remaining / self.capacity)
        new_stock = min(remaining + growth, self.capacity)
        return new_stock

class Strategy_COLLECTIVE_5(BaseStrategy):
    """
    Adaptive Collective Preservation (ACP) strategy.
    Adjusts cooperation probability based on observed opponent defection
    and the current stock level, with special handling for edge cases.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._toggle = True

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock <= 0:
            return Action.D
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        capacity = self.game_description.capacity
        round_idx = history.round_number
        if round_idx == 0 or n_opponents == 0:
            phi = 0.0
        else:
            total_defectors = 0
            for k in range(round_idx):
                opp_coop = int(history.opponent_cooperators[k])
                total_defectors += n_opponents - opp_coop
            phi = total_defectors / (round_idx * n_opponents)
        sigma = current_stock / capacity if capacity > 0 else 0.0
        b = 0.8
        alpha = 0.6
        beta = 0.5
        if round_idx == 0:
            p = b
        else:
            p = b * (1.0 - alpha * phi) * (1.0 - beta * (1.0 - sigma))
            if p < 0.0:
                p = 0.0
            elif p > 1.0:
                p = 1.0
        if sigma <= 0.1:
            p = 0.0
        elif sigma >= 0.9 and phi <= 0.2:
            p = 1.0
        elif round_idx == self.game_description.n_rounds - 1:
            if sigma >= 0.3:
                p = 0.5
            else:
                p = 0.0
        if p == 0.5:
            action = Action.C if self._toggle else Action.D
            self._toggle = not self._toggle
            return action
        else:
            return Action.C if random.random() < p else Action.D

class Strategy_COLLECTIVE_6(BaseStrategy):
    """
    Adaptive Collective Threshold (ACT) strategy.

    The agent cooperates as long as the current stock stays above a
    dynamically adjusted target that reflects the observed cooperation
    level of the opponents. The target is lowered when opponents defect
    frequently (aggressiveness α controls the sensitivity). In the
    final round the agent always defects to maximise the immediate
    payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)
        self.alpha: float = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t: int = history.round_number
        if t == self.n_rounds - 1:
            return Action.D
        if t == 0:
            p_hat: float = 1.0
        else:
            total_coop_opponents: int = int(np.sum(history.opponent_cooperators[:t]))
            opp_per_round: int = self.n_players - 1
            if opp_per_round * t == 0:
                p_hat = 1.0
            else:
                p_hat = total_coop_opponents / (opp_per_round * t)
        target_stock: float = self.capacity * (1.0 - self.alpha * (1.0 - p_hat))
        if current_stock >= target_stock:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_7(BaseStrategy):
    """
    Adaptive Collective Sustainability (ACS).

    The strategy adjusts its cooperation level based on the remaining
    rounds, current stock, a safety threshold τ and a dynamic cooperation
    target θ. It observes the opponents' cooperation in the previous round
    and decides to cooperate only when the observed cooperation meets the
    target and the required number of cooperators to keep the stock above τ
    is attainable. In the final round it always defects, and when the stock
    falls below τ it cooperates to allow regrowth. A zero stock is handled
    by cooperating (which yields zero payoff without further depletion).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        r = self.game_description.n_rounds
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        R = r - t
        tau = capacity * (R / r) * 0.5
        theta = R / r * 0.7
        if t > 0:
            C_obs = int(history.opponent_cooperators[-1])
            p_obs = C_obs / (n - 1) if n - 1 > 0 else 0.0
        else:
            C_obs = 0
            p_obs = 1.0
        if current_stock < tau:
            return Action.C
        coop_consumption = current_stock / (2 * n) if n > 0 else 0.0
        if coop_consumption == 0.0:
            required_coop = 0
        else:
            required_coop = math.ceil((current_stock - tau) / coop_consumption)
        if p_obs >= theta and required_coop <= C_obs + 1:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_8(BaseStrategy):
    """
    Dynamic Stock‑Preservation (DSP) strategy.
    Decides to cooperate or defect based on current stock levels,
    round index, and the observed cooperation ratio of opponents
    from the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        cap = float(game_description.capacity)
        self.C_threshold = cap * 0.5
        self.D_threshold = cap * 0.8
        self.coop_ratio_target = 0.6
        self.n_opponents = max(game_description.n_players - 1, 1)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the upcoming round.
        """
        t = history.round_number
        total_rounds = self.game_description.n_rounds
        if t == total_rounds - 1:
            return Action.D
        if t == 0:
            if current_stock >= self.D_threshold:
                return Action.D
            else:
                return Action.C
        k = int(history.opponent_cooperators[-1])
        observed_coop_ratio = k / self.n_opponents if self.n_opponents else 0.0
        S = current_stock
        if S < self.C_threshold:
            return Action.C
        if S < self.D_threshold:
            if observed_coop_ratio < self.coop_ratio_target:
                return Action.C
            else:
                return Action.D
        if observed_coop_ratio >= self.coop_ratio_target:
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_9(BaseStrategy):
    """
    Adaptive Collective Sustainability (ACS) strategy.

    - Cooperates when the stock is healthy and a majority of opponents cooperated in the
      previous round, or when the stock is low (to aid recovery).
    - Defects in the final round and when the stock has collapsed to zero.
    - In the first round assumes full opponent cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.majority_threshold = math.ceil((self.n - 1) / 2)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock == 0:
            return Action.D
        if t == self.r - 1:
            return Action.D
        if t == 0:
            k_prev = self.n - 1
        else:
            k_prev = int(history.opponent_cooperators[-1])
            if k_prev > self.n - 1:
                k_prev = self.n - 1
        threshold = self.capacity / 2.0
        if current_stock >= threshold:
            if k_prev >= self.majority_threshold:
                return Action.C
            else:
                return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_10(BaseStrategy):
    """
    State‑based cooperation with stock thresholds and opponent‑cooperation feedback.

    - Cooperate by default when the stock is at least 50\u202f% of capacity.
    - Defect when the stock falls to 20\u202f% or below.
    - In later rounds use the proportion of cooperating opponents from the previous round:
        * Compute pₜ₋₁ = (cooperators among n‑1 opponents) / (n‑1)
        * Adjusted target Tₜ = 0.8 × pₜ₋₁
        * Cooperate if pₜ₋₁ ≥ Tₜ **and** current stock ≥ S_safe, otherwise defect.
    - In the final round always defect (maximising immediate payoff).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._alpha = 0.5
        self._gamma = 0.2
        self._beta = 0.8
        self._S_safe = self._alpha * game_description.capacity
        self._S_crit = self._gamma * game_description.capacity
        self._n_opponents = max(game_description.n_players - 1, 1)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.game_description.n_rounds - 1:
            return Action.D
        if current_stock <= self._S_crit:
            return Action.D
        if t == 0:
            return Action.C
        coop_opponents_last = history.opponent_cooperators[-1]
        p_last = coop_opponents_last / self._n_opponents
        T = self._beta * p_last
        if p_last >= T and current_stock >= self._S_safe:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_11(BaseStrategy):
    """
    Adaptive Collective Stewardship.

    The player cooperates on the first round, defects on the last round,
    and otherwise follows a conditional rule based on the current stock level
    and the recent empirical cooperation share of opponents.

    Safeguards:
      * Grim‑trigger – two successive rounds where all opponents defect
        switch the player to permanent defect mode.
      * Stock depletion – if the stock reaches zero the player defects permanently.
      * Recovery – while in permanent defect mode, three consecutive rounds
        with sufficient stock (≥ capacity/2) and sufficient cooperation share
        (≥ 0.5) restore normal mode.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)
        self.S_target: float = self.capacity / 2.0
        self.tau: float = 0.5
        self.mode: str = 'normal'
        self.zero_coop_streak: int = 0
        self.recovery_streak: int = 0

    def _recent_coop_share(self, history: PlayerHistory, k: int=5) -> float:
        """
        Average share of opponents that cooperated over the last `k` completed rounds.
        If no prior data is available, returns 1.0 (optimistic assumption).
        """
        t = history.round_number
        if t == 0:
            return 1.0
        start = max(0, t - k)
        total_coop = int(np.sum(history.opponent_cooperators[start:t]))
        rounds = t - start
        denom = rounds * (self.n_players - 1)
        if denom == 0:
            return 1.0
        return total_coop / denom

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0:
            last_opponents_coop = int(history.opponent_cooperators[-1])
            if last_opponents_coop == 0:
                self.zero_coop_streak += 1
            else:
                self.zero_coop_streak = 0
            if self.zero_coop_streak >= 2:
                self.mode = 'permanent_defect'
            if current_stock == 0:
                self.mode = 'permanent_defect'
            if self.mode == 'permanent_defect':
                recent_share_3 = self._recent_coop_share(history, k=3)
                if current_stock >= self.S_target and recent_share_3 >= self.tau:
                    self.recovery_streak += 1
                else:
                    self.recovery_streak = 0
                if self.recovery_streak >= 3:
                    self.mode = 'normal'
                    self.recovery_streak = 0
                    self.zero_coop_streak = 0
            else:
                self.recovery_streak = 0
        if t == 0:
            return Action.C
        if t == self.n_rounds - 1:
            return Action.D
        if self.mode == 'permanent_defect':
            return Action.D
        p = self._recent_coop_share(history, k=5)
        if current_stock >= self.S_target and p >= self.tau:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_12(BaseStrategy):
    """
    Adaptive Stock‑Preservation (ASP)

    The player tries to keep the common pool above a target level (60\u202f% of capacity)
    for as many future rounds as possible.  Early in the game the player cooperates
    when the pool is healthy, otherwise defects.  In later rounds the player
    estimates whether cooperating would keep the projected stock above the target.
    If both cooperate and defect preserve the target, the player defects to obtain the
    higher immediate payoff.  The final round is always a defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n: int = game_description.n_players
        self.r: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)
        self.T: int = math.floor(self.r / 2)
        self.S_target: float = 0.6 * self.capacity
        self.C_min: int = math.ceil(self.n * 0.5)
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        capacity = self.capacity
        S_target = self.S_target
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        if t == self.r - 1:
            return Action.D
        if t == 0:
            if current_stock >= S_target:
                return Action.C
            else:
                return Action.D
        prevCoop = int(history.opponent_cooperators[-1])
        if current_stock < S_target:
            return Action.D
        cons_coop = current_stock / (2.0 * n)
        cons_def = current_stock / float(n)
        opp_coop_cons = current_stock / (2.0 * n)
        opp_def_cons = current_stock / float(n)
        n_opp = n - 1
        n_opp_def = n_opp - prevCoop
        totalCons_C = prevCoop * opp_coop_cons + n_opp_def * opp_def_cons + cons_coop
        totalCons_D = prevCoop * opp_coop_cons + n_opp_def * opp_def_cons + cons_def
        S_rem_C = max(current_stock - totalCons_C, 0.0)
        S_rem_D = max(current_stock - totalCons_D, 0.0)
        proj_C = S_rem_C + 2.0 * S_rem_C * (1.0 - S_rem_C / capacity)
        proj_D = S_rem_D + 2.0 * S_rem_D * (1.0 - S_rem_D / capacity)
        proj_C = min(proj_C, capacity)
        proj_D = min(proj_D, capacity)
        if proj_C >= S_target and proj_D >= S_target:
            return Action.D
        elif proj_C >= S_target:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_13(BaseStrategy):
    """
    Adaptive Collective Steward strategy for the Common Pool Resource game.
    Starts with cooperation, then adapts based on the observed cooperation of opponents
    in the previous round and the health of the common stock.
    """
    _THETA_LOW = 0.3
    _THETA_MID = 0.6
    _THETA_HIGH = 0.9

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0:
            return Action.D
        if t == 0:
            return Action.C
        stock_frac = current_stock / self.capacity
        opponents = self.n_players - 1
        k = int(history.opponent_cooperators[-1]) if opponents > 0 else 0
        coop_rate = k / opponents if opponents > 0 else 0.0
        if t == self.n_rounds - 1:
            return Action.C if stock_frac >= self._THETA_MID else Action.D
        if stock_frac >= self._THETA_MID:
            stock_condition = 'high'
        elif stock_frac >= self._THETA_LOW:
            stock_condition = 'moderate'
        else:
            stock_condition = 'low'
        if coop_rate >= 0.6:
            coop_condition = 'cooperative'
        elif coop_rate >= 0.3:
            coop_condition = 'mixed'
        else:
            coop_condition = 'defective'
        if stock_condition == 'high':
            if coop_condition in ('cooperative', 'mixed'):
                return Action.C
            else:
                return Action.D
        elif stock_condition == 'moderate':
            if coop_condition == 'cooperative':
                return Action.C
            elif coop_condition == 'mixed':
                return Action.C if stock_frac > self._THETA_LOW else Action.D
            else:
                return Action.D
        elif coop_condition == 'defective':
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_14(BaseStrategy):
    """
    Adaptive Collective Guard strategy for the Common Pool Resource game.

    - Cooperates when the stock is healthy (above τ_low·capacity) **and**
      the observed cooperation rate among opponents is at least λ.
    - Defects in the final round to capture the remaining payoff.
    - If the stock falls below the low threshold, cooperation resumes only
      after the stock recovers above the higher threshold τ_high
      while the cooperation rate condition still holds.
    - Starts with full cooperation in the first round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game = game_description
        self.n_players = game_description.n_players
        self.n_opponents = self.n_players - 1
        self.capacity = float(game_description.capacity)
        self.tau_low = 0.6
        self.tau_high = 0.8
        self.lambda_min = 0.5
        self.last_round_index = game_description.n_rounds - 1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.last_round_index:
            return Action.D
        low_threshold = self.tau_low * self.capacity
        high_threshold = self.tau_high * self.capacity
        if current_stock >= low_threshold:
            stock_ok = True
        else:
            stock_ok = current_stock >= high_threshold
        c_prev = history.opponent_cooperators[-1]
        coop_rate = c_prev / self.n_opponents if self.n_opponents > 0 else 0.0
        coop_ok = coop_rate >= self.lambda_min
        if stock_ok and coop_ok:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_15(BaseStrategy):
    """
    Adaptive Collective Guard strategy for the Common Pool Resource game.

    The player cooperates when the common stock is healthy and the observed
    cooperation share of opponents meets a target.  Defection is used to protect
    the resource when stock is low, to punish a slip in cooperation, or in the
    final round.  A short punishment phase (punish_len rounds) follows any
    cooperation breach, and can be refreshed if further breaches occur.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        capacity = float(game_description.capacity)
        self.safe_stock = 0.5 * capacity
        self.critical_stock = 0.2 * capacity
        self.target_coop = 0.6
        self.punish_len = 2
        self.punish_counter = 0
        self.last_coop_share = 1.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action for the current round."""
        t = history.round_number
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        if t == 0:
            action = Action.C
        elif t == self.game_description.n_rounds - 1:
            action = Action.D
        elif self.punish_counter > 0:
            if self.last_coop_share < self.target_coop:
                self.punish_counter = self.punish_len
            else:
                self.punish_counter -= 1
            action = Action.D
        else:
            coop_share = self.last_coop_share
            if current_stock < self.critical_stock:
                action = Action.D
            elif current_stock >= self.safe_stock and coop_share >= self.target_coop:
                action = Action.C
            elif coop_share < self.target_coop:
                action = Action.D
                self.punish_counter = self.punish_len
            else:
                action = Action.D
        if history.opponent_cooperators.size:
            k = int(history.opponent_cooperators[-1])
        else:
            k = 0
        self.last_coop_share = k / n_opponents if n_opponents else 0.0
        return action

class Strategy_COLLECTIVE_16(BaseStrategy):
    """
    Adaptive Collective Threshold strategy.

    Cooperates when the common‑pool stock is sufficiently high and a
    sufficient proportion of opponents have cooperated, otherwise defects.
    Special handling for first and last round, stock collapse, and a
    tit‑for‑tat‑with‑forgiveness fallback.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta_high = 0.6
        self.theta_low = 0.3
        self.phi_coop = 0.5
        self.delta = 0.1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        total_rounds = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        if current_stock <= 0.0:
            return Action.D
        t = history.round_number
        if t == total_rounds - 1:
            return Action.D
        stock_frac = current_stock / capacity if capacity > 0 else 0.0
        if t == 0:
            if stock_frac >= self.theta_high:
                return Action.C
            else:
                return Action.D
        if stock_frac >= self.theta_high:
            stock_status = 'high'
        elif stock_frac <= self.theta_low:
            stock_status = 'low'
        else:
            stock_status = 'medium'
        k = int(history.opponent_cooperators[-1]) if history.opponent_cooperators.size > 0 else 0
        p = k / n_opponents if n_opponents > 0 else 0.0
        if p >= self.phi_coop:
            coop_status = 'cooperative'
        elif p <= 1.0 - self.phi_coop:
            coop_status = 'defective'
        else:
            coop_status = 'mixed'
        last_action_bool = bool(history.my_actions[-1]) if history.my_actions.size > 0 else False
        last_action = 'C' if last_action_bool else 'D'
        if stock_status == 'high' and coop_status == 'cooperative':
            return Action.C
        if stock_status == 'low':
            return Action.D
        if stock_status == 'medium':
            if coop_status == 'cooperative':
                return Action.C
            if coop_status == 'defective':
                return Action.D
            if last_action == 'C' and p >= self.phi_coop - self.delta:
                return Action.C
            return Action.D
        return Action.D

class Strategy_PROSOCIAL_1(BaseStrategy):
    """
    Prosocial Adaptive Strategy – “Sustainable Cooperation with Conditional Rescue”.

    - Starts by cooperating (round 0).
    - Computes a linearly decreasing safety‑stock threshold.
    - Cooperates while the stock stays above that threshold.
    - When below, cooperates only if at least 60\u202f% of opponents cooperated
      in the previous round; otherwise defects.
    - In the final round, cooperates if at least 10\u202f% of capacity remains,
      else defects.
    - Handles edge cases (stock exhausted, full stock, solo player) explicitly.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        cap = float(self.game_description.capacity)
        if current_stock <= 0.0:
            return Action.D
        if current_stock >= cap:
            return Action.C
        t = history.round_number
        T = r - t
        if t == 0:
            return Action.C
        if t == r - 1:
            if current_stock > 0.1 * cap:
                return Action.C
            else:
                return Action.D
        threshold = cap * 0.5 * (T / r)
        if current_stock > threshold:
            return Action.C
        else:
            if n == 1:
                return Action.C if current_stock > threshold else Action.D
            kC = int(history.opponent_cooperators[-1])
            denominator = max(n - 1, 1)
            coop_ratio = kC / denominator
            if coop_ratio >= 0.6:
                return Action.C
            else:
                return Action.D

class Strategy_PROSOCIAL_2(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.
    Starts by cooperating, then adapts based on recent cooperation levels,
    remaining stock, and safety considerations.
    """
    SAFETY_FACTOR = 0.25
    COOP_THRESHOLD = 0.5

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = float(game_description.n_players)
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.MIN_COOP_STOCK = self.capacity * 0.3

    def _growth(self, stock: float) -> float:
        """Growth function G(S) = 2 * S * (1 - S / capacity)."""
        return 2.0 * stock * (1.0 - stock / self.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        S = self.capacity if current_stock >= self.capacity else max(0.0, current_stock)
        if S == 0.0:
            return Action.D
        t = history.round_number
        n_players = int(self.n)
        n_opponents = n_players - 1
        if t == 0:
            return Action.C
        k_last = int(history.opponent_cooperators[-1]) if history.opponent_cooperators.size > 0 else 0
        if t == self.r - 1:
            coop_cons = S / (2.0 * self.n)
            opp_coop_cons = k_last * (S / (2.0 * self.n))
            opp_defect_cons = (n_opponents - k_last) * (S / self.n)
            total_cons = coop_cons + opp_coop_cons + opp_defect_cons
            S_rem = max(0.0, S - total_cons)
            S_next = min(S_rem + self._growth(S_rem), self.capacity)
            if S_next >= self.SAFETY_FACTOR * self.capacity or S >= self.MIN_COOP_STOCK:
                return Action.C
            else:
                return Action.D
        coop_rate = k_last / n_opponents if n_opponents > 0 else 0.0
        exp_opp_cons_per = coop_rate * (S / (2.0 * self.n)) + (1.0 - coop_rate) * (S / self.n)
        total_opp_cons = n_opponents * exp_opp_cons_per
        our_cons = S / (2.0 * self.n)
        S_rem = max(0.0, S - (our_cons + total_opp_cons))
        S_next = min(S_rem + self._growth(S_rem), self.capacity)
        if coop_rate >= self.COOP_THRESHOLD and S_next >= self.SAFETY_FACTOR * self.capacity and (S >= self.MIN_COOP_STOCK):
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_3(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.
    Cooperates when the stock is healthy and opponents are mostly cooperative,
    switches to defection briefly when exploitation rises or the stock is low,
    and preserves the pool in the final round if possible.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.n_opponents = self.n - 1
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.threshold_high = 0.75 * self.capacity
        self.threshold_low = 0.3 * self.capacity
        self.defection_rate_limit = 0.3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            if current_stock >= self.threshold_low:
                return Action.C
            else:
                return Action.D
        if t == 0:
            return Action.C
        start = max(0, t - 3)
        recent_coop = 0
        recent_def = 0
        for idx in range(start, t):
            coop = int(history.opponent_cooperators[idx])
            recent_coop += coop
            recent_def += self.n_opponents - coop
        total_interactions = recent_coop + recent_def
        recent_def_rate = recent_def / total_interactions if total_interactions > 0 else 0.0
        if current_stock >= self.threshold_high and recent_def_rate <= self.defection_rate_limit:
            return Action.C
        if self.threshold_low <= current_stock < self.threshold_high:
            if recent_def_rate <= self.defection_rate_limit:
                return Action.C
            else:
                return Action.D
        if current_stock < self.threshold_low:
            if recent_def_rate > self.defection_rate_limit:
                return Action.D
            else:
                return Action.C
        return Action.C

class Strategy_PROSOCIAL_4(BaseStrategy):
    """
    Adaptive Conservation Play (ACP)

    The strategy adapts its cooperation level based on the current stock of the
    common pool and a moving average of opponent cooperation.
    - High stock + cooperative opponents → cooperate.
    - Moderate stock + at least half the opponents cooperating → cooperate.
    - Low stock or mostly defecting opponents → cooperate only if stock is
      still above a low threshold, otherwise defect.
    - In the final round the player always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.k = 3
        self.C_thresh = 0.7
        self.S_low = 0.3 * game_description.capacity
        self.S_mid = 0.6 * game_description.capacity
        self.n_opponents = game_description.n_players - 1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n_rounds = self.game_description.n_rounds
        if t == n_rounds - 1:
            return Action.D
        if t == 0:
            avg_coop = 1.0
        else:
            start = max(0, t - self.k)
            window = history.opponent_cooperators[start:t]
            total_coop = int(np.sum(window))
            window_len = t - start
            denom = window_len * self.n_opponents
            avg_coop = total_coop / denom if denom > 0 else 0.0
        S = current_stock
        if S > self.S_mid and avg_coop >= self.C_thresh:
            return Action.C
        elif S > self.S_low and avg_coop >= 0.5:
            return Action.C
        elif S > self.S_low:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_5(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.

    - Cooperates when the stock is healthy (≥ 0.5\u202f×\u202fcapacity) and opponents have
      shown sufficient cooperation (average ≥ 0.6 over the trusted recent rounds).
    - Defects when the stock is unhealthy or recent cooperation falls below the
      threshold.
    - Uses a trust‑reset mechanism: a trust counter τ tracks consecutive
      cooperative rounds (capped at k = min(3, r‑1)).  c̄ is the average opponent
      cooperation proportion over the last τ rounds.  τ is reset to 0 upon any
      defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.capacity = float(game_description.capacity)
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.alpha = 0.5 * self.capacity
        self.beta = 0.6
        self.k = min(3, max(1, self.n_rounds - 1))
        self.trust_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        opponents = self.n_players - 1
        if self.trust_counter == 0:
            c_bar = 0.0
        else:
            recent = history.opponent_cooperators[-self.trust_counter:]
            props = recent.astype(float) / opponents
            c_bar = float(np.mean(props))
        if t == 0:
            chosen_action = Action.C if current_stock >= self.alpha else Action.D
        elif t == self.n_rounds - 1:
            if current_stock >= self.alpha and c_bar >= self.beta:
                chosen_action = Action.C
            else:
                chosen_action = Action.D
        elif current_stock < self.alpha:
            chosen_action = Action.D
        elif c_bar >= self.beta:
            chosen_action = Action.C
        else:
            chosen_action = Action.D
        if chosen_action == Action.C:
            self.trust_counter = min(self.trust_counter + 1, self.k)
        else:
            self.trust_counter = 0
        return chosen_action

class Strategy_PROSOCIAL_6(BaseStrategy):
    """
    Adaptive Commons Keeper – a conditional cooperation strategy for the
    Common Pool Resource game.

    The player cooperates when the stock is above a decreasing sustainability
    target and opponents have shown at least 50\u202f% cooperation in the previous
    round.  It defects when the stock is dangerously low, when the target is
    not met, or in the final round if the stock is below half capacity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t == 0:
            o = n - 1
        else:
            o = int(history.opponent_cooperators[-1])
        if t == r - 1:
            if current_stock >= 0.5 * capacity:
                return Action.C
            else:
                return Action.D
        if current_stock <= 0.1 * capacity:
            return Action.D
        target = capacity * (r - t) / r
        coop_expect = o / (n - 1) if n > 1 else 0.0
        if current_stock >= target and coop_expect >= 0.5:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_7(BaseStrategy):
    """
    Prosocial Adaptive (PA‑C/D) strategy.
    - Cooperates when the stock is healthy and opponents have been mostly cooperative.
    - Defects otherwise, and may enter a retaliation phase of length P
      after a defection that follows a prior cooperation when opponent defection
      exceeds the tolerated level β.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.5
        self.beta = 0.3
        self.P = 3
        self.k = 3
        self.punish_cnt = 0
        self.def_rate = 0.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        round_num = history.round_number
        if round_num == 0:
            return Action.C
        if round_num == self.game_description.n_rounds - 1:
            return Action.D
        if current_stock <= 0:
            return Action.C
        if self.punish_cnt > 0:
            self.punish_cnt -= 1
            return Action.D
        c_last = int(history.opponent_cooperators[-1])
        my_last_cooperate = bool(history.my_actions[-1])
        opp_defect = n - 1 - c_last
        opp_defect_prop = opp_defect / (n - 1) if n - 1 > 0 else 0.0
        self.def_rate = (self.def_rate * (self.k - 1) + opp_defect_prop) / self.k
        sustainable = current_stock >= self.alpha * capacity
        if sustainable and self.def_rate <= self.beta:
            action = Action.C
        else:
            action = Action.D
        if action == Action.D and my_last_cooperate and (opp_defect_prop > self.beta):
            self.punish_cnt = self.P
        return action

class Strategy_PROSOCIAL_8(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.
    Cooperates when the stock is above a safety threshold and recent
    opponent cooperation is sufficiently high; otherwise defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.alpha = 0.6
        self.beta = 0.7
        self.k = min(3, self.r - 1)
        self.target_stock = self.alpha * self.capacity
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C if current_stock >= self.target_stock else Action.D
        if t == self.r - 1:
            return Action.C if current_stock >= self.target_stock else Action.D
        recent_start = max(0, t - self.k)
        recent_counts = history.opponent_cooperators[recent_start:t]
        if self.n > 1:
            recent_fractions = recent_counts.astype(float) / (self.n - 1)
        else:
            recent_fractions = np.zeros_like(recent_counts, dtype=float)
        recent_coop = float(np.mean(recent_fractions)) if recent_fractions.size else 0.0
        if current_stock >= self.target_stock and recent_coop >= self.beta:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_9(BaseStrategy):
    """
    Prosocial Adaptive Strategy (PAS).

    Cooperates by default, defects only as a bounded retaliation when the
    proportion of defecting opponents exceeds a tolerance threshold, or
    when the stock falls below a target fraction of capacity. No defection in
    the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta: float = 0.7
        self.tau: int = 2
        self.delta: float = 0.4
        self._ret_counter: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        last_round = self.game_description.n_rounds - 1
        if t == 0:
            return Action.C
        if t == last_round:
            return Action.C
        if self._ret_counter > 0:
            self._ret_counter -= 1
            return Action.D
        if current_stock < self.theta * capacity:
            return Action.C
        c = int(history.opponent_cooperators[-1]) if history.round_number > 0 else 0
        d = n - 1 - c
        opp_total = max(n - 1, 1)
        defect_ratio = d / opp_total
        if defect_ratio > self.delta:
            self._ret_counter = self.tau
            return Action.D
        return Action.C

class Strategy_PROSOCIAL_10(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.

    - Starts by cooperating.
    - Cooperates in the final round regardless of state.
    - In other rounds, cooperates if the observed cooperation rate among opponents
      is at least 0.6 *and* the stock is above the safety threshold (0.35\u202f×\u202fcapacity).
    - Otherwise, defects only when at least 50\u202f% of opponents defected in the
      previous round; in all other cases it cooperates to preserve the resource.
    """
    _TAU_FACTOR = 0.35
    _THETA_COOP = 0.6
    _PHI_DEFECT = 0.5

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._tau_stock = self._TAU_FACTOR * self.game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        t = history.round_number
        if t == 0 or t == r - 1:
            return Action.C
        c_last = int(history.opponent_cooperators[-1])
        p_coop = c_last / (n - 1) if n > 1 else 0.0
        stock_safe = current_stock >= self._tau_stock
        if p_coop >= self._THETA_COOP and stock_safe:
            return Action.C
        elif 1.0 - p_coop >= self._PHI_DEFECT:
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_11(BaseStrategy):
    """
    Adaptive cooperation strategy for the Common Pool Resource game.
    - Cooperates when the resource is low, in the first/last round,
      during a forgiveness phase after own defection, or when a sufficient
      share of opponents cooperated previously.
    - Defects otherwise.
    - Dynamically adjusts the required cooperation rate based on recent
      average stock levels.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = game_description.capacity
        self.threshold = 0.5 * self.capacity
        self.forgiveness_window = 2
        self.min_coop_rate = 0.6
        self.last_coop_count = None
        self.rounds_since_defect = self.forgiveness_window
        self.stock_history = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0:
            k = int(history.opponent_cooperators[-1])
        else:
            k = None
        cooperate = False
        if current_stock < self.threshold:
            cooperate = True
        elif t == self.n_rounds - 1:
            cooperate = True
        elif t == 0:
            cooperate = True
        elif self.rounds_since_defect < self.forgiveness_window:
            cooperate = True
        elif k is not None and k / (self.n_players - 1) >= self.min_coop_rate:
            cooperate = True
        else:
            cooperate = False
        action = Action.C if cooperate else Action.D
        if action == Action.D:
            self.rounds_since_defect = 0
        else:
            self.rounds_since_defect = min(self.rounds_since_defect + 1, self.forgiveness_window)
        self.last_coop_count = k
        self.stock_history.append(current_stock)
        if len(self.stock_history) > 3:
            self.stock_history.pop(0)
        if len(self.stock_history) == 3:
            avg_stock = float(np.mean(self.stock_history))
            if avg_stock < 0.3 * self.capacity:
                self.min_coop_rate = 0.8
            elif all((s > 0.9 * self.capacity for s in self.stock_history)):
                self.min_coop_rate = 0.5
        return action

class Strategy_PROSOCIAL_12(BaseStrategy):
    """
    Prosocial Adaptive Strategy (Cooperate‑Maintain‑Recover).

    Cooperates initially and in the final round.
    When stock falls below 30\u202f% of capacity, cooperates to protect the resource.
    Otherwise adapts to opponents' recent cooperation level using a tolerance window w.
    Defects only if cooperation is low, but uses a defect‑streak counter to
    forgive after w consecutive defections and return to cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.T_low = 0.3 * self.capacity
        self.w = max(2, self.n_rounds // 10)
        self.defectStreak = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_idx = history.round_number
        if round_idx == 0 or round_idx == self.n_rounds - 1:
            return Action.C
        if current_stock < self.T_low:
            action = Action.C
        else:
            window = min(self.w, round_idx)
            recent_coops = history.opponent_cooperators[-window:]
            avg_coop = float(np.mean(recent_coops)) if window > 0 else 0.0
            possible_coops = window * (self.n_players - 1)
            coop_ratio = avg_coop / possible_coops if possible_coops > 0 else 0.0
            if coop_ratio >= 0.6:
                action = Action.C
            elif coop_ratio >= 0.4:
                action = Action.C
            else:
                action = Action.D
        if action == Action.D:
            self.defectStreak += 1
        else:
            self.defectStreak = max(0, self.defectStreak - 1)
        if self.defectStreak >= self.w:
            action = Action.C
            self.defectStreak = 0
        return action

class Strategy_PROSOCIAL_13(BaseStrategy):
    """
    Adaptive Sustainability‑Threshold (AST) strategy.
    Starts cooperating, then adapts its action based on the current stock level and a
    moving average of opponent cooperation (last k=3 rounds).  The stock thresholds
    are 0.5·capacity (sustainability) and 0.2·capacity (critical risk).  The
    strategy may randomise its choice when cooperation is moderate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_opponents = self.n_players - 1
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.sustain_thresh = 0.5 * self.capacity
        self.risk_thresh = 0.2 * self.capacity
        self.k = 3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock <= 0.0:
            return Action.D
        round_idx = history.round_number
        if round_idx == 0:
            return Action.C
        if round_idx == self.n_rounds - 1:
            return Action.C if current_stock >= self.risk_thresh else Action.D
        recent = history.opponent_cooperators[-self.k:]
        fractions = recent.astype(float) / self.n_opponents
        coop_ratio = float(np.mean(fractions))
        if current_stock >= self.sustain_thresh:
            if coop_ratio >= 0.7:
                return Action.C
            elif coop_ratio >= 0.4:
                return Action.C if random.random() <= coop_ratio else Action.D
            else:
                return Action.C if random.random() <= coop_ratio else Action.D
        elif current_stock >= self.risk_thresh:
            return Action.C if coop_ratio >= 0.5 else Action.D
        else:
            return Action.D

class Strategy_PROSOCIAL_14(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.

    The willingness to cooperate is calibrated each round based on:
    • The observed cooperation ratio of opponents in the previous round.
    • The current health of the common pool (stock / capacity).
    • An end‑game boost in the last round.
    • A safety override when the resource is critically low.

    The algorithm follows the specification provided in the prompt.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        if current_stock <= 0.0:
            return Action.D
        t = history.round_number
        if t == 0:
            p = 1.0
        else:
            try:
                coop_count = history.opponent_cooperators[-1]
                p = coop_count / (n - 1)
            except Exception:
                p = 0.5
        s_ratio = current_stock / capacity
        if s_ratio >= 0.75:
            base_pi = 0.9 * p + 0.05
        elif s_ratio >= 0.35:
            base_pi = 0.6 * p + 0.2
        else:
            base_pi = 0.3 * p + 0.3
        if t == r - 1:
            base_pi = max(base_pi, 0.8)
        if s_ratio < 0.1:
            return Action.D
        if random.random() <= base_pi:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_15(BaseStrategy):
    """
    Adaptive Prosocial Cooperation.

    - First round: cooperate.
    - If stock falls below β·capacity (β = 0.2) → cooperate.
    - Otherwise, detect exploitation:
        * previous payoff ≤ τ·π_C  (τ = 0.8)
        * observed cooperation fraction of opponents < 0.5
    - When exploited and I have not defected in the previous k rounds (k = 2),
      defect for this round only.
    - After any cooperative move the exploitation counter is reset, allowing a
      new possible defection after at least k cooperative rounds.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._beta = 0.2
        self._tau = 0.8
        self._k = 2
        self._last_defect_round = -10 ** 9

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if current_stock <= 0:
            return Action.C
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        if current_stock < self._beta * capacity:
            return Action.C
        pi_C = current_stock / (2 * n) if n > 0 else 0.0
        exploited = False
        if pi_C > 0:
            prev_payoff = history.my_payoffs[-1]
            c_prev = history.opponent_cooperators[-1]
            f_prev = c_prev / (n - 1) if n - 1 > 0 else 1.0
            exploited = prev_payoff <= self._tau * pi_C and f_prev < 0.5
        rounds_since_last_defect = t - self._last_defect_round
        can_defect = exploited and rounds_since_last_defect > self._k
        if can_defect:
            self._last_defect_round = t
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_16(BaseStrategy):
    """
    Prosocial Adaptive Strategy.

    - Cooperates when the stock is at or below a safety threshold or when
      only a few rounds remain.
    - Otherwise estimates a sustainable total draw; if full cooperation is
      sustainable it cooperates, else it mixes cooperation and defection.
    - The cooperation probability is increased when many opponents
      cooperated in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.6

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        remaining = r - t
        safe_stock = self.alpha * capacity
        if current_stock <= safe_stock or remaining <= 2:
            return Action.C
        if t == 0:
            return Action.C
        S = current_stock
        max_total_cons = S * (1.0 - self.alpha)
        coop_total = S * 0.5
        defect_total = S
        if coop_total <= max_total_cons:
            return Action.C
        if coop_total == defect_total:
            p = 0.0
        else:
            p = (max_total_cons - defect_total) / (coop_total - defect_total)
            p = 0.0 if p < 0.0 else 1.0 if p > 1.0 else p
        k = history.opponent_cooperators[-1] if history.round_number > 0 else 0
        boost = 0.2 * (k / (n - 1)) if n > 1 else 0.0
        p = p + boost
        p = 1.0 if p > 1.0 else p
        return Action.C if random.random() < p else Action.D

class Strategy_INDIVIDUALISTIC_1(BaseStrategy):
    """
    Projected‑Payoff Greedy strategy for the common pool resource game.
    The agent estimates the opponent cooperation rate from the last observed round,
    projects the immediate and next‑round stock levels for each of its two possible
    actions (cooperate or defect), assumes the same cooperation rate will persist,
    and chooses the action that maximizes the projected cumulative payoff until
    the end of the game.  In the final round it always defects, and if the stock is
    depleted it also defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        if t == r - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.D
        if t == 0:
            p = 0.5
        else:
            c_obs = int(history.opponent_cooperators[-1])
            p = c_obs / (n - 1) if n - 1 > 0 else 0.0
        remaining = r - t - 1
        S = float(current_stock)
        payoff_C_now = S / (2 * n)
        payoff_D_now = S / n
        opp_consume_coop = S / (2 * n)
        opp_consume_def = S / n
        expected_opp_cons = p * opp_consume_coop + (1.0 - p) * opp_consume_def
        total_cons_C = payoff_C_now + (n - 1) * expected_opp_cons
        S_remain_C = S - total_cons_C
        if S_remain_C < 0.0:
            S_remain_C = 0.0
        growth_C = 2.0 * S_remain_C * (1.0 - S_remain_C / capacity)
        S_next_C = S_remain_C + growth_C
        if S_next_C > capacity:
            S_next_C = capacity
        total_cons_D = payoff_D_now + (n - 1) * expected_opp_cons
        S_remain_D = S - total_cons_D
        if S_remain_D < 0.0:
            S_remain_D = 0.0
        growth_D = 2.0 * S_remain_D * (1.0 - S_remain_D / capacity)
        S_next_D = S_remain_D + growth_D
        if S_next_D > capacity:
            S_next_D = capacity
        future_payoff_C = S_next_C / (2 * n) if remaining > 0 else 0.0
        future_payoff_D = S_next_D / n if remaining > 0 else 0.0
        total_proj_C = payoff_C_now + remaining * future_payoff_C
        total_proj_D = payoff_D_now + remaining * future_payoff_D
        if total_proj_C > total_proj_D:
            return Action.C
        elif total_proj_D > total_proj_C:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_2(BaseStrategy):
    """
    Self‑Preserving Greedy strategy.
    Chooses cooperation when the immediate payoff plus the expected future value
    of preserving stock (adjusted by observed opponent cooperation) outweighs the
    immediate defect payoff. Otherwise defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        S = max(0.0, min(current_stock, capacity))
        if S <= 0.0:
            return Action.D
        pi_C = S / (2 * n) if n else 0.0
        pi_D = S / n if n else 0.0
        if t == r - 1:
            return Action.D if pi_D > pi_C else Action.C
        M = (r - t - 1) / r
        if capacity == 0.0:
            growth_factor = 0.0
        else:
            growth_factor = 2.0 * (1.0 - S / capacity)
            if growth_factor < 0.0:
                growth_factor = 0.0
        raw_future_value = M * (capacity * growth_factor)
        if t == 0:
            adjusted_future_value = raw_future_value
        else:
            c_obs = int(history.opponent_cooperators[-1])
            M_adj = M * (c_obs / (n - 1)) if n > 1 else 0.0
            adjusted_future_value = M_adj * (capacity * growth_factor)
        if pi_C + adjusted_future_value >= pi_D:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_3(BaseStrategy):
    """
    Adaptive Expected‑Value Cooperation (AEVC).

    The strategy keeps an estimate `p̂` of the probability that an opponent will
    cooperate, updates it from the observed number of cooperating opponents,
    and selects an action by comparing the expected utility of cooperating vs.
    defecting.  Future utility is approximated by a limited‑depth look‑ahead
    (max 3 rounds); beyond that horizon the marginal gain of a defector in a
    final round (`Δ_last = stock/(2n)`) is used as a linear approximation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_hat = 1.0
        self._max_lookahead = 3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        S = max(current_stock, 0.0)
        if S == 0.0:
            return Action.C
        if t == 0:
            self.p_hat = 1.0
        else:
            k_prev = int(history.opponent_cooperators[-1])
            prev = self.p_hat
            self.p_hat = (k_prev + (n - 1) * prev) / (n - 1)
            if self.p_hat < 0.0:
                self.p_hat = 0.0
            elif self.p_hat > 1.0:
                self.p_hat = 1.0
        pi_C = S / (2.0 * n)
        pi_D = S / (1.0 * n)
        if t == r - 1:
            return Action.D

        def expected_total_consumption(my_payoff: float) -> float:
            opp_exp = self.p_hat * (S / (2.0 * n)) + (1.0 - self.p_hat) * (S / (1.0 * n))
            return my_payoff + (n - 1) * opp_exp

        def project_stock(total_consumption: float) -> float:
            S_rem = max(S - total_consumption, 0.0)
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity) if capacity != 0 else 0.0
            return min(S_rem + growth, capacity)
        total_cons_C = expected_total_consumption(pi_C)
        total_cons_D = expected_total_consumption(pi_D)
        S_next_C = project_stock(total_cons_C)
        S_next_D = project_stock(total_cons_D)
        if t == r - 2:
            delta_last = lambda stock: stock / (2.0 * n)
            U_C = pi_C + delta_last(S_next_C)
            U_D = pi_D + delta_last(S_next_D)
            return Action.C if U_C >= U_D else Action.D

        def future_value(start_round: int, stock: float, depth: int) -> float:
            """
            Expected total utility from `start_round` onward given the
            current stock and a fixed opponent cooperation estimate.
            """
            if start_round >= r or depth == 0:
                remaining = r - start_round
                return stock / (2.0 * n) * remaining
            pi_c = stock / (2.0 * n)
            pi_d = stock / (1.0 * n)
            opp_exp = self.p_hat * (stock / (2.0 * n)) + (1.0 - self.p_hat) * (stock / (1.0 * n))
            tot_c = pi_c + (n - 1) * opp_exp
            tot_d = pi_d + (n - 1) * opp_exp
            S_c = max(stock - tot_c, 0.0)
            growth_c = 2.0 * S_c * (1.0 - S_c / capacity) if capacity != 0 else 0.0
            S_next_c = min(S_c + growth_c, capacity)
            S_d = max(stock - tot_d, 0.0)
            growth_d = 2.0 * S_d * (1.0 - S_d / capacity) if capacity != 0 else 0.0
            S_next_d = min(S_d + growth_d, capacity)
            val_c = pi_c + future_value(start_round + 1, S_next_c, depth - 1)
            val_d = pi_d + future_value(start_round + 1, S_next_d, depth - 1)
            return max(val_c, val_d)
        V_future_C = future_value(t + 1, S_next_C, self._max_lookahead)
        V_future_D = future_value(t + 1, S_next_D, self._max_lookahead)
        U_C = pi_C + V_future_C
        U_D = pi_D + V_future_D
        return Action.C if U_C >= U_D else Action.D

class Strategy_INDIVIDUALISTIC_4(BaseStrategy):
    """
    Adaptive Individual‑Return Maximiser (AIRM)

    The agent estimates opponent cooperation from the previous round,
    forecasts the stock after taking each possible action, evaluates the
    expected future payoff for the remaining rounds, and selects the action
    with the higher total (immediate + future) value while never allowing the
    stock to drop below a safety threshold (5\u202f% of capacity).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.S_min = float(game_description.capacity) * 0.05

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        t = history.round_number
        R = self.game_description.n_rounds - t - 1
        if t == 0:
            p = 0.5
        else:
            k_t = int(history.opponent_cooperators[-1])
            p = k_t / (n - 1) if n - 1 > 0 else 0.0
        payC = current_stock / (2.0 * n) if n else 0.0
        payD = current_stock / float(n) if n else 0.0
        opp_consume_ifC = p * (current_stock / (2.0 * n)) + (1.0 - p) * (current_stock / float(n))
        opp_total_consume = (n - 1) * opp_consume_ifC
        consC = opp_total_consume + current_stock / (2.0 * n)
        consD = opp_total_consume + current_stock / float(n)
        SremC = max(current_stock - consC, 0.0)
        SremD = max(current_stock - consD, 0.0)

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / capacity)
        SnextC = min(SremC + grow(SremC), capacity)
        SnextD = min(SremD + grow(SremD), capacity)

        def exp_pay(stock: float) -> float:
            return p * (stock / (2.0 * n)) + (1.0 - p) * (stock / float(n))
        futureC = exp_pay(SnextC) * R
        futureD = exp_pay(SnextD) * R
        valueC = float('-inf') if SremC < self.S_min else payC + futureC
        valueD = float('-inf') if SremD < self.S_min else payD + futureD
        return Action.C if valueC >= valueD else Action.D

class Strategy_INDIVIDUALISTIC_5(BaseStrategy):
    """
    Adaptive Stock‑Threshold with Conditional Retaliation.

    - Cooperates when the observed stock is at least 60\u202f% of capacity,
      the opponents cooperated as a majority in the previous round,
      and no recent majority defection streak exists.
    - Defects in the final round, when stock is low, or after any
      majority‑defection streak until cooperation is observed again.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.theta = 0.6 * self.capacity
        self.coop_last: bool = True
        self.defect_streak: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            k = int(history.opponent_cooperators[-1])
            if k >= self.n / 2:
                self.coop_last = True
                self.defect_streak = 0
            else:
                self.coop_last = False
                self.defect_streak += 1
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if current_stock >= self.theta and self.coop_last and (self.defect_streak == 0):
            return Action.C
        return Action.D

class Strategy_INDIVIDUALISTIC_6(BaseStrategy):
    """
    Conditional‑Future‑Value Threshold (CFVT).

    At each round the player compares the immediate gain from defecting
    with the expected loss of future stock value.  The decision uses the
    observed cooperation rate of opponents in the previous round to
    estimate their next‑round behaviour and predicts the stock level for
    the next round under both cooperating and defecting actions.  Cooperation
    is chosen when the discounted future loss outweighs the instant profit;
    otherwise the player defects.  The last round is always a defection,
    because no future value remains.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock <= 0.0:
            return Action.D
        t = history.round_number
        n = self.n
        cap = self.capacity
        delta_inst = current_stock / (2.0 * n)
        if t >= self.r - 1:
            return Action.D
        if t == 0:
            p_hat = 1.0
        else:
            k_prev = int(history.opponent_cooperators[-1])
            denom = n - 1
            p_hat = k_prev / denom if denom > 0 else 0.0
        cons_coop = current_stock / (2.0 * n)
        cons_def = current_stock / n
        opp_expected = p_hat * cons_coop + (1.0 - p_hat) * cons_def
        my_cons_C = current_stock / (2.0 * n)
        my_cons_D = current_stock / n
        total_C = my_cons_C + (n - 1) * opp_expected
        total_D = my_cons_D + (n - 1) * opp_expected
        S_next_C = max(current_stock - total_C, 0.0)
        S_next_D = max(current_stock - total_D, 0.0)

        def grow(S):
            growth = 2.0 * S * (1.0 - S / cap) if cap > 0 else 0.0
            return min(S + growth, cap)
        S_next_star_C = grow(S_next_C)
        S_next_star_D = grow(S_next_D)
        h = self.r - t - 1
        future_horizon = h + 1
        if future_horizon <= 0:
            V = 0.0
        else:
            V = S_next_star_C / (2.0 * n * future_horizon)
        delta_future = V * (S_next_star_C - S_next_star_D)
        if delta_inst <= delta_future:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_7(BaseStrategy):
    """
    Adaptive Individualist (AI‑C/D) strategy.
    Estimates opponent cooperation probability from observed counts,
    forecasts one‑step stock under cooperation and defection,
    projects future cooperative payoff, and chooses C when the
    net expected advantage exceeds a proportional tolerance.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._total_coop = 0.0
        self._total_opp_moves = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        R = r - t
        if t > 0:
            obs_coop = int(history.opponent_cooperators[-1])
            self._total_coop += obs_coop
            self._total_opp_moves += n - 1
        if self._total_opp_moves > 0:
            p_hat = self._total_coop / self._total_opp_moves
        else:
            p_hat = 0.5
        c_payoff = current_stock / (2.0 * n)
        d_payoff = current_stock / float(n)
        exp_opp_coop = (n - 1) * p_hat
        exp_opp_def = (n - 1) * (1.0 - p_hat)
        consumption_c = c_payoff + exp_opp_coop * c_payoff + exp_opp_def * d_payoff
        s_rem_c = max(current_stock - consumption_c, 0.0)
        growth_c = 2.0 * s_rem_c * (1.0 - s_rem_c / capacity)
        s_next_c = min(s_rem_c + growth_c, capacity)
        consumption_d = d_payoff + exp_opp_coop * c_payoff + exp_opp_def * d_payoff
        s_rem_d = max(current_stock - consumption_d, 0.0)
        growth_d = 2.0 * s_rem_d * (1.0 - s_rem_d / capacity)
        s_next_d = min(s_rem_d + growth_d, capacity)
        if R > 1:
            future_c = (R - 1) * (s_next_c / (2.0 * n))
            future_d = (R - 1) * (s_next_d / (2.0 * n))
        else:
            future_c = 0.0
            future_d = 0.0
        if current_stock <= 0.05 * capacity:
            return Action.D
        net_coop = c_payoff + future_c - (d_payoff + future_d)
        epsilon = 0.02 * current_stock / n
        if net_coop > epsilon:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_8(BaseStrategy):
    """
    Adaptive Threshold Defection (ATD)

    The player cooperates only when the discounted expected future benefit
    of preserving the common pool outweighs the immediate gain from defecting.
    Beliefs about opponent cooperation are updated from observed counts.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._discount = 0.9

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t == r - 1:
            return Action.D
        if t == 0:
            opp_coop_rate = 1.0
        else:
            total_coop_opps = int(np.sum(history.opponent_cooperators))
            denom = t * (n - 1)
            opp_coop_rate = total_coop_opps / denom if denom > 0 else 0.0
        expected_coop_opp = opp_coop_rate * (n - 1)
        expected_def_opp = (1.0 - opp_coop_rate) * (n - 1)
        S = max(current_stock, 0.0)
        cons_self_C = S / (2.0 * n)
        cons_opp_C = expected_coop_opp * (S / (2.0 * n)) + expected_def_opp * (S / n)
        S_remain_C = S - (cons_self_C + cons_opp_C)
        growth_C = 2.0 * S_remain_C * (1.0 - S_remain_C / capacity) if capacity > 0 else 0.0
        S_next_if_C = min(S_remain_C + growth_C, capacity)
        cons_self_D = S / n
        cons_opp_D = expected_coop_opp * (S / (2.0 * n)) + expected_def_opp * (S / n)
        S_remain_D = S - (cons_self_D + cons_opp_D)
        growth_D = 2.0 * S_remain_D * (1.0 - S_remain_D / capacity) if capacity > 0 else 0.0
        S_next_if_D = min(S_remain_D + growth_D, capacity)
        horizon = r - t - 1
        value_future_C = 0.0
        S_proj = S_next_if_C
        for k in range(1, horizon + 1):
            cons_self = S_proj / (2.0 * n)
            cons_opp = expected_coop_opp * (S_proj / (2.0 * n)) + expected_def_opp * (S_proj / n)
            S_remain = S_proj - (cons_self + cons_opp)
            growth = 2.0 * S_remain * (1.0 - S_remain / capacity) if capacity > 0 else 0.0
            S_proj = min(S_remain + growth, capacity)
            value_future_C += self._discount ** k * (S_proj / (2.0 * n))
        value_future_D = 0.0
        S_proj = S_next_if_D
        for k in range(1, horizon + 1):
            cons_self = S_proj / n
            cons_opp = expected_coop_opp * (S_proj / (2.0 * n)) + expected_def_opp * (S_proj / n)
            S_remain = S_proj - (cons_self + cons_opp)
            growth = 2.0 * S_remain * (1.0 - S_remain / capacity) if capacity > 0 else 0.0
            S_proj = min(S_remain + growth, capacity)
            value_future_D += self._discount ** k * (S_proj / n)
        immediate_C = S / (2.0 * n)
        immediate_D = S / n
        total_C = immediate_C + value_future_C
        total_D = immediate_D + value_future_D
        return Action.C if total_C > total_D else Action.D

class Strategy_INDIVIDUALISTIC_10(BaseStrategy):
    """
    Individual‑payoff‑maximising strategy for the Common Pool Resource game.

    The agent projects the stock level for the next round under the two
    alternatives (cooperating vs defecting) using a simple linear reserve
    estimate and the observed number of cooperating opponents from the
    previous round.  It then estimates the cumulative payoff over the
    remaining rounds assuming the same pattern will repeat, and picks the
    action with the higher projected total.  Special handling is applied
    for the last round, zero stock and the first round (no history).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock < 1e-12:
            return Action.D
        if t == self.r - 1:
            return Action.D
        if t == 0:
            observed_coop = math.floor((self.n - 1) / 2)
        else:
            observed_coop = int(history.opponent_cooperators[-1])
            observed_coop = max(0, min(observed_coop, self.n - 1))
        r_rem = self.r - t
        sustainable_per_round = self.capacity * (1.0 - 1.0 / (r_rem + 1))
        safe_share = sustainable_per_round / (2.0 * self.n)
        defect_share = sustainable_per_round / self.n
        decay_factor = r_rem / (r_rem + 1)
        expected_coop = int(round(observed_coop * decay_factor))
        expected_coop = max(0, min(expected_coop, self.n - 1))
        coop_payoff = current_stock / (2.0 * self.n)
        defect_payoff = current_stock / self.n
        total_if_C = (expected_coop + 1) * coop_payoff + (self.n - 1 - expected_coop) * defect_payoff
        total_if_D = expected_coop * coop_payoff + (self.n - expected_coop) * defect_payoff

        def next_stock(total_consumption: float) -> float:
            remaining = current_stock - total_consumption
            if remaining < 0:
                remaining = 0.0
            growth = 2.0 * remaining * (1.0 - remaining / self.capacity)
            nxt = remaining + growth
            return min(nxt, self.capacity)
        stock_next_C = next_stock(total_if_C)
        stock_next_D = next_stock(total_if_D)
        future_rounds = r_rem - 1
        payoff_C = coop_payoff + future_rounds * (stock_next_C / (2.0 * self.n))
        payoff_D = defect_payoff + future_rounds * (stock_next_D / self.n)
        if payoff_D >= payoff_C:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_11(BaseStrategy):
    """
    Threshold‑Defector strategy for the Common Pool Resource game.

    The player looks at the current stock level, the observed number of
    cooperating opponents in the previous round, and the round index.
    It applies hard stock thresholds, a safety‑margin check on the expected
    next stock, and an intermediate‐stock adaptive rule that compares the
    expected immediate payoff of cooperating versus defecting.
    """
    MIN_COOP_FACTOR = 0.6
    MAX_DEF_FACTOR = 0.3
    SAFETY_MARGIN_FACTOR = 0.1
    LAST_ROUND_DEFECT = True
    PAYOFF_TOLERANCE = 0.05

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._capacity = float(game_description.capacity)
        self._n = int(game_description.n_players)
        self._r = int(game_description.n_rounds)
        self._min_coop_stock = self.MIN_COOP_FACTOR * self._capacity
        self._max_def_stock = self.MAX_DEF_FACTOR * self._capacity
        self._safety_margin = self.SAFETY_MARGIN_FACTOR * self._capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        S = max(0.0, float(current_stock))
        n = self._n
        capacity = self._capacity
        round_idx = history.round_number
        if S == 0.0:
            return Action.D
        coop_obs = int(history.opponent_cooperators[-1]) if round_idx > 0 else 0
        if round_idx == self._r - 1 and self.LAST_ROUND_DEFECT:
            return Action.D
        if S <= self._max_def_stock:
            return Action.D
        if S >= self._min_coop_stock:
            exp_cons = (coop_obs + 1) * (S / (2.0 * n)) + (n - 1 - coop_obs) * (S / n)
            S_remain = max(0.0, S - exp_cons)
            growth = 2.0 * S_remain * (1.0 - S_remain / capacity)
            S_next = min(S_remain + growth, capacity)
            if S_next >= self._safety_margin:
                return Action.C
            return Action.D
        denom = max(1, n - 1)
        p_hat = coop_obs / denom
        opp_cons_if_coop = p_hat * (S / (2.0 * n)) + (1.0 - p_hat) * (S / n)
        coop_cons = opp_cons_if_coop * (n - 1) + S / (2.0 * n)
        defect_cons = opp_cons_if_coop * (n - 1) + S / n
        S_rem_coop = max(0.0, S - coop_cons)
        S_rem_def = max(0.0, S - defect_cons)
        growth_coop = 2.0 * S_rem_coop * (1.0 - S_rem_coop / capacity)
        growth_def = 2.0 * S_rem_def * (1.0 - S_rem_def / capacity)
        S_next_coop = min(S_rem_coop + growth_coop, capacity)
        S_next_def = min(S_rem_def + growth_def, capacity)
        payoff_coop = S / (2.0 * n)
        payoff_def = S / n
        payoff_ok = payoff_coop >= payoff_def * (1.0 - self.PAYOFF_TOLERANCE)
        if S_next_coop >= self._safety_margin and payoff_ok:
            return Action.C
        return Action.D

class Strategy_INDIVIDUALISTIC_12(BaseStrategy):
    """
    Adaptive Stock‑Threshold with Conditional Cooperation (ASTCC).

    The agent cooperates when the common pool stock is sufficiently high
    and the observed cooperation rate among opponents in the previous
    round meets a majority threshold. It always defects in the final round
    and otherwise follows the adaptive rule described in the specification.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.target = 0.6 * self.capacity
        self.min_stock_coop = 0.3 * self.capacity
        self.coop_threshold = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            if current_stock >= self.target:
                return Action.C
            else:
                return Action.D
        last_opponents_coop = int(history.opponent_cooperators[-1])
        coop_rate = last_opponents_coop / (self.n - 1)
        env_coop = coop_rate >= self.coop_threshold
        stock_ok = current_stock >= self.target and current_stock >= self.min_stock_coop
        if env_coop and stock_ok:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_13(BaseStrategy):
    """
    Adaptive Threshold Cooperation (ATC).

    The player estimates the opponents' cooperation probability from the
    most recent round, projects the deterministic evolution of the common
    pool under the two possible current actions (cooperate or defect),
    and computes the expected cumulative payoff over the remaining
    horizon assuming the chosen action is repeated.  The action with the
    higher projected payoff is taken.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.total_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        capacity = self.capacity
        t = history.round_number
        R = self.total_rounds - t
        if current_stock <= 0:
            return Action.D
        if R <= 0:
            return Action.D
        if t == 0:
            p_hat = 0.5
        else:
            k = int(history.opponent_cooperators[-1])
            p_hat = k / (n - 1) if n > 1 else 0.0
        c_C = current_stock / (2 * n) if n else 0.0
        c_D = current_stock / n if n else 0.0
        if R == 1:
            return Action.D

        def expected_cumulative(start_stock: float, cooperate: bool) -> float:
            stock = start_stock
            total_payoff = 0.0
            for _ in range(R):
                if cooperate:
                    payoff = stock / (2 * n) if n else 0.0
                    my_consume = payoff
                else:
                    payoff = stock / n if n else 0.0
                    my_consume = payoff
                total_payoff += payoff
                exp_opp_consume = (n - 1) * (p_hat * (stock / (2 * n) if n else 0.0) + (1 - p_hat) * (stock / n if n else 0.0))
                stock_opp = stock - exp_opp_consume
                stock_after = stock_opp - my_consume
                stock_after = max(stock_after, 0.0)
                growth = 2 * stock_after * (1 - stock_after / capacity) if capacity else 0.0
                stock = min(stock_after + growth, capacity)
            return total_payoff
        U_C = expected_cumulative(current_stock, cooperate=True)
        U_D = expected_cumulative(current_stock, cooperate=False)
        return Action.C if U_C >= U_D else Action.D

class Strategy_INDIVIDUALISTIC_14(BaseStrategy):
    """
    Dynamic Threshold‑Defect strategy for the Common Pool Resource game.

    The player cooperates when the stock is sufficiently high and recent
    opponent behavior is cooperative, or when the marginal future benefit
    of preserving stock (Δ) is positive. The player always defects in the
    final round and when the stock falls below a dynamically decreasing
    threshold.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        if current_stock <= 0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            k = n - 1
        else:
            k = int(history.opponent_cooperators[-1])
        S = current_stock
        c = S / (2 * n) if n else 0.0
        d = S / n if n else 0.0
        S_remain_C = S - n * c
        growth_C = 2 * S_remain_C * (1 - S_remain_C / capacity) if capacity else 0.0
        S_plus_C = min(S_remain_C + growth_C, capacity)
        S_remain_D = S - n * d
        growth_D = 2 * S_remain_D * (1 - S_remain_D / capacity) if capacity else 0.0
        S_plus_D = min(S_remain_D + growth_D, capacity)
        Delta = (S_plus_C - S_plus_D) / n if n else 0.0
        R = r - t - 1
        tau_t = capacity * (0.5 + 0.3 * (R / r))
        if S >= tau_t and k >= math.floor(0.6 * (n - 1)):
            return Action.C
        elif Delta > 0 and S >= tau_t * 0.8:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_15(BaseStrategy):
    """
    Self‑Preserving Conditional Cooperation.
    Cooperates when the stock is ample and opponents have shown enough
    cooperation, otherwise defects. Uses a one‑step look‑ahead to weigh
    immediate vs. future payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.theta1 = 0.6 * self.capacity
        self.theta2 = 0.4 * self.capacity
        self.gamma = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        cap = self.capacity
        S = float(current_stock)
        if history.round_number == 0:
            return Action.C if S >= self.theta1 else Action.D
        if history.round_number == self.game_description.n_rounds - 1:
            return Action.D
        k = int(history.opponent_cooperators[-1])
        coop_share = S / (2.0 * n)
        defect_share = S / n
        opp_coop = k
        opp_def = n - 1 - k
        opp_consumption = opp_coop * coop_share + opp_def * defect_share
        total_if_C = opp_consumption + coop_share
        total_if_D = opp_consumption + defect_share
        S_rem_C = max(S - total_if_C, 0.0)
        S_rem_D = max(S - total_if_D, 0.0)

        def next_stock(S_rem: float) -> float:
            growth = 2.0 * S_rem * (1.0 - S_rem / cap)
            nxt = S_rem + growth
            return cap if nxt > cap else nxt
        stock_after_C = next_stock(S_rem_C)
        stock_after_D = next_stock(S_rem_D)
        future_value_C = stock_after_C / cap * coop_share
        future_value_D = stock_after_D / cap * defect_share
        if S >= self.theta1 and coop_share + future_value_C > (defect_share + future_value_D) * self.gamma:
            return Action.C
        if S <= self.theta2:
            return Action.D
        if k >= self.gamma * (n - 1):
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_16(BaseStrategy):
    """
    Dynamic Threshold Strategy (DTS) for the common‑pool resource game.

    The strategy maintains a belief `p̂` about the probability that any given
    opponent will cooperate.  Each round it evaluates the expected value of
    cooperating versus defecting, taking into account:
      * the immediate payoff,
      * the predicted stock for the next round under each own action,
      * a simple estimate of future payoffs weighted by the expected next‑round
        stock level.

    It cooperates when the expected value of cooperating is at least as good as
    defecting (except in the final round, where it always defects).  The belief
    is updated each round with a smoothing factor λ = 0.7 based on the observed
    number of cooperating opponents in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_hat = 0.5
        self._lambda = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t > 0:
            coop_obs = int(history.opponent_cooperators[-1])
            if n > 1:
                observed_frac = coop_obs / (n - 1)
            else:
                observed_frac = 0.0
            self.p_hat = self._lambda * self.p_hat + (1.0 - self._lambda) * observed_frac
        if current_stock <= 0.0:
            self.p_hat = 0.0
            return Action.D
        payoff_C = current_stock / (2.0 * n) if n else 0.0
        payoff_D = current_stock / float(n) if n else 0.0
        exp_coops = (n - 1) * self.p_hat
        per_coop = current_stock / (2.0 * n) if n else 0.0
        per_def = current_stock / float(n) if n else 0.0
        cons_if_C = payoff_C + exp_coops * per_coop + (n - 1 - exp_coops) * per_def
        cons_if_D = payoff_D + exp_coops * per_coop + (n - 1 - exp_coops) * per_def
        rem_C = max(current_stock - cons_if_C, 0.0)
        rem_D = max(current_stock - cons_if_D, 0.0)
        next_C = rem_C + 2.0 * rem_C * (1.0 - rem_C / capacity)
        next_D = rem_D + 2.0 * rem_D * (1.0 - rem_D / capacity)
        next_C = min(next_C, capacity)
        next_D = min(next_D, capacity)
        remaining_rounds = r - t - 1
        future_est = (payoff_C + payoff_D) / 2.0 * remaining_rounds
        EV_C = payoff_C + future_est * (next_C / capacity) if capacity else payoff_C
        EV_D = payoff_D + future_est * (next_D / capacity) if capacity else payoff_D
        if t == r - 1:
            return Action.D
        if EV_C >= EV_D:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_1(BaseStrategy):
    """
    Adaptive Opportunist Strategy for the Common Pool Resource Game.

    The decision logic follows the specification:
    - Defect in the final round.
    - Cooperate when the stock is critically low or when urgency exceeds
      observed cooperation.
    - Defect opportunistically under several conditions (high opponent
      cooperation, low opponent cooperation with abundant stock, or in the
      final quarter of the game).
    - Otherwise cooperate with probability 1‑p (inverse‑mirroring).
    """
    _SIGMA_LOW = 0.2
    _SIGMA_MID = 0.5
    _SIGMA_HIGH = 0.7
    _TAU_LOW = 0.2
    _TAU_HIGH = 0.5
    _RHO = 0.3

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._n_players = game_description.n_players
        self._n_opponents = self._n_players - 1
        self._n_rounds = game_description.n_rounds
        self._capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            The history seen by this player (my actions, payoffs, opponent
            cooperation counts).
        current_stock : float
            The common pool stock level at the start of the round.

        Returns
        -------
        Action
            Action.C (cooperate) or Action.D (defect).
        """
        if current_stock <= 0.0:
            return Action.C
        if history.round_number == 0:
            pass
        t = history.round_number
        R = self._n_rounds - t
        s = current_stock / self._capacity
        k = history.opponent_cooperators[-1] if t > 0 else 0
        p = k / self._n_opponents if self._n_opponents > 0 else 0.0
        rho_R = self._RHO * (R / self._n_rounds)
        if R == 1:
            return Action.D
        if s <= self._SIGMA_LOW:
            return Action.C
        if s <= self._SIGMA_MID and rho_R > p:
            return Action.C
        opportunistic = p >= self._TAU_HIGH and s >= self._SIGMA_MID or (p <= self._TAU_LOW and s >= self._SIGMA_HIGH) or (R <= self._n_rounds / 4 and s > self._SIGMA_LOW)
        if opportunistic:
            return Action.D
        q = 1.0 - p
        if random.random() < q:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_2(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    Cooperates only when the stock is endangered and opponents have shown
    a cooperative tendency, otherwise defects to maximise immediate payoff.
    Implements the additional “sudden‑stock‑drop” and “near‑capacity”
    adjustments described in the specification.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._force_coop_next = False

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        R = r - t
        if t == r - 1:
            return Action.D
        if self._force_coop_next:
            self._force_coop_next = False
            action = Action.C
        else:
            if t == 0:
                avg_coop = 0.5
            else:
                total_coop = int(np.sum(history.opponent_cooperators))
                denom = t * (n - 1)
                avg_coop = total_coop / denom if denom > 0 else 0.5
            low_thresh = capacity * (0.25 + 0.75 * R / r)
            if current_stock <= 0:
                action = Action.D
            elif current_stock > 0.9 * capacity:
                action = Action.D
            elif current_stock < low_thresh and avg_coop > 0.5:
                action = Action.C
            else:
                action = Action.D
        if t > 0:
            prev_R = r - (t - 1)
            prev_low_thresh = capacity * (0.25 + 0.75 * prev_R / r)
            coop_threshold = math.ceil(0.6 * (n - 1))
            if current_stock < prev_low_thresh and history.opponent_cooperators[-1] >= coop_threshold:
                self._force_coop_next = True
        return action

class Strategy_OPPORTUNISTIC_3(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource Game.

    The player defects by default and only cooperates when:
      1. The current stock is above a high‑stock threshold (θ = 0.65·capacity).
      2. The recent empirical cooperation rate of opponents is at least τ = 0.55.
      3. A one‑step‑ahead projection shows higher total expected payoff
         (immediate plus linearly projected future) from cooperating than from defecting.

    The opponent cooperation estimate uses the last k = 3 rounds (or fewer if fewer rounds
    have been played). In the very first round a neutral prior avg_coop = 0.5 is used.
    The last round is always a defection because no future benefit can be harvested.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.n_opponents = self.n - 1
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta = 0.65 * self.capacity
        self.tau = 0.55
        self.k = 3
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == self.r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        if t == 0:
            avg_coop = 0.5
        else:
            start = max(0, t - self.k)
            recent = history.opponent_cooperators[start:t]
            coop_sum = int(np.sum(recent))
            denom = self.n_opponents * (t - start)
            avg_coop = coop_sum / denom if denom > 0 else 0.0
        coop_possible = S >= self.theta and avg_coop >= self.tau
        payoff_C = S / (2.0 * self.n)
        payoff_D = S / float(self.n)
        opp_consump_C = self.n_opponents * avg_coop * payoff_C
        S_rem_C = S - payoff_C - opp_consump_C
        if S_rem_C < 0.0:
            S_rem_C = 0.0
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / self.capacity)
        S_next_C = min(S_rem_C + growth_C, self.capacity)
        opp_consump_D = self.n_opponents * avg_coop * payoff_D
        S_rem_D = S - payoff_D - opp_consump_D
        if S_rem_D < 0.0:
            S_rem_D = 0.0
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / self.capacity)
        S_next_D = min(S_rem_D + growth_D, self.capacity)
        remaining = self.r - t - 1
        exp_future_C = remaining * (S_next_C / (2.0 * self.n))
        exp_future_D = remaining * (S_next_D / float(self.n))
        total_exp_C = payoff_C + exp_future_C
        total_exp_D = payoff_D + exp_future_D
        if coop_possible and total_exp_C > total_exp_D:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_4(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy (OCPS).

    The agent keeps a short‑term estimate of opponents' cooperation rate,
    evaluates the immediate payoff of cooperating vs defecting, and adds a
    discounted estimate of future payoffs based on the projected stock level.
    Cooperation is chosen only when the stock is healthy enough and the
    expected long‑term value of cooperating exceeds that of defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.m = 3
        self.delta = 0.9
        self.tau = 0.4
        self.min_stock_ratio = 0.05

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide Action.C (cooperate) or Action.D (defect) for the current round."""
        t = history.round_number
        n = self.n
        capacity = self.capacity
        remaining = self.r - t - 1
        if current_stock <= 0.0:
            return Action.D
        if current_stock <= self.min_stock_ratio * capacity:
            return Action.D
        if remaining == 0:
            return Action.D
        if t == 0:
            hat_p = 1.0
        else:
            c_prev = int(history.opponent_cooperators[-1])
            start = max(0, t - self.m)
            end = t - 1
            if start < end:
                recent_sum = int(np.sum(history.opponent_cooperators[start:end]))
            else:
                recent_sum = 0
            denom = (self.m + 1) * (n - 1)
            hat_p = (c_prev + recent_sum) / denom if denom != 0 else 0.0
        hat_C = (n - 1) * hat_p
        pi_C = current_stock / (2 * n) if n else 0.0
        pi_D = current_stock / n if n else 0.0
        opp_cons_from_C = current_stock / (2 * n)
        opp_cons_from_D = current_stock / n
        exp_opp_cons = hat_C * opp_cons_from_C + (n - 1 - hat_C) * opp_cons_from_D

        def action_value(is_cooperate: bool) -> float:
            my_cons = pi_C if is_cooperate else pi_D
            total_cons = exp_opp_cons + my_cons
            stock_rem = max(current_stock - total_cons, 0.0)
            growth = 2 * stock_rem * (1 - stock_rem / capacity) if capacity else 0.0
            stock_next = min(stock_rem + growth, capacity)
            per_round = stock_next / (2 * n) if is_cooperate else stock_next / n
            future_val = remaining * per_round
            immediate = pi_C if is_cooperate else pi_D
            return immediate + self.delta * future_val
        V_C = action_value(True)
        V_D = action_value(False)
        if current_stock > self.tau * capacity and V_C > V_D:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_5(BaseStrategy):
    """
    Opportunistic Adaptive Strategy (OAS).

    The agent mostly values immediate payoff (weight w ≥ 0.85) and only
    cooperates when recent opponent cooperation is high and the resource
    stock is abundant enough that a one‑step‑ahead forecast favors cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:

        def growth(x: float) -> float:
            return 2.0 * x * (1.0 - x / self.capacity)
        if current_stock <= 0.0:
            return Action.D
        if history.round_number == 0:
            return Action.D
        t = history.round_number
        c_opp = int(history.opponent_cooperators[-1])
        payoff_C = current_stock / (2.0 * self.n)
        payoff_D = current_stock / self.n
        total_cons_C = (c_opp + 1) * payoff_C + (self.n - 1 - c_opp) * payoff_D
        total_cons_D = c_opp * payoff_C + (self.n - c_opp) * payoff_D
        S_rem_C = max(0.0, current_stock - total_cons_C)
        S_rem_D = max(0.0, current_stock - total_cons_D)
        S_next_C = min(S_rem_C + growth(S_rem_C), self.capacity)
        S_next_D = min(S_rem_D + growth(S_rem_D), self.capacity)
        exp_future_C = S_next_C / (2.0 * self.n)
        exp_future_D = S_next_D / self.n
        if self.r > 1:
            w = 0.85 + 0.1 * (t / (self.r - 1))
        else:
            w = 0.95
        w = max(0.0, min(1.0, w))
        utility_C = w * payoff_C + (1.0 - w) * exp_future_C
        utility_D = w * payoff_D + (1.0 - w) * exp_future_D
        if utility_D > utility_C:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_6(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource Game.
    The player balances immediate consumption against a discounted
    expectation of future payoff, using the empirical cooperation
    rate of opponents observed so far.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.delta = 0.9

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            p = 0.5
        else:
            c_hist = int(history.opponent_cooperators.sum())
            denom = t * (self.n - 1)
            p = c_hist / denom if denom > 0 else 0.5
        payoff_C = S / (2 * self.n)
        payoff_D = S / self.n
        opp_consume_if_C = S / (2 * self.n)
        opp_consume_if_D = S / self.n
        exp_opp_consume = (self.n - 1) * (p * opp_consume_if_C + (1 - p) * opp_consume_if_D)

        def forecast_stock(my_consume: float) -> float:
            total_consume = my_consume + exp_opp_consume
            remaining_stock = S - total_consume
            if remaining_stock < 0:
                remaining_stock = 0.0
            growth = 2.0 * remaining_stock * (1.0 - remaining_stock / self.capacity)
            new_stock = remaining_stock + growth
            if new_stock > self.capacity:
                new_stock = self.capacity
            return new_stock
        S_next_C = forecast_stock(payoff_C)
        S_next_D = forecast_stock(payoff_D)
        exp_future_C = self.delta * (p * (S_next_C / (2 * self.n)) + (1 - p) * (S_next_C / self.n))
        exp_future_D = self.delta * (p * (S_next_D / (2 * self.n)) + (1 - p) * (S_next_D / self.n))
        total_C = payoff_C + exp_future_C
        total_D = payoff_D + exp_future_D
        return Action.C if total_C > total_D else Action.D

class Strategy_OPPORTUNISTIC_7(BaseStrategy):
    """
    Opportunistic Conditional Defection (OCD).

    The player evaluates the current stock level and the observed cooperation of opponents
    in the previous round.  When the pool is abundant and many opponents previously
    cooperated, it defects to capture a larger share.  When the pool is scarce and
    cooperation was low, it cooperates to help preserve future stock.  In intermediate
    situations it compares the immediate gain from defecting with an estimate of the
    future stock loss caused by that defect and chooses the action that maximizes
    expected cumulative payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.high_stock = 0.75 * self.capacity
        self.low_stock = 0.3 * self.capacity
        self.coop_ratio = 0.6
        self.initial_cprev = math.floor(0.5 * (self.n - 1))

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        S = min(current_stock, self.capacity)
        if S == 0.0:
            return Action.C
        t = history.round_number
        T = self.r - 1
        if t == 0:
            cPrev = self.initial_cprev
        else:
            cPrev = int(history.opponent_cooperators[-1])
        if t == T:
            return Action.D
        if S >= self.high_stock and cPrev / (self.n - 1) >= self.coop_ratio:
            return Action.D
        if S <= self.low_stock and cPrev / (self.n - 1) < self.coop_ratio:
            return Action.C
        marginal_gain = S / (2 * self.n)
        S_after_defect = S - S / self.n
        growth = 2 * S_after_defect * (1 - S_after_defect / self.capacity)
        new_stock = min(S_after_defect + growth, self.capacity)
        expected_loss = new_stock - S
        if expected_loss < 0.5 * marginal_gain:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_8(BaseStrategy):
    """
    Opportunistic Conditional Defector.

    The agent estimates the cooperation probability of opponents from the
    previous round, evaluates the immediate payoff of cooperating vs defecting,
    projects the resulting stock for the next round, and adds the expected
    payoff in that next round.  The action with the larger summed value is
    chosen; ties are resolved aggressively in favour of defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        r = self.game_description.n_rounds
        t = history.round_number
        S = float(current_stock)
        if t == r - 1:
            return Action.D
        if t == 0:
            p = 0.5
        else:
            k = int(history.opponent_cooperators[-1])
            p = k / (n - 1) if n > 1 else 0.5
        payoff_C = S / (2.0 * n) if n else 0.0
        payoff_D = S / float(n) if n else 0.0
        exp_cons_opponent = p * payoff_C + (1.0 - p) * payoff_D
        total_cons_C = payoff_C + (n - 1) * exp_cons_opponent
        total_cons_D = payoff_D + (n - 1) * exp_cons_opponent
        S_rem_C = S - total_cons_C
        S_rem_D = S - total_cons_D
        if S_rem_C < 0.0:
            S_rem_C = 0.0
        if S_rem_D < 0.0:
            S_rem_D = 0.0

        def growth(stock_rem: float) -> float:
            if capacity == 0.0:
                return 0.0
            return 2.0 * stock_rem * (1.0 - stock_rem / capacity)
        S_next_C = S_rem_C + growth(S_rem_C)
        S_next_D = S_rem_D + growth(S_rem_D)
        if S_next_C > capacity:
            S_next_C = capacity
        if S_next_D > capacity:
            S_next_D = capacity
        next_payoff_C = S_next_C / (2.0 * n) if n else 0.0
        next_payoff_D = S_next_D / float(n) if n else 0.0
        value_C = payoff_C + next_payoff_C
        value_D = payoff_D + next_payoff_D
        if value_D > value_C:
            return Action.D
        elif value_C > value_D:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_9(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    The player defects in almost all situations, cooperating only when the
    common stock is low *and* the observed cooperation rate among opponents in the
    previous round is at least 30\u202f%.  In the final round the player always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._S_high = 0.75 * game_description.capacity
        self._S_mid = 0.4 * game_description.capacity
        self._p_coop_min = 0.3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.game_description.n_rounds - 1:
            return Action.D
        if t == 0:
            coop_rate = 0.5
        else:
            coop_prev = int(history.opponent_cooperators[-1])
            denom = self.game_description.n_players - 1
            coop_rate = coop_prev / denom if denom != 0 else 0.0
        if current_stock >= self._S_high:
            return Action.D
        elif current_stock >= self._S_mid:
            return Action.D
        elif coop_rate >= self._p_coop_min:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_10(BaseStrategy):
    """
    Opportunistic Conditional Defection (OCD).
    Estimates opponent cooperation, exploits high-stock situations,
    and cooperates when stock is low and future rounds remain.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta_coop = 0.4 * self.game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        t = history.round_number
        R = self.game_description.n_rounds - t - 1
        if current_stock <= 0.0:
            return Action.D
        if t == 0:
            p_hat = 0.5
        else:
            k_t = int(history.opponent_cooperators[-1])
            p_hat = k_t / (n - 1) if n > 1 else 0.0
        c_C = current_stock / (2 * n)
        c_D = current_stock / n
        exp_coop = (n - 1) * p_hat
        exp_def = (n - 1) * (1.0 - p_hat)

        def projected_stock(own_consumption: float) -> float:
            total_consumption = exp_coop * (current_stock / (2 * n)) + exp_def * (current_stock / n) + own_consumption
            remaining = current_stock - total_consumption
            remaining = max(remaining, 0.0)
            growth = 2 * remaining * (1.0 - remaining / capacity)
            next_stock = remaining + growth
            return min(next_stock, capacity)
        S_next_C = projected_stock(c_C)
        S_next_D = projected_stock(c_D)
        delta_margin = 0.05 * current_stock / n
        if R == 0:
            return Action.D
        if current_stock >= self.theta_coop:
            payoff_diff = c_D - c_C
            if payoff_diff >= delta_margin and (S_next_D >= self.theta_coop or R <= 2):
                return Action.D
            else:
                return Action.C
        elif R >= 3 and p_hat > 0.6:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_11(BaseStrategy):
    """
    Greedy Stock‑Preserving Defector.

    At each round the agent compares the expected total payoff of cooperating
    versus defecting, taking into account the impact on the common‑pool stock.
    The future value of the stock is approximated linearly: the remaining
    rounds are valued as if the player would cooperate forever from then on.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        capacity = self.capacity
        r_total = self.r
        t = history.round_number
        if t == r_total - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            k = n - 1
        else:
            k = int(history.opponent_cooperators[-1])
        r_rem = r_total - t - 1

        def next_stock(after_consumption: float) -> float:
            s_rem = current_stock - after_consumption
            if s_rem < 0.0:
                s_rem = 0.0
            grow = 2.0 * s_rem * (1.0 - s_rem / capacity) if capacity != 0 else 0.0
            s_next = s_rem + grow
            if s_next > capacity:
                s_next = capacity
            return s_next
        cons_C = k * (current_stock / (2.0 * n)) + (n - 1 - k) * (current_stock / n) + current_stock / (2.0 * n)
        s_next_C = next_stock(cons_C)
        cons_D = k * (current_stock / (2.0 * n)) + (n - 1 - k) * (current_stock / n) + current_stock / n
        s_next_D = next_stock(cons_D)

        def future_value(stock: float) -> float:
            return r_rem * (stock / (2.0 * n))
        payoff_C = current_stock / (2.0 * n) + future_value(s_next_C)
        payoff_D = current_stock / n + future_value(s_next_D)
        if payoff_C > payoff_D:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_12(BaseStrategy):
    """
    Opportunist‑CPR Strategy:
    Cooperates only when the projected long‑run benefit of preserving the
    common pool outweighs the immediate gain from defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._p0 = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            p_est = self._p0
        else:
            k_prev = int(history.opponent_cooperators[-1])
            p_est = k_prev / (n - 1) if n > 1 else 0.0

        def growth(x: float) -> float:
            """Growth function g(x) = 2 * x * (1 - x / capacity)."""
            return 2.0 * x * (1.0 - x / capacity)

        def expected_next_stock(action: Action, S: float) -> float:
            """Expected stock after the current round given our action."""
            if action is Action.C:
                self_cons = S / (2.0 * n)
            else:
                self_cons = S / n
            opp_cons = (n - 1) * (p_est * (S / (2.0 * n)) + (1.0 - p_est) * (S / n))
            total_cons = opp_cons + self_cons
            S_rem = S - total_cons
            S_next = S_rem + growth(S_rem)
            return S_next if S_next <= capacity else capacity

        def projected_value(initial_action: Action, S: float) -> float:
            """Cumulative expected payoff from now until the end of the game
            assuming we keep the same action after the first round."""
            horizon = r - t - 1
            total = 0.0
            S_tmp = S
            if initial_action is Action.C:
                total += S_tmp / (2.0 * n)
                S_tmp = expected_next_stock(Action.C, S_tmp)
            else:
                total += S_tmp / n
                S_tmp = expected_next_stock(Action.D, S_tmp)
            for _ in range(horizon):
                if initial_action is Action.C:
                    total += S_tmp / (2.0 * n)
                else:
                    total += S_tmp / n
                avg_cons = p_est * (S_tmp / (2.0 * n)) + (1.0 - p_est) * (S_tmp / n)
                S_next = S_tmp - avg_cons + growth(S_tmp - avg_cons)
                S_tmp = S_next if S_next <= capacity else capacity
            return total
        V_C = projected_value(Action.C, current_stock)
        V_D = projected_value(Action.D, current_stock)
        return Action.C if V_C >= V_D else Action.D

class Strategy_OPPORTUNISTIC_13(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource Game.

    The player cooperates when the stock is sufficiently high, opponents have shown
    enough cooperation, and preserving the stock promises a higher future payoff.
    Otherwise it defects, especially in the last round or when the stock is low.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game = game_description
        self.n = self.game.n_players
        self.capacity = float(self.game.capacity)
        self.theta1 = 0.6 * self.capacity
        self.theta2 = 0.3 * self.capacity
        self.tau = 0.45

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        S = float(current_stock)
        if t == self.game.n_rounds - 1:
            return Action.D
        if S <= self.theta2:
            return Action.D
        c = int(history.opponent_cooperators[-1]) if t > 0 else 0
        if self.n > 1:
            p_hat = c / (self.n - 1)
        else:
            p_hat = 0.0
        coop_consume = S / (2 * self.n)
        defect_consume = S / self.n
        total_consume_coop = (c + 1) * coop_consume + (self.n - 1 - c) * defect_consume
        S_after_coop = max(S - total_consume_coop, 0.0)
        growth_coop = 2.0 * S_after_coop * (1.0 - S_after_coop / self.capacity)
        S_next_coop = min(S_after_coop + growth_coop, self.capacity)
        total_consume_def = c * coop_consume + (self.n - c) * defect_consume
        S_after_def = max(S - total_consume_def, 0.0)
        growth_def = 2.0 * S_after_def * (1.0 - S_after_def / self.capacity)
        S_next_def = min(S_after_def + growth_def, self.capacity)
        immediate_gain = defect_consume - coop_consume
        future_gain = S_next_coop - S_next_def
        if S > self.theta1 and p_hat >= self.tau and (future_gain >= 0.0):
            return Action.C
        if immediate_gain >= future_gain:
            return Action.D
        return Action.C

class Strategy_OPPORTUNISTIC_14(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource game.

    The player defects by default and only cooperates when:
      * the stock is at least 40\u202f% of capacity,
      * the observed fraction of cooperating opponents in the previous round
        is at least 0.6 (or assumed 0.5 in the first round), and
      * the discounted expected value of cooperating exceeds that of defecting.

    The expected value combines the immediate payoff with a discounted
    projection of the next‑round stock level.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.stock_th = 0.4 * float(game_description.capacity)
        self.coop_frac_th = 0.6
        self.discount = 0.9

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return Action.C or Action.D for the current round."""
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        S = float(current_stock)
        if S == 0.0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            obs_frac = 0.5
        else:
            c_prev = int(history.opponent_cooperators[-1])
            obs_frac = c_prev / (n - 1)

        def projected_next_stock(cons_self: float) -> float:
            """
            Compute the projected stock after this round assuming opponents
            behave according to `obs_frac`.
            """
            opp_cons_per = obs_frac * (S / (2 * n)) + (1.0 - obs_frac) * (S / n)
            opp_total = (n - 1) * opp_cons_per
            total_consumption = cons_self + opp_total
            S_rem = max(0.0, S - total_consumption)
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity)
            return min(S_rem + growth, capacity)
        cons_self_C = S / (2.0 * n)
        S_next_C = projected_next_stock(cons_self_C)
        payoff_C = cons_self_C
        value_C = payoff_C + self.discount * S_next_C
        cons_self_D = S / float(n)
        S_next_D = projected_next_stock(cons_self_D)
        payoff_D = cons_self_D
        value_D = payoff_D + self.discount * S_next_D
        if S >= self.stock_th and obs_frac >= self.coop_frac_th and (value_C > value_D):
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_15(BaseStrategy):
    """
    Adaptive probe‑then‑cooperate/defect strategy for the Common Pool Resource game.

    - First PROBE_ROUNDS rounds (up to 3) are spent defecting to gather data.
    - In the final round we always defect to maximise immediate payoff.
    - Afterwards we estimate opponents' cooperation probability from the observed
      counts of cooperating opponents and decide between C and D by comparing
      expected total payoff, while enforcing a safety guard that avoids dropping
      the stock below MIN_SURVIVAL (10\u202f% of capacity).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.PROBE_ROUNDS = min(3, self.r - 1)
        self.MIN_SURVIVAL = 0.1 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action (C or D) for the current round."""
        t = history.round_number
        n = self.n
        r = self.r
        capacity = self.capacity
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        if t < self.PROBE_ROUNDS:
            return Action.D
        total_observed = int(np.sum(history.opponent_cooperators))
        denom = (n - 1) * t
        p = total_observed / denom if denom != 0 else 0.0
        coop_consume = current_stock / (2 * n)
        defect_consume = current_stock / n
        opp_consume_exp = p * (current_stock / (2 * n)) + (1.0 - p) * (current_stock / n)
        total_consump_coop = coop_consume + (n - 1) * opp_consume_exp
        total_consump_defect = defect_consume + (n - 1) * opp_consume_exp
        S_after_coop = max(0.0, current_stock - total_consump_coop)
        S_after_defect = max(0.0, current_stock - total_consump_defect)

        def grow(x: float) -> float:
            return min(x + 2 * x * (1 - x / capacity), capacity)
        S_next_coop = grow(S_after_coop)
        S_next_defect = grow(S_after_defect)
        horizon = r - t - 1
        exp_future_coop = horizon * (S_next_coop / (2 * n))
        exp_future_defect = horizon * (S_next_defect / n)
        payoff_coop = current_stock / (2 * n) + exp_future_coop
        payoff_defect = current_stock / n + exp_future_defect
        if S_next_defect < self.MIN_SURVIVAL and S_next_coop >= self.MIN_SURVIVAL:
            return Action.C
        if payoff_defect >= payoff_coop:
            return Action.D
        return Action.C

class Strategy_OPPORTUNISTIC_16(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy (OCPS).

    The agent defects by default, but may cooperate when:
      * the resource stock is above a configurable fraction τ of capacity,
      * a sufficient proportion θ of opponents cooperated in the previous round,
      * and a one‑step look‑ahead value that mixes immediate payoff with a
        λ‑weighted forecast of the next‑round stock does not favour defecting
        by more than a bias β.
    The final round is always a defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.tau = 0.55
        self.theta = 0.45
        self.lamb = 0.4
        self.beta = 0.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t == self.game_description.n_rounds - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.D
        if t == 0:
            k = 0
        else:
            k = int(history.opponent_cooperators[-1])
        coop_payoff = current_stock / (2.0 * n)
        defect_payoff = current_stock / n
        opp_coop = k * coop_payoff
        opp_def = (n - 1 - k) * defect_payoff
        total_cons_coop = opp_coop + opp_def + coop_payoff
        total_cons_defect = opp_coop + opp_def + defect_payoff
        S_rem_coop = max(current_stock - total_cons_coop, 0.0)
        S_rem_defect = max(current_stock - total_cons_defect, 0.0)

        def growth(S_rem: float) -> float:
            if capacity == 0.0:
                return 0.0
            return 2.0 * S_rem * (1.0 - S_rem / capacity)
        growth_coop = growth(S_rem_coop)
        growth_defect = growth(S_rem_defect)
        S_next_coop = min(S_rem_coop + growth_coop, capacity)
        S_next_defect = min(S_rem_defect + growth_defect, capacity)
        value_coop = coop_payoff + self.lamb * S_next_coop
        value_defect = defect_payoff + self.lamb * S_next_defect
        if value_defect - value_coop > self.beta:
            return Action.D
        stock_ok = current_stock / capacity >= self.tau if capacity > 0 else False
        coop_ratio_ok = k / (n - 1) >= self.theta if n - 1 > 0 else False
        if stock_ok and coop_ratio_ok:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_1(BaseStrategy):
    """
    Adaptive Forward‑Looking Cooperation (AFLC).

    The agent decides to cooperate or defect based on:
    - The remaining number of rounds vs. a cooperation horizon (2\u202f×\u202fn).
    - A safety stock target that linearly decreases as the game proceeds.
    - The observed cooperation rate of opponents.
    - A forced defection in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            Contains the past actions, payoffs and opponent cooperation counts.
        current_stock : float
            The common pool stock at the start of the current round.

        Returns
        -------
        Action
            Action.C for cooperation, Action.D for defection.
        """
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            theta = self.capacity * (self.r / self.r)
            cooperate = self.capacity >= theta
            return Action.C if cooperate else Action.D
        R = self.r - t
        H = 2 * self.n
        theta = self.capacity * (R / self.r)
        cooperate = R > H and current_stock >= theta
        total_opponent_coops = int(history.opponent_cooperators.sum())
        denominator = t * (self.n - 1)
        p_hat = 0.5 if denominator == 0 else total_opponent_coops / denominator
        if not cooperate:
            if p_hat >= 0.5 and current_stock >= 0.2 * self.capacity:
                cooperate = True
        return Action.C if cooperate else Action.D

class Strategy_SELFINTERESTED_2(BaseStrategy):
    """
    Self‑Interested Adaptive (SIA) strategy.

    The agent compares the immediate gain from defecting with the
    expected loss in future rounds that results from a lower stock level.
    It defects in the final round, otherwise only when the extra
    immediate profit outweighs the projected future loss and the stock
    is safely above a dynamic threshold.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        S = min(max(current_stock, 0.0), capacity)
        if S == 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        if t == 0:
            prev_coops = n
        else:
            my_last_coop = bool(history.my_actions[-1])
            prev_coops = int(history.opponent_cooperators[-1]) + (1 if my_last_coop else 0)
        remaining_rounds = r - t
        base_coop_share = S / (2.0 * n)
        base_def_share = S / float(n)
        expected_total_consumption = prev_coops * S / (2.0 * n) + (n - prev_coops) * S / float(n)
        S_prime = S - expected_total_consumption
        growth = 2.0 * S_prime * (1.0 - S_prime / capacity)
        S_next = min(S_prime + growth, capacity)
        stock_ratio = S_next / S
        future_payoff_if_coop = remaining_rounds * base_coop_share * stock_ratio
        future_payoff_if_def = remaining_rounds * base_def_share * stock_ratio
        immediate_gain = base_def_share - base_coop_share
        projected_future_loss = future_payoff_if_coop - future_payoff_if_def
        safety_threshold = capacity * (1.0 - 0.5 / float(remaining_rounds))
        if immediate_gain > projected_future_loss and S > safety_threshold:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_3(BaseStrategy):
    """
    Future‑Value Threshold (FVT) strategy for the common‑pool resource game.
    Estimates opponent cooperation from the previous round, projects the stock
    after cooperating or defecting, and adds a future‑value term proportional
    to the remaining rounds. The action with the higher total estimated value
    is taken.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t: int = history.round_number
        if t == 0:
            p_hat: float = 0.5
        else:
            k: int = int(history.opponent_cooperators[-1])
            p_hat = k / (self.n_players - 1)
        n: int = self.n_players
        S: float = current_stock
        cC: float = S / (2 * n) if n > 0 else 0.0
        cD: float = S / n if n > 0 else 0.0
        exp_opp_cons_per: float = p_hat * (S / (2 * n)) + (1 - p_hat) * (S / n)
        E_opp_cons: float = (n - 1) * exp_opp_cons_per

        def project_stock(own_cons: float) -> float:
            S_rem: float = S - (own_cons + E_opp_cons)
            S_rem = max(min(S_rem, self.capacity), 0.0)
            growth: float = 2 * S_rem * (1 - S_rem / self.capacity) if self.capacity != 0 else 0.0
            growth = max(growth, 0.0)
            S_next: float = S_rem + growth
            S_next = max(min(S_next, self.capacity), 0.0)
            return S_next
        S_next_C: float = project_stock(cC)
        S_next_D: float = project_stock(cD)
        R: int = self.n_rounds - t - 1
        FV: float = R * (1 / (2 * n)) if n > 0 else 0.0
        value_cooperate: float = cC + FV * (S_next_C - S)
        value_defect: float = cD + FV * (S_next_D - S)
        if value_cooperate >= value_defect:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_4(BaseStrategy):
    """
    Self‑Interested Adaptive (SIA‑CP) strategy for the Common Pool Resource game.

    Decision logic:
      • Last round → always defect.
      • Stock == 0 → defect (payoff is zero regardless).
      • Stock ≤ 0.5\u202f×\u202fcapacity → cooperate.
      • Otherwise (stock > 0.5\u202f×\u202fcapacity) → defect, with two optional overrides
        based on the previous round’s cooperation count:
          – If all opponents cooperated previously and stock ≤ 0.2\u202f×\u202fcapacity → cooperate.
          – If no opponent cooperated previously and stock > 0.7\u202f×\u202fcapacity → defect.
      • In any other situation the default action is defect.
    The rule uses only public information: round index, current stock,
    and the number of cooperating opponents in the last round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        if t == self.n_rounds - 1:
            return Action.D
        half_capacity = 0.5 * self.capacity
        if current_stock <= half_capacity:
            return Action.C
        if t > 0:
            k_prev = int(history.opponent_cooperators[-1])
            if k_prev == self.n_players - 1 and current_stock <= 0.2 * self.capacity:
                return Action.C
            if k_prev == 0 and current_stock > 0.7 * self.capacity:
                return Action.D
        return Action.D

class Strategy_SELFINTERESTED_5(BaseStrategy):
    """
    Immediate-payoff greedy strategy for the Common Pool Resource game.
    It maintains a belief `p_hat` about opponent cooperation, updates it each round,
    but chooses actions solely based on the higher immediate consumption:
        - Defect in the last round.
        - Defect whenever the stock is positive (defect always yields higher immediate payoff).
        - Defect if the stock is depleted.
    The belief is kept for completeness and possible extensions.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)
        self.p_hat: float = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t: int = history.round_number
        if t > 0 and self.n_players > 1:
            c: int = int(history.opponent_cooperators[-1])
            frac: float = c / (self.n_players - 1)
            self.p_hat = ((t - 1) * self.p_hat + frac) / t
        if t >= self.n_rounds - 1:
            return Action.D
        if current_stock <= 0:
            return Action.D
        payoff_C: float = current_stock / (2 * self.n_players) if self.n_players > 0 else 0.0
        payoff_D: float = current_stock / self.n_players if self.n_players > 0 else 0.0
        if payoff_D > payoff_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_6(BaseStrategy):
    """
    Dynamic Threshold Self‑Interest Strategy for the Common Pool Resource Game.

    The agent defects in the first and last round. In intermediate rounds it
    cooperates only when the stock is sufficiently high (both above a fixed
    high‑stock threshold and above a dynamic threshold that declines as the
    horizon shortens) **and** when at least half of the opponents cooperated in
    the previous round. Otherwise it defects.
    """
    THRESH_HIGH_FACTOR = 0.6
    THRESH_LOW_FACTOR = 0.2
    COOP_FRACTION = 0.5
    ALPHA = 0.5
    BETA = 0.1

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        cap = float(game_description.capacity)
        self.thresh_high = self.THRESH_HIGH_FACTOR * cap
        self.thresh_low = self.THRESH_LOW_FACTOR * cap
        self.n_opponents = max(game_description.n_players - 1, 1)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            Immutable record of past actions, payoffs and observed opponent
            cooperation counts.
        current_stock : float
            The common‑pool stock level at the beginning of this round.

        Returns
        -------
        Action
            Action.C to cooperate, Action.D to defect.
        """
        t = history.round_number
        r = self.game_description.n_rounds
        if t == 0:
            return Action.D
        if t == r - 1:
            return Action.D
        S = float(current_stock)
        if S <= self.thresh_low:
            return Action.D
        dynamic_thresh = self.game_description.capacity * (self.ALPHA * (1.0 - t / (r - 1)) + self.BETA)
        obs_c = int(history.opponent_cooperators[-1])
        coop_share = obs_c / self.n_opponents
        if S >= max(self.thresh_high, dynamic_thresh) and coop_share >= self.COOP_FRACTION:
            return Action.C
        return Action.D

class Strategy_SELFINTERESTED_7(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy (SI‑ADAPT) for the Common Pool Resource Game.
    Uses an exponential moving average of opponent cooperation (trend) to decide
    whether to cooperate, with a defect mode that can be triggered by low
    cooperation and low stock, and forgiveness based on stock recovery and
    improved cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.L = 5
        self.K = 3
        self.alpha = 0.6
        self.tau = 0.4
        self.gamma = 0.7
        self.trend = 1.0
        self.defect_mode = False

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if self.capacity == 0:
            return Action.D
        if history.round_number > 0:
            last_obs = int(history.opponent_cooperators[-1])
            opp_rate = last_obs / (self.n - 1) if self.n > 1 else 0.0
            self.trend = self.alpha * opp_rate + (1.0 - self.alpha) * self.trend

        def avg_coop_last(k: int) -> float:
            if history.round_number == 0:
                return 1.0
            recent = history.opponent_cooperators[-k:] if history.round_number >= k else history.opponent_cooperators
            rates = recent.astype(float) / (self.n - 1) if self.n > 1 else np.zeros_like(recent, dtype=float)
            return float(np.mean(rates))
        recent_K_avg = avg_coop_last(self.K)
        recent_L_avg = avg_coop_last(self.L)
        stock_prop = current_stock / self.capacity if self.capacity else 0.0
        if not self.defect_mode:
            if history.round_number >= self.K and recent_K_avg < self.tau and (stock_prop <= 0.4):
                self.defect_mode = True
        elif stock_prop > self.gamma and recent_L_avg > 0.6:
            self.defect_mode = False
            self.trend = 1.0
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            return Action.C
        if self.defect_mode:
            return Action.D
        p_star = max(0.3, 0.5 * self.trend)
        if stock_prop >= p_star:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_8(BaseStrategy):
    """
    Dynamic Greedy‑Sustain strategy.

    The agent estimates the cooperation probability of opponents from its
    observed history, projects the common‑pool stock under both possible
    actions, and compares the immediate gain of defecting with the
    expected loss in future payoff (using a simple lower‑bound value
    function).  Defection is chosen only when the present advantage
    outweighs the projected future loss; otherwise the agent cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._cum_cooperators = 0
        self._cum_opponent_obs = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if t == r - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.D
        if current_stock <= 0.05 * capacity:
            return Action.C
        if t == 0:
            p_hat = 0.5
        else:
            total_coop = int(np.sum(history.opponent_cooperators))
            total_obs = t * (n - 1)
            p_hat = total_coop / total_obs if total_obs > 0 else 0.5
        cons_per_coop = current_stock / (2 * n)
        cons_per_def = current_stock / n
        exp_opp_cons = p_hat * cons_per_coop + (1 - p_hat) * exp_opp_cons if False else p_hat * cons_per_coop + (1 - p_hat) * cons_per_def
        total_exp_opp_cons = (n - 1) * exp_opp_cons

        def project_stock(our_cons):
            total_cons = total_exp_opp_cons + our_cons
            remaining = current_stock - total_cons
            if remaining < 0.0:
                remaining = 0.0
            growth = 2.0 * remaining * (1.0 - remaining / capacity)
            projected = remaining + growth
            return min(projected, capacity)
        our_cons_C = current_stock / (2 * n)
        our_cons_D = current_stock / n
        S_prime_C = project_stock(our_cons_C)
        S_prime_D = project_stock(our_cons_D)
        remaining_rounds = r - t - 1

        def V(x):
            return x / (2 * n) * remaining_rounds
        delta_V = V(S_prime_C) - V(S_prime_D)
        pi_C = our_cons_C
        pi_D = our_cons_D
        if pi_D + delta_V >= pi_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_9(BaseStrategy):
    """
    One‑step look‑ahead strategy for the Common Pool Resource game.
    It estimates the opponents' cooperation probability from the previous round,
    projects the next‑round stock for each of its two possible actions (C or D),
    and selects the action that maximizes the sum of the current payoff and the
    expected payoff in the next round. Edge cases (first round, last round,
    exhausted stock) are handled explicitly.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == total_rounds - 1:
            return Action.D if current_stock > 0.0 else Action.C
        if t == 0:
            p = 0.5
        else:
            coop_last = int(history.opponent_cooperators[-1])
            p = coop_last / (n - 1) if n > 1 else 0.0
        cons_C = current_stock / (2.0 * n)
        cons_D = current_stock / float(n)
        opp_cons_if_coop = current_stock / (2.0 * n)
        opp_cons_if_def = current_stock / float(n)
        expected_opp_cons = p * (n - 1) * opp_cons_if_coop + (1.0 - p) * (n - 1) * opp_cons_if_def
        total_cons_C = cons_C + expected_opp_cons
        total_cons_D = cons_D + expected_opp_cons
        S_rem_C = max(0.0, current_stock - total_cons_C)
        S_rem_D = max(0.0, current_stock - total_cons_D)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / capacity) if capacity != 0 else 0.0
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / capacity) if capacity != 0 else 0.0
        S_next_C = min(S_rem_C + growth_C, capacity)
        S_next_D = min(S_rem_D + growth_D, capacity)
        payoff_C = cons_C + S_next_C / (2.0 * n)
        payoff_D = cons_D + S_next_D / float(n)
        if payoff_D >= payoff_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_10(BaseStrategy):
    """
    Adaptive cooperation strategy for the Common Pool Resource game.

    The player estimates opponents' cooperation probability from the previous round,
    projects the stock trajectory for both cooperating and defecting in the current
    round, and evaluates the discounted future payoff using a stock‑ and horizon‑
    scaled weighting factor θ_coop.  The action with the higher weighted sum of
    immediate and projected future payoff is selected.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t >= self.r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        R = self.r - t - 1
        theta = S / self.capacity * (R / (self.r - 1))
        if t == 0:
            p_hat = 0.0
        else:
            opp_coop = int(history.opponent_cooperators[-1])
            p_hat = opp_coop / (self.n - 1) if self.n > 1 else 0.0
        payoff_C = S / (2 * self.n)
        payoff_D = S / self.n
        exp_other_cons = p_hat * (S / (2 * self.n)) + (1.0 - p_hat) * (S / self.n)
        total_other_cons = (self.n - 1) * exp_other_cons
        cons_C = payoff_C + total_other_cons
        cons_D = payoff_D + total_other_cons
        S_rem_C = max(0.0, S - cons_C)
        S_rem_D = max(0.0, S - cons_D)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / self.capacity)
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / self.capacity)
        S_next_C = min(S_rem_C + growth_C, self.capacity)
        S_next_D = min(S_rem_D + growth_D, self.capacity)

        def _future_value(start_stock: float) -> float:
            V = 0.0
            S_sim = start_stock
            for _ in range(R):
                exp_payoff = p_hat * (S_sim / (2 * self.n)) + (1.0 - p_hat) * (S_sim / self.n)
                V += exp_payoff
                total_cons = self.n * exp_payoff
                S_rem = max(0.0, S_sim - total_cons)
                growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
                S_sim = min(S_rem + growth, self.capacity)
            return V
        V_C = _future_value(S_next_C)
        V_D = _future_value(S_next_D)
        value_C = payoff_C + theta * V_C
        value_D = payoff_D + theta * V_D
        return Action.C if value_C >= value_D else Action.D

class Strategy_SELFINTERESTED_11(BaseStrategy):
    """
    Self‑Interested Adaptive Coop/Defect Strategy.

    The player computes the immediate payoff for cooperating (C) and defecting (D)
    given the observed number of opponent cooperators in the previous round.
    It then projects the stock one step ahead, checks a safety threshold L,
    and adds a discounted future value (weight γ) if the projected stock is
    sufficiently high.  The action with the higher total estimated value is
    chosen.  In the final round the future value is ignored, so the player
    always prefers defect (the higher immediate payoff).  Edge cases such as
    zero stock are handled gracefully.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.35
        self.gamma_default = 0.8

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        L = self.alpha * capacity
        remaining = r - t
        gamma = self.gamma_default
        if remaining <= 2:
            gamma = 0.5
        if t == 0:
            k = (n - 1) // 2
        else:
            k = int(history.opponent_cooperators[-1])
        if t == r - 1:
            payoff_coop = current_stock / (2 * n)
            payoff_def = current_stock / n
            return Action.C if payoff_coop >= payoff_def else Action.D

        def evaluate(is_coop: bool) -> float:
            c = k + 1 if is_coop else k
            d = n - c
            cons_coop = current_stock / (2 * n)
            cons_def = current_stock / n
            total_consumption = c * cons_coop + d * cons_def
            S_prime = max(current_stock - total_consumption, 0.0)
            growth = 2 * S_prime * (1 - S_prime / capacity) if capacity != 0 else 0.0
            S_hat = min(S_prime + growth, capacity)
            immediate = cons_coop if is_coop else cons_def
            if S_hat >= L:
                future_payoff = S_hat / n
                future = gamma * future_payoff
            else:
                future = 0.0
            return immediate + future
        value_c = evaluate(True)
        value_d = evaluate(False)
        return Action.C if value_c > value_d else Action.D

class Strategy_SELFINTERESTED_12(BaseStrategy):
    """
    Adaptive Self‑Interest (ASI) strategy.

    The agent balances the immediate payoff from cooperating or defecting
    against a linear estimate of future payoffs based on the projected stock.
    It also includes deterministic edge‑case handling:
      * last round → always defect,
      * two consecutive rounds where all opponents defect → permanent defection lock,
      * three consecutive rounds where all opponents cooperate and the stock is high → cooperate,
      * zero stock → defect.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self._defection_lock = False

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if self._defection_lock:
            return Action.D
        if t >= 2 and history.opponent_cooperators[-1] == 0 and (history.opponent_cooperators[-2] == 0):
            self._defection_lock = True
            return Action.D
        if t >= 3 and all((c == self.n - 1 for c in history.opponent_cooperators[-3:])) and (S >= 0.9 * self.capacity):
            return Action.C
        if t == self.r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        n = self.n
        capacity = self.capacity
        remaining_rounds = self.r - t - 1
        coop_cons = S / (2.0 * n)
        defect_cons = S / n
        if t == 0:
            exp_coop = n - 1
        else:
            exp_coop = int(history.opponent_cooperators[-1])
        exp_def = n - 1 - exp_coop

        def projected_stock(my_cons: float) -> float:
            total_cons = my_cons + exp_coop * (S / (2.0 * n)) + exp_def * (S / n)
            S_rem = max(0.0, S - total_cons)
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity)
            return min(S_rem + growth, capacity)
        stock_if_C = projected_stock(coop_cons)
        stock_if_D = projected_stock(defect_cons)
        payoff_C = coop_cons
        payoff_D = defect_cons

        def continuation_value(stock_future: float, my_action_is_C: bool) -> float:
            p_coop = (exp_coop + (1 if my_action_is_C else 0)) / n
            exp_per_round = stock_future * (p_coop / (2.0 * n) + (1.0 - p_coop) / n)
            return remaining_rounds * exp_per_round
        total_C = payoff_C + continuation_value(stock_if_C, True)
        total_D = payoff_D + continuation_value(stock_if_D, False)
        return Action.C if total_C > total_D else Action.D

class Strategy_SELFINTERESTED_13(BaseStrategy):
    """
    Adaptive Stock‑Threshold (AST) strategy for the Common Pool Resource game.

    The player estimates opponents' cooperation probability, computes a
    dynamic stock threshold θ(R) = capacity·(1\xa0‑\xa0β/R) and decides each round
    whether to cooperate (C) or defect (D) based on expected total returns
    over the remaining horizon.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.beta = 0.5
        self.L = 5
        self.c_hist = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0:
            last_c = int(history.opponent_cooperators[-1])
            self.c_hist.append(last_c)
        if current_stock <= 0.0:
            return Action.D
        if not self.c_hist:
            p_hat = 0.0
        else:
            recent = self.c_hist[-self.L:]
            proportions = [c / (self.n - 1) for c in recent]
            p_hat = float(np.mean(proportions))
        R = self.r - t
        if t == self.r - 1:
            payoff_c = current_stock / (2 * self.n)
            payoff_d = current_stock / self.n
            return Action.C if payoff_c >= payoff_d else Action.D
        cons_coop = current_stock / (2 * self.n)
        cons_def = current_stock / self.n
        exp_opponent_cons = (p_hat * cons_coop + (1 - p_hat) * cons_def) * (self.n - 1)
        my_cons_c = cons_coop
        stock_after_c = current_stock - (my_cons_c + exp_opponent_cons)
        stock_after_c = max(stock_after_c, 0.0)
        growth_c = 2 * stock_after_c * (1 - stock_after_c / self.capacity)
        next_stock_c = min(stock_after_c + growth_c, self.capacity)
        my_cons_d = cons_def
        stock_after_d = current_stock - (my_cons_d + exp_opponent_cons)
        stock_after_d = max(stock_after_d, 0.0)
        growth_d = 2 * stock_after_d * (1 - stock_after_d / self.capacity)
        next_stock_d = min(stock_after_d + growth_d, self.capacity)
        avg_future_c = next_stock_c / (2 * self.n) * (R - 1)
        avg_future_d = next_stock_d / self.n * (R - 1)
        total_c = current_stock / (2 * self.n) + avg_future_c
        total_d = current_stock / self.n + avg_future_d
        theta = self.capacity * (1 - self.beta / R)
        if current_stock >= theta and total_c >= total_d:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_14(BaseStrategy):
    """
    Adaptive Threshold with Observed Cooperation (ATOC).

    The agent adapts its cooperation decision based on the remaining rounds,
    the observed cooperation level of opponents in the previous round, the
    current stock level, and a simple estimate of future loss caused by
    cooperating now.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = int(game_description.n_players)
        self.n_rounds = int(game_description.n_rounds)
        self.capacity = float(game_description.capacity)
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        remaining = self.n_rounds - t
        if remaining == 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.D
        if t == 0:
            coop_est = 0.0
        else:
            k_prev = int(history.opponent_cooperators[-1])
            coop_est = k_prev / (self.n_players - 1) if self.n_players > 1 else 0.0
        expected_share_if_C = current_stock / (2 * self.n_players)
        expected_share_if_D = current_stock / self.n_players
        stock_loss = current_stock / (2 * self.n_players)
        future_loss = stock_loss * (remaining - 1) / 2.0
        net_defect = expected_share_if_D - future_loss
        net_cooperate = expected_share_if_C
        if coop_est >= 0.5 and net_cooperate >= net_defect:
            return Action.C
        elif coop_est >= 0.7 and expected_share_if_D > net_cooperate:
            return Action.D
        elif net_defect > net_cooperate:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_15(BaseStrategy):
    """
    Self‑Interested Adaptive Trigger‑Lookahead Strategy.

    The agent evaluates, each round, the immediate payoff of cooperating
    versus defecting and adds a one‑step look‑ahead estimate of future payoff.
    The estimate assumes opponents keep the cooperation frequency observed in
    the previous round and that the projected stock after the next round
    remains roughly constant for the remaining horizon.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t: int = history.round_number
        S: float = float(current_stock)
        n: int = self.n_players
        r: int = self.n_rounds
        cap: float = self.capacity
        if t == r - 1 or S <= 0.0:
            return Action.D
        if t == 0:
            p: float = 0.0
            c_prev: int = 0
        else:
            c_prev = int(history.opponent_cooperators[-1])
            p = c_prev / (n - 1) if n - 1 > 0 else 0.0
        coop_cons = S / (2.0 * n)
        defect_cons = S / n
        opp_cons = p * (n - 1) * coop_cons + (1.0 - p) * (n - 1) * defect_cons

        def project_next_stock(my_cons: float) -> float:
            """Return projected stock after this round given my consumption."""
            S_rem = S - (my_cons + opp_cons)
            if S_rem <= 0.0:
                return 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / cap)
            S_next = S_rem + growth
            return cap if S_next > cap else S_next
        my_cons_C = coop_cons
        S_next_C = project_next_stock(my_cons_C)
        imm_C = coop_cons
        my_cons_D = defect_cons
        S_next_D = project_next_stock(my_cons_D)
        imm_D = defect_cons
        h: int = r - t - 1
        if h <= 0:
            fut_C = fut_D = 0.0
        else:
            avg_C = S_next_C / (2.0 * n)
            avg_D = S_next_D / n
            fut_C = h * avg_C
            fut_D = h * avg_D
        if imm_C + fut_C >= imm_D + fut_D:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_16(BaseStrategy):
    """
    Future‑Preserving Defector.
    The agent defects when the immediate extra payoff outweighs the
    projected loss of future stock, otherwise it cooperates.
    In the final round it always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        cap = self.capacity
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        t = history.round_number
        R = r - t
        if t == r - 1:
            return Action.D
        if t == 0:
            p = 0.5
        else:
            k = int(history.opponent_cooperators[-1])
            p = k / (n - 1) if n > 1 else 0.0
        coop_cons = S / (2.0 * n)
        defect_cons = S / n
        exp_opp_cons = p * (S / (2.0 * n)) + (1.0 - p) * (S / n)
        stock_after_coop = S - (coop_cons + (n - 1) * exp_opp_cons)
        stock_after_def = S - (defect_cons + (n - 1) * exp_opp_cons)
        stock_after_coop = max(stock_after_coop, 0.0)
        stock_after_def = max(stock_after_def, 0.0)

        def growth(next_stock):
            return 2.0 * next_stock * (1.0 - next_stock / cap)
        S_next_coop = min(stock_after_coop + growth(stock_after_coop), cap)
        S_next_def = min(stock_after_def + growth(stock_after_def), cap)

        def target_stock(R_val):
            return cap * (1.0 - 1.0 / (R_val + 1.0))
        S_target_R = target_stock(R)
        R_future = R - 1
        if R_future <= 0:
            future_payoff_coop = 0.0
            future_payoff_def = 0.0
        else:
            S_target_future = target_stock(R_future)
            base_future = R_future * (S_target_future / (2.0 * n))
            adj_coop = min(1.0, S_next_coop / S_target_future) if S_target_future > 0 else 1.0
            adj_def = min(1.0, S_next_def / S_target_future) if S_target_future > 0 else 1.0
            future_payoff_coop = base_future * adj_coop
            future_payoff_def = base_future * adj_def
        total_coop = coop_cons + future_payoff_coop
        total_def = defect_cons + future_payoff_def
        if total_def > total_coop and S > S_target_R:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_1(BaseStrategy):
    """
    Selfish Adaptive Strategy – Threshold-Defect.
    Cooperates on the first round to obtain an observation,
    defects on the final round, and otherwise defects when the
    projected stock after defecting stays above a safety margin
    (25\u202f% of capacity); otherwise cooperates to preserve the resource.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._alpha = 0.25

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if t == 0:
            return Action.C
        if t == total_rounds - 1:
            return Action.D
        c = int(history.opponent_cooperators[-1])
        denom = n - 1
        p_hat = c / denom if denom > 0 else 0.0
        S = float(current_stock)
        cons_self = S / n
        coop_cons = S / (2 * n)
        def_cons = S / n
        expected_opp_cons = p_hat * coop_cons + (1.0 - p_hat) * def_cons
        cons_opp = (n - 1) * expected_opp_cons
        S_after = S - cons_self - cons_opp
        if S_after < 0.0:
            S_after = 0.0
        growth = 2.0 * S_after * (1.0 - S_after / capacity) if capacity > 0 else 0.0
        S_next = S_after + growth
        if S_next > capacity:
            S_next = capacity
        safe_margin = self._alpha * capacity
        if S_next >= safe_margin:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_2(BaseStrategy):
    """
    Greedy‑Preserve strategy for the Common Pool Resource game.

    The player defects in the last round, otherwise balances the immediate
    payoff against a one‑step discounted estimate of future payoff.
    Cooperation is forced when the stock is very low or when the resource
    is abundant and opponents are largely defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta_low = 0.35 * self.capacity
        self.theta_high = 0.65 * self.capacity
        self.gamma = 0.9
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            if current_stock > self.theta_high:
                return Action.D
            return Action.C
        opp_coop_prev = int(history.opponent_cooperators[-1])
        f_est = opp_coop_prev / (self.n - 1) if self.n > 1 else 0.0
        if current_stock <= self.theta_low:
            return Action.C
        if current_stock >= self.theta_high and f_est < 0.2:
            return Action.D
        coop_others = f_est * (self.n - 1)
        def_others = (1.0 - f_est) * (self.n - 1)

        def next_stock(S: float, total_cons: float) -> float:
            S_rem = S - total_cons
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
            S_next = S_rem + growth
            if S_next > self.capacity:
                S_next = self.capacity
            return S_next
        cons_self_C = current_stock / (2.0 * self.n)
        total_cons_C = (coop_others + 1.0) * cons_self_C + def_others * (current_stock / self.n)
        S_next_C = next_stock(current_stock, total_cons_C)
        payoff_C = cons_self_C
        future_val_C = self.gamma * (f_est * (S_next_C / (2.0 * self.n)) + (1.0 - f_est) * (S_next_C / self.n))
        total_val_C = payoff_C + future_val_C
        cons_self_D = current_stock / self.n
        total_cons_D = coop_others * (current_stock / (2.0 * self.n)) + (def_others + 1.0) * cons_self_D
        S_next_D = next_stock(current_stock, total_cons_D)
        payoff_D = cons_self_D
        future_val_D = self.gamma * (f_est * (S_next_D / (2.0 * self.n)) + (1.0 - f_est) * (S_next_D / self.n))
        total_val_D = payoff_D + future_val_D
        if total_val_D > total_val_C:
            return Action.D
        return Action.C

class Strategy_SELFISH_3(BaseStrategy):
    """
    One‑Step Look‑Ahead (OSLA) selfish adaptive strategy.

    The player keeps an exponential moving average (EMA) `k̂` of the number of
    cooperating opponents observed each round.  Using the current stock `S`,
    it evaluates the expected total payoff of cooperating versus defecting:
    immediate payoff plus a single‑step estimate of future payoff assuming the
    same action will be repeated.  The player cooperates only if the total
    expected payoff of cooperating exceeds that of defecting *and* the stock is
    above a safety threshold (10\u202f% of capacity).  In the final round the
    player always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.k_hat: float | None = None
        self.alpha: float = 0.7
        self.safety_threshold: float = 0.1 * self.game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        S = current_stock
        if S <= 0.0:
            return Action.D
        if self.k_hat is None:
            self.k_hat = (n - 1) / 2.0
        elif t > 0:
            last_obs = int(history.opponent_cooperators[-1])
            self.k_hat = self.alpha * last_obs + (1.0 - self.alpha) * self.k_hat
            if self.k_hat < 0:
                self.k_hat = 0.0
            elif self.k_hat > n - 1:
                self.k_hat = float(n - 1)
        if t == r - 1:
            return Action.D

        def next_stock(stock_after_consumption: float) -> float:
            growth = 2.0 * stock_after_consumption * (1.0 - stock_after_consumption / capacity)
            new_stock = stock_after_consumption + growth
            return min(new_stock, capacity)
        my_cons_C = S / (2.0 * n)
        opp_cons_C = self.k_hat * S / (2.0 * n)
        opp_cons_D = (n - 1.0 - self.k_hat) * S / n
        total_cons_C = my_cons_C + opp_cons_C + opp_cons_D
        S_remain_C = max(S - total_cons_C, 0.0)
        S_next_C = next_stock(S_remain_C)
        my_cons_D = S / n
        total_cons_D = my_cons_D + opp_cons_C + opp_cons_D
        S_remain_D = max(S - total_cons_D, 0.0)
        S_next_D = next_stock(S_remain_D)
        remaining = r - t - 1
        future_payoff_C = remaining * (S_next_C / (2.0 * n))
        future_payoff_D = remaining * (S_next_D / n)
        total_C = S / (2.0 * n) + future_payoff_C
        total_D = S / n + future_payoff_D
        if total_C > total_D and S > self.safety_threshold:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_4(BaseStrategy):
    """
    Adaptive Self‑Preserving Defection (ASPD).

    The strategy estimates the cooperation probability of opponents from past
    observations, projects the stock after a cooperative or defective move,
    adds the discounted future gain of preserving the stock, and picks the
    action with the higher total (defecting on ties).  It also defects
    immediately when the stock falls below 5\u202f% of capacity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.total_rounds = game_description.n_rounds

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        capacity = self.capacity
        t = history.round_number
        R = self.total_rounds - t - 1
        if current_stock <= capacity * 0.05:
            return Action.D
        if t == 0:
            p_hat = 0.0
        else:
            total_coops = int(np.sum(history.opponent_cooperators))
            denom = (n - 1) * t
            p_hat = total_coops / denom if denom != 0 else 0.0
        E_coop = current_stock / (2 * n)
        E_def = current_stock / n
        opp_exp_consumption = p_hat * (current_stock / (2 * n)) + (1 - p_hat) * (current_stock / n)
        C_expected = (n - 1) * opp_exp_consumption + current_stock / (2 * n)
        D_expected = (n - 1) * opp_exp_consumption + current_stock / n

        def project_stock(expected_consumption: float) -> float:
            remaining = current_stock - expected_consumption
            if remaining < 0:
                remaining = 0.0
            growth = 2.0 * remaining * (1.0 - remaining / capacity)
            new_stock = remaining + growth
            if new_stock > capacity:
                new_stock = capacity
            return new_stock
        S_prime_c = project_stock(C_expected)
        S_prime_d = project_stock(D_expected)
        Future_gain_coop = R * S_prime_c / (2 * n)
        Future_gain_def = R * S_prime_d / n
        total_coop = E_coop + Future_gain_coop
        total_def = E_def + Future_gain_def
        if total_def > total_coop:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_5(BaseStrategy):
    """
    Selfish Adaptive Threshold‑Growth Strategy (SATS).

    The player forecasts the next stock level under both cooperate (C) and defect (D)
    actions, assumes opponents keep the same cooperation probability as observed in the
    previous round, and values the remaining horizon accordingly.  It defects in the
    final round or when the stock is exhausted; otherwise it cooperates only if the
    expected total utility (immediate payoff + projected future value) is strictly
    larger than that of defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        capacity = self.capacity
        t = history.round_number
        if t == 0:
            c_prev = math.ceil((n - 1) / 2)
        else:
            c_prev = int(history.opponent_cooperators[-1])
        denom_opp = max(1, n - 1)
        p = c_prev / denom_opp
        pi_C = current_stock / (2 * n)
        pi_D = current_stock / n
        opp_cons_per = p * (current_stock / (2 * n)) + (1 - p) * (current_stock / n)
        cons_Opp = (n - 1) * opp_cons_per
        cons_C = pi_C + cons_Opp
        Srem_C = max(0.0, current_stock - cons_C)
        growth_C = 2 * Srem_C * (1 - Srem_C / capacity) if capacity != 0 else 0.0
        Snext_C = min(Srem_C + growth_C, capacity)
        cons_D = pi_D + cons_Opp
        Srem_D = max(0.0, current_stock - cons_D)
        growth_D = 2 * Srem_D * (1 - Srem_D / capacity) if capacity != 0 else 0.0
        Snext_D = min(Srem_D + growth_D, capacity)
        h = r - t - 1
        if h > 0:
            V_C = h * (p * (Snext_C / (2 * n)) + (1 - p) * (Snext_C / n))
            V_D = h * (p * (Snext_D / (2 * n)) + (1 - p) * (Snext_D / n))
        else:
            V_C = V_D = 0.0
        if t == r - 1 or current_stock <= 0.0:
            return Action.D
        total_C = pi_C + V_C
        total_D = pi_D + V_D
        if total_C > total_D:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_6(BaseStrategy):
    """
    Forward‑looking cooperation rule for the Common Pool Resource game.
    It estimates the opponents' cooperation rate, projects the stock after
    both possible actions, evaluates a simple linear proxy for future
    payoffs and decides:

    * Defect in the final round.
    * Otherwise cooperate only if the expected utility of cooperating
      exceeds that of defecting *and* the stock is above 30\u202f% of capacity
      *and* the estimated opponent cooperation rate is at least 0.5.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        t = history.round_number
        S = float(current_stock)
        if S < 0.0:
            S = 0.0
        elif S > capacity:
            S = float(capacity)
        if t == r - 1:
            return Action.D
        if t > 0:
            H = history.opponent_cooperators.tolist()
            coop_rate = sum(H) / ((n - 1) * t)
        else:
            coop_rate = 0.5
        pay_C = S / (2 * n)
        pay_D = S / n
        exp_opp_cons = coop_rate * pay_C + (1 - coop_rate) * pay_D
        exp_opp_total = (n - 1) * exp_opp_cons
        stock_after_C = S - (pay_C + exp_opp_total)
        stock_after_D = S - (pay_D + exp_opp_total)

        def grow(x: float) -> float:
            grown = x + 2 * x * (1 - x / capacity)
            return grown if grown < capacity else float(capacity)
        proj_stock_C = grow(max(stock_after_C, 0.0))
        proj_stock_D = grow(max(stock_after_D, 0.0))
        remaining = r - t
        future_val_C = remaining * (proj_stock_C / (2 * n))
        future_val_D = remaining * (proj_stock_D / (2 * n))
        U_C = pay_C + future_val_C
        U_D = pay_D + future_val_D
        if U_C > U_D and S > capacity * 0.3 and (coop_rate >= 0.5):
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_7(BaseStrategy):
    """
    Selfish Adaptive “Stock‑Threshold Conditional Defection” strategy.

    The player keeps an exponential‑moving average (EMA) estimate of the
    opponents’ cooperation probability (p̂).  For each round it projects the
    expected stock after the current round for both possible actions (C or D)
    using that estimate, adds a one‑step discounted future payoff, and computes
    the net advantage Δ of defecting versus cooperating.

    Decision rule:
        * Last round → always defect.
        * Stock ≤ ε (practically depleted) → defect.
        * If Δ > 0 and stock > T_high → defect.
        * If Δ ≤ 0 and stock > T_low → cooperate.
        * Otherwise → defect.
    """
    _lambda: float = 0.9
    _delta: float = 0.9
    _epsilon_factor: float = 0.01
    _t_high_factor: float = 0.6
    _t_low_factor: float = 0.3

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_estimate: float | None = None

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        t = history.round_number
        eps = self._epsilon_factor * capacity
        T_high = self._t_high_factor * capacity
        T_low = self._t_low_factor * capacity
        if t == 0:
            self.p_estimate = 0.5
        else:
            k_prev = int(history.opponent_cooperators[-1])
            prev_est = self.p_estimate if self.p_estimate is not None else 0.5
            self.p_estimate = self._lambda * prev_est + (1.0 - self._lambda) * (k_prev / (n - 1))
        payoff_def = current_stock / n
        payoff_coop = current_stock / (2.0 * n)
        p_hat = float(self.p_estimate)
        exp_coop = (n - 1) * p_hat
        exp_def = n - 1 - exp_coop
        opp_consume = exp_coop * (current_stock / (2.0 * n)) + exp_def * (current_stock / n)
        total_cons_def = payoff_def + opp_consume
        S_rem_def = max(current_stock - total_cons_def, 0.0)
        growth_def = 2.0 * S_rem_def * (1.0 - S_rem_def / capacity)
        S_next_def = min(S_rem_def + growth_def, capacity)
        total_cons_coop = payoff_coop + opp_consume
        S_rem_coop = max(current_stock - total_cons_coop, 0.0)
        growth_coop = 2.0 * S_rem_coop * (1.0 - S_rem_coop / capacity)
        S_next_coop = min(S_rem_coop + growth_coop, capacity)
        future_gain_def = S_next_def / n * self._delta
        future_gain_coop = S_next_coop / (2.0 * n) * self._delta
        delta = payoff_def - payoff_coop + (future_gain_def - future_gain_coop)
        if t == r - 1:
            return Action.D
        if current_stock <= eps:
            return Action.D
        if delta > 0.0 and current_stock > T_high:
            return Action.D
        if delta <= 0.0 and current_stock > T_low:
            return Action.C
        return Action.D

class Strategy_SELFISH_8(BaseStrategy):
    """
    Selfish Adaptive Defection‑Cooperation (SADC) strategy.
    Chooses the action that maximises the sum of immediate payoff and an
    estimated future payoff, assuming opponents continue cooperating with the
    empirically observed probability.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.total_rounds = game_description.n_rounds

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        R = self.total_rounds - t - 1
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            p = 0.5
        else:
            coop_sum = int(np.sum(history.opponent_cooperators))
            denom = (self.n - 1) * t
            p = coop_sum / denom if denom > 0 else 0.5
            p = max(0.0, min(1.0, p))
        n = self.n
        S = float(current_stock)
        pi_C = S / (2.0 * n)
        pi_D = S / float(n)
        exp_opp_cons = p * (S / (2.0 * n)) + (1.0 - p) * (S / float(n))
        cons_C = pi_C + (n - 1) * exp_opp_cons
        S_rem_C = max(0.0, S - cons_C)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / self.capacity)
        S_next_C = min(S_rem_C + growth_C, self.capacity)
        cons_D = pi_D + (n - 1) * exp_opp_cons
        S_rem_D = max(0.0, S - cons_D)
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / self.capacity)
        S_next_D = min(S_rem_D + growth_D, self.capacity)
        pi_future_C = S_next_C / (2.0 * n) * (1.0 - p) + S_next_C / float(n) * p
        pi_future_D = S_next_D / (2.0 * n) * (1.0 - p) + S_next_D / float(n) * p
        V_C = pi_future_C * R if R > 0 else 0.0
        V_D = pi_future_D * R if R > 0 else 0.0
        if pi_C + V_C > pi_D + V_D:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_9(BaseStrategy):
    """
    Selfish Threshold‑Based Adaptive Strategy.
    Defects unless the projected stock after defecting would fall below a
    safety threshold (Reserve + remaining_rounds * CoopMargin). In the last
    round it always defects. Handles edge cases such as zero stock and first
    round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self._reserve = 0.15 * self.capacity
        self._coop_margin = 0.05 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if t == 0:
            k_prev = 0
        else:
            k_prev = int(history.opponent_cooperators[-1])
        opponents = self.n - 1
        p = k_prev / opponents if opponents > 0 else 0.0
        coop_consumption = S / (2 * self.n)
        defect_consumption = S / self.n
        opp_expected = p * coop_consumption + (1.0 - p) * defect_consumption
        total_opp_consumption = opponents * opp_expected
        my_c = coop_consumption
        S_c = max(S - (my_c + total_opp_consumption), 0.0)
        growth_c = 2.0 * S_c * (1.0 - S_c / self.capacity) if self.capacity > 0 else 0.0
        S_next_c = min(S_c + growth_c, self.capacity)
        my_d = defect_consumption
        S_d = max(S - (my_d + total_opp_consumption), 0.0)
        growth_d = 2.0 * S_d * (1.0 - S_d / self.capacity) if self.capacity > 0 else 0.0
        S_next_d = min(S_d + growth_d, self.capacity)
        remaining_rounds = self.r - t - 1
        safe_needed = self._reserve + remaining_rounds * self._coop_margin
        if S_next_d < safe_needed:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_10(BaseStrategy):
    """
    Selfish Adaptive Threshold Strategy.
    Uses a one‑step look‑ahead combined with a steady‑state estimate
    of future payoffs.  A small greedy bias (ε=0.01) breaks ties
    in favour of defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self._epsilon = 0.01

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock <= 0:
            return Action.D
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            k_t = 0
            coop_ratio = 0.5
        else:
            k_t = int(history.opponent_cooperators[-1])
            coop_ratio = (k_t + 1) / self.n
        pay_C = current_stock / (2 * self.n)
        pay_D = current_stock / self.n
        total_cons_C = (k_t + 1) * (current_stock / (2 * self.n)) + (self.n - (k_t + 1)) * (current_stock / self.n)
        total_cons_D = k_t * (current_stock / (2 * self.n)) + (self.n - k_t) * (current_stock / self.n)

        def next_stock(total_cons: float) -> float:
            s_rem = max(0.0, current_stock - total_cons)
            growth = 2.0 * s_rem * (1.0 - s_rem / self.capacity)
            return min(s_rem + growth, self.capacity)
        S_next_C = next_stock(total_cons_C)
        S_next_D = next_stock(total_cons_D)
        if S_next_C == 0.0 and S_next_D == 0.0:
            return Action.D
        T = self.r - t
        remaining = max(0, T - 1)

        def expected_future(S_next: float) -> float:
            if remaining == 0:
                return 0.0
            steady_coop = S_next / (2 * self.n) * coop_ratio
            steady_def = S_next / self.n * (1.0 - coop_ratio)
            return remaining * (steady_coop + steady_def)
        future_C = expected_future(S_next_C)
        future_D = expected_future(S_next_D)
        value_C = pay_C + future_C
        value_D = pay_D + future_D
        if value_D >= value_C - self._epsilon:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_11(BaseStrategy):
    """
    Greedy‑Sustain selfish adaptive strategy.
    Cooperates only when the projected stock after cooperating stays above a safety
    fraction of the capacity; otherwise defects. Defects unconditionally in the final
    round to maximise immediate payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self._theta = 0.3
        self._epsilon = 1e-06
        self._default_coop_rate = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        S_t = float(current_stock)
        if t == self.r - 1:
            return Action.D
        if t == 0:
            k_prev = None
        else:
            k_prev = int(history.opponent_cooperators[-1])
        n_opp = self.n - 1
        if k_prev is None:
            coop_rate = self._default_coop_rate
        else:
            coop_rate = k_prev / n_opp if n_opp > 0 else 0.0
        exp_coop_opp = coop_rate * n_opp
        exp_def_opp = n_opp - exp_coop_opp
        cons_self_C = S_t / (2 * self.n)
        cons_self_D = S_t / self.n
        cons_opp = exp_coop_opp * (S_t / (2 * self.n)) + exp_def_opp * (S_t / self.n)
        S_after_C = S_t - (cons_self_C + cons_opp)
        S_after_C = max(S_after_C, 0.0)
        growth_C = 2 * S_after_C * (1 - S_after_C / self.capacity)
        S_next_C = min(S_after_C + growth_C, self.capacity)
        S_after_D = S_t - (cons_self_D + cons_opp)
        S_after_D = max(S_after_D, 0.0)
        growth_D = 2 * S_after_D * (1 - S_after_D / self.capacity)
        S_next_D = min(S_after_D + growth_D, self.capacity)
        if S_next_C >= self._theta * self.capacity:
            payoff_C = cons_self_C
            payoff_D = cons_self_D
            if payoff_C >= payoff_D * (1 - self._epsilon):
                return Action.C
        return Action.D

class Strategy_SELFISH_12(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy for the Common Pool Resource Game.
    The agent estimates opponent cooperation from the previous round,
    weighs the immediate loss of cooperating against the future benefit of a
    larger stock, and includes a penalty when opponents appear mostly
    defecting.
    """
    _MIN_COOP_FRAC = 0.5
    _COOP_PENALTY = 1.0

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        n = self.n
        r = self.r
        capacity = self.capacity
        if t == r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        if t == 0:
            p_est = 0.5
        else:
            coop_last = int(history.opponent_cooperators[-1])
            p_est = coop_last / (n - 1) if n - 1 > 0 else 0.0
        delta_self = S / (2.0 * n)
        delta_def = S / float(n)
        loss_now = delta_def - delta_self
        growth_factor = 2.0 * (1.0 - S / capacity) if capacity != 0 else 0.0
        per_opp_cons = p_est * (S / (2.0 * n)) + (1.0 - p_est) * (S / float(n))
        exp_cons_opp = (n - 1) * per_opp_cons
        S_rem_if_C = S - (delta_self + exp_cons_opp)
        S_rem_if_D = S - (delta_def + exp_cons_opp)
        S_rem_if_C = max(S_rem_if_C, 0.0)
        S_rem_if_D = max(S_rem_if_D, 0.0)
        S_next_C = min(S_rem_if_C + growth_factor * S_rem_if_C, capacity)
        S_next_D = min(S_rem_if_D + growth_factor * S_rem_if_D, capacity)
        remaining_rounds = r - t - 1
        future_val_C = remaining_rounds * (S_next_C / (2.0 * n))
        future_val_D = remaining_rounds * (S_next_D / (2.0 * n))
        util_C = delta_self + future_val_C
        util_D = delta_def + future_val_D
        if p_est < self._MIN_COOP_FRAC:
            util_C -= self._COOP_PENALTY * loss_now * remaining_rounds
        if util_C >= util_D:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_13(BaseStrategy):
    """
    Selfish Adaptive Cooperation–Defection (SACD) strategy.
    Uses one‑step lookahead on expected payoff, with exponential smoothing
    of opponents' past cooperation levels (λ = 0.7) and a tiny tie‑breaker ε.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.lam = 0.7
        self.epsilon = 0.01

    def _estimate_other_cooperators(self, history: PlayerHistory) -> int:
        """
        Exponential smoothing estimate of the number of opponents that will cooperate.
        Returns an integer count.
        """
        hist = history.opponent_cooperators.tolist()
        if not hist:
            return math.floor((self.n - 1) / 2)
        lam = self.lam
        w = 1.0
        est = 0.0
        for cnt in reversed(hist):
            est += w * cnt
            w *= lam
        norm = (1.0 - lam ** len(hist)) / (1.0 - lam) if lam != 1.0 else float(len(hist))
        if norm == 0.0:
            return 0
        return int(round(est / norm))

    def _expected_payoff(self, action: Action, k_other: int, stock: float) -> float:
        """
        Computes immediate payoff plus one‑step lookahead payoff under the assumption
        that the same action will be repeated next round.
        """
        n = self.n
        cap = self.capacity
        if action == Action.C:
            my_share = stock / (2.0 * n)
            total_cons = (k_other + 1) * (stock / (2.0 * n)) + (n - k_other - 1) * (stock / n)
        else:
            my_share = stock / n
            total_cons = k_other * (stock / (2.0 * n)) + (n - k_other) * (stock / n)
        s_rem = stock - total_cons
        if s_rem < 0.0:
            s_rem = 0.0
        growth = 2.0 * s_rem * (1.0 - s_rem / cap) if cap != 0.0 else 0.0
        s_next = min(s_rem + growth, cap)
        my_next = s_next / (2.0 * n) if action == Action.C else s_next / n
        return my_share + my_next

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if current_stock <= 0.0:
            return Action.D
        if t == self.r - 1:
            return Action.D
        k_est = self._estimate_other_cooperators(history)
        payoff_C = self._expected_payoff(Action.C, k_est, current_stock)
        payoff_D = self._expected_payoff(Action.D, k_est, current_stock)
        eps = self.epsilon
        if payoff_D > payoff_C + eps:
            return Action.D
        if payoff_C > payoff_D + eps:
            return Action.C
        return Action.D

class Strategy_SELFISH_14(BaseStrategy):
    """
    Selfish Adaptive Strategy.

    The player estimates opponents' cooperation probability from the previous
    round, forecasts the stock after taking a cooperative or defective action,
    projects the remaining horizon assuming the same cooperation level,
    and selects the action with the higher estimated cumulative selfish payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock <= 0.0:
            return Action.C
        t = history.round_number
        R = self.r - t - 1
        if t == 0:
            c_last = self.n - 1
        else:
            c_last = int(history.opponent_cooperators[-1])
        denom = self.n - 1
        p_est = c_last / denom if denom > 0 else 0.0
        my_coop_cons = current_stock / (2.0 * self.n)
        my_def_cons = current_stock / float(self.n)
        opp_coop_cons = my_coop_cons
        opp_def_cons = my_def_cons
        opp_expected_cons = p_est * (self.n - 1) * opp_coop_cons + (1.0 - p_est) * (self.n - 1) * opp_def_cons
        expected_total_coop = my_coop_cons + opp_expected_cons
        S_rem_coop = max(0.0, current_stock - expected_total_coop)
        growth_coop = 2.0 * S_rem_coop * (1.0 - S_rem_coop / self.capacity) if self.capacity > 0 else 0.0
        S_next_coop = min(S_rem_coop + growth_coop, self.capacity)
        expected_total_def = my_def_cons + opp_expected_cons
        S_rem_def = max(0.0, current_stock - expected_total_def)
        growth_def = 2.0 * S_rem_def * (1.0 - S_rem_def / self.capacity) if self.capacity > 0 else 0.0
        S_next_def = min(S_rem_def + growth_def, self.capacity)
        if R > 0:
            future_coop = R * (S_next_coop / (2.0 * self.n))
            future_def = R * (S_next_def / float(self.n))
        else:
            future_coop = future_def = 0.0
        value_coop = my_coop_cons + future_coop
        value_def = my_def_cons + future_def
        if value_coop > value_def:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_15(BaseStrategy):
    """
    Selfish Adaptive Cooperation/Defection Rule.
    The agent estimates the opponents' cooperation probability from the
    previous round (or uses 0.5 initially) and projects the impact of its
    current action on the future stock.  It cooperates only when the
    expected total payoff (current + future) is larger than defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n_players
        r = self.n_rounds
        cap = self.capacity
        S = float(current_stock)
        if t >= r:
            return Action.D
        if t == r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        if t == 0:
            p_hat = 0.5
        else:
            c_prev = int(history.opponent_cooperators[-1])
            p_hat = c_prev / (n - 1) if n - 1 > 0 else 0.0
        cons_C = S / (2 * n)
        cons_D = S / n
        exp_cons_C = (n - 1) * (p_hat * cons_C + (1.0 - p_hat) * cons_D) + cons_C
        exp_cons_D = (n - 1) * (p_hat * cons_C + (1.0 - p_hat) * cons_D) + cons_D

        def expected_next_stock(exp_cons: float) -> float:
            S_rem = max(0.0, S - exp_cons)
            growth = 2.0 * S_rem * (1.0 - S_rem / cap) if cap != 0 else 0.0
            return min(S_rem + growth, cap)
        S_next_C = expected_next_stock(exp_cons_C)
        S_next_D = expected_next_stock(exp_cons_D)
        R = r - t
        V = (R - 1) / n if n != 0 else 0.0
        immediate_diff = cons_D - cons_C
        delta = V * (S_next_C - S_next_D) - immediate_diff
        if delta > 0.0:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_16(BaseStrategy):
    """
    Selfish Adaptive “Stock‑Threshold Trigger”.

    - Round\u202f0: always cooperate to obtain a baseline.
    - Final round: always defect (no future to protect).
    - Other rounds:
        • Compute the remaining rounds and a target stock level
          (capacity\u202f×\u202fremaining / total_rounds).
        • If the current stock is at least the target, cooperate only when
          at least half of the opponents cooperated in the previous round;
          otherwise defect.
        • If the stock is below the target, defect (extraction is optimal).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if t >= 1 and len(history.opponent_cooperators) >= 1:
            k = int(history.opponent_cooperators[-1])
        else:
            k = 0
        remaining = self.r - t
        target = self.capacity * (remaining / self.r)
        if current_stock >= target:
            if k >= self.n / 2:
                return Action.C
            else:
                return Action.D
        else:
            return Action.D
