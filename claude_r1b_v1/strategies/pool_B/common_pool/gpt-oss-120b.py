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
