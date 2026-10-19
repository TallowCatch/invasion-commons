"""
R1b Pool A (audit-aware) strategies for common_pool.

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
    Altruistic Adaptive Cooperation Strategy.

    - Cooperates when the resource is abundant and the community’s average
      defection rate is low.
    - Scales back cooperation as the stock falls or as exploitation rises.
    - Guarantees cooperation in the final round (no audit risk).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._alpha = 0.5
        self._beta = 0.3
        self._gamma = 0.1
        self._delta = game_description.audit_prob * game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n_players = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        round_idx = history.round_number
        last_round = self.game_description.n_rounds - 1
        if round_idx == last_round:
            return Action.C
        stock_frac = current_stock / capacity if capacity > 0 else 0.0
        coop_cap = max(self._gamma, min(1.0, stock_frac / self._alpha))
        if round_idx == 0:
            avg_defect_rate = 0.0
        else:
            n_opponents = n_players - 1
            past_coops = history.opponent_cooperators[:-1]
            defect_counts = n_opponents - past_coops
            total_defects = int(np.sum(defect_counts))
            avg_defect_rate = total_defects / (round_idx * n_players)
        pressure = min(1.0, avg_defect_rate / self._beta)
        p_coop = coop_cap * (1.0 - pressure)
        if self._delta > 0.0:
            stock_term = current_stock + 1e-06
            reduction = min(1.0, self._delta / stock_term)
            p_coop *= 1.0 - reduction
        p_coop = max(0.0, min(1.0, p_coop))
        cooperate = random.random() < p_coop
        return Action.C if cooperate else Action.D

class Strategy_BENEVOLENT_1(BaseStrategy):
    """
    Benevolent Adaptive Strategy.

    - Starts cooperative.
    - Defects only when the expected net gain from defection (ignoring
      the stock impact) is positive, the environment is not healthy,
      and the stock is not critical.
    - Allows at most L (=3) consecutive defections as a mild
      punishment; afterwards cooperates again.
    - In the final round, defects only if the expected net gain is
      positive and opponents were mostly defecting.
    - Never defects when the stock is zero.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.tau = 0.7
        self.sigma = 0.4
        self.theta = 0.2
        self.L = 3
        self.consecutive_defections = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            self.consecutive_defections = 0
            return Action.C
        if t > 0:
            opponents_coop = int(history.opponent_cooperators[-1])
            p = opponents_coop / (self.n - 1) if self.n > 1 else 0.0
        else:
            p = 0.0
        G = S / (2 * self.n) - self.q * self.F
        if t == self.r - 1:
            if G > 0.0 and p < self.tau and (S - S / self.n >= 0.0):
                self.consecutive_defections = 0
                return Action.D
            else:
                self.consecutive_defections = 0
                return Action.C
        if t == 0:
            self.consecutive_defections = 0
            return Action.C
        if G <= 0.0:
            self.consecutive_defections = 0
            return Action.C
        if S >= self.sigma * self.capacity and p >= self.tau:
            self.consecutive_defections = 0
            return Action.C
        projected_stock_after_my_defect = S - S / self.n
        if S < self.theta * self.capacity or p < self.tau:
            if projected_stock_after_my_defect < self.theta * self.capacity:
                self.consecutive_defections = 0
                return Action.C
            if self.consecutive_defections < self.L:
                self.consecutive_defections += 1
                return Action.D
            else:
                self.consecutive_defections = 0
                return Action.C
        self.consecutive_defections = 0
        return Action.C

class Strategy_COLLECTIVE_1(BaseStrategy):
    """
    Collective Adaptive Strategy (CAP‑S) for the Common Pool Resource game.

    The agent starts by cooperating, then adapts its willingness to defect
    based on:
      * current stock level,
      * observed cooperation rate of opponents (sliding window),
      * expected audit fine (q·F),
      * audit feedback (raises cooperation threshold if many audited defections are fined),
      * a hard protection rule when the stock becomes scarce.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.tau_stock = 0.3 * self.capacity
        self.tau_coop = 0.6
        self.tau_low = 0.4
        self.tau_defect_last = 0.5 * self.capacity
        self.audit_count = 0
        self.penalty_cnt = 0
        self.forced_cooperate = False

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            if bool(history.my_audited[-1]):
                self.audit_count += 1
                if history.my_fines[-1] > 0.0:
                    self.penalty_cnt += 1
            if self.audit_count > 0 and self.penalty_cnt / self.audit_count > 0.2:
                self.tau_coop = min(1.0, self.tau_coop + 0.1)
        if not self.forced_cooperate and current_stock <= 0.1 * self.capacity:
            self.forced_cooperate = True
        if self.forced_cooperate:
            return Action.C
        t = history.round_number
        n = self.n_players
        opponents = n - 1
        if t == 0:
            return Action.C
        expected_fine = self.audit_prob * self.fine
        gain_defect = current_stock / (2 * n) - expected_fine
        c_last = history.opponent_cooperators[-1]
        p_hat_last = c_last / opponents if opponents > 0 else 0.0
        w = min(3, t)
        recent_p_hats = []
        for k in range(1, w + 1):
            c_k = history.opponent_cooperators[-k]
            recent_p_hats.append(c_k / opponents if opponents > 0 else 0.0)
        p_hat_window = float(np.mean(recent_p_hats)) if recent_p_hats else p_hat_last
        if t == self.n_rounds - 1:
            if current_stock > 0 and p_hat_last >= self.tau_low:
                return Action.C
            if current_stock >= self.tau_defect_last:
                return Action.D
            else:
                return Action.C
        if current_stock < self.tau_stock:
            return Action.C
        if gain_defect <= 0:
            return Action.C
        if p_hat_window >= self.tau_coop:
            return Action.D
        if p_hat_window < self.tau_low:
            return Action.C
        prob_defect = (p_hat_window - self.tau_low) / (self.tau_coop - self.tau_low)
        if random.random() < prob_defect:
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_2(BaseStrategy):
    """
    Stock‑Guarded Generous Tit‑for‑Tol (SG‑GTT).

    The strategy guards the common pool by cooperating when the stock is low
    or when the expected fine makes defection unattractive.  It retaliates
    briefly (one round) if opponents' cooperation falls below a tolerance level.
    In the final round, where audits are impossible, it defects only if the
    stock is sufficiently high.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.E_fine = self.q * self.F
        self.tau_low = 0.3 * self.capacity
        self.tau_high = 0.7 * self.capacity
        self.alpha = 0.2
        self.retaliation_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            last_coop_rate = 1.0
        else:
            n_opponents = self.n - 1
            if n_opponents == 0:
                last_coop_rate = 1.0
            else:
                last_coop_rate = float(history.opponent_cooperators[-1]) / n_opponents
        if t == self.r - 1:
            if S < self.tau_low:
                return Action.C
            else:
                return Action.D
        extra_consumption = S / (2.0 * self.n) if self.n > 0 else 0.0
        incentive_ok = self.E_fine <= extra_consumption
        if S < self.tau_low:
            return Action.C
        elif S > self.tau_high and (not incentive_ok):
            return Action.C
        elif self.retaliation_counter > 0:
            self.retaliation_counter -= 1
            return Action.D
        elif last_coop_rate < 1.0 - self.alpha:
            self.retaliation_counter = 1
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_1(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.

    The player monitors:
      * the current stock,
      * an exponential moving average of opponents' defection rate,
      * personal audit/fine history.

    Decision logic (all but the final round):
      1. Determine a tolerance δ based on the stock level:
            - stock ≤ 0.4·capacity → δ = δ_low (0.15)
            - stock ≥ 0.8·capacity → δ = δ_high (0.30)
            - otherwise            → δ = (δ_low + δ_high)/2
      2. Cooperate if the estimated opponent defection rate D_avg ≤ δ
         **and** the stock is above the critical low threshold.
      3. Otherwise defect.

    Audits:
      - After each round the strategy updates the number of fines received.
      - If more than half of the audited rounds ended in a fine,
        the player switches to unconditional cooperation for the rest
        of the game.

    The last round is always played as cooperation (no audits).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_opponents = self.n_players - 1
        self.capacity = float(game_description.capacity)
        self.S_low_factor = 0.4
        self.S_high_factor = 0.8
        self.delta_low = 0.15
        self.delta_high = 0.3
        self.alpha = 0.5
        self.D_avg = 0.0
        self.audit_fines = 0
        self.forced_cooperate = False

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            return Action.C
        prev_action_cooperate = bool(history.my_actions[-1])
        prev_audited = bool(history.my_audited[-1])
        if not prev_action_cooperate and prev_audited:
            self.audit_fines += 1
        if self.n_opponents > 0:
            C_t = int(history.opponent_cooperators[-1])
            d_t = (self.n_opponents - C_t) / self.n_opponents
        else:
            d_t = 0.0
        self.D_avg = self.alpha * d_t + (1.0 - self.alpha) * self.D_avg
        audited_sofar = int(np.sum(history.my_audited))
        if audited_sofar > 0 and self.audit_fines > math.ceil(0.5 * audited_sofar):
            self.forced_cooperate = True
        current_round = history.round_number
        last_round_index = self.game_description.n_rounds - 1
        if current_round == last_round_index:
            return Action.C
        if self.forced_cooperate:
            return Action.C
        S = float(current_stock)
        S_low = self.capacity * self.S_low_factor
        S_high = self.capacity * self.S_high_factor
        if S <= S_low:
            delta = self.delta_low
        elif S >= S_high:
            delta = self.delta_high
        else:
            delta = (self.delta_low + self.delta_high) / 2.0
        if self.D_avg <= delta and S > S_low:
            return Action.C
        else:
            return Action.D

class Strategy_INDIVIDUALISTIC_1(BaseStrategy):
    """
    Expected‑return strategy.

    At each round the agent weighs the immediate payoff of cooperating
    versus defecting (including the expected fine) and adds a projection of
    future returns under the assumption that everybody will cooperate from
    the next round onward.  The action with the higher total expected return
    is chosen; ties are broken in favour of Defect (strict individualism).

    Opponent cooperation probability is estimated as the average of the
    observed opponent‑cooperation share over the last three rounds (or all
    available rounds if fewer).  If no history exists the estimate defaults
    to 1.0 (optimistic).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._window = 3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        cap = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        F = self.game_description.fine
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        R = total_rounds - t - 1
        if t == 0:
            coop_est = 1.0
        else:
            recent = min(self._window, t)
            recent_counts = history.opponent_cooperators[-recent:]
            coop_shares = recent_counts.astype(float) / (n - 1)
            coop_est = float(np.mean(coop_shares))
        pay_C = current_stock / (2.0 * n)
        pay_D = current_stock / n - q * F
        opp_cons = coop_est * (current_stock / (2.0 * n)) + (1.0 - coop_est) * (current_stock / n)
        tot_cons_C = current_stock / (2.0 * n) + (n - 1) * opp_cons
        tot_cons_D = current_stock / n + (n - 1) * opp_cons
        S_rem_C = max(0.0, current_stock - tot_cons_C)
        S_rem_D = max(0.0, current_stock - tot_cons_D)

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / cap) if cap != 0.0 else 0.0
        next_S_C = min(S_rem_C + grow(S_rem_C), cap)
        next_S_D = min(S_rem_D + grow(S_rem_D), cap)
        if R > 0:
            future_C = R * (next_S_C / (2.0 * n))
            future_D = R * (next_S_D / (2.0 * n))
        else:
            future_C = future_D = 0.0
        total_C = pay_C + future_C
        total_D = pay_D + future_D
        return Action.D if total_D >= total_C else Action.C

class Strategy_OPPORTUNISTIC_1(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.
    Estimates opponent cooperation with a simple exponential smoothing,
    compares expected immediate payoffs (including expected fine),
    and defects only when it is profitable and the forecasted stock stays
    above a minimal sustainability threshold.  Defects in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.alpha = 0.5
        self.epsilon = 1e-09
        self.p_est = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            self.p_est = 0.5
        else:
            k_prev = int(history.opponent_cooperators[-1])
            if self.n > 1:
                obs = k_prev / (self.n - 1)
            else:
                obs = 0.0
            self.p_est = self.alpha * self.p_est + (1.0 - self.alpha) * obs
        if t == self.r - 1:
            return Action.D
        if S <= 0.0:
            return Action.D
        coop_val = S / (2.0 * self.n)
        defect_val = S / self.n - (self.q * self.F if self.q > 0.0 else 0.0)

        def forecast_stock(S_cur: float, action_self: str, p: float) -> float:
            coop_opp = (self.n - 1) * p
            def_opp = self.n - 1 - coop_opp
            cons_self = S_cur / (2.0 * self.n) if action_self == 'C' else S_cur / self.n
            cons_opp = coop_opp * (S_cur / (2.0 * self.n)) + def_opp * (S_cur / self.n)
            S_rem = S_cur - (cons_self + cons_opp)
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
            return min(S_rem + growth, self.capacity)
        min_needed = (self.r - t - 1) * (self.capacity / (2.0 * self.n))
        if defect_val > coop_val + self.epsilon:
            S_next = forecast_stock(S, 'D', self.p_est)
            if S_next >= min_needed:
                return Action.D
            else:
                return Action.C
        elif coop_val > defect_val + self.epsilon:
            return Action.C
        else:
            return Action.D if random.random() < 0.7 else Action.C

class Strategy_SELFINTERESTED_1(BaseStrategy):
    """
    Adaptive Expected‑Value with Defection‑Retaliation (AEV‑DR).

    The agent balances immediate expected profit (adjusted for audit risk) with a
    weighted estimate of future profit that depends on the stock level.  When the
    observed proportion of defecting opponents exceeds a tolerance β, the agent
    enters a retaliation phase of γ rounds, during which it always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game = game_description
        self.n = self.game.n_players
        self.r = self.game.n_rounds
        self.capacity = self.game.capacity
        self.audit_q = self.game.audit_prob
        self.fine_F = self.game.fine
        self.alpha = 0.6
        self.beta = 0.5
        self.gamma = 2
        self.retaliation_counter = 0
        self.last_defect_round = -float('inf')

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0:
            return Action.C
        if t == self.r - 1:
            coop_now = current_stock / (2 * self.n)
            defect_now = current_stock / self.n - self.audit_q * self.fine_F
            return Action.D if defect_now > coop_now else Action.C
        coop_now = current_stock / (2 * self.n)
        defect_now = current_stock / self.n - self.audit_q * self.fine_F
        if t == 0:
            p_obs = 0.5
            cooperators_last = 0
        else:
            cooperators_last = int(history.opponent_cooperators[-1])
            p_obs = cooperators_last / (self.n - 1)
        opp_cons_if_coop = p_obs * (current_stock / (2 * self.n)) + (1 - p_obs) * (current_stock / self.n)
        our_cons_coop = current_stock / (2 * self.n)
        total_cons_coop = our_cons_coop + (self.n - 1) * opp_cons_if_coop
        our_cons_def = current_stock / self.n
        total_cons_def = our_cons_def + (self.n - 1) * opp_cons_if_coop
        S_rem_coop = max(current_stock - total_cons_coop, 0.0)
        S_rem_def = max(current_stock - total_cons_def, 0.0)
        growth_coop = 2 * S_rem_coop * (1 - S_rem_coop / self.capacity) if S_rem_coop > 0 else 0.0
        growth_def = 2 * S_rem_def * (1 - S_rem_def / self.capacity) if S_rem_def > 0 else 0.0
        S_future_coop = min(S_rem_coop + growth_coop, self.capacity)
        S_future_def = min(S_rem_def + growth_def, self.capacity)
        rounds_left = self.r - t - 1
        future_val_coop = S_future_coop / (2 * self.n) * rounds_left
        future_val_def = S_future_def / self.n * rounds_left
        ev_coop = coop_now + self.alpha * future_val_coop
        ev_def = defect_now + self.alpha * future_val_def
        if t > 0:
            defection_ratio = 1.0 - cooperators_last / (self.n - 1)
            if defection_ratio > self.beta:
                self.last_defect_round = t
                self.retaliation_counter = self.gamma
        if self.retaliation_counter > 0:
            self.retaliation_counter -= 1
            return Action.D
        return Action.C if ev_coop >= ev_def else Action.D

class Strategy_SELFISH_1(BaseStrategy):
    """
    Selfish Adaptive Strategy for the Common Pool Resource Game.

    The agent estimates the probability that opponents cooperate based on the
    previous round, evaluates immediate expected payoffs (including expected
    fine), projects the stock for each possible own action, and adds a
    myopic continuation value for the remaining rounds.  It defects when the
    total expected value of defecting exceeds that of cooperating.  In the
    final round it simply picks the action with the higher certain payoff
    (defect if stock > 0, otherwise cooperate).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        capacity = self.capacity
        q = self.audit_prob
        F = self.fine
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            p_coop = 1.0
        else:
            p_coop = history.opponent_cooperators[-1] / (n - 1)
        if t == r - 1:
            return Action.D if current_stock > 0 else Action.C
        payoff_C = current_stock / (2 * n)
        expected_fine = q * F
        payoff_D = current_stock / n - expected_fine

        def projected_stock(own_consume: float) -> float:
            opp_consume = p_coop * (current_stock / (2 * n)) + (1 - p_coop) * (current_stock / n)
            total_consume = own_consume + (n - 1) * opp_consume
            S_rem = current_stock - total_consume
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2 * S_rem * (1 - S_rem / capacity)
            S_next = S_rem + growth
            if S_next > capacity:
                S_next = capacity
            return S_next
        own_cons_C = current_stock / (2 * n)
        own_cons_D = current_stock / n
        S_next_C = projected_stock(own_cons_C)
        S_next_D = projected_stock(own_cons_D)
        remaining = r - t - 1
        exp_next_C = p_coop * (S_next_C / (2 * n)) + (1 - p_coop) * (S_next_C / n - expected_fine)
        exp_next_D = p_coop * (S_next_D / (2 * n)) + (1 - p_coop) * (S_next_D / n - expected_fine)
        V_C = exp_next_C * remaining
        V_D = exp_next_D * remaining
        total_C = payoff_C + V_C
        total_D = payoff_D + V_D
        return Action.D if total_D > total_C else Action.C

class Strategy_SELFISH_2(BaseStrategy):
    """
    Selfish Adaptive Strategy (SA‑Defender).

    The agent compares expected net defect payoff with cooperative payoff,
    respects a stock‑preservation threshold, and conditionally cooperates
    based on the observed cooperation rate of opponents in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.stock_min = self.capacity * 0.4

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        q = self.audit_prob
        F = self.fine
        S = float(current_stock)
        if S <= 0.0:
            return Action.D
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        c_prev = int(history.opponent_cooperators[-1]) if t > 0 else 0
        Cpay = S / (2.0 * n)
        Dpay = S / float(n)
        DpayNet = Dpay - q * F
        immediate_action = Action.D if DpayNet > Cpay else Action.C
        if S < self.stock_min:
            S_after_C = S - n * Cpay
            S_after_D = S - ((n - c_prev) * Cpay + c_prev * Dpay)
            return Action.C if S_after_C >= self.stock_min else Action.D
        if t == 0:
            return immediate_action
        coop_rate = c_prev / (n - 1) if n > 1 else 0.0
        if coop_rate >= 0.6:
            return Action.C
        else:
            return Action.D
