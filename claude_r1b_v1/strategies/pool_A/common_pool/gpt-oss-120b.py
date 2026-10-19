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

class Strategy_ALTRUISTIC_2(BaseStrategy):
    """
    Altruistic Adaptive Strategy (AAS)

    The player cooperates in the first and last round.
    In intermediate rounds it cooperates unless:
      * the stock is safely abundant,
      * the expected net gain from defecting (accounting for audit risk) exceeds the
        guaranteed cooperative payoff,
      * at least 60\u202f% of opponents cooperated in the previous round,
      * and the player has not just defected while seeing a very high cooperation level
        (≥\u202f80\u202f%) – in that case it forgives and cooperates.

    The strategy also refrains from defecting when the stock is close to collapse
    (S < 2·capacity/(2n)).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0 or t == self.r - 1:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        sustainable_threshold = self.capacity / (2 * self.n)
        if current_stock < 2 * sustainable_threshold:
            return Action.C
        opp_coop = int(history.opponent_cooperators[-1])
        p = opp_coop / (self.n - 1) if self.n > 1 else 0.0
        gain_C = current_stock / (2 * self.n)
        gain_D = current_stock / self.n - self.audit_prob * self.fine
        if gain_D <= gain_C:
            return Action.C
        if p >= 0.6:
            my_last_action = bool(history.my_actions[-1])
            if not my_last_action and p >= 0.8:
                return Action.C
            return Action.D
        return Action.C

class Strategy_ALTRUISTIC_3(BaseStrategy):
    """
    Altruistic Adaptive Policy (AA‑Policy).

    The agent cooperates by default and defects only when:
      * the pool is abundant,
      * expected net gain from defecting (Δ) is positive,
      * recent opponent cooperation is low.
    Defections are limited to at most `m` consecutive rounds and are
    followed by a forced cooperation round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game = game_description
        self.n = self.game.n_players
        self.r = self.game.n_rounds
        self.capacity = float(self.game.capacity)
        self.audit_prob = self.game.audit_prob
        self.fine = self.game.fine
        self.alpha = 0.3
        self.theta = 0.6
        self.m = 2
        self.consec_defect = 0
        self.coop_history: list[int] = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_num = history.round_number
        if round_num > 0:
            recent_coop = int(history.opponent_cooperators[-1])
            self.coop_history.append(recent_coop)
            k = min(5, round_num)
            while len(self.coop_history) > k:
                self.coop_history.pop(0)
        if round_num == self.r - 1:
            self.consec_defect = 0
            return Action.C
        if current_stock < self.alpha * self.capacity:
            self.consec_defect = 0
            return Action.C
        delta = current_stock / (2.0 * self.n) - self.audit_prob * self.fine
        if delta <= 0.0:
            self.consec_defect = 0
            return Action.C
        if not self.coop_history:
            p = 1.0
        else:
            avg_coop = sum(self.coop_history) / len(self.coop_history)
            denom = max(self.n - 1, 1)
            p = avg_coop / denom
        if p >= self.theta:
            self.consec_defect = 0
            return Action.C
        if self.consec_defect < self.m:
            self.consec_defect += 1
            return Action.D
        else:
            self.consec_defect = 0
            return Action.C

class Strategy_ALTRUISTIC_4(BaseStrategy):
    """
    Altruistic Adaptive Strategy:
    - Cooperate in the first and last rounds.
    - Cooperate when the stock falls below a safety threshold.
    - Defect only when opponents are sufficiently cooperative,
      the stock is comfortably high, and the expected fine does not
      outweigh the extra payoff from defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.beta = 0.5
        self.gamma = 0.6
        self.epsilon = 0.05

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            return Action.C
        if history.round_number == self.r - 1:
            return Action.C
        opp_coop_prev = int(history.opponent_cooperators[-1])
        if self.n > 1:
            coop_rate = opp_coop_prev / (self.n - 1)
        else:
            coop_rate = 0.0
        stock_ratio = current_stock / self.capacity if self.capacity != 0 else 0.0
        exp_fine = self.q * self.F
        low_stock = stock_ratio < self.beta - self.epsilon
        high_coop = coop_rate > self.gamma
        marginal_gain = current_stock / (2 * self.n) if self.n != 0 else 0.0
        fine_too_high = exp_fine > marginal_gain
        if low_stock:
            return Action.C
        if high_coop and (not fine_too_high):
            return Action.D
        return Action.C

class Strategy_ALTRUISTIC_5(BaseStrategy):
    """
    Altruistic Stock‑Preserving Strategy.

    The agent always cooperates (Action.C) regardless of the current stock,
    round number, opponent behavior, audits or fines. This guarantees the
    minimal per‑round extraction (stock/(2\u202fn)) and thus maximises the
    sustainability of the common pool over the whole horizon.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            The player’s own history up to (but not including) the current round.
        current_stock : float
            The stock level at the start of this round.

        Returns
        -------
        Action
            Always Action.C (cooperate).
        """
        return Action.C

class Strategy_ALTRUISTIC_6(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC) strategy.
    The player never defects; it always chooses Action.C.
    It tracks a generosity factor `g` that increments (capped at 3)
    when the proportion of cooperating opponents falls below 20%
    for two consecutive rounds. This factor is kept for potential
    analysis but does not affect the action choice.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._prev_low_coop = False
        self._g = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round.

        Parameters
        ----------
        history : PlayerHistory
            Read‑only record of past rounds for this player.
        current_stock : float
            Current common‑pool stock level (0 ≤ stock ≤ capacity).

        Returns
        -------
        Action
            Always Action.C (cooperate).
        """
        if history.round_number == 0:
            self._prev_low_coop = False
            self._g = 0
            return Action.C
        opp_coop = int(history.opponent_cooperators[-1])
        n_players = self.game_description.n_players
        if n_players > 1:
            p_coop = opp_coop / (n_players - 1)
        else:
            p_coop = 0.0
        low_coop_now = p_coop < 0.2
        if low_coop_now and self._prev_low_coop:
            self._g = min(self._g + 1, 3)
        self._prev_low_coop = low_coop_now
        return Action.C

class Strategy_ALTRUISTIC_7(BaseStrategy):
    """
    Altruistic Adaptive Strategy (AAS).

    The player cooperates when the expected net payoff from defecting
    (including expected fine) is not higher than the guaranteed cooperative
    payoff, or when the stock is healthy and recent cooperation among opponents
    is high. Otherwise it defects. In the final round audits are absent, so it
    always cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.theta_H = 0.6 * self.capacity
        self.theta_L = 0.3 * self.capacity
        self.tau = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == self.game_description.n_rounds - 1:
            q_eff = 0.0
        else:
            q_eff = self.audit_prob
        if t == 0:
            return Action.C
        if current_stock >= 2 * self.n * q_eff * self.fine:
            return Action.C
        c_prev = int(history.opponent_cooperators[-1])
        coop_rate = c_prev / (self.n - 1) if self.n > 1 else 0.0
        if current_stock >= self.theta_H and coop_rate >= self.tau:
            return Action.C
        if current_stock <= self.theta_L or coop_rate < self.tau:
            return Action.D
        return Action.D

class Strategy_ALTRUISTIC_8(BaseStrategy):
    """
    Altruistic Conditional‑Cooperation Strategy.
    Cooperates by default, defects only when the expected payoff advantage
    justifies it and the common stock is comfortably above a safety target.
    Implements limited retaliation when opponent cooperation falls below a
    tolerance, and always returns to cooperation after being fined.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.safety_factor = 0.4
        self.cooperation_tolerance = 0.6
        self.defection_tolerance = 0.2
        self.min_stock_threshold = 1e-06

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if S <= self.min_stock_threshold:
            return Action.C
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        opp_coop_count = int(history.opponent_cooperators[-1])
        opp_coop_rate = opp_coop_count / (self.n - 1) if self.n > 1 else 0.0
        my_last_action_bool = bool(history.my_actions[-1])
        my_last_action = Action.C if my_last_action_bool else Action.D
        my_last_audit = bool(history.my_audited[-1])
        r_rem = self.r - t - 1
        target_stock = self.capacity * self.safety_factor * (r_rem / self.r)
        coop_payoff = S / (2 * self.n)
        expected_fine = self.audit_prob * self.fine
        defect_payoff = S / self.n - expected_fine
        if defect_payoff <= coop_payoff:
            return Action.C
        if S > target_stock and opp_coop_rate < self.cooperation_tolerance:
            punish_prob = (self.cooperation_tolerance - opp_coop_rate) / self.cooperation_tolerance
            punish_prob = max(0.0, min(1.0, punish_prob))
            if my_last_action == Action.D and my_last_audit:
                return Action.C
            if random.random() < punish_prob:
                return Action.D
            else:
                return Action.C
        else:
            return Action.C

class Strategy_ALTRUISTIC_9(BaseStrategy):
    """
    Altruistic Conditional‑Cooperation Strategy.
    Cooperates when the stock is healthy and opponents are sufficiently cooperative,
    self‑restricts when the stock is low, and uses a temporary defect phase as a warning
    when opponents betray cooperation while the stock is still viable.
    """
    Tstock_factor = 0.6
    Mstock_factor = 0.3
    Cmin = 0.4
    Cooldown = 2

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        capacity = float(game_description.capacity)
        self.Tstock = self.Tstock_factor * capacity
        self.Mstock = self.Mstock_factor * capacity
        self.mode = 'Cooperate'
        self.cooldown_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        n_players = self.game_description.n_players
        n_opponents = n_players - 1
        round_idx = history.round_number
        total_rounds = self.game_description.n_rounds
        if round_idx == 0:
            self.mode = 'Cooperate'
            self.cooldown_counter = 0
            return Action.C
        if round_idx == total_rounds - 1:
            self.mode = 'Cooperate'
            self.cooldown_counter = 0
            return Action.C
        opp_coop = int(history.opponent_cooperators[-1])
        p = opp_coop / n_opponents if n_opponents > 0 else 0.0
        if self.mode == 'Defect':
            action = Action.D
            self.cooldown_counter -= 1
            if self.cooldown_counter <= 0:
                self.mode = 'Cooperate'
                self.cooldown_counter = 0
            return action
        S = current_stock
        if S >= self.Tstock and p >= self.Cmin:
            return Action.C
        if S < self.Mstock:
            return Action.C
        self.mode = 'Defect'
        self.cooldown_counter = self.Cooldown
        return Action.D

class Strategy_ALTRUISTIC_10(BaseStrategy):
    """
    Adaptive Altruist (AA) strategy for the Common Pool Resource game.

    The player starts cooperating, then decides each round based on:
      • Expected net gain from defecting (considering the expected fine).
      • The observed cooperation rate of the whole population so far.
      • The current stock relative to a safety threshold.

    It cooperates in the first and last rounds, and otherwise follows the
    conditional rules described in the specification.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.safety_stock = 0.4 * self.capacity
        self.coop_thr = 0.6
        self.def_thr = 0.3
        self.exp_fine = self.q * self.F

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0 or t == self.r - 1:
            return Action.C
        S = float(current_stock)
        gain_defect = S / (2.0 * self.n) if self.n > 0 else 0.0
        net_gain_defect = gain_defect - self.exp_fine
        if net_gain_defect <= 0.0:
            return Action.C
        total_moves = self.n * t
        if total_moves == 0:
            obs_coop_rate = 0.0
        else:
            my_coops = int(np.sum(history.my_actions))
            opp_coops = int(np.sum(history.opponent_cooperators))
            obs_coop_rate = (my_coops + opp_coops) / total_moves
        if S < self.safety_stock:
            if obs_coop_rate >= self.coop_thr:
                return Action.C
            else:
                return Action.D
        elif obs_coop_rate <= self.def_thr:
            return Action.D
        else:
            return Action.C

class Strategy_ALTRUISTIC_11(BaseStrategy):
    """
    Altruistic Adaptive Cooperation (AAC).

    The agent starts by cooperating. It tracks the average cooperation rate of opponents
    and the current stock level to decide whether to continue cooperating or, with a
    probability that grows when opponents defect, to defect for a possible short‑term
    gain after accounting for expected audit fines.
    """
    low_stock_threshold = 0.3
    coop_tolerance = 0.7
    min_coop_rate = 0.3
    min_stock_threshold = 0.1

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        stock = float(current_stock)
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        opp_counts = history.opponent_cooperators
        avg_opp_coop_rate = float(opp_counts.sum()) / ((self.n - 1) * t)
        stock_ratio = stock / self.capacity if self.capacity != 0 else 0.0
        if stock_ratio < self.low_stock_threshold:
            return Action.C
        if avg_opp_coop_rate >= self.coop_tolerance:
            return Action.C
        denom = self.coop_tolerance - self.min_coop_rate
        if denom <= 0:
            p_conditional = 0.0
        else:
            p_conditional = (avg_opp_coop_rate - self.min_coop_rate) / denom
            p_conditional = max(0.0, min(1.0, p_conditional))
        expected_fine = self.audit_prob * self.fine
        defect_consumption = stock / self.n
        coop_consumption = stock / (2 * self.n)
        net_gain = defect_consumption - expected_fine - coop_consumption
        if net_gain > 0.0 and random.random() < p_conditional:
            return Action.D
        else:
            return Action.C

class Strategy_ALTRUISTIC_12(BaseStrategy):
    """
    Altruistic Common‑Pool Strategy:
    Cooperates when the stock is healthy and opponents are not overly selfish,
    otherwise defects only under limited, calculated conditions.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.threshold_high = 0.6 * self.capacity
        self.threshold_low = 0.3 * self.capacity
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.expected_fine = self.audit_prob * self.fine
        self.epsilon = 0.15
        self.defect_ratio_high = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        coops_prev = int(history.opponent_cooperators[-1])
        opponents = self.n - 1
        defects_prev = opponents - coops_prev
        defect_ratio = defects_prev / opponents if opponents > 0 else 0.0
        stock_t = float(current_stock)
        if stock_t >= self.threshold_high:
            if defect_ratio <= self.defect_ratio_high:
                return Action.C
            coop_payoff = stock_t / (2.0 * self.n)
            defect_payoff = stock_t / self.n
            gain_defect = defect_payoff - (coop_payoff + self.expected_fine)
            if gain_defect >= 0.1 * defect_payoff:
                return Action.D
            else:
                return Action.C
        if stock_t <= self.threshold_low:
            return Action.C
        coop_prob = max(0.0, 1.0 - defect_ratio) + self.epsilon
        coop_prob = min(1.0, coop_prob)
        if t % 2 == 0 and coop_prob >= 0.5 or (t % 2 == 1 and coop_prob > 0.5):
            return Action.C
        else:
            return Action.D

class Strategy_ALTRUISTIC_13(BaseStrategy):
    """
    Altruistic Sustainable‑Cooperation Strategy.

    The agent starts by cooperating, keeps a moving average of the opponents'
    cooperation ratio, and defects only when the common pool is plentiful,
    the expected net gain from defecting is positive, and opponents have been
    sufficiently cooperative.  Adaptive punishment makes the average
    cooperation threshold harder to reach after consecutive rounds of low
    cooperation.  If the stock ever falls below a quarter of capacity the
    agent switches to forced cooperation for the rest of the game.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.safety_level = self.capacity / 2.0
        self.defect_threshold = 2.0 * self.n * self.q * self.F
        self.emergency = False
        self.avg_coop = 1.0
        self.base_lambda = 0.3
        self.punish_lambda = 0.6
        self.current_lambda = self.base_lambda
        self.low_coop_streak = 0
        self.punish_timer = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if not self.emergency and current_stock < self.capacity / 4.0:
            self.emergency = True
        if history.round_number > 0:
            opp_coop_cnt = int(history.opponent_cooperators[-1])
            opp_coop_ratio = opp_coop_cnt / (self.n - 1) if self.n > 1 else 0.0
            if opp_coop_ratio < 0.3:
                self.low_coop_streak += 1
            else:
                self.low_coop_streak = 0
            if self.low_coop_streak >= 2 and self.punish_timer == 0:
                self.punish_timer = 3
            self.current_lambda = self.punish_lambda if self.punish_timer > 0 else self.base_lambda
            self.avg_coop = self.current_lambda * opp_coop_ratio + (1.0 - self.current_lambda) * self.avg_coop
            if self.punish_timer > 0:
                self.punish_timer -= 1
        t = history.round_number
        if t == 0 or t == self.r - 1:
            return Action.C
        if self.emergency:
            return Action.C
        if current_stock < self.safety_level:
            return Action.C
        delta = current_stock / (2.0 * self.n) - self.q * self.F
        if delta > 0.0 and current_stock > self.defect_threshold and (self.avg_coop >= 0.6):
            return Action.D
        return Action.C

class Strategy_ALTRUISTIC_14(BaseStrategy):
    """
    Altruistic Adaptive Cooperation Strategy.

    The agent cooperates by default, defects only with a bounded probability
    when the stock is abundant, recent cooperation is low, and the expected
    extra gain from defecting exceeds the expected fine.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.tau = 0.5 * self.capacity
        self.gamma = 0.6
        self.p_max = 0.2
        self.expected_fine = self.audit_prob * self.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        S = float(current_stock)
        if t == self.r - 1:
            return Action.C
        if t == 0:
            coop_confidence = 1.0
        else:
            C_prev = int(history.opponent_cooperators[-1])
            coop_confidence = C_prev / (self.n - 1) if self.n > 1 else 1.0
        if S < self.tau:
            return Action.C
        if coop_confidence >= self.gamma:
            return Action.C
        delta = S / (2.0 * self.n) if self.n else 0.0
        if self.expected_fine >= delta:
            return Action.C
        if delta <= 0.0:
            return Action.C
        stock_ratio = S / self.capacity if self.capacity != 0 else 0.0
        p_defect_raw = (1.0 - coop_confidence) * stock_ratio * (delta - self.expected_fine) / delta
        p_defect = min(self.p_max, max(0.0, p_defect_raw))
        if random.random() < p_defect:
            return Action.D
        else:
            return Action.C

class Strategy_ALTRUISTIC_15(BaseStrategy):
    """
    Altruistic Conditional Cooperation (ACC).

    Starts with cooperation, cooperates in the final round, and otherwise
    defects only when the expected net personal gain is positive,
    the stock stays above a safety margin, and opponents have shown
    sufficient cooperation in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.beta = 0.3
        self.alpha = 0.8

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            return Action.C
        if history.round_number == self.r - 1:
            return Action.C
        coop_others_last = int(history.opponent_cooperators[-1])
        coop_ratio = coop_others_last / (self.n - 1) if self.n > 1 else 0.0
        S = float(current_stock)
        p = S / self.capacity if self.capacity != 0 else 0.0
        extra_gain = S / (2.0 * self.n)
        expected_fine = self.audit_prob * self.fine
        net_gain = extra_gain - expected_fine
        defect_others = self.n - 1 - coop_others_last
        cons_others = coop_others_last * (S / (2.0 * self.n)) + defect_others * (S / self.n)
        cons_self_C = S / (2.0 * self.n)
        S_C = S - (cons_others + cons_self_C)
        if S_C < 0.0:
            S_C = 0.0
        growth_C = 2.0 * S_C * (1.0 - S_C / self.capacity) if self.capacity != 0 else 0.0
        S_next_C = min(S_C + growth_C, self.capacity)
        cons_self_D = S / self.n
        S_D = S - (cons_others + cons_self_D)
        if S_D < 0.0:
            S_D = 0.0
        growth_D = 2.0 * S_D * (1.0 - S_D / self.capacity) if self.capacity != 0 else 0.0
        S_next_D = min(S_D + growth_D, self.capacity)
        if net_gain > 0.0 and p >= self.beta and ((S_D / self.capacity if self.capacity != 0 else 0.0) >= self.beta) and (coop_ratio >= 0.6):
            return Action.D
        else:
            return Action.C

class Strategy_ALTRUISTIC_16(BaseStrategy):
    """
    Altruistic Adaptive Common‑Pool Strategy (AACPS).

    Uses the observed stock transition to infer how many opponents
    cooperated in the previous round and adapts cooperation based on
    stock health, majority cooperation, audit risk and remaining horizon.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.prev_stock = None
        self.last_coop_ratio = 0.0

    def _expected_fine_cost(self) -> float:
        """Expected fine per defect (audit probability × fine)."""
        return self.q * self.F

    def _estimate_cooperators(self, prev_stock: float, next_stock: float) -> int:
        """
        Estimate how many opponents cooperated in the previous round
        from the stock transition (prev_stock -> next_stock after growth).
        """
        if prev_stock <= 0:
            return 0
        if next_stock >= self.capacity:
            rem = self.capacity
        else:
            denom = 1.0 + 2.0 * (1.0 - next_stock / self.capacity)
            rem = next_stock / denom if denom != 0 else 0.0
        C = 2.0 * self.n * rem / prev_stock
        C_int = int(round(C))
        return max(0, min(self.n, C_int))

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_idx = history.round_number
        if round_idx == 0:
            self.prev_stock = float(current_stock)
            return Action.C
        if self.prev_stock is not None:
            estimated_coop = self._estimate_cooperators(self.prev_stock, float(current_stock))
            self.last_coop_ratio = estimated_coop / self.n
        self.prev_stock = float(current_stock)
        if current_stock <= 0.0:
            return Action.C
        if round_idx == self.r - 1:
            return Action.C
        stock_t = float(current_stock)
        stock_threshold = self.capacity * 0.5
        coop_threshold = 0.6
        future_factor = (self.r - round_idx) / self.r
        stock_ok = stock_t >= stock_threshold
        majority_ok = self.last_coop_ratio >= coop_threshold
        fine_ok = self._expected_fine_cost() < stock_t / (4.0 * self.n)
        condition = stock_ok and majority_ok or (future_factor > 0.7 and stock_t > 0.3 * self.capacity and (self.last_coop_ratio > 0.4))
        condition = condition and fine_ok
        return Action.C if condition else Action.D

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

class Strategy_BENEVOLENT_2(BaseStrategy):
    """
    Adaptive Benevolent Cooperation (ABC).

    The player cooperates while:
      • the stock is at least 40\u202f% of capacity,
      • at least 50\u202f% of opponents cooperated in the previous round,
      • the stock has not dropped more than 15\u202f% since the previous round.

    If any condition fails the player defects only when the expected monetary
    gain (defect payoff minus cooperate payoff minus expected fine) is positive;
    otherwise it stays cooperative.  In the final round the player always
    cooperates (no audit, no future stock concerns).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._theta_stock = 0.4 * game_description.capacity
        self._theta_opp = 0.5
        self._theta_decline = 0.15
        self._prev_stock = float(game_description.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        r = self.game_description.n_rounds
        if t == r - 1:
            chosen = Action.C
        elif current_stock == 0:
            chosen = Action.C
        else:
            stock_ok = current_stock >= self._theta_stock
            if t == 0:
                c_prev = 0.0
            else:
                coop_cnt = int(history.opponent_cooperators[-1])
                c_prev = coop_cnt / (n - 1) if n > 1 else 0.0
            opp_ok = c_prev >= self._theta_opp
            if t > 0:
                S_prev = self._prev_stock
                if S_prev > 0:
                    decline = (S_prev - current_stock) / S_prev
                    trend_ok = decline <= self._theta_decline
                else:
                    trend_ok = True
            else:
                trend_ok = True
            if stock_ok and opp_ok and trend_ok:
                chosen = Action.C
            else:
                expected_fine = q * F
                net_gain = current_stock / n - current_stock / (2 * n) - expected_fine
                chosen = Action.D if net_gain > 0 else Action.C
        self._prev_stock = float(current_stock)
        return chosen

class Strategy_BENEVOLENT_3(BaseStrategy):
    """
    Benevolent Sustainable‑Cooperation (BSC) strategy.

    Cooperates by default, defects only when the pool is comfortably
    above a safety margin, the expected net gain from defection is
    positive, and the projected stock after a single defection would
    still stay above the safety margin.  The strategy also reacts to
    low cooperation of opponents by defecting for a single round,
    otherwise it resumes evaluation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.safety = 0.4 * self.capacity
        self.auditThresh = 2 * self.n * self.q * self.F
        self.minStock = max(self.safety, self.auditThresh)
        self.tolerance = 0.6

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        stock = max(0.0, float(current_stock))
        opp_coop_last = int(history.opponent_cooperators[-1])
        opp_ratio = opp_coop_last / (self.n - 1) if self.n > 1 else 1.0
        if opp_ratio < self.tolerance:
            return Action.D
        delta_payoff = stock / self.n - stock / (2 * self.n)
        exp_fine = self.q * self.F
        net_gain = delta_payoff - exp_fine
        if stock < self.minStock or net_gain <= 0:
            return Action.C
        total_consumption_if_one_D = stock / 2.0 + stock / (2.0 * self.n)
        stock_remaining = stock - total_consumption_if_one_D
        growth = 2.0 * stock_remaining * (1.0 - stock_remaining / self.capacity)
        projected_stock = min(stock_remaining + growth, self.capacity)
        if projected_stock >= self.safety:
            return Action.D
        return Action.C

class Strategy_BENEVOLENT_4(BaseStrategy):
    """
    Sustainable Reciprocity with Risk‑Adjusted Defection.

    The strategy cooperates to protect the common pool when stock is low,
    rewards sustained opponent cooperation, and allows a controlled,
    risk‑adjusted probability of defection when the stock is abundant and
    opponents are mostly defecting.  It also accounts for the expected fine
    from audits when evaluating the profitability of defection.
    """
    _THETA_LOW = 0.35
    _THETA_HIGH = 0.75
    _THETA_COOP = 0.6

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def _max_defect_fraction(self, stock: float, remaining_rounds: int) -> float:
        """
        Upper bound on the probability of defecting in a given round such that,
        even in the worst case (defect every remaining round), the stock will
        not fall below the low‑stock threshold at the end of the game.
        """
        if self.capacity == 0:
            return 0.0
        worst_consumption = remaining_rounds * (stock / self.n)
        required_stock = self._THETA_LOW * self.capacity
        if stock - worst_consumption >= required_stock:
            return 1.0
        if worst_consumption <= 0:
            return 0.0
        return max(0.0, (stock - required_stock) / worst_consumption)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if self.n <= 0 or self.r <= 0 or self.capacity <= 0:
            return Action.C
        stock = max(0.0, float(current_stock))
        stock_ratio = stock / self.capacity if self.capacity > 0 else 0.0
        expected_fine = self.audit_prob * self.fine
        gain_C = stock / (2 * self.n) if self.n > 0 else 0.0
        net_gain_D = stock / self.n - expected_fine if self.n > 0 else -expected_fine
        delta_gain = 0.05 * gain_C
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C if gain_C >= net_gain_D else Action.D
        if t > 0:
            total_coop = int(np.sum(history.opponent_cooperators[:t]))
            coop_ratio = total_coop / ((self.n - 1) * t) if self.n > 1 else 0.0
        else:
            coop_ratio = 0.0
        if stock_ratio < self._THETA_LOW:
            return Action.C
        if t > 0 and coop_ratio > self._THETA_COOP:
            return Action.C
        if stock_ratio > self._THETA_HIGH and (t == 0 or coop_ratio < self._THETA_COOP):
            remaining = self.r - t
            safe_defect_prob = self._max_defect_fraction(stock, remaining)
            if net_gain_D - gain_C > delta_gain and random.random() < safe_defect_prob:
                return Action.D
            else:
                return Action.C
        return Action.C

class Strategy_BENEVOLENT_5(BaseStrategy):
    """
    Benevolent Conditional Cooperation (BCC)

    Starts by cooperating. In subsequent rounds it defects only when:
      - the remaining stock is comfortably above a dynamic safety threshold,
      - the expected monetary gain from defecting (extra consumption minus expected fine) is positive,
      - at least half of the opponents cooperated in the previous round.

    Otherwise it cooperates, never defects in the final round, and always cooperates
    when the stock is depleted or the expected gain is non‑positive.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._n_opponents = max(1, self.game_description.n_players - 1)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if t == 0:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.C
        c_prev = int(history.opponent_cooperators[-1])
        f_prev = c_prev / self._n_opponents
        extra_gain = current_stock / (2.0 * n)
        expected_fine = q * F
        net_gain = extra_gain - expected_fine
        t_remaining = r - t
        min_safe_stock = capacity * 0.3 * (t_remaining / r)
        if net_gain <= 0.0:
            return Action.C
        if current_stock >= min_safe_stock and net_gain > 0.0 and (f_prev >= 0.5):
            return Action.D
        return Action.C

class Strategy_BENEVOLENT_6(BaseStrategy):
    """
    Benevolent Adaptive Strategy:
    - Cooperates by default, uses defection only as a signal when opponents
      exceed a tolerated number of defections and the stock is healthy.
    - Never defects when the stock falls below a safety reserve.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self._alpha = 0.4
        self._beta = 0.2
        self.safety_stock = self._alpha * self.capacity
        self.tolerance_def = math.ceil(self._beta * self.n)
        self.prev_defectors = 0
        self.my_last_action = None

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        if t > 0:
            last_op_coop = int(history.opponent_cooperators[-1])
            self.prev_defectors = self.n - 1 - last_op_coop
            self.my_last_action = Action.C if bool(history.my_actions[-1]) else Action.D
        if t == self.r - 1:
            action = Action.C
        elif t == 0:
            action = Action.C
        elif current_stock < self.safety_stock:
            action = Action.C
        elif self.prev_defectors > self.tolerance_def:
            action = Action.D
        else:
            expected_gain = current_stock / (2.0 * self.n) - self.q * self.F
            action = Action.C
        self.my_last_action = action
        return action

class Strategy_BENEVOLENT_7(BaseStrategy):
    """
    Benevolent Adaptive Common‑Pool Strategy.

    Starts cooperatively, uses expected‑fine adjusted payoff comparison,
    observes recent cooperation, and switches to defection only when the
    stock is low or when defection is not economically worse than cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.r = game_description.n_rounds
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.theta_target = 0.6
        self.theta_min = 0.2
        self.rho_min = 0.3
        self.S_target = self.theta_target * self.capacity
        self.S_min = self.theta_min * self.capacity
        self.delta_cons = self.capacity * (self.theta_target - self.theta_min)
        self.expected_fine = self.audit_prob * self.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        S = max(0.0, float(current_stock))
        if S == 0.0:
            return Action.C
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        c_obs = int(history.opponent_cooperators[-1])
        if self.n > 1:
            rho = c_obs / (self.n - 1)
        else:
            rho = 1.0
        pi_C = S / (2 * self.n) if self.n else 0.0
        pi_D_raw = S / self.n if self.n else 0.0
        pi_D = pi_D_raw - self.expected_fine
        if S <= self.S_min:
            return Action.D
        if pi_D <= pi_C:
            return Action.C
        if rho < self.rho_min:
            if S > self.S_target:
                return Action.C
            else:
                return Action.D
        return Action.C

class Strategy_BENEVOLENT_8(BaseStrategy):
    """
    Benevolent Adaptive Strategy:
    Starts and ends with cooperation, cooperates when stock is low,
    otherwise defects only if the extra expected gain (after expected fine)
    is positive, the stock is comfortably high, and opponents have
    historically cooperated at or above the target level.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.S_safe = 0.3 * self.capacity
        self.S_high = 0.5 * self.capacity
        self.p_target = 0.8

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0 or t == self.r - 1:
            return Action.C
        if S <= 0.0:
            return Action.C
        if S < self.S_safe:
            return Action.C
        if t > 0:
            total_coop_opponents = int(np.sum(history.opponent_cooperators))
            denom = t * (self.n - 1)
            p_coop = total_coop_opponents / denom if denom > 0 else 1.0
        else:
            p_coop = 1.0
        extra_consumption = S / (2.0 * self.n)
        expected_fine = self.audit_prob * self.fine
        expected_gain_defect = extra_consumption - expected_fine
        if expected_gain_defect > 0.0 and S > self.S_high and (p_coop >= self.p_target):
            return Action.D
        else:
            return Action.C

class Strategy_BENEVOLENT_9(BaseStrategy):
    """
    Benevolent Adaptive Strategy (BAS)

    Cooperates by default, retaliates for a limited number of rounds when a
    sufficient fraction of opponents defect, preserves the stock when it is
    critically low, and defects only when the expected monetary gain after
    accounting for audit risk exceeds cooperating.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.epsilon_low = 0.1
        self.theta_defect = 0.5
        self.tau_retaliate = 2
        self.retaliation_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        round_num = history.round_number
        if round_num == 0:
            self.retaliation_counter = 0
            return Action.C
        if round_num == self.r - 1:
            return Action.C
        S = float(current_stock)
        if S < self.epsilon_low * self.capacity:
            return Action.C
        if self.retaliation_counter > 0:
            self.retaliation_counter -= 1
            return Action.D
        c_prev = int(history.opponent_cooperators[-1])
        n_opponents = self.n - 1
        if n_opponents > 0:
            delta_defection = (n_opponents - c_prev) / n_opponents
        else:
            delta_defection = 0.0
        if delta_defection >= self.theta_defect:
            self.retaliation_counter = self.tau_retaliate
            return Action.D
        delta = S / (2 * self.n)
        phi = S / self.n - self.audit_prob * self.fine
        if delta >= phi:
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_10(BaseStrategy):
    """
    Benevolent Conditional Cooperation.
    Cooperates in the first and last round. In intermediate rounds,
    cooperates unless the stock is above a safety threshold (θ_low) and
    the proportion of defecting opponents in the previous round exceeds τ,
    in which case it defects for that round only.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.theta_low = 0.35 * self.capacity
        self.theta_high = 0.7 * self.capacity
        self.tau = 0.2

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        if current_stock <= self.theta_low:
            return Action.C
        coop_prev = int(history.opponent_cooperators[-1])
        opponents = self.n - 1
        defect_prev = opponents - coop_prev
        p_def = defect_prev / opponents if opponents > 0 else 0.0
        if p_def > self.tau:
            return Action.D
        return Action.C

class Strategy_BENEVOLENT_11(BaseStrategy):
    """
    Benevolent Adaptive Strategy.

    The player cooperates in the last round and otherwise decides based on:
      * immediate extra payoff from defecting (Δ_t) vs expected fine (E_t)
      * a safety‑margin stock threshold T_t that ensures survivable cooperation
      * a bias derived from the average opponent cooperation rate.

    Defect only when the net expected gain exceeds the bias **and** the current
    stock is above the safety threshold.  Otherwise cooperate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.C
        R = self.r - t
        if t == 0:
            avg_coop = 0.0
        else:
            coop_fractions = history.opponent_cooperators[:t] / (self.n - 1)
            avg_coop = float(coop_fractions.mean())
        delta_t = current_stock / (2.0 * self.n)
        expected_fine = self.q * self.F
        half_stock = current_stock / 2.0
        growth = 2.0 * half_stock * (1.0 - half_stock / self.capacity)
        stock_coop_next = min(half_stock + growth, self.capacity)
        T_t = R / self.r * stock_coop_next
        safe = current_stock >= T_t
        bias = 0.2 * (0.5 - avg_coop)
        net_gain = delta_t - expected_fine
        if net_gain > bias and safe:
            return Action.D
        else:
            return Action.C

class Strategy_BENEVOLENT_12(BaseStrategy):
    """
    Benevolent Adaptive Strategy.

    - Cooperate in the first and last round.
    - In intermediate rounds follow a hierarchy of safeguards, sustainability
      checks, expected‑payoff comparison, and opponent‑cooperation reinforcement.
    - Defect only when the expected net payoff from defecting exceeds cooperation
      **and** the projected stock after defecting stays above a modest safety floor.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return Action.C (cooperate) or Action.D (defect) for the current round."""
        n = self.game_description.n_players
        rounds = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        if capacity == 0.0:
            return Action.C
        t = history.round_number
        if t == 0:
            return Action.C
        if t == rounds - 1:
            return Action.C
        c_opponents = int(history.opponent_cooperators[-1])
        p_coop = c_opponents / (n - 1)
        exp_fine = audit_prob * fine
        delta_C = current_stock / (2 * n)
        delta_D = current_stock / n - exp_fine

        def projected_stock(my_coop: bool) -> float:
            """Return the stock at the start of the next round if we act as `my_coop`."""
            cons_me = current_stock / (2 * n) if my_coop else current_stock / n
            cons_opp_coop = current_stock / (2 * n)
            cons_opp_def = current_stock / n
            total_consumption = cons_me + c_opponents * cons_opp_coop + (n - 1 - c_opponents) * cons_opp_def
            remain = current_stock - total_consumption
            if remain < 0.0:
                remain = 0.0
            growth = 2.0 * remain * (1.0 - remain / capacity)
            next_stock = remain + growth
            if next_stock > capacity:
                next_stock = capacity
            return next_stock
        if current_stock <= 0.1 * capacity:
            return Action.C
        stock_next_if_coop = projected_stock(my_coop=True)
        if stock_next_if_coop - 0.5 * capacity >= 0.0:
            return Action.C
        if delta_D <= delta_C:
            return Action.C
        if p_coop >= 0.6:
            return Action.C
        stock_next_if_def = projected_stock(my_coop=False)
        if delta_D > delta_C and stock_next_if_def >= 0.3 * capacity:
            return Action.D
        return Action.C

class Strategy_BENEVOLENT_13(BaseStrategy):
    """
    Benevolent Adaptive Strategy for the Common Pool Resource Game.
    Uses observed opponent cooperation, stock levels, and audit parameters
    to decide when to cooperate or defect, limiting defections to short bursts.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.Cap = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.n_players = int(game_description.n_players)
        self.high_stock = 0.7 * self.Cap
        self.medium_stock = 0.4 * self.Cap
        self.low_stock = 0.2 * self.Cap
        self.cooperate_rate_threshold_high = 0.6
        self.cooperate_rate_threshold_low = 0.3
        self.max_consecutive_defect = 2
        self.last_defect = False
        self.consec_defect = 0

    def _opp_coop_rate(self, history: PlayerHistory, m: int) -> float:
        """Average proportion of opponents that cooperated over the last m rounds."""
        rounds_available = history.round_number
        if rounds_available == 0:
            return 0.0
        m = min(m, rounds_available)
        recent = history.opponent_cooperators[-m:]
        denom = self.n_players - 1
        if denom == 0:
            return 0.0
        proportions = recent.astype(float) / denom
        return float(np.mean(proportions))

    def _expected_defect_payoff(self, stock: float) -> float:
        """Expected payoff from defecting, accounting for audit probability."""
        return stock / self.n_players - self.q * self.F

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            self.last_defect = False
            self.consec_defect = 0
            return Action.C
        if history.round_number == self.game_description.n_rounds - 1:
            self.last_defect = False
            self.consec_defect = 0
            return Action.C
        stock = float(current_stock)
        coop_payoff = stock / (2.0 * self.n_players)
        if self._expected_defect_payoff(stock) <= coop_payoff:
            self.last_defect = False
            self.consec_defect = 0
            return Action.C
        if stock <= self.low_stock:
            self.last_defect = False
            self.consec_defect = 0
            return Action.C
        recent_m = min(3, history.round_number)
        opp_rate = self._opp_coop_rate(history, recent_m)
        if opp_rate >= self.cooperate_rate_threshold_high and stock >= self.medium_stock:
            self.last_defect = False
            self.consec_defect = 0
            return Action.C
        if opp_rate <= self.cooperate_rate_threshold_low and stock >= self.high_stock:
            if self.consec_defect < self.max_consecutive_defect:
                self.last_defect = True
                self.consec_defect += 1
                return Action.D
            else:
                self.last_defect = False
                self.consec_defect = 0
                return Action.C
        self.last_defect = False
        self.consec_defect = 0
        return Action.C

class Strategy_BENEVOLENT_14(BaseStrategy):
    """
    Benevolent Adaptive Cooperation (BAC) strategy.

    The agent cooperates in the first and last round.
    In intermediate rounds it balances the observed stock level,
    the recent cooperation of opponents, and its own recent
    defections.  Defections are limited to a sliding window of length K=3
    and are only chosen when the stock is moderate, opponent cooperation
    is low and the net incentive (including expected audit fines) is
    negative.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.S_safe = 0.5 * self.capacity
        self.S_critical = 0.2 * self.capacity
        self.p_coop_min = 0.6
        self.E_fine = self.q * self.F
        self.K = 3

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0 or t == self.r - 1:
            return Action.C
        stock_t = float(current_stock)
        opp_cnt = int(history.opponent_cooperators[-1])
        opp_total = self.n - 1
        opp_coop_t = opp_cnt / opp_total if opp_total > 0 else 0.0
        start = max(0, t - self.K)
        recent_actions = history.my_actions[start:t]
        defect_history = int(np.sum(~recent_actions))
        incentive = stock_t / self.capacity
        denominator = stock_t / self.n if stock_t > 0 else 1.0
        incentive -= self.E_fine / denominator
        incentive += opp_coop_t - defect_history / self.K
        if stock_t <= self.S_critical:
            return Action.C
        elif stock_t >= self.S_safe and opp_coop_t >= self.p_coop_min and (incentive > 0):
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_15(BaseStrategy):
    """
    Benevolent Adaptive Cooperation Strategy.

    Starts by cooperating, updates a trust measure based on the observed
    fraction of cooperating opponents, and decides each round whether to
    cooperate or defect using:
    - a safety‑stock rule,
    - the expected net gain from defection (including expected fines),
    - and a probability that blends trust with the penalised incentive to defect.
    The final round is always cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.trust: float = 1.0
        self.last_observed_coop: float | None = None

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        if history.round_number == 0:
            return Action.C
        if history.round_number == r - 1:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        opp_coop_cnt = history.opponent_cooperators[-1]
        observed_coop = opp_coop_cnt / (n - 1) if n > 1 else 0.0
        self.last_observed_coop = observed_coop
        self.trust = 0.9 * self.trust + 0.1 * observed_coop
        payoff_C = current_stock / (2.0 * n)
        payoff_D = current_stock / float(n)
        expected_fine = q * F
        net_gain_D = payoff_D - expected_fine - payoff_C
        safety_threshold = 0.25 * capacity
        if current_stock < safety_threshold:
            return Action.C
        cooperate_score = self.trust - 0.5 * max(net_gain_D, 0.0)
        prob_cooperate = max(0.0, min(cooperate_score, 1.0))
        if random.random() < prob_cooperate:
            return Action.C
        else:
            return Action.D

class Strategy_BENEVOLENT_16(BaseStrategy):
    """
    Benevolent Adaptive Strategy for the Common Pool Resource game.
    The agent cooperates initially and later decides based on:
      • observed cooperation rate of opponents,
      • a linearly decreasing target stock,
      • expected payoff accounting for audit risk.
    It prefers actions that keep the projected stock at or above the target,
    defecting only when it is safe and profitable after expected fines.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.theta = 0.5
        self.E_fine = self.audit_prob * self.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        stock = float(current_stock)
        if stock <= 1e-06:
            return Action.C
        if t == 0:
            return Action.C
        if t == self.r - 1:
            if stock <= self.capacity / (2 * self.n):
                return Action.C
            else:
                return Action.D
        coop_opponents = int(history.opponent_cooperators[-1])
        opponents = self.n - 1
        coop_rate = coop_opponents / opponents if opponents > 0 else 0.0
        c_self = stock / (2 * self.n)
        d_self = stock / self.n
        c_opp = stock / (2 * self.n)
        d_opp = stock / self.n
        exp_opp_cons = coop_rate * c_opp + (1.0 - coop_rate) * d_opp
        total_cons_C = c_self + opponents * exp_opp_cons
        total_cons_D = d_self + opponents * exp_opp_cons
        s_after_C = max(stock - total_cons_C, 0.0)
        s_after_D = max(stock - total_cons_D, 0.0)
        growth_C = 2.0 * s_after_C * (1.0 - s_after_C / self.capacity)
        growth_D = 2.0 * s_after_D * (1.0 - s_after_D / self.capacity)
        proj_stock_C = min(s_after_C + growth_C, self.capacity)
        proj_stock_D = min(s_after_D + growth_D, self.capacity)
        target = self.theta * self.capacity * ((self.r - t) / self.r)
        feasible_C = proj_stock_C >= target
        feasible_D = proj_stock_D >= target
        payoff_C = c_self
        payoff_D = d_self - self.E_fine
        if feasible_C and (not feasible_D or payoff_C >= payoff_D):
            return Action.C
        if feasible_D:
            return Action.D
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

class Strategy_COLLECTIVE_3(BaseStrategy):
    """
    Collective Safeguard strategy.
    Cooperates when the stock is low or below a safety threshold.
    Defects in the final round (never audited) and otherwise only
    defects when the stock is abundant (≥\u202fT) *and* the recent average
    number of cooperating opponents is at least half of the total players.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.T = max(2 * self.n * self.q * self.F, self.capacity * 0.3)
        self.alpha = 0.05
        self.stock_t = self.capacity
        self.coop_history: list[bool] = []
        self.opp_coop_counts: list[int] = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            self.coop_history.append(bool(history.my_actions[-1]))
            self.opp_coop_counts.append(int(history.opponent_cooperators[-1]))
        self.stock_t = float(current_stock)
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if self.stock_t <= self.alpha * self.capacity:
            return Action.C
        if self.stock_t >= self.T:
            k = min(3, t)
            if k == 0:
                majority_coop = True
            else:
                recent = self.opp_coop_counts[-k:]
                avg = sum(recent) / k
                majority_coop = avg >= self.n / 2
            return Action.D if majority_coop else Action.C
        return Action.C

class Strategy_COLLECTIVE_4(BaseStrategy):
    """
    Sustainable Collective‑Threshold (SCT) strategy.

    - Starts with cooperation.
    - Protects the resource when stock falls below a safety buffer.
    - Defects when opponents have been insufficiently cooperative
      and the expected net gain (Δ) from defecting is positive.
    - Otherwise defects only on scheduled slots (≤\u202f20\u202f% of rounds)
      when Δ > 0.
    - Adapts to consecutive low/high cooperation streaks:
        * 3+ low‑cooperation rounds → continuous defection.
        * 3+ high‑cooperation rounds → return to the base schedule.
    - In the final round, defects if Δ > 0, else cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.alpha = 0.5
        self.theta = 0.6
        self.K = max(1, math.floor(self.r * 0.2))
        self.low_streak = 0
        self.high_streak = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n
        B = self.alpha * self.capacity
        if t == 0:
            return Action.C
        last_coop_cnt = history.opponent_cooperators[-1]
        obsCoop = last_coop_cnt / (n - 1) if n > 1 else 0.0
        if obsCoop < self.theta:
            self.low_streak += 1
        else:
            self.low_streak = 0
        if obsCoop > self.theta:
            self.high_streak += 1
        else:
            self.high_streak = 0
        if t == self.r - 1:
            delta = current_stock / n - self.q * self.F - current_stock / (2 * n)
            return Action.D if delta > 0 else Action.C
        delta = current_stock / n - self.q * self.F - current_stock / (2 * n)
        if self.low_streak >= 3:
            return Action.D
        if current_stock < B:
            return Action.C
        if obsCoop < self.theta:
            return Action.D if delta > 0 else Action.C
        if delta > 0:
            if t % self.K == 0:
                return Action.D
            else:
                return Action.C
        else:
            return Action.C

class Strategy_COLLECTIVE_5(BaseStrategy):
    """
    Adaptive Collective Stewardship (ACS).

    The agent cooperates when defecting is not strictly better after accounting for the
    expected fine, or when the stock is low or the observed cooperation level among opponents
    meets a stock‑dependent target. In the final round it always defects to maximise the
    immediate payoff.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == self.r - 1:
            return Action.D
        if t == 0:
            return Action.C
        obsCoop = int(history.opponent_cooperators[-1])
        p = obsCoop / (self.n - 1) if self.n > 1 else 0.0
        Gc = S / (2 * self.n) if self.n else 0.0
        Gd = S / self.n - self.audit_prob * self.fine if self.n else -self.audit_prob * self.fine
        low_stock = 0.3 * self.capacity
        high_stock = 0.8 * self.capacity
        coop_target = S / self.capacity if self.capacity else 0.0
        if Gd <= Gc:
            return Action.C
        if S <= low_stock:
            return Action.C
        if S >= high_stock and p >= coop_target:
            return Action.C
        return Action.D

class Strategy_COLLECTIVE_6(BaseStrategy):
    """
    Adaptive Collective Threshold strategy.

    Cooperates when the common pool stock is at least half the capacity
    and the observed recent cooperation level exceeds the confidence
    threshold that balances the expected fine from audits. Defects
    otherwise, except in the final round where it always cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.S_safe = self.capacity / 2.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.game_description.n_rounds - 1:
            return Action.C
        if t == 0:
            recent_coop = 1.0
        else:
            opp_coop = int(history.opponent_cooperators[-1])
            n_opponents = self.n - 1
            recent_coop = opp_coop / n_opponents if n_opponents else 1.0
        if current_stock <= 0.0:
            p_conf = 0.0
        else:
            cons_per_coop = current_stock / (2.0 * self.n)
            if cons_per_coop == 0.0:
                p_conf = 0.0
            else:
                denom = 1.0 + self.audit_prob * self.fine / cons_per_coop
                p_conf = 1.0 / denom
        if current_stock >= self.S_safe and recent_coop >= p_conf:
            return Action.C
        else:
            return Action.D

class Strategy_COLLECTIVE_7(BaseStrategy):
    """
    Adaptive Sustainable Threshold (AST) strategy.
    Cooperates when the expected gain from defection is not higher than cooperation,
    when the common stock falls below a safety threshold, or when recent opponent
    behavior shows high trust. Defects otherwise, especially in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players: int = game_description.n_players
        self.n_rounds: int = game_description.n_rounds
        self.capacity: float = float(game_description.capacity)
        self.audit_prob: float = game_description.audit_prob
        self.fine: float = game_description.fine
        self.theta: float = 0.6
        self.gamma: float = 0.1
        self.defect_count: int = 0
        self.prev_coop_obs: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide the action for the current round based on the AST rules.
        """
        t: int = history.round_number
        if t == self.n_rounds - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        n = self.n_players
        q = self.audit_prob
        F = self.fine
        exp_defect: float = current_stock / n - q * F
        coop_pay: float = current_stock / (2 * n)
        remaining_rounds: int = self.n_rounds - t
        safety_fraction: float = max(self.theta, 1.0 - self.gamma * (remaining_rounds / self.n_rounds))
        safety_stock: float = self.capacity * safety_fraction
        if t == 0:
            trust: float = 1.0
            coop_obs_t: int = 0
        else:
            coop_obs_t = int(history.opponent_cooperators[-1])
            trust = coop_obs_t / (n - 1) if n > 1 else 1.0
        if exp_defect <= coop_pay:
            action = Action.C
        elif current_stock <= safety_stock:
            action = Action.C
        elif trust < 0.4:
            action = Action.D
        elif current_stock > 2 * n * q * F:
            action = Action.D
        else:
            action = Action.C
        self.prev_coop_obs = coop_obs_t
        self.defect_count += n - 1 - coop_obs_t
        return action

class Strategy_COLLECTIVE_8(BaseStrategy):
    """
    Collective Adaptive Strategy (CAS).

    The agent cooperates when:
      • the common stock is above a fraction of capacity,
      • the observed fraction of defecting opponents is low,
      • enough rounds remain to reap the collective benefit.

    Otherwise it evaluates the expected net gain from defecting,
    taking the audit probability and fine into account, and defects
    only if that net gain is positive. In the final round it always
    defects (no audit possible).
    """
    _THETA_STOCK = 0.5
    _THETA_DEFECT = 0.3
    _THETA_FUTURE = 2

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action for the upcoming round."""
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            defect_ratio = 0.0
        else:
            opp_coop = int(history.opponent_cooperators[-1])
            opp_total = self.n - 1
            defect_ratio = (opp_total - opp_coop) / opp_total if opp_total else 0.0
        stock_ok = current_stock >= self._THETA_STOCK * self.capacity
        defection_ok = defect_ratio <= self._THETA_DEFECT
        rounds_left = self.r - t
        future_ok = rounds_left > self._THETA_FUTURE
        if stock_ok and defection_ok and future_ok:
            return Action.C
        expected_gain = current_stock / (2.0 * self.n)
        expected_fine = self.q * self.F
        net_gain = expected_gain - expected_fine
        if net_gain > 0.0:
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_9(BaseStrategy):
    """
    Adaptive Collective Stewardship strategy.
    Balances expected profit from defection against a target stock level,
    adapting defect probability based on current stock and observed
    cooperativeness of opponents.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.q = game_description.audit_prob
        self.F = game_description.fine
        self.target_stock = 0.5 * self.capacity
        self.exp_fine = self.q * self.F

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:

        def c_gain(stock: float) -> float:
            return stock / (2 * self.n)

        def d_gain(stock: float) -> float:
            return stock / self.n
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            if d_gain(current_stock) > c_gain(current_stock):
                return Action.D
            else:
                return Action.C
        opp_coop_cnt_last = int(history.opponent_cooperators[-1])
        opp_coop_ratio = opp_coop_cnt_last / (self.n - 1) if self.n > 1 else 0.0
        C = c_gain(current_stock)
        D = d_gain(current_stock)
        exp_D = D - self.exp_fine
        if current_stock >= self.target_stock and exp_D > C:
            excess = exp_D - C
            scale = excess / (D + 1e-09)
            p_defect = min(1.0, scale * (1 - opp_coop_ratio) + 0.1)
            if random.random() < p_defect:
                return Action.D
            else:
                return Action.C
        elif opp_coop_ratio < 0.25 and current_stock < self.target_stock:
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_10(BaseStrategy):
    """
    State‑based trigger strategy for the Common Pool Resource game.

    - Starts with cooperation to signal goodwill.
    - In the final round always defects (no audit, maximize payoff).
    - Otherwise guards the stock with a safety target and decides based on
      expected payoffs, audit risk, and observed opponent defection rate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        fine = self.game_description.fine
        t = history.round_number
        if t == 0:
            return Action.C
        if t == r - 1:
            return Action.D
        remaining = r - t
        theta = max(0.2, remaining / r)
        target_stock = capacity * theta
        e_defect = current_stock / n - q * fine
        e_coop = current_stock / (2 * n)
        if history.opponent_cooperators.size > 0:
            last_coop = int(history.opponent_cooperators[-1])
        else:
            last_coop = 0
        if n > 1:
            p_defect = (n - 1 - last_coop) / (n - 1)
        else:
            p_defect = 0.0
        tau_defect = 0.4 * (remaining / r)
        if current_stock < target_stock:
            return Action.C
        if e_defect > e_coop:
            if p_defect > tau_defect:
                return Action.C
            else:
                return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_11(BaseStrategy):
    """
    Adaptive Cooperative Threshold (ACT)

    Dynamically balances immediate profit, expected audit fines,
    and preservation of the common‑pool stock.  The decision combines:
      • a utility estimate assuming all players continue to cooperate,
      • a collective bias based on the observed proportion of opponent defections,
      • safety/excess overrides that force cooperation when the stock is low,
        and relax it when the stock is abundant.
    """
    _ALPHA = 0.3
    _BETA = 0.7
    _GAMMA = 0.05

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        S = float(current_stock)
        payoff_C = S / (2.0 * n)
        expected_fine = q * F
        payoff_D = S / n - expected_fine
        T = r - t - 1
        if T > 0:
            S_after_cons = S - n * payoff_C
            growth = 2.0 * S_after_cons * (1.0 - S_after_cons / capacity)
            S_next = min(S_after_cons + growth, capacity)
            V_future = T * S_next / (2.0 * n)
        else:
            V_future = 0.0
        U_C = payoff_C + V_future
        U_D = payoff_D + V_future
        base_action = 'C' if U_C >= U_D else 'D'
        if t > 0:
            c_obs = int(history.opponent_cooperators[t - 1])
            delta = (n - 1 - c_obs) / (n - 1)
        else:
            delta = 0.0
        if delta <= self._GAMMA:
            bias_action = 'C'
        elif delta >= 1.0 - self._GAMMA:
            bias_action = 'D'
        else:
            bias_action = base_action
        if S <= self._ALPHA * capacity:
            final_action = 'C'
        elif S >= self._BETA * capacity and bias_action == 'C' and (delta <= self._GAMMA):
            final_action = 'C'
        else:
            final_action = bias_action
        if t == r - 1:
            final_action = 'C' if payoff_C >= payoff_D else 'D'
        return Action.C if final_action == 'C' else Action.D

class Strategy_COLLECTIVE_12(BaseStrategy):
    """
    Collective Adaptive Strategy for the Common Pool Resource Game.

    The agent keeps a running estimate of opponents' cooperation,
    adapts its willingness to cooperate based on the current stock
    level, the observed cooperation level, and the audit probability,
    and always cooperates in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = game_description.audit_prob
        self.F = game_description.fine
        self.high_thr = 0.6 * self.capacity
        self.low_thr = 0.3 * self.capacity
        self.coop_thr = 0.7
        self.coop_est = 1.0
        self.t = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            opp_coop = int(history.opponent_cooperators[-1])
            opp_prop = opp_coop / (self.n - 1) if self.n > 1 else 0.0
            self.coop_est = (self.t * self.coop_est + opp_prop) / (self.t + 1)
        if self.t == self.r - 1:
            action = Action.C
        else:
            if current_stock >= self.high_thr and self.coop_est >= self.coop_thr:
                base_prob = 0.9
            elif current_stock <= self.low_thr:
                base_prob = 0.2
            else:
                base_prob = 0.5 * self.coop_est + 0.3 * (1.0 - self.q)
            explore = 0.05
            coop_prob = base_prob * (1.0 - explore) + explore * random.random()
            coop_prob = max(0.0, min(1.0, coop_prob))
            action = Action.C if random.random() < coop_prob else Action.D
        self.t += 1
        self.stock = float(current_stock)
        return action

class Strategy_COLLECTIVE_13(BaseStrategy):
    """
    Sustainable cooperate‑until‑threshold‑break strategy.
    Starts with cooperation, never defects in the final round,
    and defects only when the expected net payoff (considering audit risk)
    exceeds cooperation, subject to a stock safety check and a limit
    on consecutive defections.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.theta_stock = 0.3 * self.capacity
        self.theta_coop = (self.n - 1) // 2
        self.T_defect = 2
        self.defect_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            self.defect_counter = 0
            return Action.C
        if t == self.game_description.n_rounds - 1:
            self.defect_counter = 0
            return Action.C
        S_t = float(current_stock)
        if S_t < self.theta_stock:
            self.defect_counter = 0
            return Action.C
        c_obs = int(history.opponent_cooperators[-1]) if history.opponent_cooperators.size else 0
        c_obs = max(0, min(c_obs, self.n - 1))
        if c_obs >= self.theta_coop:
            self.defect_counter = 0
            return Action.C
        q = float(self.game_description.audit_prob)
        F = float(self.game_description.fine)
        pi_C = S_t / (2.0 * self.n)
        pi_D = S_t / float(self.n) - q * F
        if pi_D > pi_C and self.defect_counter < self.T_defect:
            self.defect_counter += 1
            return Action.D
        else:
            self.defect_counter = 0
            return Action.C

class Strategy_COLLECTIVE_14(BaseStrategy):
    """
    Sustainability‑first, expected‑payoff and reputation‑based strategy.

    * Cooperate when the stock falls below a sustainability threshold τ.
    * Otherwise compute the expected net payoff of defecting (accounting for
      audit probability q and fine F) and defect only if it strictly dominates
      cooperating.
    * Maintain a sliding window of the last w rounds; if the empirical
      defection rate of opponents exceeds θ, enter a short punishment phase of
      p rounds during which the agent defects (unless overridden by the safety
      filter).
    * First round: cooperate to seed trust.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.tau = 0.3
        self.w = 3
        self.theta = 0.25
        self.p = 2
        self.punishment_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        S = max(0.0, min(current_stock, self.capacity))
        if history.round_number == 0:
            return Action.C
        if S < self.tau * self.capacity:
            return Action.C
        if self.punishment_counter > 0:
            self.punishment_counter -= 1
            return Action.D
        n_opp = self.n - 1
        start_idx = max(0, history.round_number - self.w)
        window_len = history.round_number - start_idx
        if window_len > 0 and n_opp > 0:
            opp_coops = history.opponent_cooperators[start_idx:history.round_number]
            total_defections = np.sum((n_opp - opp_coops).astype(float))
            defection_rate = total_defections / (window_len * n_opp)
        else:
            defection_rate = 0.0
        if defection_rate > self.theta:
            self.punishment_counter = self.p - 1
            return Action.D
        payoff_C = S / (2 * self.n) if self.n > 0 else 0.0
        last_round = history.round_number == self.game_description.n_rounds - 1
        if last_round:
            payoff_D = S / self.n if self.n > 0 else 0.0
        else:
            payoff_D = S / self.n - self.audit_prob * self.fine if self.n > 0 else 0.0
        if S == 0.0:
            return Action.C
        epsilon = 1e-12
        if payoff_D > payoff_C + epsilon:
            return Action.D
        else:
            return Action.C

class Strategy_COLLECTIVE_15(BaseStrategy):
    """
    Adaptive Collective Sustainability (ACS)

    The player balances immediate profit with resource sustainability
    using an exponential moving average of opponent defection, a stock
    fragility threshold, and a short‑term grim‑trigger mechanism.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._lambda = 0.8
        self._defect_rate_max = 0.6
        self._grace_period = 2
        self._stock_min_frac = 0.4
        self.defect_rate = 0.0
        self._consec_low_coop = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        stock_min = self._stock_min_frac * capacity
        opponents = n - 1
        low_coop_threshold = opponents // 2
        round_number = history.round_number
        if round_number > 0:
            coop_last = int(history.opponent_cooperators[-1])
            if opponents > 0:
                defect_prop = (opponents - coop_last) / opponents
            else:
                defect_prop = 0.0
            self.defect_rate = self._lambda * self.defect_rate + (1.0 - self._lambda) * defect_prop
            if coop_last < low_coop_threshold:
                self._consec_low_coop += 1
            else:
                self._consec_low_coop = 0
        if round_number == self.game_description.n_rounds - 1:
            return Action.D
        S = current_stock
        EC = S / (2 * n) if n else 0.0
        ED = S / n - q * F if n else 0.0
        immediate_prefer = Action.D if ED > EC else Action.C
        if S < stock_min:
            sustainability = Action.C
        else:
            sustainability = immediate_prefer
        if self.defect_rate > self._defect_rate_max:
            hostility = Action.D
        else:
            hostility = sustainability
        if self._consec_low_coop >= self._grace_period:
            action = Action.D
        else:
            action = hostility
        return action

class Strategy_COLLECTIVE_16(BaseStrategy):
    """
    Trust‑based conditional cooperator/defector.
    Starts by cooperating, tracks opponent defection rate to adjust a trust score,
    and triggers a fixed‑length forced‑defection (punishment) block when trust falls
    below a threshold.  In regular rounds the decision weighs the expected net gain
    from defection (extra consumption minus expected fine) against stock health
    and the current trust level.  In the final round the player defects only if the
    stock is comfortably above a health threshold.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.T_stock = 0.5
        self.T_defect = 0.25
        self.T_punish = 3
        self.trust = 1.0
        self.punish_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number == 0:
            self.trust = 1.0
            self.punish_counter = 0
            return Action.C
        opp_coop = int(history.opponent_cooperators[-1])
        if self.n > 1:
            opp_defect_rate = (self.n - 1 - opp_coop) / (self.n - 1)
        else:
            opp_defect_rate = 0.0
        if opp_defect_rate > self.T_defect:
            self.trust = self.trust * (1.0 - opp_defect_rate)
        else:
            self.trust = min(1.0, self.trust + 0.05)
        if self.punish_counter == 0 and self.trust < 0.3:
            self.punish_counter = self.T_punish
        t = history.round_number
        last_round = self.game_description.n_rounds - 1
        if t == last_round:
            if current_stock > self.capacity * self.T_stock:
                chosen_action = Action.D
            else:
                chosen_action = Action.C
        else:
            benefit = current_stock / (2 * self.n)
            expected_fine = self.audit_prob * self.fine
            net_gain = benefit - expected_fine
            if self.punish_counter > 0:
                chosen_action = Action.D
            elif current_stock < self.capacity * self.T_stock:
                chosen_action = Action.C
            elif net_gain > 0 and self.trust > 0.5:
                chosen_action = Action.D
            else:
                chosen_action = Action.C
        if self.punish_counter > 0:
            self.punish_counter -= 1
        return chosen_action

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

class Strategy_PROSOCIAL_2(BaseStrategy):
    """
    Prosocial Adaptive Strategy:
    - Starts with unconditional cooperation.
    - Cooperates when stock is low or when expected defect payoff is not higher than cooperation payoff.
    - Uses an exponential moving average of opponent defection rates to apply modest retaliation.
    - Always cooperates in the final round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.alpha = 0.3
        self.beta = 0.2
        self.theta_stock = 0.4
        self.theta_defect = 0.35
        self.avg_defection = 0.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        r = self.game_description.n_rounds
        if t == 0:
            return Action.C
        if t == r - 1:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        n_opponents = self.n - 1
        if n_opponents > 0:
            coop_last = int(history.opponent_cooperators[-1])
            defect_last = n_opponents - coop_last
            p_last = defect_last / n_opponents
            self.avg_defection = self.alpha * p_last + (1.0 - self.alpha) * self.avg_defection
        else:
            self.avg_defection = 0.0
        S = float(current_stock)
        Δ = S / (2.0 * self.n)
        Γ = S / float(self.n)
        E_D = Γ - self.audit_prob * self.fine
        if S / self.capacity < self.theta_stock:
            baseline = True
        else:
            baseline = E_D <= Δ
        if self.avg_defection > self.theta_defect:
            effective_Δ = Δ * (1.0 - self.beta)
        else:
            effective_Δ = Δ
        if baseline and E_D <= effective_Δ:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_3(BaseStrategy):
    """
    Adaptive Cooperative Guard

    Starts cooperating, continues as long as the pool is healthy,
    opponents were mostly cooperative in the previous round, and the
    expected net gain from defecting (considering the audit risk)
    is not positive. Defects otherwise, but always cooperates in the
    final round and when the stock is depleted.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._n = game_description.n_players
        self._total_opp = max(self._n - 1, 0)
        self._stock_thresh = game_description.capacity * 0.4
        self._cooperate_frac_thresh = 0.5
        self._audit_q = game_description.audit_prob
        self._fine_F = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if self.game_description.capacity == 0 or current_stock == 0:
            return Action.C
        if t == 0:
            return Action.C
        if t == self.game_description.n_rounds - 1:
            return Action.C
        stock_ok = current_stock > self._stock_thresh
        opp_coop_last = int(history.opponent_cooperators[-1])
        opp_coop_frac = opp_coop_last / self._total_opp if self._total_opp > 0 else 0.0
        behavior_ok = opp_coop_frac >= self._cooperate_frac_thresh
        expected_defect_gain = current_stock / (2 * self._n)
        expected_fine = self._audit_q * self._fine_F
        net_defect = expected_defect_gain - expected_fine
        if stock_ok and behavior_ok and (net_defect <= 0):
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_4(BaseStrategy):
    """
    Prosocial Adaptive Strategy:
    - Cooperate on the first and last round.
    - Cooperate when stock is very low.
    - Otherwise defect only if:
        * Expected net monetary gain from defecting (benefit minus expected fine,
          adjusted for recent audit penalty) is positive,
        * Opponents' cooperation rate in the previous round exceeds 50%,
        * The predicted stock after universal cooperation stays above a safety
          threshold (25% of capacity).
    - After a personal audit while defecting, the net‑gain threshold is raised
      by 0.5\u202f×\u202ffine to discourage repeated risky defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.q = game_description.audit_prob
        self.F = game_description.fine
        self.safe_threshold = 0.25 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        if current_stock <= 0.01 * self.capacity:
            return Action.C
        opp_coop_last = int(history.opponent_cooperators[-1])
        n_opponents = self.n - 1
        coop_rate = opp_coop_last / n_opponents if n_opponents > 0 else 0.0
        benefit = current_stock / self.n - current_stock / (2 * self.n)
        expected_fine = self.q * self.F
        net_gain = benefit - expected_fine
        penalty = 0.0
        if t > 0:
            last_defected = not history.my_actions[-1]
            last_audited = bool(history.my_audited[-1])
            if last_defected and last_audited:
                penalty = 0.5 * self.F
        effective_gain = net_gain - penalty
        S_remain_allC = current_stock - self.n * current_stock / (2 * self.n)
        growth_allC = 2 * S_remain_allC * (1 - S_remain_allC / self.capacity)
        S_next_if_allC = min(S_remain_allC + growth_allC, self.capacity)
        if effective_gain > 0 and coop_rate > 0.5 and (S_next_if_allC >= self.safe_threshold):
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_5(BaseStrategy):
    """
    Prosocial conditional defector with audit‑aware safeguards.
    Cooperates by default, defects temporarily when cooperation drops
    or the stock is low, but only if the expected net gain after fines
    exceeds the cooperative payoff. The final round is always cooperative.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.base_coop_threshold = 0.6
        self.high_coop_threshold = 0.7
        self.alarm_stock_factor = 0.2
        self.safety_margin = 0.05 * self.capacity

    def _next_stock(self, stock: float, everyone_cooperate: bool) -> float:
        """Return stock after one round assuming all players take the same action."""
        if stock <= 0.0:
            return 0.0
        if everyone_cooperate:
            consumption_per = stock / (2 * self.n)
        else:
            consumption_per = stock / self.n
        total_consumption = self.n * consumption_per
        remaining = stock - total_consumption
        if remaining < 0.0:
            remaining = 0.0
        growth = 2.0 * remaining * (1.0 - remaining / self.capacity)
        new_stock = remaining + growth
        if new_stock > self.capacity:
            new_stock = self.capacity
        return new_stock

    def _project_stock(self, start_stock: float, rounds_left: int, coop: bool) -> float:
        s = start_stock
        for _ in range(rounds_left):
            s = self._next_stock(s, coop)
            if s == 0.0:
                break
        return s

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0 and bool(history.my_audited[-1]) and (history.my_fines[-1] > 0.0):
            coop_threshold = self.high_coop_threshold
        else:
            coop_threshold = self.base_coop_threshold
        if t == 0:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        my_last_coop = bool(history.my_actions[-1])
        opponents_coop = int(history.opponent_cooperators[-1])
        coop_rate = (opponents_coop + (1 if my_last_coop else 0)) / self.n
        if t == self.r - 1:
            return Action.C
        if current_stock <= self.alarm_stock_factor * self.capacity and coop_rate < 0.5:
            intended_action = Action.D
        elif coop_rate < 0.3:
            intended_action = Action.D
        else:
            rounds_left = self.r - t
            stock_coop = self._project_stock(current_stock, rounds_left, coop=True)
            stock_def = self._project_stock(current_stock, rounds_left, coop=False)
            if stock_coop >= stock_def + self.safety_margin and coop_rate >= coop_threshold:
                intended_action = Action.C
            else:
                intended_action = Action.C
        if intended_action == Action.D:
            expected_gain = current_stock / self.n - self.audit_prob * self.fine
            coop_gain = current_stock / (2 * self.n)
            if expected_gain <= coop_gain:
                intended_action = Action.C
        return intended_action

class Strategy_PROSOCIAL_6(BaseStrategy):
    """
    Prosocial Conditional‑Cooperation Strategy.

    Cooperates in the first round and when the expected payoff from cooperating
    is at least as good as defecting (taking audits into account). If defecting
    is more profitable, cooperation is retained only if the common‑pool stock
    remains above a sustainability threshold τ and the recent observed
    cooperation fraction among opponents exceeds θ.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.k = 3
        self.theta = 0.75
        self.tau = game_description.capacity * 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        total_rounds = self.game_description.n_rounds
        audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        t = history.round_number
        if t == 0:
            return Action.C
        if current_stock == 0:
            return Action.C
        payoff_coop = current_stock / (2 * n)
        effective_audit = 0.0 if t == total_rounds - 1 else audit_prob
        expected_defect = current_stock / n - effective_audit * fine
        if payoff_coop >= expected_defect:
            return Action.C
        start_idx = max(0, t - self.k)
        end_idx = t
        recent_window = history.opponent_cooperators[start_idx:end_idx]
        rounds_considered = end_idx - start_idx
        if rounds_considered == 0:
            recent_frac = 0.0
        else:
            total_coop = float(np.sum(recent_window))
            opp_cnt = n - 1
            recent_frac = total_coop / (rounds_considered * opp_cnt)
        if current_stock > self.tau and recent_frac >= self.theta:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_7(BaseStrategy):
    """
    Prosocial Adaptive Strategy (Cooperate‑Monitor‑Adjust, CMA).

    Core idea:
    - Default to cooperation.
    - Defect when the expected net gain after accounting for expected fine exceeds
      cooperative payoff, when the stock is critically low, or when opponents have
      shown sustained defection.
    - Retaliation is limited to at most two consecutive rounds and is reset when
      opponents cooperate sufficiently.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.stock_low = 0.3 * self.capacity
        self.defect_ratio_high = 0.5
        self.epsilon = 0.01
        self.recent_defections = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            self.recent_defections = 0
            return Action.C
        prev_opp_coop_cnt = history.opponent_cooperators[-1]
        opp_ratio_prev = prev_opp_coop_cnt / (self.n - 1) if self.n > 1 else 0.0
        prev_my_action = history.my_actions[-1]
        if not prev_my_action and opp_ratio_prev < 1.0 - self.defect_ratio_high:
            self.recent_defections += 1
        else:
            self.recent_defections = 0
        if t == self.r - 1:
            return Action.C
        denom = max(self.n, 1)
        C_gain = current_stock / (2.0 * denom)
        D_gain = current_stock / denom
        expected_fine = self.audit_prob * self.fine
        net_D = D_gain - expected_fine
        action = Action.C
        if net_D > C_gain + self.epsilon:
            action = Action.D
        if current_stock < self.stock_low:
            action = Action.D
        if t > 0:
            opp_coop_cnt = history.opponent_cooperators[-1]
            opp_ratio = opp_coop_cnt / (self.n - 1) if self.n > 1 else 0.0
            if opp_ratio < 1.0 - self.defect_ratio_high:
                if self.recent_defections < 2:
                    action = Action.D
                else:
                    action = Action.C
            else:
                pass
        if current_stock <= 0.0:
            action = Action.C
        return action

class Strategy_PROSOCIAL_8(BaseStrategy):
    """
    Adaptive Conditional Cooperation (ACC).

    - Cooperates by default.
    - Defects when stock is low *and* opponents have recently defected frequently (Trigger\u202fA).
    - Defects as retaliation when many opponents defected in the previous round (Trigger\u202fC).
    - Switches to cooperation during recovery phases (Trigger\u202fD).
    - Uses audit probability and fine to avoid unprofitable defections (Trigger\u202fB).
    - Guarantees cooperation in the final round and when stock is zero.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.theta_stock = 0.3
        self.theta_defect = 0.4
        self.window = 3
        self.prev_stock: float | None = None
        self.opponent_defect_fractions: list[float] = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t > 0 and len(self.opponent_defect_fractions) < t:
            coop_prev = int(history.opponent_cooperators[-1])
            defect_prev = self.n - 1 - coop_prev
            frac = defect_prev / (self.n - 1) if self.n > 1 else 0.0
            self.opponent_defect_fractions.append(frac)
        if current_stock <= 0.0:
            self.prev_stock = current_stock
            return Action.C
        if t == self.r - 1:
            self.prev_stock = current_stock
            return Action.C
        G_coop = current_stock / (2 * self.n) if self.n else 0.0
        G_defect = current_stock / self.n - self.q * self.F if self.n else -self.q * self.F
        w = min(self.window, len(self.opponent_defect_fractions))
        delta = sum(self.opponent_defect_fractions[-w:]) / w if w > 0 else 0.0
        stock_margin = current_stock / self.capacity
        if stock_margin < self.theta_stock and delta > self.theta_defect:
            self.prev_stock = current_stock
            return Action.D
        if t > 0:
            coop_prev = int(history.opponent_cooperators[-1])
            defectors_prev = self.n - 1 - coop_prev
            if defectors_prev >= math.ceil(0.75 * (self.n - 1)):
                self.opponent_defect_fractions = []
                self.prev_stock = current_stock
                return Action.D
        if self.prev_stock is not None and self.prev_stock < 0.15 * self.capacity and (current_stock > 0.25 * self.capacity):
            self.prev_stock = current_stock
            return Action.C
        if G_defect <= G_coop:
            self.prev_stock = current_stock
            return Action.C
        self.prev_stock = current_stock
        return Action.C

class Strategy_PROSOCIAL_9(BaseStrategy):
    """
    Conditional cooperation with stock thresholds, EMA trust estimation,
    and limited retaliation. Starts with cooperation, cooperates when
    the resource is low, otherwise follows trust and expected‑gain logic.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.alpha = 0.3
        self.tol = 0.6
        self.max_retaliation = 2
        self.excess_defect_threshold = 0.4
        cap = float(game_description.capacity)
        self.low_stock = 0.3 * cap
        self.high_stock = 0.7 * cap
        self.p_est = 1.0
        self.defect_seq = 0
        self.retaliation_counter = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        t = history.round_number
        if t > 0:
            c_obs = int(history.opponent_cooperators[-1])
            obs = c_obs / (n - 1) if n > 1 else 0.0
            self.p_est = self.alpha * obs + (1.0 - self.alpha) * self.p_est
            if obs == 1.0:
                self.defect_seq = 0
            else:
                self.defect_seq += 1
            if obs < 1.0 - self.excess_defect_threshold:
                self.retaliation_counter = self.max_retaliation
        if t == 0:
            return Action.C
        if t == r - 1:
            return Action.C if current_stock > 0 else Action.D
        if current_stock <= self.low_stock:
            return Action.C
        if self.retaliation_counter > 0:
            self.retaliation_counter -= 1
            return Action.D
        if self.p_est >= self.tol and current_stock >= self.high_stock:
            return Action.C
        exp_fine = audit_prob * fine
        diff = current_stock / (2.0 * n) - exp_fine
        if diff > 0.0 and self.p_est < self.tol:
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_10(BaseStrategy):
    """
    Prosocial Adaptive Cooperation (PAC)

    The agent starts by cooperating, then evaluates:
      * Expected net gain from defecting (extra consumption minus expected fine)
      * Sustainability of the commons (stock / capacity)
      * Recent opponent cooperation levels (moving average)
      * A short‑term penalty after being audited for defection

    It cooperates in the last round, protects the pool when stock is low,
    and defects only when the expected monetary gain outweighs the audit risk
    and the environment signals insufficient reciprocity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.stock_history: list[float] = []
        self.opp_coop_ratio_history: list[float] = []
        self.penalty_counter: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        stock = max(0.0, min(current_stock, self.capacity))
        self.stock_history.append(stock)
        if t == 0:
            return Action.C
        opp_coop_cnt = int(history.opponent_cooperators[-1])
        opp_ratio = opp_coop_cnt / (self.n - 1) if self.n > 1 else 0.0
        self.opp_coop_ratio_history.append(opp_ratio)
        if history.my_audited[-1] and (not history.my_actions[-1]):
            self.penalty_counter = 2
        if self.penalty_counter > 0:
            self.penalty_counter -= 1
        effective_opp_ratio = opp_ratio
        if len(self.opp_coop_ratio_history) >= 2:
            prev = self.opp_coop_ratio_history[-2]
            if opp_ratio - prev >= 0.3:
                self.penalty_counter = 0
                effective_opp_ratio = 1.0
        recent_stock = self.stock_history[-3:]
        avg_stock_ratio = sum(recent_stock) / len(recent_stock) / self.capacity
        recent_opp = self.opp_coop_ratio_history[-3:]
        avg_opp_coop = sum(recent_opp) / len(recent_opp)
        if t == self.r - 1:
            return Action.C
        if stock < 0.2 * self.capacity:
            return Action.C
        extra_consumption = stock / (2 * self.n)
        net_defect_gain = extra_consumption - self.audit_prob * self.fine
        if self.penalty_counter > 0:
            net_defect_gain -= self.fine / 2.0
        if net_defect_gain <= 0.0:
            return Action.C
        scaling = 0.3 + 0.2 * avg_stock_ratio
        coop_threshold = 0.5 + scaling * avg_stock_ratio
        if t == self.r - 2:
            coop_threshold += 0.05
        if effective_opp_ratio >= coop_threshold:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_11(BaseStrategy):
    """
    Trust‑based conditional cooperation for the Common Pool Resource game.

    - Maintains a running estimate `trust` of the opponents' cooperation rate.
    - Adjusts the decision to cooperate or defect based on expected net payoff,
      audit risk, current stock health, and the trust level.
    - Uses a probabilistic cooperation when the stock is moderate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.trust: float = 1.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n_players = self.game_description.n_players
        capacity = float(self.game_description.capacity)
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if t > 0:
            c_prev = int(history.opponent_cooperators[-1])
            n_opponents = n_players - 1
            opp_coop_rate = c_prev / n_opponents if n_opponents > 0 else 0.0
            self.trust = ((t - 1) * self.trust + opp_coop_rate) / t
        if t == total_rounds - 1:
            audit_prob = 0.0
        else:
            audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        net_defect = current_stock / n_players - audit_prob * fine
        pay_coop = current_stock / (2 * n_players)
        if net_defect <= pay_coop:
            return Action.C
        h = current_stock / capacity if capacity > 0 else 0.0
        if self.trust >= 0.5 and h >= 0.4:
            return Action.C
        if h < 0.2:
            return Action.D
        if random.random() < h:
            return Action.C
        else:
            return Action.D

class Strategy_PROSOCIAL_12(BaseStrategy):
    """
    Prosocial Adaptive Strategy for the Common Pool Resource Game.
    - Starts and ends with cooperation.
    - Uses a short‑term window (up to 3 rounds) to estimate opponents' cooperation rate.
    - Adjusts behaviour to stock level and observed cooperation.
    - Defection is allowed only when the stock is scarce, opponents are not very cooperative,
      and the expected net gain after accounting for audit risk is positive.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.C
        if current_stock <= 0:
            return Action.C
        window = min(3, t)
        if window > 0:
            recent_counts = history.opponent_cooperators[-window:]
            total_coops = int(np.sum(recent_counts))
            opp_coop_rate = total_coops / ((self.n - 1) * window)
        else:
            opp_coop_rate = 1.0
        stock_ratio = current_stock / self.capacity
        decide_cooperate = True
        if stock_ratio >= 0.6:
            if opp_coop_rate >= 0.5:
                decide_cooperate = True
            else:
                decide_cooperate = random.random() < opp_coop_rate
        elif 0.3 <= stock_ratio < 0.6:
            if opp_coop_rate >= 0.7:
                decide_cooperate = True
            else:
                p = 0.5 * opp_coop_rate + 0.25
                decide_cooperate = random.random() < p
        elif opp_coop_rate >= 0.8:
            decide_cooperate = True
        else:
            decide_cooperate = False
        if not decide_cooperate:
            expected_gain = current_stock / (2 * self.n) - self.audit_prob * self.fine
            if expected_gain <= 0:
                decide_cooperate = True
        return Action.C if decide_cooperate else Action.D

class Strategy_PROSOCIAL_13(BaseStrategy):
    """
    Prosocial Adaptive Cooperation Strategy.

    - Starts by cooperating.
    - Maintains an exponential moving average of opponent cooperation.
    - Tracks a streak of rounds with significant opponent defection.
    - Cooperates when the stock is low, retaliates briefly after repeated defection,
      and otherwise cooperates with probabilities that depend on the estimated
      cooperation level of others.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.k = 5
        self.defect_threshold = 0.3
        self.max_defect_streak = 2
        self.stock_low = 0.25 * self.capacity
        self.stock_high = 0.75 * self.capacity
        self.opp_coop_rate = 1.0
        self.defect_streak = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.n_rounds - 1:
            return Action.C
        n_opponents = self.n_players - 1
        if n_opponents == 0:
            observed_coop = 1.0
        else:
            observed_coop = history.opponent_cooperators[-1] / n_opponents
        self.opp_coop_rate = 0.8 * self.opp_coop_rate + 0.2 * observed_coop
        if observed_coop < 1.0 - self.defect_threshold:
            self.defect_streak += 1
        else:
            self.defect_streak = max(0, self.defect_streak - 1)
        if current_stock <= 0.0 or current_stock < self.stock_low:
            return Action.C
        if self.defect_streak >= self.max_defect_streak:
            self.defect_streak = 0
            return Action.D
        if self.opp_coop_rate >= 0.7:
            return Action.C
        elif self.opp_coop_rate >= 0.4:
            return Action.C if random.random() < 0.8 else Action.D
        else:
            return Action.C if random.random() < 0.5 else Action.D

class Strategy_PROSOCIAL_14(BaseStrategy):
    """
    Adaptive Conditional Cooperation (ACC):
    Starts by cooperating. In intermediate rounds it cooperates unless
    the common pool is healthy, opponents are mostly defecting, and
    the expected gain from defection (accounting for audit risk) exceeds
    cooperation by a small margin. In the final round it cooperates unless
    the stock is already depleted.
    """
    MIN_STOCK = 0.45
    MIN_COOP = 0.6
    GAIN_MARGIN = 0.02

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        q = float(self.game_description.audit_prob)
        F = float(self.game_description.fine)
        t = history.round_number
        if t == 0:
            return Action.C
        if t == r - 1:
            return Action.C if current_stock > 0 else Action.D
        stock_ratio = current_stock / capacity if capacity != 0 else 0.0
        low_stock = stock_ratio < self.MIN_STOCK
        opp_coop_last = int(history.opponent_cooperators[-1])
        coop_ratio = opp_coop_last / (n - 1) if n - 1 > 0 else 1.0
        low_coop = coop_ratio < self.MIN_COOP
        E_coop = current_stock / (2.0 * n) if n != 0 else 0.0
        E_def = current_stock / n - q * F if n != 0 else -q * F
        profitable_def = E_def - E_coop > self.GAIN_MARGIN
        if low_stock:
            return Action.C
        elif low_coop and profitable_def:
            return Action.D
        else:
            return Action.C

class Strategy_PROSOCIAL_15(BaseStrategy):
    """
    Adaptive Prosocial Guard.

    Cooperates when the common pool is healthy and opponents appear cooperative.
    Defects only if the expected net gain from defecting (after accounting for
    expected fines) is positive *and* either the stock is below a threshold or
    the observed cooperation rate is low.  After a defection the strategy
    requires two consecutive rounds in which both the stock ratio and the
    cooperation rate are above their thresholds before returning to cooperation.
    The final round is always cooperative (no audits).
    """
    _THETA_STOCK = 0.5
    _THETA_COOP = 0.6

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.c_obs: int = 0
        self.d_obs: int = 0
        self.last_action: Action = Action.C
        self.defecting: bool = False
        self.recovery_counter: int = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if t == 0:
            expected_gain = current_stock / (2 * n) - q * F
            stock_ratio = current_stock / capacity
            if expected_gain <= 0:
                action = Action.C
            elif stock_ratio < self._THETA_STOCK:
                action = Action.D
            else:
                action = Action.C
            self.last_action = action
            if action == Action.D:
                self.defecting = True
                self.recovery_counter = 0
            return action
        k = int(history.opponent_cooperators[-1])
        self.c_obs += k
        self.d_obs += n - 1 - k
        stock_ratio = current_stock / capacity
        coop_rate = self.c_obs / ((n - 1) * t) if t > 0 else 0.0
        expected_gain = current_stock / (2 * n) - q * F
        if t == total_rounds - 1:
            action = Action.C
            self.defecting = False
            self.recovery_counter = 0
            self.last_action = action
            return action
        if self.defecting:
            if stock_ratio >= self._THETA_STOCK and coop_rate >= self._THETA_COOP:
                self.recovery_counter += 1
            else:
                self.recovery_counter = 0
            if self.recovery_counter >= 2:
                action = Action.C
                self.defecting = False
                self.recovery_counter = 0
            else:
                action = Action.D
        elif expected_gain <= 0:
            action = Action.C
        elif stock_ratio < self._THETA_STOCK or coop_rate < self._THETA_COOP:
            action = Action.D
        else:
            action = Action.C
        if action == Action.D:
            self.defecting = True
            self.recovery_counter = 0
        else:
            pass
        self.last_action = action
        return action

class Strategy_PROSOCIAL_16(BaseStrategy):
    """
    Prosocial Adaptive Cooperation (PAC) strategy for the Common Pool Resource game.

    The player starts by cooperating, continues to cooperate in the final round,
    and otherwise adjusts its action based on stock health, observed opponent
    cooperation, and an expected‑penalty calculation.

    Internal state:
        - defect_streak: consecutive rounds the player has defected.
        - cooperate_streak: consecutive rounds the player has cooperated.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.theta_stock = 0.6
        self.theta_coop = 0.5
        self.theta_forgive = 0.3
        self.defect_streak = 0
        self.cooperate_streak = 0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == 0:
            self.cooperate_streak = 1
            self.defect_streak = 0
            return Action.C
        if t == self.r - 1:
            self.cooperate_streak += 1
            self.defect_streak = 0
            return Action.C
        if current_stock <= 0.0:
            self.cooperate_streak += 1
            self.defect_streak = 0
            return Action.C
        c_prev = int(history.opponent_cooperators[-1])
        coop_rate = c_prev / (self.n - 1) if self.n > 1 else 0.0
        if current_stock < self.theta_stock * self.capacity:
            if coop_rate >= self.theta_coop:
                self.cooperate_streak += 1
                self.defect_streak = 0
                return Action.C
            else:
                self.defect_streak += 1
                self.cooperate_streak = 0
                return Action.D
        if coop_rate >= self.theta_coop:
            self.cooperate_streak += 1
            self.defect_streak = 0
            return Action.C
        elif coop_rate >= self.theta_forgive and self.cooperate_streak > 0:
            self.cooperate_streak += 1
            self.defect_streak = 0
            return Action.C
        else:
            expected_fine = self.audit_prob * self.fine
            marginal_gain = current_stock / (2 * self.n)
            if marginal_gain > expected_fine:
                self.defect_streak += 1
                self.cooperate_streak = 0
                return Action.D
            else:
                self.cooperate_streak += 1
                self.defect_streak = 0
                return Action.C

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

class Strategy_INDIVIDUALISTIC_2(BaseStrategy):
    """
    Self‑Interested Sustainable (SIS) strategy.
    Balances the immediate net gain from defecting against the
    expected future value of the stock that would be preserved
    by cooperating, while accounting for audit probability and fine.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        C = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if t == r - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        S = float(current_stock)
        R = r - t - 1
        if t == 0:
            p = 0.5
        else:
            c_last = int(history.opponent_cooperators[-1])
            denom = max(1, n - 1)
            p = c_last / denom

        def next_stock(my_consumption: float, opp_cons: float) -> float:
            total_cons = my_consumption + opp_cons
            S_rem = max(0.0, S - total_cons)
            growth = 2.0 * S_rem * (1.0 - S_rem / C) if C != 0 else 0.0
            return min(S_rem + growth, C)
        opp_cons = (n - 1) * (p * (S / n) + (1.0 - p) * (S / (2.0 * n)))
        our_coop = S / (2.0 * n)
        S_next_coop = next_stock(our_coop, opp_cons)
        our_def = S / n
        S_next_def = next_stock(our_def, opp_cons)
        immediate_gain = S / (2.0 * n) - q * F
        delta_S = S_next_coop - S_next_def
        future_value = delta_S * R * (1.0 / (2.0 * n))
        if immediate_gain > future_value:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_3(BaseStrategy):
    """
    Threshold‑Defect‑Cooperate (TDC) strategy.
    Decides based on the current stock, audit probability, fine,
    and the recent proportion of opponent defections.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        n = game_description.n_players
        q = game_description.audit_prob
        F = game_description.fine
        self._threshold_T = 2.0 * n * q * F
        self._alpha = 0.3
        self._beta = 0.5
        self._capacity = float(game_description.capacity)
        self._n_players = n
        self._n_opponents = n - 1
        self._total_rounds = game_description.n_rounds

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action for the current round."""
        t = history.round_number
        if t == self._total_rounds - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            opp_def_rate = 0.0
        else:
            coop_last = int(history.opponent_cooperators[-1])
            defections_last = self._n_opponents - coop_last
            opp_def_rate = defections_last / self._n_opponents if self._n_opponents > 0 else 0.0
        effective_threshold = self._threshold_T * (1.0 + self._alpha * opp_def_rate)
        if current_stock < self._beta * self._capacity:
            return Action.C
        if current_stock > effective_threshold:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_4(BaseStrategy):
    """
    Adaptive Expected‑Value Threshold (AEVT) strategy.

    The player keeps an estimate of the probability that any opponent cooperates
    (initially 0.5).  At each round it computes the immediate expected payoff of
    cooperating vs. defecting (including the expected fine) and a one‑step look‑ahead
    value of the stock that would result from each pure action.  The discounted
    future value is added with weight λ = (remaining rounds)/total rounds (λ = 0
    in the final round).  The action with the higher total expected value is
    chosen, unless either action would drive the projected stock below a safety
    floor (5\u202f% of capacity), in which case cooperation is forced.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._safety = 0.05 * self.game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        cap = self.game_description.capacity
        fine = self.game_description.fine
        t = history.round_number
        S = current_stock
        if t == 0:
            p_hat = 0.5
        else:
            observed_coop_cnt = int(history.opponent_cooperators[-1])
            p_hat = observed_coop_cnt / (n - 1) if n > 1 else 0.0
        q = 0.0 if t == r - 1 else self.game_description.audit_prob
        expected_fine = q * fine
        payoff_C = S / (2 * n)
        payoff_D = S / n - expected_fine
        opp_expected_consume_one = p_hat * (S / (2 * n)) + (1 - p_hat) * (S / n)
        opp_consume = (n - 1) * opp_expected_consume_one
        total_consume_C = payoff_C + opp_consume
        S_rem_C = max(S - total_consume_C, 0.0)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / cap) if cap != 0 else 0.0
        S_next_C = min(S_rem_C + growth_C, cap)
        total_consume_D = payoff_D + opp_consume
        S_rem_D = max(S - total_consume_D, 0.0)
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / cap) if cap != 0 else 0.0
        S_next_D = min(S_rem_D + growth_D, cap)
        lam = 0.0 if t == r - 1 else (r - t) / r
        future_val_C = lam * (S_next_C / (2 * n))
        future_val_D = lam * (S_next_D / (2 * n))
        exp_total_C = payoff_C + future_val_C
        exp_total_D = payoff_D + future_val_D
        if S_next_C < self._safety and S_next_D < self._safety:
            return Action.C
        if S_next_C < self._safety:
            return Action.C
        if S_next_D < self._safety:
            return Action.D
        if exp_total_D > exp_total_C:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_5(BaseStrategy):
    """
    Adaptive Individualist (AI) strategy.

    The player tracks an exponential moving average of opponents' cooperation,
    computes a dynamic safety‑stock threshold, and decides each round whether
    the immediate private gain from defecting (after expected fine) outweighs
    the discounted future loss caused by a lower stock.  Cooperation is forced
    when the stock falls below the safety threshold or when the future loss
    exceeds the net gain.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._gamma = 0.95
        self._expected_opp_coop = 0.5
        self._base_safety = 0.3 * game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if t > 0:
            k_prev = int(history.opponent_cooperators[-1])
            coop_rate = k_prev / (n - 1) if n > 1 else 0.0
            self._expected_opp_coop = 0.7 * self._expected_opp_coop + 0.3 * coop_rate
        safety_stock = self._base_safety * (1 + (0.5 - self._expected_opp_coop))
        safety_stock = max(0.0, min(safety_stock, capacity))
        if current_stock <= 0.0:
            return Action.C
        extra_consumption = current_stock / (2 * n) if n else 0.0
        expected_fine = q * F
        net_gain_now = extra_consumption - expected_fine
        if t == self.game_description.n_rounds - 1:
            future_loss = 0.0
        else:
            my_cons = current_stock / n
            opp_cons = self._expected_opp_coop * (current_stock / (2 * n)) + (1 - self._expected_opp_coop) * (current_stock / n)
            total_cons = my_cons + (n - 1) * opp_cons
            S_after = max(0.0, current_stock - total_cons)
            growth = 0.0
            if capacity:
                growth = 2 * S_after * (1 - S_after / capacity)
            S_next = min(S_after + growth, capacity)
            future_loss = self._gamma * (capacity - S_next) / (2 * n) if n else 0.0
        if current_stock < safety_stock:
            return Action.C
        if net_gain_now > future_loss:
            return Action.D
        return Action.C

class Strategy_INDIVIDUALISTIC_6(BaseStrategy):
    """
    Self‑Preserving Opportunist.
    Balances immediate gain (with expected audit fine) against a weighted
    estimate of future stock, using an exponential moving average of opponents'
    cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._c_bar = 1.0
        self._lambda = 0.7
        self._alpha = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        gd = self.game_description
        n = gd.n_players
        r = gd.n_rounds
        cap = float(gd.capacity)
        q = gd.audit_prob
        F = gd.fine
        t = history.round_number
        S = float(current_stock)
        if t > 0 and n > 1:
            c_hat = history.opponent_cooperators[-1] / (n - 1)
            self._c_bar = self._lambda * c_hat + (1.0 - self._lambda) * self._c_bar
        pay_C = S / (2.0 * n)
        expected_fine = 0.0 if t == r - 1 else q * F
        pay_D = S / n - expected_fine

        def forecast_stock(my_action: Action) -> float:
            opp_coop_est = self._c_bar * (n - 1)
            my_cons = S / (2.0 * n) if my_action is Action.C else S / n
            coop_cons = opp_coop_est * (S / (2.0 * n))
            defect_cons = (n - 1 - opp_coop_est) * (S / n)
            total_cons = my_cons + coop_cons + defect_cons
            S_rem = S - total_cons
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / cap)
            new_stock = S_rem + growth
            if new_stock > cap:
                new_stock = cap
            return new_stock
        future_stock_C = forecast_stock(Action.C)
        future_stock_D = forecast_stock(Action.D)
        remaining = r - t - 1
        total_remaining = r - t
        factor = remaining / total_remaining if total_remaining > 0 else 0.0
        future_gain_C = future_stock_C * factor
        future_gain_D = future_stock_D * factor
        U_C = pay_C + self._alpha * future_gain_C
        U_D = pay_D + self._alpha * future_gain_D
        if t == r - 1:
            action = Action.D
        elif S <= 0.0:
            action = Action.C
        elif U_D > U_C:
            action = Action.D
        else:
            action = Action.C
        return action

class Strategy_INDIVIDUALISTIC_7(BaseStrategy):
    """
    Self‑Preserving Expected‑Benefit strategy for the Common Pool Resource game.

    The player maintains an estimate `p_est` of how likely any given opponent
    will cooperate.  This estimate is updated each round from the observed
    opponent cooperation count and from audit outcomes (defectors become more
    pessimistic when fined, more optimistic when not fined).  The decision
    balances the immediate expected payoff (including the expected fine) with
    a one‑step look‑ahead of the expected future stock, weighted by a linear
    discount factor `γ`.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_est: float = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            prev_t = history.round_number - 1
            n_players = self.game_description.n_players
            n_opponents = n_players - 1
            c_t = int(history.opponent_cooperators[-1])
            if prev_t > 0 and n_opponents > 0:
                self.p_est = (self.p_est * prev_t + c_t / n_opponents) / (prev_t + 1)
            elif n_opponents > 0:
                self.p_est = c_t / n_opponents
            delta = 0.1
            last_action_coop = bool(history.my_actions[-1])
            was_audited = bool(history.my_audited[-1])
            if not last_action_coop:
                if was_audited:
                    self.p_est = max(0.0, self.p_est - delta)
                else:
                    self.p_est = min(1.0, self.p_est + delta)
            self.p_est = max(0.0, min(1.0, self.p_est))
        t = history.round_number
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        r = self.game_description.n_rounds
        S = max(0.0, float(current_stock))
        if S == 0.0:
            return Action.C
        E_coop = S / (2 * n)
        E_def = S / n - q * F
        if t == r - 1:
            return Action.D if E_def > E_coop else Action.C
        gamma = (r - t - 1) / (r - t) if r - t != 0 else 0.0
        n_opponents = n - 1

        def expected_next_stock(my_consume: float) -> float:
            opp_cons_per = self.p_est * (S / (2 * n)) + (1.0 - self.p_est) * (S / n)
            total_consume = opp_cons_per * n_opponents + my_consume
            S_remaining = S - total_consume
            if S_remaining < 0.0:
                S_remaining = 0.0
            if capacity != 0:
                growth = 2.0 * S_remaining * (1.0 - S_remaining / capacity)
            else:
                growth = 0.0
            S_next = S_remaining + growth
            return min(S_next, float(capacity))
        S_next_coop = expected_next_stock(S / (2 * n))
        S_next_def = expected_next_stock(S / n)
        exp_next_payoff_coop = S_next_coop / (2 * n) * self.p_est + S_next_coop / n * (1.0 - self.p_est)
        exp_next_payoff_def = S_next_def / (2 * n) * self.p_est + S_next_def / n * (1.0 - self.p_est)
        total_coop = E_coop + gamma * exp_next_payoff_coop
        total_def = E_def + gamma * exp_next_payoff_def
        return Action.D if total_def > total_coop else Action.C

class Strategy_INDIVIDUALISTIC_8(BaseStrategy):
    """
    Adaptive Expected‑Return Maximizer (AERM).

    The agent repeatedly evaluates the expected total monetary return
    from cooperating versus defecting in the current round, using the
    empirical cooperation rate of opponents observed so far.  It
    projects the stock dynamics under each possible current action,
    estimates future payoffs over the remaining horizon (assuming the
    same opponent behaviour continues), and selects the action with
    the higher expected cumulative payoff.  Ties are broken in favour of
    cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def _project_stock(self, S: float, p: float, my_action: str) -> float:
        """Return the expected stock after one round."""
        opp_cons = p * (S / (2 * self.n)) + (1 - p) * (S / self.n)
        my_cons = S / (2 * self.n) if my_action == 'C' else S / self.n
        total_cons = (self.n - 1) * opp_cons + my_cons
        S_rem = S - total_cons
        if S_rem < 0.0:
            S_rem = 0.0
        growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
        new_stock = S_rem + growth
        if new_stock > self.capacity:
            new_stock = self.capacity
        return new_stock

    def _expected_future_payoff(self, S_start: float, p: float, k: int) -> float:
        """Simulate forward k steps using the same myopic decision rule."""
        if k <= 0:
            return 0.0
        payoff = 0.0
        S = S_start
        for step in range(1, k + 1):
            audit_factor = 0.0 if step == k else self.q
            val_C = S / (2 * self.n)
            val_D = S / self.n - audit_factor * self.F
            S_C = self._project_stock(S, p, 'C')
            S_D = self._project_stock(S, p, 'D')
            if val_C >= val_D:
                payoff += val_C
                S = S_C
            else:
                payoff += val_D
                S = S_D
        return payoff

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            p = 1.0
        else:
            total_opponent_actions = t * (self.n - 1)
            total_coops = int(np.sum(history.opponent_cooperators))
            p = total_coops / total_opponent_actions if total_opponent_actions > 0 else 0.0
        if S <= 0.0:
            return Action.C
        if t == self.r - 1:
            payoff_C = S / (2 * self.n)
            payoff_D = S / self.n
            return Action.C if payoff_C >= payoff_D else Action.D
        payoff_C_now = S / (2 * self.n)
        payoff_D_now = S / self.n - self.q * self.F
        stock_if_C = self._project_stock(S, p, 'C')
        stock_if_D = self._project_stock(S, p, 'D')
        remaining = self.r - t - 1
        future_C = self._expected_future_payoff(stock_if_C, p, remaining)
        future_D = self._expected_future_payoff(stock_if_D, p, remaining)
        total_C = payoff_C_now + future_C
        total_D = payoff_D_now + future_D
        return Action.C if total_C >= total_D else Action.D

class Strategy_INDIVIDUALISTIC_9(BaseStrategy):
    """
    A forward‑looking, audit‑aware strategy for the Common Pool Resource game.
    It compares the immediate net gain from defecting (extra consumption minus
    expected fine) with an estimate of the discounted loss caused by the
    resulting stock depletion.  The agent defects only when the immediate
    advantage outweighs the projected future loss; otherwise it cooperates.
    Special handling is applied for the first round, the final round, and
    the degenerate case of zero stock.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._observed_coops: list[int] = []

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        R = r - t - 1
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D if current_stock > 0 else Action.C
        if t == 0:
            p_hat = 0.5
        else:
            total_obs = sum(self._observed_coops)
            total_turns = len(self._observed_coops) * (n - 1)
            p_hat = 0.0 if total_turns == 0 else total_obs / total_turns
        if t > 0:
            self._observed_coops.append(int(history.opponent_cooperators[-1]))
        immediate_gain = current_stock / (2 * n) - q * F
        if immediate_gain <= 0:
            return Action.C
        extra_consume = current_stock / (2 * n)
        delta_stock = extra_consume
        growth_factor = 1 + 2 * (1 - 2 * current_stock / capacity)
        delta_S_next = -delta_stock * growth_factor
        future_loss = -delta_S_next * (R + 1) * (current_stock / (2 * n)) / max(capacity, 1e-12)
        if immediate_gain > future_loss:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_10(BaseStrategy):
    """
    Safety‑stock, finite‑memory, forward‑looking strategy for the common‑pool
    resource game.  It estimates opponents' cooperation probability from the
    last ``m`` rounds, evaluates the immediate expected payoff of cooperating
    versus defecting (including expected fines), projects the stock one step
    ahead for each alternative, and then computes a discounted future value
    assuming the chosen action is kept for the remaining rounds.  A safety‑
    stock threshold ``λ·capacity`` prevents a defect that would push the stock
    below this level unless cooperation is clearly superior.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = game_description.audit_prob
        self.F = float(game_description.fine)
        self.lamb = 0.4
        self.eps = 0.01
        self.m = 5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n
        capacity = self.capacity
        q = self.q if t < self.r - 1 else 0.0
        F = self.F
        if t == 0:
            p_hat = 0.0
        else:
            start = max(0, t - self.m)
            observed = history.opponent_cooperators[start:t]
            denom = (n - 1) * (t - start)
            p_hat = float(observed.sum()) / denom if denom > 0 else 0.0
        coop_pay = current_stock / (2.0 * n)
        defect_pay = current_stock / n - q * F
        exp_opp_cons = p_hat * (current_stock / (2.0 * n)) + (1.0 - p_hat) * (current_stock / n)
        total_opp_cons = (n - 1) * exp_opp_cons
        cons_coop = current_stock / (2.0 * n)
        cons_def = current_stock / n
        S_after_coop = max(0.0, current_stock - (total_opp_cons + cons_coop))
        S_after_def = max(0.0, current_stock - (total_opp_cons + cons_def))

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / capacity)
        S_next_coop = min(S_after_coop + grow(S_after_coop), capacity)
        S_next_def = min(S_after_def + grow(S_after_def), capacity)

        def future_value_coop(start_stock: float) -> float:
            cur = start_stock
            total = 0.0
            for future_t in range(t + 1, self.r):
                exp_opp = p_hat * (cur / (2.0 * n)) + (1.0 - p_hat) * (cur / n)
                total_cons = (n - 1) * exp_opp + cur / (2.0 * n)
                cur = max(0.0, cur - total_cons)
                cur = min(cur + grow(cur), capacity)
                total += cur / (2.0 * n)
            return total

        def future_value_def(start_stock: float) -> float:
            cur = start_stock
            total = 0.0
            for future_t in range(t + 1, self.r):
                audit_prob = self.q if future_t < self.r - 1 else 0.0
                exp_opp = p_hat * (cur / (2.0 * n)) + (1.0 - p_hat) * (cur / n)
                total_cons = (n - 1) * exp_opp + cur / n
                cur = max(0.0, cur - total_cons)
                cur = min(cur + grow(cur), capacity)
                total += cur / n - audit_prob * F
            return total
        FV_coop = future_value_coop(S_next_coop)
        FV_def = future_value_def(S_next_def)
        U_coop = coop_pay + FV_coop
        U_def = defect_pay + FV_def
        safety = self.lamb * capacity
        if S_next_def < safety and U_coop + self.eps > U_def:
            return Action.C
        if U_def > U_coop + self.eps:
            return Action.D
        else:
            return Action.C

class Strategy_INDIVIDUALISTIC_11(BaseStrategy):
    """
    Self‑Preserving Greedy strategy.

    At each round the player compares the immediate extra payoff from defecting
    with the expected fine and the estimated future loss caused by the extra
    stock consumption.  Cooperation is chosen when the net expected gain is
    non‑positive; otherwise the player defects.  In the final round, where no
    future loss exists and audits do not occur, the player defects if any stock
    remains.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._eps = 1e-12

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        q = self.game_description.audit_prob
        F = self.game_description.fine
        if current_stock <= self._eps:
            return Action.C
        if t == r - 1:
            return Action.D if current_stock > 0 else Action.C
        delta_pay = current_stock / (2.0 * n)
        delta_stock = delta_pay
        remaining_rounds = r - t - 1
        v_unit = remaining_rounds / (2.0 * n)
        future_loss = v_unit * delta_stock
        expected_fine = q * F
        net_gain = delta_pay - expected_fine - future_loss
        return Action.D if net_gain > 0 else Action.C

class Strategy_INDIVIDUALISTIC_12(BaseStrategy):
    """
    Adaptive Individualist (AI‑CPC) strategy.
    Estimates opponents' cooperation probability (p̂) and the marginal
    future value of one unit of stock (V) from observed history.
    Defects only when the immediate expected gain (including audit risk)
    exceeds the expected future loss from the extra consumption.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.q = game_description.audit_prob
        self.F = game_description.fine
        self.p_hat_initial = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if t == 0:
            p_hat = self.p_hat_initial
        else:
            total_coop = int(np.sum(history.opponent_cooperators))
            p_hat = total_coop / ((self.n - 1) * t)
        pi_C = current_stock / (2 * self.n)
        pi_D = current_stock / self.n - self.q * self.F
        delta_inst = pi_D - pi_C
        delta_cons = pi_C
        future_rounds = self.r - t - 1
        if future_rounds > 0 and current_stock > 0.0:
            exp_per_capita = p_hat * (current_stock / (2 * self.n)) + (1 - p_hat) * (current_stock / self.n - self.q * self.F)
            V = future_rounds * exp_per_capita / current_stock
        else:
            V = 0.0
        delta_future = delta_cons * V
        delta_net = delta_inst - delta_future
        return Action.D if delta_net > 0 else Action.C

class Strategy_INDIVIDUALISTIC_13(BaseStrategy):
    """
    Conditional‑Threshold Adaptive Play (CTAP).

    The agent uses the current common‑pool stock, the observed cooperation
    of opponents in the previous round, and its own audit history to decide
    between cooperating (C) and defecting (D).  The decision follows the
    specification:

    * Seed cooperation in round 0 only if the stock is above a safety
      threshold (60\u202f% of capacity).
    * In the final round, always defect because no future consequences
      exist and the round is never audited.
    * Otherwise compute the immediate net gain from defecting (Δ) after
      accounting for the expected fine.  If Δ ≤\u202f0 the agent cooperates.
      If Δ >\u202f0 an adjusted gain (adj) is derived by penalising defect when
      many opponents previously cooperated, when the agent was recently
      audited, and when the stock is low.  The agent defects only when
      adj >\u202f0.

    All required parameters are taken from the provided ``CommonPoolDescription``
    and ``PlayerHistory`` objects; no external state is used.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.S_safe = self.capacity * 0.6
        self.expected_fine = self.audit_prob * self.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action for the current round."""
        t = history.round_number
        n = self.n
        capacity = self.capacity
        S_safe = self.S_safe
        expected_fine = self.expected_fine
        stock = float(current_stock) if current_stock > 0 else 0.0
        if t == self.r - 1:
            return Action.D
        if t == 0:
            return Action.C if stock >= S_safe else Action.D
        obs_coop = int(history.opponent_cooperators[-1])
        w = obs_coop / (n - 1) if n > 1 else 0.0
        was_audited = bool(history.my_audited[-1])
        C_share = stock / (2 * n) if stock > 0 else 0.0
        D_share = stock / n if stock > 0 else 0.0
        delta = D_share - C_share - expected_fine
        if delta <= 0:
            return Action.C
        adj = delta * (1 - w)
        if was_audited:
            adj += expected_fine * 0.5
        if stock < S_safe:
            adj += stock / capacity * 0.2
        return Action.D if adj > 0 else Action.C

class Strategy_INDIVIDUALISTIC_14(BaseStrategy):
    """
    Adaptive Individual‑Benefit Forecast (AIBF).

    The strategy estimates the cooperation probability of opponents from the previous
    round, forecasts the stock after the current round for each possible personal
    action, includes the expected audit fine, and adds a forecast of future payoff
    assuming full cooperation thereafter.  The action with the higher total
    estimated value is chosen; ties are broken in favour of cooperation.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            p = 0.5
        else:
            c_prev = int(history.opponent_cooperators[t - 1])
            p = c_prev / (self.n - 1) if self.n > 1 else 0.0
        future_rounds = self.r - t - 1
        best_action = Action.C
        best_value = -float('inf')
        eps = 1e-12
        for act in (Action.C, Action.D):
            if act is Action.C:
                self_consume = S / (2.0 * self.n)
                audit_cost = 0.0
            else:
                self_consume = S / float(self.n)
                audit_cost = 0.0 if t == self.r - 1 else self.audit_prob * self.fine
            opp_consume = p * (S / (2.0 * self.n)) + (1.0 - p) * (S / float(self.n))
            total_consume = self_consume + (self.n - 1) * opp_consume
            S_remain = S - total_consume
            if S_remain < 0.0:
                S_remain = 0.0
            growth = 2.0 * S_remain * (1.0 - S_remain / self.capacity) if self.capacity != 0 else 0.0
            S_next = S_remain + growth
            if S_next > self.capacity:
                S_next = self.capacity
            immediate_payoff = self_consume - audit_cost
            future_payoff = future_rounds * (S_next / (2.0 * self.n))
            total_value = immediate_payoff + future_payoff
            if total_value > best_value + eps:
                best_value = total_value
                best_action = act
            elif abs(total_value - best_value) <= eps:
                best_action = Action.C
        return best_action

class Strategy_INDIVIDUALISTIC_15(BaseStrategy):
    """
    Self‑Preserving Expected‑Value strategy for the Common Pool Resource Game.
    It balances immediate payoff (including expected fine) with a discounted
    estimate of future stock value, using a simple belief about opponents'
    cooperation derived from the previous round's observed cooperators.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == self.r - 1:
            payoff_c = current_stock / (2 * self.n)
            payoff_d = current_stock / self.n
            return Action.C if payoff_c >= payoff_d else Action.D
        if t == 0:
            p = 0.5
        else:
            k_prev = int(history.opponent_cooperators[-1])
            p = 0.5 + 0.5 * (k_prev / (self.n - 1))
        cons_self_c = current_stock / (2 * self.n)
        cons_self_d = current_stock / self.n
        opp_cons_expect = p * (current_stock / (2 * self.n)) + (1 - p) * (current_stock / self.n)
        cons_opp = (self.n - 1) * opp_cons_expect
        lam = (self.r - t - 1) / (2 * self.n)

        def projected_stock(cons_self: float) -> float:
            total_cons = cons_self + cons_opp
            s_rem = max(0.0, current_stock - total_cons)
            growth = 2.0 * s_rem * (1.0 - s_rem / self.capacity) if self.capacity != 0 else 0.0
            s_next = min(s_rem + growth, self.capacity)
            return s_next
        e_c = current_stock / (2 * self.n)
        s_next_c = projected_stock(cons_self_c)
        future_c = lam * s_next_c
        u_c = e_c + future_c
        e_d = current_stock / self.n - self.q * self.F
        s_next_d = projected_stock(cons_self_d)
        future_d = lam * s_next_d
        u_d = e_d + future_d
        return Action.C if u_c >= u_d else Action.D

class Strategy_INDIVIDUALISTIC_16(BaseStrategy):
    """
    Adaptive Expected‑Return Tracker (AERT).

    The agent estimates the probability that a random opponent cooperates,
    projects the stock dynamics under each possible own action, and chooses
    the action that maximizes the sum of the immediate expected payoff and a
    discounted continuation value.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.lambda_ = 0.8
        self.tau = 0.6
        self.p_hat = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            C_obs = int(history.opponent_cooperators[-1])
            opp_prob = C_obs / (self.n - 1) if self.n > 1 else 0.0
            self.p_hat = self.lambda_ * self.p_hat + (1.0 - self.lambda_) * opp_prob
        t = history.round_number
        last_round = t == self.r - 1
        if current_stock <= 0.0:
            return Action.C
        if last_round:
            return Action.D
        U_C = current_stock / (2.0 * self.n)
        U_D = current_stock / self.n - self.audit_prob * self.fine
        coop_opp_cons = self.p_hat * (self.n - 1) * (current_stock / (2.0 * self.n))
        defect_opp_cons = (1.0 - self.p_hat) * (self.n - 1) * (current_stock / self.n)
        total_cons_coop = coop_opp_cons + defect_opp_cons + current_stock / (2.0 * self.n)
        total_cons_def = coop_opp_cons + defect_opp_cons + current_stock / self.n
        S_rem_coop = max(0.0, current_stock - total_cons_coop)
        S_rem_def = max(0.0, current_stock - total_cons_def)
        if self.capacity != 0.0:
            growth_coop = 2.0 * S_rem_coop * (1.0 - S_rem_coop / self.capacity)
            growth_def = 2.0 * S_rem_def * (1.0 - S_rem_def / self.capacity)
        else:
            growth_coop = growth_def = 0.0
        S_next_coop = min(S_rem_coop + growth_coop, self.capacity)
        S_next_def = min(S_rem_def + growth_def, self.capacity)
        remaining_rounds = self.r - t - 1
        V_coop = self.tau * remaining_rounds * (S_next_coop / self.n)
        V_def = self.tau * remaining_rounds * (S_next_def / self.n)
        EV_coop = U_C + V_coop
        EV_def = U_D + V_def
        return Action.D if EV_def > EV_coop else Action.C

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

class Strategy_OPPORTUNISTIC_2(BaseStrategy):
    """
    Opportunistic Adaptive Strategy (OAS).

    The strategy keeps an exponential moving average of the opponents' cooperation
    level and uses a stock‑safety threshold to decide when to defect.
    It evaluates the immediate net gain from defecting (including expected fine)
    together with a simple forecast of future payoffs under the assumed continuation
    of the observed opponent behaviour.  Defection is chosen only if the total
    expected advantage is positive and the stock stays above a safety level;
    otherwise the player cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n_players = game_description.n_players
        self.n_rounds = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = float(game_description.fine)
        self._beta = 0.2
        self._alpha = 0.8
        self._opp_coop_avg = 1.0
        self._defect_score = 0.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n_players
        r = self.n_rounds
        capacity = self.capacity
        q = self.audit_prob
        F = self.fine
        if t > 0:
            coop_cnt = int(history.opponent_cooperators[-1])
            opp_coop_now = coop_cnt / (n - 1) if n > 1 else 0.0
            self._opp_coop_avg = (1 - self._beta) * self._opp_coop_avg + self._beta * opp_coop_now
        remaining = r - t
        tau = capacity * (remaining / r) * self._alpha
        if remaining <= 0:
            return Action.C
        if current_stock <= 0.0:
            return Action.C
        pay_C = current_stock / (2 * n)
        pay_D = current_stock / n
        exp_fine = 0.0 if t == r - 1 else q * F
        gain_now = pay_D - exp_fine - pay_C
        exp_opp_cons_per = self._opp_coop_avg * (current_stock / (2 * n)) + (1.0 - self._opp_coop_avg) * (current_stock / n)

        def projected_stock(own_payoff: float) -> float:
            total_cons = own_payoff + (n - 1) * exp_opp_cons_per
            stock_rem = current_stock - total_cons
            if stock_rem < 0.0:
                stock_rem = 0.0
            growth = 2.0 * stock_rem * (1.0 - stock_rem / capacity)
            new_stock = stock_rem + growth
            if new_stock > capacity:
                new_stock = capacity
            return new_stock
        stock_if_C = projected_stock(pay_C)
        stock_if_D = projected_stock(pay_D)
        future_rounds = remaining - 1
        if future_rounds > 0:
            future_C = future_rounds * (stock_if_C / (2 * n))
            exp_future_fine = q * F
            future_D = future_rounds * (stock_if_D / n) - future_rounds * exp_future_fine
        else:
            future_C = 0.0
            future_D = 0.0
        total_gain = gain_now + (future_D - future_C)
        if current_stock < tau:
            action = Action.C
        elif total_gain > 0.0 and current_stock >= tau:
            action = Action.D
        else:
            action = Action.C
        self._defect_score += gain_now
        return action

class Strategy_OPPORTUNISTIC_3(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    The agent maintains a Bayesian estimate of opponent cooperation (Beta(α,β))
    and a discounted estimate of the value of a higher stock for future rounds.
    Each round it compares the immediate gain from defection (including expected
    fine) with the gain from cooperation plus the projected future benefit of a
    larger stock.  The policy becomes aggressively defective when opponents appear
    mostly selfish and temporarily more cooperative when the pool is healthy and
    opponents are highly cooperative.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.alpha = 1.0
        self.beta = 1.0
        self.delta = 0.95
        self.future_value_est = 0.0
        self.theta_low = 0.2
        self.theta_high = 0.8
        self.prev_stock = self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = max(0.0, min(current_stock, self.capacity))
        if t > 0:
            c_obs = int(history.opponent_cooperators[-1])
            self.alpha += c_obs
            self.beta += self.n - 1 - c_obs
        coop_est = self.alpha / (self.alpha + self.beta)
        if t < self.r - 1:
            stock_change = S - self.prev_stock
            weight = (self.r - t - 1) / self.r
            self.future_value_est = self.delta * self.future_value_est + stock_change * weight
        else:
            self.future_value_est = 0.0
        q_eff = 0.0 if t == self.r - 1 else self.audit_prob
        pay_C = S / (2.0 * self.n)
        base_D = S / self.n
        pay_D = base_D - q_eff * self.fine
        S_after_C = S - self.n * pay_C
        S_after_C = max(0.0, S_after_C)
        growth_C = 2.0 * S_after_C * (1.0 - S_after_C / self.capacity)
        S_C = min(S_after_C + growth_C, self.capacity)
        exp_opponent_cons = coop_est * (S / (2.0 * self.n)) + (1.0 - coop_est) * (S / self.n)
        total_cons_D = pay_D + (self.n - 1) * exp_opponent_cons
        S_after_D = S - total_cons_D
        S_after_D = max(0.0, S_after_D)
        growth_D = 2.0 * S_after_D * (1.0 - S_after_D / self.capacity)
        S_D = min(S_after_D + growth_D, self.capacity)
        delta_future = self.future_value_est * (S_D - S_C) / self.capacity
        eff_D = pay_D + delta_future
        if S <= 0.0:
            action = Action.C
        elif t == self.r - 1:
            action = Action.D if S > 0.0 else Action.C
        elif coop_est < self.theta_low:
            action = Action.D
        else:
            if coop_est > self.theta_high and S > 0.5 * self.capacity:
                self.future_value_est *= 1.2
            action = Action.D if eff_D > pay_C else Action.C
        self.prev_stock = S
        return action

class Strategy_OPPORTUNISTIC_4(BaseStrategy):
    """
    Opportunistic Adaptive Strategy.
    Defects when the immediate expected gain from defecting,
    plus the weighted expected change in the next‑round advantage,
    is positive. Otherwise cooperates.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        if self.r >= 10:
            self.gamma = 0.7
        elif self.r <= 3:
            self.gamma = 0.3
        else:
            self.gamma = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide C or D for the current round."""
        if current_stock <= 0.0:
            return Action.C
        t = history.round_number
        n = self.n
        q = self.q
        F = self.F
        cap = self.capacity
        gamma = self.gamma
        if t == 0:
            k_est = n - 1
        else:
            k_est = int(history.opponent_cooperators[-1])
        delta = current_stock / (2.0 * n) - q * F
        coop_cons = (k_est + 1) * (current_stock / (2.0 * n)) + (n - k_est - 1) * (current_stock / n)
        defect_cons = k_est * (current_stock / (2.0 * n)) + (n - k_est) * (current_stock / n)

        def post_stock(consumption: float) -> float:
            """Compute the stock after consumption and logistic growth."""
            s_rem = max(0.0, current_stock - consumption)
            growth = 2.0 * s_rem * (1.0 - s_rem / cap) if cap > 0 else 0.0
            return min(s_rem + growth, cap)
        s_next_c = post_stock(coop_cons)
        s_next_d = post_stock(defect_cons)
        delta_next_c = s_next_c / (2.0 * n) - q * F
        delta_next_d = s_next_d / (2.0 * n) - q * F
        adv = delta + gamma * (delta_next_d - delta_next_c)
        if t == self.r - 1:
            return Action.D if delta > 0.0 else Action.C
        else:
            return Action.D if adv > 0.0 else Action.C

class Strategy_OPPORTUNISTIC_5(BaseStrategy):
    """
    Opportunistic Threshold‑Defect Strategy.

    - Defects when the immediate expected net gain (after audit risk) exceeds cooperation,
      *and* the projected stock after this round stays above a sustainability threshold.
    - Cooperates otherwise, including the final round where there is no audit risk.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        F = self.game_description.fine
        if t == r - 1:
            return Action.D
        immediate_defect = current_stock / n - q * F
        immediate_coop = current_stock / (2 * n)
        if immediate_defect <= immediate_coop:
            return Action.C
        if t == 0:
            p_def = 0.5
        else:
            last_opponents_coop = int(history.opponent_cooperators[-1])
            p_def = (n - 1 - last_opponents_coop) / (n - 1)
        exp_consumption_per_opponent = p_def * current_stock / n + (1.0 - p_def) * current_stock / (2 * n)
        exp_consumption_opponents = (n - 1) * exp_consumption_per_opponent
        exp_consumption_self = current_stock / n
        total_exp_consumption = exp_consumption_self + exp_consumption_opponents
        S_rem = current_stock - total_exp_consumption
        if S_rem < 0.0:
            S_rem = 0.0
        growth = 2.0 * S_rem * (1.0 - S_rem / capacity) if capacity != 0.0 else 0.0
        S_next = S_rem + growth
        if S_next > capacity:
            S_next = capacity
        S_thresh = 2.0 * n * q * F
        if S_next < S_thresh:
            return Action.C
        else:
            return Action.D

class Strategy_OPPORTUNISTIC_6(BaseStrategy):
    """
    Opportunistic Expected‑Gain Strategy for the Common Pool Resource Game.

    Core idea:
        - Compute the expected net gain of defecting (extra consumption minus
          expected fine). Defect if this gain is positive.
        - Otherwise, look at the opponent defection rate in the previous round.
          If opponents are heavily defecting (≥ θ) and the stock is relatively high
          (≥ α·capacity), add a bonus Δ that may tip the balance toward defection.
        - Edge handling:
            * First round – only the core rule.
            * Final round – audits never occur, so always defect if stock > 0.
            * Zero stock – cooperate (any action yields 0 payoff).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self._theta = 0.6
        self._alpha = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n
        r = self.r
        capacity = self.capacity
        q = self.audit_prob
        F = self.fine
        if current_stock <= 0:
            return Action.C
        if t == r - 1:
            return Action.D if current_stock > 0 else Action.C
        expected_fine = q * F
        profit_D = current_stock / n
        profit_C = current_stock / (2 * n)
        net_gain = profit_D - profit_C - expected_fine
        if net_gain > 0:
            return Action.D
        if t > 0:
            c_prev = int(history.opponent_cooperators[-1])
            opponents = n - 1
            if opponents > 0:
                p_prev = (opponents - c_prev) / opponents
                if p_prev >= self._theta and current_stock >= self._alpha * capacity:
                    delta = (p_prev - self._theta) * current_stock / (2 * n)
                    if net_gain + delta > 0:
                        return Action.D
        return Action.C

class Strategy_OPPORTUNISTIC_7(BaseStrategy):
    """
    Opportunistic defector with risk‑aware projection.
    At each round the agent compares the expected net payoff from defecting
    (including the expected fine) with cooperating, while penalising aggressive
    defection when the projected stock after the round would be low.
    It estimates opponents' cooperation rates from history and uses a small
    risk‑penalty α = 0.1·max(c_pay,d_pay).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        total_rounds = self.game_description.n_rounds
        t = history.round_number
        if current_stock <= 0:
            return Action.C
        if t == total_rounds - 1:
            c_pay = current_stock / (2 * n) if n else 0.0
            d_pay = current_stock / n if n else 0.0
            if d_pay >= c_pay:
                return Action.D
            return Action.C
        c_pay = current_stock / (2 * n) if n else 0.0
        d_pay = current_stock / n if n else 0.0
        exp_fine = audit_prob * fine
        if t == 0:
            p_C = 1.0
        else:
            avg_coop = float(np.mean(history.opponent_cooperators))
            p_C = avg_coop / (n - 1) if n > 1 else 0.0
            p_C = max(0.0, min(1.0, p_C))
        p_D = 1.0 - p_C
        k_c = (n - 1) * p_C
        k_d = (n - 1) * p_D
        expected_consumption = k_c * c_pay + k_d * d_pay + c_pay
        stock_remain = current_stock - expected_consumption
        if stock_remain <= 0:
            return Action.C
        growth = 2 * stock_remain * (1 - stock_remain / capacity) if capacity != 0 else 0.0
        projected_stock = min(stock_remain + growth, capacity)
        if projected_stock <= 0:
            return Action.C
        risk = 1.0 - projected_stock / capacity if capacity != 0 else 1.0
        alpha = 0.1 * max(c_pay, d_pay)
        d_exp = d_pay - exp_fine
        score = d_exp - c_pay - alpha * risk
        if score > 0 or abs(score) < 1e-12:
            return Action.D
        return Action.C

class Strategy_OPPORTUNISTIC_8(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    - Defects when the audit‑adjusted expected payoff exceeds the cooperative payoff,
      unless defecting would push the projected stock below a safety threshold.
    - Never cooperates in the final round (no future value to preserve).
    - If the stock is depleted, cooperates to avoid unnecessary fines.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._n = game_description.n_players
        self._r = game_description.n_rounds
        self._capacity = float(game_description.capacity)
        self._audit_prob = float(game_description.audit_prob)
        self._fine = float(game_description.fine)
        self._danger_threshold = self._capacity * 0.1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == self._r - 1:
            return Action.D
        n = self._n
        q = self._audit_prob
        F = self._fine
        capacity = self._capacity
        coop_payoff = current_stock / (2.0 * n)
        expected_defect = current_stock / n - q * F
        S_after_def = current_stock - current_stock / n
        S_after_coop = current_stock - current_stock / (2.0 * n)

        def grow(x: float) -> float:
            grown = x + 2.0 * x * (1.0 - x / capacity)
            return min(grown, capacity)
        projected_def = grow(S_after_def)
        projected_coop = grow(S_after_coop)
        if projected_def < self._danger_threshold and projected_coop >= self._danger_threshold:
            return Action.C
        if expected_defect > coop_payoff:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_9(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource Game.
    It balances immediate gain from defection (adjusted for audit risk)
    against the future value of preserving stock, using the observed
    cooperation rate of opponents in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        capacity = self.capacity
        q = self.audit_prob
        F = self.fine
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            p_coop = 0.5
        else:
            k_last = int(history.opponent_cooperators[-1])
            p_coop = k_last / (n - 1) if n - 1 > 0 else 0.0
        payoff_C = S / (2.0 * n)
        payoff_D = S / n - q * F
        opp_exp_consumption = p_coop * payoff_C + (1.0 - p_coop) * payoff_D
        total_opp_consumption = (n - 1) * opp_exp_consumption
        expect_coop_consumption = payoff_C + total_opp_consumption
        expect_defect_consumption = payoff_D + total_opp_consumption
        S_if_C = max(0.0, S - expect_coop_consumption)
        S_if_D = max(0.0, S - expect_defect_consumption)

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / capacity)
        S_next_C = min(S_if_C + grow(S_if_C), capacity)
        S_next_D = min(S_if_D + grow(S_if_D), capacity)
        remaining = r - t
        future_rounds = remaining - 1
        if future_rounds > 0:
            value_per_unit = future_rounds * (1.0 / (2.0 * n))
        else:
            value_per_unit = 0.0
        future_val_C = (S_next_C - S) * value_per_unit
        future_val_D = (S_next_D - S) * value_per_unit
        U_C = payoff_C + future_val_C
        U_D = payoff_D + future_val_D
        if U_D > U_C:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_10(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    - Defects in the final round (no audit, no future).
    - Otherwise compares expected payoff of defecting (including expected fine)
      with the guaranteed payoff of cooperating.
    - Refuses to defect when the stock falls below 25\u202f% of capacity.
    - When defecting looks attractive, checks whether the projected stock after
      this round (assuming opponents keep their estimated cooperation rate) would
      stay above the safety threshold; if not, cooperates to avoid collapse.
    - Estimates opponents' cooperation rate from the count of cooperating
      opponents in the previous round; assumes full cooperation at the very start.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.safety_thresh = 0.25 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if current_stock < self.safety_thresh:
            return Action.C
        if t == 0:
            coop_rate = 1.0
        else:
            coop_cnt = int(history.opponent_cooperators[-1])
            coop_rate = coop_cnt / (self.n - 1) if self.n > 1 else 0.0
        coop_pay = current_stock / (2 * self.n)
        defect_pay = current_stock / self.n - self.audit_prob * self.fine
        if defect_pay <= coop_pay:
            return Action.C
        opp_exp_consumption = coop_rate * current_stock / (2 * self.n) + (1 - coop_rate) * current_stock / self.n
        total_opp_consumption = (self.n - 1) * opp_exp_consumption
        my_consumption = current_stock / self.n
        projected_stock = current_stock - (total_opp_consumption + my_consumption)
        if projected_stock >= self.safety_thresh:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_11(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    - Defects when the expected net payoff (including expected fine) exceeds
      cooperating, provided the stock is not critically low.
    - Uses the opponent cooperation count from the previous round to
      forecast next‑round stock and gives extra weight to future stock when
      the current stock is scarce.
    - Always defects in the final round (no audit risk and no future benefit).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.THRESH = 2.0 * self.n * self.q * self.F
        self.MIN_SAFE = self.capacity * 0.05
        self.LAST_ROUND = self.r - 1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        if history.round_number == 0:
            if S > self.THRESH and S > self.MIN_SAFE:
                return Action.D
            else:
                return Action.C
        if history.round_number == self.LAST_ROUND:
            return Action.D
        coop_cnt = int(history.opponent_cooperators[-1])
        ev_defect = S / self.n - self.q * self.F
        ev_coop = S / (2.0 * self.n)
        cons_def = S / self.n
        cons_coop = S / (2.0 * self.n)
        opp_cons = coop_cnt * (S / (2.0 * self.n)) + (self.n - 1 - coop_cnt) * (S / self.n)
        S_rem_def = max(0.0, S - (cons_def + opp_cons))
        S_rem_coop = max(0.0, S - (cons_coop + opp_cons))
        if self.capacity > 0.0:
            grow_def = 2.0 * S_rem_def * (1.0 - S_rem_def / self.capacity)
            grow_coop = 2.0 * S_rem_coop * (1.0 - S_rem_coop / self.capacity)
        else:
            grow_def = grow_coop = 0.0
        S_next_def = min(S_rem_def + grow_def, self.capacity)
        S_next_coop = min(S_rem_coop + grow_coop, self.capacity)
        weight_future = 1.0 - S / self.capacity if self.capacity > 0.0 else 0.0
        score_def = ev_defect + weight_future * (S_next_def - S_next_coop)
        score_coop = ev_coop
        if score_def > score_coop and S > self.MIN_SAFE:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_12(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    The agent compares the net immediate gain from defecting (gain minus expected fine)
    with a forward‑looking estimate of future stock loss.  If the immediate gain
    exceeds a fraction γ of the estimated loss it defects, otherwise it cooperates.
    Special handling is applied for the first round, the last round, and very low
    stock levels.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.gamma = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        capacity = self.capacity
        q = self.q
        F = self.F
        gamma = self.gamma
        t = history.round_number
        stock = float(current_stock)
        if stock <= 0.0 or stock < 1e-06:
            return Action.C
        if t == r - 1:
            return Action.D if stock > 0.0 else Action.C
        if t == 0:
            pCoop = 0.5
        else:
            total_coop_opp = int(np.sum(history.opponent_cooperators[:t]))
            denom = (n - 1) * t
            pCoop = total_coop_opp / denom if denom > 0 else 0.5
        gain_defect = stock / (2.0 * n)
        expected_fine = q * F if t < r - 1 else 0.0
        net_immediate = gain_defect - expected_fine
        if net_immediate <= 0.0:
            return Action.C
        remaining = r - t - 1
        extra_consumption = stock / (2.0 * n)
        growth_derivative = 2.0 * (1.0 - 2.0 * stock / capacity)
        loss_factor = 1.0 - growth_derivative
        loss_factor = max(loss_factor, 0.0)
        future_loss = remaining * extra_consumption * loss_factor
        threshold = gamma * future_loss
        if net_immediate > threshold:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_13(BaseStrategy):
    """
    Audit‑adjusted threshold strategy for the Common Pool Resource game.
    Defects when the stock is comfortably above a dynamic threshold that
    depends on the audit probability, fine, and recent opponent behaviour.
    Always defects in the final round (no audit then). Otherwise cooperates
    if the stock is too low to risk further depletion.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = game_description.capacity
        self.q = game_description.audit_prob
        self.F = game_description.fine
        self.alpha = 1.0
        self.beta = 1.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if t == self.r - 1:
            return Action.D
        if t == 0:
            C_obs = self.n - 1
        else:
            C_obs = int(history.opponent_cooperators[-1])
        theta0 = 2 * self.n * self.q * self.F
        if self.n > 1:
            p_def = (self.n - 1 - C_obs) / (self.n - 1)
        else:
            p_def = 0.0
        theta = theta0 * (1.0 + self.alpha * p_def)
        S_min = self.beta * theta
        if current_stock > theta and current_stock > S_min:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_14(BaseStrategy):
    """
    Opportunistic Common‑Pool Strategy.

    The player defects when the expected immediate gain (extra consumption
    minus expected fine) is positive and the resulting stock stays above a
    safety threshold; otherwise it cooperates.  The decision also looks one
    step ahead by adding the projected next‑round stock value to the payoff
    estimate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        r = self.game_description.n_rounds
        R = r - t
        if t == r - 1:
            delta = current_stock / (2 * self.n)
            return Action.D if delta > 0 else Action.C
        if t == 0:
            p_hat = 0.5
        else:
            opp_coop = history.opponent_cooperators[-1]
            opp_total = self.n - 1
            p_hat = opp_coop / opp_total if opp_total > 0 else 0.0
        S = current_stock
        n = self.n
        delta = S / (2 * n)
        expected_fine = self.audit_prob * self.fine
        G = delta - expected_fine
        if G <= 0:
            return Action.C
        coop_cons = S / (2 * n)
        defect_cons = S / n
        opp_cons = p_hat * coop_cons + (1 - p_hat) * defect_cons
        total_cons_coop = coop_cons + (n - 1) * opp_cons
        total_cons_def = defect_cons + (n - 1) * opp_cons
        S_rem_coop = max(S - total_cons_coop, 0.0)
        S_rem_def = max(S - total_cons_def, 0.0)

        def growth(x: float) -> float:
            return 2 * x * (1 - x / self.capacity)
        S_next_coop = min(S_rem_coop + growth(S_rem_coop), self.capacity)
        S_next_def = min(S_rem_def + growth(S_rem_def), self.capacity)
        if S_next_def < 0.05 * self.capacity and R > 1:
            return Action.C
        payoff_coop = coop_cons
        payoff_def = defect_cons - expected_fine
        score_coop = payoff_coop + S_next_coop
        score_def = payoff_def + S_next_def
        return Action.D if score_def > score_coop else Action.C

class Strategy_OPPORTUNISTIC_15(BaseStrategy):
    """
    Opportunistic Stock‑Preserving Defector.

    The player defects whenever the expected immediate gain (after accounting for
    audit risk) is positive **and** the projected stock after defecting stays
    above a safety threshold that depends on the remaining horizon.
    Otherwise it cooperates to preserve the common pool.
    The threshold adapts each round and uses the observed number of cooperating
    opponents from the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = game_description.audit_prob
        self.F = game_description.fine

    def _sigma(self, m: int) -> float:
        """Minimum proportion of capacity that must remain."""
        proportional = (m - 1) / self.r if self.r else 0.0
        return max(0.05, proportional)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0:
            return Action.C
        n = self.n
        capacity = self.capacity
        q = self.q
        F = self.F
        m = self.r - t
        safety = capacity * self._sigma(m)
        delta_immediate = current_stock / (2 * n) - q * F
        if t == self.r - 1:
            return Action.D
        if t == 0:
            expected_consumption_if_D = current_stock / n + (n - 1) * (current_stock / (2 * n))
            projected_stock_after_D = current_stock - expected_consumption_if_D
            if delta_immediate > 0 and projected_stock_after_D >= safety:
                return Action.D
            else:
                return Action.C
        k_last = int(history.opponent_cooperators[-1])
        opp_total = n - 1
        p_coop = k_last / opp_total if opp_total > 0 else 0.0
        coop_payoff = current_stock / (2 * n)
        defect_payoff = current_stock / n
        exp_opp_cons = (p_coop * coop_payoff + (1.0 - p_coop) * defect_payoff) * opp_total
        total_cons_C = coop_payoff + exp_opp_cons
        S_prime_C = max(0.0, current_stock - total_cons_C)
        growth_C = 2.0 * S_prime_C * (1.0 - S_prime_C / capacity) if capacity else 0.0
        S_next_C = min(S_prime_C + growth_C, capacity)
        total_cons_D = defect_payoff + exp_opp_cons
        S_prime_D = max(0.0, current_stock - total_cons_D)
        growth_D = 2.0 * S_prime_D * (1.0 - S_prime_D / capacity) if capacity else 0.0
        S_next_D = min(S_prime_D + growth_D, capacity)
        if delta_immediate > 0 and S_next_D >= safety:
            return Action.D
        else:
            return Action.C

class Strategy_OPPORTUNISTIC_16(BaseStrategy):
    """
    Opportunistic Adaptive Strategy for the Common Pool Resource Game.

    - Starts with cooperation to gather information.
    - Defects in the final round (no audit, maximal immediate payoff).
    - Cooperates when the stock falls below 20\u202f% of capacity.
    - Otherwise compares expected net payoff of defecting (including expected fine)
      to cooperating and projects the resulting stock.
    - Defects only if it strictly dominates and the projected stock stays above the
      safety threshold; a recent surge of opponent defections (≥\u202f60\u202f% within the
      last three rounds) further encourages defection.
    - All decisions rely solely on public parameters, the observable opponent
      cooperation counts, and the current stock.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.low_stock = 0.2 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action (Action.C or Action.D) for the upcoming round."""
        t = history.round_number
        if t == 0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if current_stock < self.low_stock:
            return Action.C
        total_opp_coops = int(history.opponent_cooperators.sum())
        denom = (self.n - 1) * t
        p_coop = total_opp_coops / denom if denom > 0 else 0.0
        coop_pay = current_stock / (2 * self.n)
        defect_pay = current_stock / self.n - self.audit_prob * self.fine
        opp_expected_cons = p_coop * (current_stock / (2 * self.n)) + (1.0 - p_coop) * (current_stock / self.n)
        total_cons_coop = (self.n - 1) * opp_expected_cons + current_stock / (2 * self.n)
        total_cons_defect = (self.n - 1) * opp_expected_cons + current_stock / self.n

        def project_stock(consumed: float) -> float:
            s_rem = current_stock - consumed
            if s_rem < 0.0:
                s_rem = 0.0
            growth = 2.0 * s_rem * (1.0 - s_rem / self.capacity) if self.capacity != 0 else 0.0
            new_stock = s_rem + growth
            return new_stock if new_stock <= self.capacity else self.capacity
        stock_next_if_C = project_stock(total_cons_coop)
        stock_next_if_D = project_stock(total_cons_defect)
        if defect_pay > coop_pay and stock_next_if_D >= self.low_stock:
            window_start = max(0, t - 3)
            window_len = t - window_start
            recent_coops = int(history.opponent_cooperators[window_start:t].sum())
            recent_coop_ratio = recent_coops / ((self.n - 1) * window_len) if window_len > 0 else 0.0
            recent_defect_ratio = 1.0 - recent_coop_ratio
            if recent_defect_ratio > 0.6:
                return Action.D
            return Action.D
        else:
            return Action.C

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

class Strategy_SELFINTERESTED_2(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy for the Common Pool Resource game.

    At each round the agent evaluates the expected total payoff of cooperating
    (C) versus defecting (D) – accounting for the immediate payoff, the expected
    audit fine, and the impact of the chosen action on the future stock level.
    The action with the larger expected total payoff is taken.

    Edge handling:
    * If the stock is zero → cooperate (no payoff either way and avoids audit risk).
    * In the final round → defect (no audit, D strictly dominates C for any positive stock).
    * First round has no observed opponent behaviour; it assumes all opponents
      cooperated in the previous round (k = n‑1) as an optimistic seed.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        fine = self.game_description.fine
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        if t == 0:
            k = n - 1
        else:
            k = int(history.opponent_cooperators[-1])
        c_C = S / (2.0 * n)
        c_D = S / n
        C_tot = c_C * (2.0 * n - 1.0 - k)
        D_tot = c_C * (2.0 * n - k)
        S_rem_C = max(0.0, S - C_tot)
        S_rem_D = max(0.0, S - D_tot)

        def next_stock(S_rem: float) -> float:
            if capacity == 0.0:
                return 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity)
            S_next = S_rem + growth
            return min(S_next, capacity)
        S_next_C = next_stock(S_rem_C)
        S_next_D = next_stock(S_rem_D)
        pi_imm_C = c_C
        pi_imm_D = c_D - q * fine
        pi_future_C = S_next_C / (2.0 * n)
        pi_future_D = S_next_D / (2.0 * n)
        remaining = r - t - 1
        if remaining <= 0:
            future_C = 0.0
            future_D = 0.0
        else:
            lam = remaining / (remaining + 1.0)
            future_C = lam * remaining * pi_future_C
            future_D = lam * remaining * pi_future_D
        Pi_C = pi_imm_C + future_C
        Pi_D = pi_imm_D + future_D
        if Pi_D > Pi_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_3(BaseStrategy):
    """
    Self‑interested adaptive strategy for the common‑pool resource game.
    It balances immediate expected profit with a simple forecast of the
    next round’s stock, respects a safety‑stock threshold, and enforces
    a short retaliation burst when opponents defect heavily.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.theta = 0.3
        self.alpha = 0.5
        self.m = 2
        self.punish = False
        self.punish_counter = 0
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        if current_stock <= 0.0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if current_stock < self.theta * self.capacity:
            return Action.C
        if t == 0:
            k_prev = (self.n - 1) // 2
        else:
            k_prev = int(history.opponent_cooperators[-1])
        U_C = current_stock / (2.0 * self.n)
        U_D = current_stock / self.n - self.q * self.F
        opp_consume = k_prev * (current_stock / (2.0 * self.n)) + (self.n - 1 - k_prev) * (current_stock / self.n)
        cons_C = opp_consume + current_stock / (2.0 * self.n)
        S_rem_C = max(current_stock - cons_C, 0.0)
        grow_C = 2.0 * S_rem_C * (1.0 - S_rem_C / self.capacity) if self.capacity > 0 else 0.0
        S_next_C = min(S_rem_C + grow_C, self.capacity)
        cons_D = opp_consume + current_stock / self.n
        S_rem_D = max(current_stock - cons_D, 0.0)
        grow_D = 2.0 * S_rem_D * (1.0 - S_rem_D / self.capacity) if self.capacity > 0 else 0.0
        S_next_D = min(S_rem_D + grow_D, self.capacity)
        remaining_rounds = self.r - 1 - t
        future_C = remaining_rounds * (S_next_C / (2.0 * self.n))
        future_D = remaining_rounds * (S_next_D / (2.0 * self.n))
        V_C = U_C + future_C
        V_D = U_D + future_D
        if self.n > 1:
            defect_share = (self.n - 1 - k_prev) / (self.n - 1)
        else:
            defect_share = 0.0
        if defect_share > self.alpha:
            self.punish = True
            self.punish_counter = 1
            return Action.D
        if self.punish:
            if self.punish_counter < self.m:
                self.punish_counter += 1
                return Action.D
            else:
                self.punish = False
                self.punish_counter = 0
        if V_D > V_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_4(BaseStrategy):
    """
    Self‑Interest with Expected‑Future Adjustment.

    The player compares the expected total utility of cooperating versus defecting,
    taking into account the immediate payoff (including the expected audit fine),
    an estimate of opponents' defect rate derived from past observations, and
    a simple projection of stock dynamics for the remaining rounds.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        cap = float(self.game_description.capacity)
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        S = current_stock
        p_audit = q if t < r - 1 else 0.0
        payoff_C_now = S / (2.0 * n) if n else 0.0
        payoff_D_now = S / float(n) - p_audit * F
        opp_moves = t * (n - 1)
        if opp_moves == 0:
            est_def_rate = 0.5
        else:
            opp_defects = int((n - 1 - history.opponent_cooperators).sum())
            est_def_rate = opp_defects / opp_moves
        exp_opp_C = (n - 1) * (1.0 - est_def_rate)
        exp_opp_D = (n - 1) * est_def_rate
        per_coop = S / (2.0 * n)
        per_def = S / float(n)
        cons_if_C = payoff_C_now + exp_opp_C * per_coop + exp_opp_D * per_def
        cons_if_D = payoff_D_now + exp_opp_C * per_coop + exp_opp_D * per_def
        S_rem_C = max(0.0, S - cons_if_C)
        S_rem_D = max(0.0, S - cons_if_D)

        def grow(stock: float) -> float:
            grown = stock + 2.0 * stock * (1.0 - stock / cap)
            return min(grown, cap)
        S_next_C = grow(S_rem_C)
        S_next_D = grow(S_rem_D)
        remaining = r - t - 1
        if remaining > 0:
            future_C_per = S_next_C / (2.0 * n)
            future_D_per = S_next_D / float(n) - q * F
            future_C = remaining * future_C_per
            future_D = remaining * future_D_per
        else:
            future_C = future_D = 0.0
        total_C = payoff_C_now + future_C
        total_D = payoff_D_now + future_D
        if S == 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        return Action.D if total_D > total_C else Action.C

class Strategy_SELFINTERESTED_5(BaseStrategy):
    """
    Expected‑payoff greedy strategy with a stock‑preservation safeguard.
    It estimates the fraction of cooperating opponents from the previous round,
    but the decision only depends on the current stock, audit probability and fine.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_estimate = 1.0

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if history.round_number > 0:
            last_coop = int(history.opponent_cooperators[-1])
            denom = max(self.game_description.n_players - 1, 1)
            self.p_estimate = last_coop / denom
        n = self.game_description.n_players
        capacity = self.game_description.capacity
        audit_prob = self.game_description.audit_prob
        fine = self.game_description.fine
        total_rounds = self.game_description.n_rounds
        if history.round_number == total_rounds - 1:
            q = 0.0
        else:
            q = audit_prob
        S = current_stock
        E_C = S / (2 * n)
        E_D = S / n - q * fine
        theta = 0.25
        eps = 1e-06
        if S < theta * capacity and E_C >= E_D - eps:
            return Action.C
        if E_D > E_C + eps:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_6(BaseStrategy):
    """
    Conditional Self‑Interest (CSI) strategy.

    The player looks at the most recent observation of how many opponents
    cooperated, estimates their future behaviour, projects the stock for the
    next round under the two possible own actions, adds the expected audit
    fine, and chooses the action with the higher one‑step‑ahead expected
    payoff.  Special handling is included for the final round, zero or very
    low stock, and ties (randomised).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self._eps = 0.01 * game_description.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if current_stock <= 0.0 or current_stock < self._eps:
            return Action.C
        payoff_C = current_stock / (2.0 * n)
        payoff_D_nominal = current_stock / float(n)
        if t == r - 1:
            if payoff_D_nominal > payoff_C:
                return Action.D
            if payoff_D_nominal < payoff_C:
                return Action.C
            return random.choice([Action.C, Action.D])
        expected_fine = q * F
        if t == 0:
            c_prev = n - 1
        else:
            c_prev = int(history.opponent_cooperators[-1])
        c_prev = max(0, min(c_prev, n - 1))

        def _project_stock(my_coop: bool) -> float:
            if my_coop:
                coop_cnt = c_prev + 1
                defect_cnt = n - 1 - c_prev
            else:
                coop_cnt = c_prev
                defect_cnt = n - 1 - c_prev + 1
            total_cons = coop_cnt * (current_stock / (2.0 * n)) + defect_cnt * (current_stock / float(n))
            S_rem = current_stock - total_cons
            if S_rem < 0.0:
                S_rem = 0.0
            grow = 2.0 * S_rem * (1.0 - S_rem / capacity)
            next_stock = S_rem + grow
            if next_stock > capacity:
                next_stock = capacity
            return next_stock
        stock_next_C = _project_stock(my_coop=True)
        stock_next_D = _project_stock(my_coop=False)
        U_C = payoff_C
        U_D = payoff_D_nominal - expected_fine
        expected_fine_next = 0.0 if t + 1 == r - 1 else q * F
        future_U_C = stock_next_C / (2.0 * n)
        future_U_D = stock_next_D / float(n) - expected_fine_next
        E_C = U_C + future_U_C
        E_D = U_D + future_U_D
        if E_D > E_C:
            return Action.D
        if E_D < E_C:
            return Action.C
        return random.choice([Action.C, Action.D])

class Strategy_SELFINTERESTED_7(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy.
    Chooses between cooperation (C) and defection (D) by comparing the
    immediate expected payoff plus a linear estimate of future payoff,
    while respecting a safety threshold on the projected stock level.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.safety_threshold = 0.05 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if t == 0:
            k = self.n - 1
        else:
            k = int(history.opponent_cooperators[-1])
        pay_C = S / (2.0 * self.n)
        if t == self.r - 1:
            pay_D = S / self.n
        else:
            pay_D = S / self.n - self.audit_prob * self.fine
        cons_C = (k + 1) * (S / (2.0 * self.n)) + (self.n - 1 - k) * (S / self.n)
        S_rem_C = max(0.0, S - cons_C)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / self.capacity)
        S_next_C = min(S_rem_C + growth_C, self.capacity)
        cons_D = k * (S / (2.0 * self.n)) + (self.n - k) * (S / self.n)
        S_rem_D = max(0.0, S - cons_D)
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / self.capacity)
        S_next_D = min(S_rem_D + growth_D, self.capacity)
        remaining_rounds = self.r - t - 1
        future_factor = 1.0 / (2.0 * self.n)
        future_C = remaining_rounds * future_factor * S_next_C
        future_D = remaining_rounds * future_factor * S_next_D
        total_C = pay_C + future_C
        total_D = pay_D + future_D
        if total_D > total_C and S_next_D >= self.safety_threshold:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_8(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy (SAAS).

    The strategy compares the expected immediate payoff of defecting
    (including an expected fine) with cooperating, checks a safety
    stock threshold for the future, and adjusts the audit risk
    temporarily after two consecutive audited defections.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.base_q = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n
        capacity = self.capacity
        F = self.fine

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / capacity) if x > 0 else 0.0
        if t >= 2:
            recent = [(bool(history.my_actions[t - 2]), bool(history.my_audited[t - 2])), (bool(history.my_actions[t - 1]), bool(history.my_audited[t - 1]))]
            if all((not act and aud for act, aud in recent)):
                q_eff = min(1.0, self.base_q * 1.5)
            else:
                q_eff = self.base_q
        else:
            q_eff = self.base_q
        if t == self.r - 1:
            u_c = current_stock / (2.0 * n) if current_stock > 0 else 0.0
            u_d = current_stock / n if current_stock > 0 else 0.0
            return Action.D if u_d > u_c else Action.C
        u_c = current_stock / (2.0 * n) if current_stock > 0 else 0.0
        u_d = current_stock / n - q_eff * F if current_stock > 0 else 0.0
        if t == 0:
            opp_coop = (n - 1) // 2
        else:
            opp_coop = int(history.opponent_cooperators[-1])
        opp_def = n - 1 - opp_coop
        delta_c = current_stock / (2.0 * n) if current_stock > 0 else 0.0
        delta_d = current_stock / n if current_stock > 0 else 0.0
        opp_consumption = opp_coop * (current_stock / (2.0 * n)) + opp_def * (current_stock / n)
        cons_c = delta_c + opp_consumption
        cons_d = delta_d + opp_consumption
        s_rem_c = current_stock - cons_c
        s_rem_d = current_stock - cons_d
        s_next_c = min(s_rem_c + grow(s_rem_c), capacity)
        s_next_d = min(s_rem_d + grow(s_rem_d), capacity)
        tau = (self.r - t - 1) / self.r
        s_min = tau * capacity
        if u_d <= u_c:
            return Action.C
        if s_next_d < s_min:
            return Action.C
        remaining_rounds = self.r - t - 1
        diff_future = (s_next_d - s_next_c) / (2.0 * n) * remaining_rounds
        if diff_future + (u_d - u_c) > 0.0:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_9(BaseStrategy):
    """
    Self‑Interested Adaptive (SIA‑A) strategy.
    Considers expected audit fine, current stock, and observed opponent cooperation.
    Exploits the last round, otherwise decides by comparing immediate net gain of defecting
    with the expected payoff of cooperating and a one‑step stock projection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self.expected_fine = self.q * self.F

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Return the action for the current round."""
        t = history.round_number
        n = self.n
        capacity = self.capacity
        q = self.q
        F = self.F
        expected_fine = self.expected_fine

        def payoff_C(S: float) -> float:
            return S / (2.0 * n) if S > 0 else 0.0

        def payoff_D(S: float) -> float:
            return S / float(n) if S > 0 else 0.0

        def net_gain_D(S: float) -> float:
            """Expected payoff of defecting this round (after expected fine)."""
            return payoff_D(S) - expected_fine

        def next_stock(S: float, consumption: float) -> float:
            """Compute stock for the following round after consumption and growth."""
            remaining = S - consumption
            if remaining < 0.0:
                remaining = 0.0
            growth = 2.0 * remaining * (1.0 - remaining / capacity) if capacity > 0 else 0.0
            new_stock = remaining + growth
            if new_stock > capacity:
                new_stock = capacity
            return new_stock
        if current_stock <= 0.0:
            return Action.C
        if t == self.game_description.n_rounds - 1:
            return Action.D
        if t == 0:
            if net_gain_D(current_stock) > payoff_C(current_stock):
                return Action.D
            else:
                return Action.C
        k_C = int(history.opponent_cooperators[-1])
        opponents = n - 1
        p_coop = k_C / opponents if opponents > 0 else 0.0
        if net_gain_D(current_stock) <= payoff_C(current_stock):
            return Action.C
        opp_expected_consumption = p_coop * payoff_C(current_stock) + (1.0 - p_coop) * payoff_D(current_stock)
        total_opp_consumption = opponents * opp_expected_consumption
        my_cons_C = payoff_C(current_stock)
        my_cons_D = payoff_D(current_stock)
        total_cons_C = my_cons_C + total_opp_consumption
        total_cons_D = my_cons_D + total_opp_consumption
        S_next_C = next_stock(current_stock, total_cons_C)
        S_next_D = next_stock(current_stock, total_cons_D)

        def expected_next_payoff(S_next: float) -> float:
            return payoff_C(S_next) * p_coop + payoff_D(S_next) * (1.0 - p_coop)
        val_C = payoff_C(current_stock) + expected_next_payoff(S_next_C)
        val_D = net_gain_D(current_stock) + expected_next_payoff(S_next_D)
        if val_D > val_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_11(BaseStrategy):
    """
    Adaptive Expected‑Benefit Defector (AED).

    Estimates opponents' cooperation probability with an exponential
    moving average (α = 0.7) and compares the expected immediate gain from
    defecting (including expected fine) against the expected future‑payoff
    loss caused by the extra consumption.  Defects when the net benefit Δ
    is positive; otherwise cooperates.  Handles edge cases such as the first
    round, the final round, zero stock, and missing audit risk.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.p_estimate = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if t == 0:
            p_hat = 0.5
        else:
            alpha = 0.7
            coop_last = int(history.opponent_cooperators[-1])
            p_obs = coop_last / (n - 1) if n > 1 else 0.0
            p_hat = alpha * self.p_estimate + (1.0 - alpha) * p_obs
        self.p_estimate = p_hat
        if current_stock <= 0:
            return Action.C
        if t == r - 1:
            return Action.D
        R = r - t
        E_C = current_stock / (2.0 * n)
        E_D = current_stock / n - q * F
        delta_v = R * current_stock / (4.0 * n * n)
        delta = E_D - E_C - delta_v
        return Action.D if delta > 0 else Action.C

class Strategy_SELFINTERESTED_12(BaseStrategy):
    """
    Expected‑Utility‑Defection (EUD)

    The agent estimates the probability p̂ that a random opponent will cooperate,
    updates it each round from the observed number of cooperating opponents,
    and compares the expected total (immediate + future) utility of cooperating
    versus defecting.  Audits are accounted for via the expected fine q·F.
    The agent always defects in the final round (no audit, no future stock).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        S = float(current_stock)
        n = self.n
        capacity = self.capacity
        q = self.q
        F = self.F
        if t == self.r - 1:
            return Action.D
        if S <= 0.0:
            return Action.C
        if t == 0:
            p_hat = 0.5
        else:
            opp_coop_last = int(history.opponent_cooperators[-1])
            p_hat = opp_coop_last / (n - 1) if n > 1 else 0.0
        coop_now = S / (2.0 * n)
        defect_now = S / n - q * F
        cons_self_C = coop_now
        cons_self_D = S / n
        opp_cons_C = S / (2.0 * n) * p_hat
        opp_cons_D = S / n * (1.0 - p_hat)
        opp_expected = (n - 1) * (opp_cons_C + opp_cons_D)
        total_C = cons_self_C + opp_expected
        total_D = cons_self_D + opp_expected

        def next_stock(S_curr: float, total_cons: float) -> float:
            S_rem = S_curr - total_cons
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity)
            return min(S_rem + growth, capacity)
        S_next_C = next_stock(S, total_C)
        S_next_D = next_stock(S, total_D)
        future_rounds = self.r - t - 1
        if future_rounds <= 0:
            future_C = future_D = 0.0
        else:

            def exp_pay_per_round(stock: float) -> float:
                coop_part = p_hat * (stock / (2.0 * n))
                defect_part = (1.0 - p_hat) * (stock / n - q * F)
                return coop_part + defect_part
            future_C = future_rounds * exp_pay_per_round(S_next_C)
            future_D = future_rounds * exp_pay_per_round(S_next_D)
        value_C = coop_now + future_C
        value_D = defect_now + future_D
        if value_D > value_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFINTERESTED_13(BaseStrategy):
    """
    Self‑interested Adaptive Strategy (SA‑AI).

    The player maintains a Laplace‑smoothed estimate `p` of the probability that a
    random opponent cooperates, based on all observed opponent cooperation counts.
    At each round it evaluates the expected total utility of cooperating vs.
    defecting: immediate payoff (including the expected fine) plus a one‑step
    look‑ahead continuation value that incorporates the expected stock evolution.
    The action with the higher expected utility is chosen.  If the stock is
    depleted, the player cooperates to avoid the expected fine.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        remain = r - t
        if t == 0:
            p = 0.5
        else:
            total_obs = t * (n - 1)
            total_coop = int(np.sum(history.opponent_cooperators))
            p = (total_coop + 1) / (total_obs + 2)
        if current_stock <= 0.0:
            return Action.C
        S = current_stock
        U_C = S / (2 * n)
        U_D = S / n - q * F

        def stock_update(S_curr: float, total_consumption: float) -> float:
            S_rem = S_curr - total_consumption
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / capacity)
            S_new = S_rem + growth
            if S_new > capacity:
                S_new = capacity
            return S_new
        opp_cons_expect = p * (S / (2 * n)) + (1 - p) * (S / n)
        opp_total_expect = (n - 1) * opp_cons_expect
        cons_self_C = S / (2 * n)
        cons_self_D = S / n
        S_next_C = stock_update(S, opp_total_expect + cons_self_C)
        S_next_D = stock_update(S, opp_total_expect + cons_self_D)
        if remain <= 1:
            V_C = V_D = 0.0
        else:
            U_C_next = S_next_C / (2 * n)
            U_D_next = S_next_C / n - q * F
            V_C = (remain - 1) * (p * U_C_next + (1 - p) * U_D_next)
            U_C_next2 = S_next_D / (2 * n)
            U_D_next2 = S_next_D / n - q * F
            V_D = (remain - 1) * (p * U_C_next2 + (1 - p) * U_D_next2)
        if U_C + V_C >= U_D + V_D:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_14(BaseStrategy):
    """
    Dynamic Expected‑Return Trigger (DERT).

    The agent cooperates in the first round, defects in the last round,
    and in intermediate rounds compares the expected one‑step payoff
    (including expected audit fine) plus a one‑step‑ahead continuation
    value.  Cooperation is kept when this total is not worse than defecting
    and the stock is still above 10\u202f% of capacity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)
        self._default_prev_opp_coop = self.n - 1

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        n, r, C, q, F = (self.n, self.r, self.capacity, self.q, self.F)
        t = history.round_number
        if current_stock <= 0.0:
            return Action.D
        if t == r - 1:
            return Action.D
        if t == 0:
            return Action.C
        opp_coop_last = history.opponent_cooperators[-1] if history.opponent_cooperators.size > 0 else self._default_prev_opp_coop
        c_est = opp_coop_last / (n - 1) if n > 1 else 0.0
        pay_C = current_stock / (2.0 * n)
        pay_D = current_stock / n - q * F
        if pay_D < 0.0:
            return Action.C
        exp_def_opponents = (1.0 - c_est) * (n - 1)
        exp_total_cons_if_C = pay_C + exp_def_opponents * pay_D
        S_rem_C = current_stock - exp_total_cons_if_C
        S_rem_C = max(S_rem_C, 0.0)
        growth_C = 2.0 * S_rem_C * (1.0 - S_rem_C / C) if S_rem_C > 0.0 else 0.0
        stock_if_C = min(S_rem_C + growth_C, C)
        exp_total_cons_if_D = pay_D + exp_def_opponents * pay_D
        S_rem_D = current_stock - exp_total_cons_if_D
        S_rem_D = max(S_rem_D, 0.0)
        growth_D = 2.0 * S_rem_D * (1.0 - S_rem_D / C) if S_rem_D > 0.0 else 0.0
        stock_if_D = min(S_rem_D + growth_D, C)
        cont_if_C = stock_if_C / (2.0 * n) if stock_if_C > 1e-08 else 0.0
        cont_if_D = stock_if_D / n - q * F if stock_if_D > 1e-08 else 0.0
        tot_C = pay_C + cont_if_C
        tot_D = pay_D + cont_if_D
        if tot_C >= tot_D and current_stock > 0.1 * C:
            return Action.C
        else:
            return Action.D

class Strategy_SELFINTERESTED_15(BaseStrategy):
    """
    Self‑Interested Adaptive Strategy (SI‑Adapt) for the Common Pool Resource game.
    The agent compares the immediate expected payoff of cooperating vs defecting,
    adds a future‑value estimate based on the projected stock after the current
    action, and chooses the action with the higher total expected value.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        pay_C = S / (2.0 * self.n)
        pay_D = S / self.n - self.q * self.F
        if t == self.r - 1:
            return Action.D if pay_D >= pay_C else Action.C
        if t == 0:
            return Action.D if pay_D > pay_C else Action.C
        cntC_prev = int(history.opponent_cooperators[-1])
        pC = cntC_prev / (self.n - 1) if self.n > 1 else 0.0
        cons_self_C = S / (2.0 * self.n)
        cons_self_D = S / self.n
        opp_cons_per = pC * S / (2.0 * self.n) + (1.0 - pC) * S / self.n
        cons_opp = (self.n - 1) * opp_cons_per
        stock_if_C = S - (cons_self_C + cons_opp)
        stock_if_D = S - (cons_self_D + cons_opp)

        def grow(stock: float) -> float:
            if stock <= 0.0:
                return 0.0
            growth = 2.0 * stock * (1.0 - stock / self.capacity)
            new_stock = stock + growth
            return new_stock if new_stock <= self.capacity else self.capacity
        S_next_C = grow(stock_if_C)
        S_next_D = grow(stock_if_D)
        R = self.r - t - 1
        future_val_C = R * (S_next_C / (2.0 * self.n))
        future_val_D = R * (S_next_D / (2.0 * self.n))
        total_C = pay_C + future_val_C
        total_D = pay_D + future_val_D
        return Action.D if total_D > total_C else Action.C

class Strategy_SELFINTERESTED_16(BaseStrategy):
    """
    Expected‑payoff + future‑stock penalty strategy for the Common Pool Resource game.

    The agent compares the immediate expected payoff of cooperating (U_C) with that of
    defecting (U_D) reduced by a penalty that reflects the value of preserving the
    stock when a projected post‑defection stock would fall below a safety threshold.
    The projection uses the opponent cooperation rate observed in the previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta = 0.3
        self.prior_coop = 0.5

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = float(self.game_description.capacity)
        q = float(self.game_description.audit_prob)
        F = float(self.game_description.fine)
        t = history.round_number
        S = current_stock
        if t == r - 1:
            return Action.D
        if S <= 0.0:
            return Action.C
        if t == 0:
            p = self.prior_coop
        else:
            last_k = int(history.opponent_cooperators[-1])
            p = last_k / (n - 1) if n - 1 > 0 else 0.0
        Uc = S / (2 * n)
        Ud = S / n - q * F
        coop_share = S / (2 * n)
        defect_share = S / n
        exp_cons_per_opp = p * coop_share + (1 - p) * defect_share
        opp_total_exp = (n - 1) * exp_cons_per_opp
        total_consumption_D = opp_total_exp + defect_share
        S_rem_D = max(0.0, S - total_consumption_D)
        growth_D = 2 * S_rem_D * (1 - S_rem_D / capacity)
        S_next_D = min(S_rem_D + growth_D, capacity)
        P = 0.0
        if t < r - 1 and S_next_D < self.theta * capacity:
            remaining_rounds = r - t - 1
            lam = capacity / remaining_rounds if remaining_rounds > 0 else 0.0
            P = lam * (self.theta * capacity - S_next_D)
        if Uc >= Ud - P:
            return Action.C
        else:
            return Action.D

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

class Strategy_SELFISH_3(BaseStrategy):
    """
    Selfish Adaptive Defection‑Cooperation rule.

    At each round the player compares the expected immediate net gain of
    defecting (including the expected fine) with an estimate of the future
    loss caused by the extra consumption.  The decision logic is:

        * If the stock is zero → cooperate (nothing to gain).
        * Compute G = S/(2n) - q·F.
            – If G ≤ 0 → cooperate (defection not worth the fine).
        * Compute an estimated future loss:
            future_loss_est = (S * R) / (2 n²),
            where R is the number of rounds remaining after the current one.
        * Defect if G > future_loss_est, otherwise cooperate.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        R = self.r - t - 1
        if current_stock <= 0.0:
            return Action.C
        G = current_stock / (2.0 * self.n) - self.audit_prob * self.fine
        if G <= 0.0:
            return Action.C
        future_loss_est = current_stock * R / (2.0 * self.n * self.n)
        if G > future_loss_est:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_4(BaseStrategy):
    """
    Selfish Adaptive Strategy – Expected‑Gain Threshold.
    Chooses to defect only when the expected net gain (including audit risk)
    exceeds the cooperative payoff plus a dynamic safety margin that depends on
    remaining rounds and current stock scarcity.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        payoff_C = current_stock / (2.0 * self.n)
        payoff_D_no_fine = current_stock / float(self.n)
        if t == self.r - 1:
            expected_fine = 0.0
        else:
            expected_fine = self.audit_prob * self.fine
        expected_gain_D = payoff_D_no_fine - expected_fine
        rounds_left = self.r - t
        if self.capacity == 0:
            stock_factor = 0.0
        else:
            stock_factor = 1.0 - current_stock / self.capacity
        horizon_factor = rounds_left / float(self.r)
        raw_gain = payoff_D_no_fine - payoff_C
        safety_margin = raw_gain * (stock_factor * horizon_factor)
        if expected_gain_D > payoff_C + safety_margin:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_5(BaseStrategy):
    """
    Selfish Adaptive Strategy.

    The player compares the immediate expected payoff of defecting versus cooperating,
    subtracts an estimate of future loss caused by the extra consumption when defecting,
    and chooses the action with the higher net advantage. The estimate of future loss is
    scaled by the observed proportion of cooperating opponents in the previous round,
    making the strategy more tolerant when others cooperate and more aggressive when
    they mostly defect.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        S = current_stock
        if S <= 0:
            return Action.C
        if t == r - 1:
            return Action.D
        E_C = S / (2 * n) if n else 0.0
        E_D = S / n - q * F if n else 0.0
        delta_cons = S / (2 * n) if n else 0.0
        R = r - t - 1
        if t > 0:
            k = int(history.opponent_cooperators[-1])
            coop_ratio = k / (n - 1) if n > 1 else 0.0
        else:
            coop_ratio = 0.5
        delta_future = delta_cons * (R / (R + 1)) * (1.0 - coop_ratio)
        adv = E_D - E_C - delta_future
        return Action.D if adv > 0 else Action.C

class Strategy_SELFISH_6(BaseStrategy):
    """
    Selfish Adaptive Defect‑When‑Profitable (SADWP).

    The player compares the expected immediate benefit of defecting
    (including the expected audit fine) with the effect that the
    choice has on the stock for the next round.  It defects when the
    sum of these two components is positive, otherwise it cooperates.
    The rule is applied selfishly and uses only the observable stock,
    the game parameters and the number of cooperating opponents in the
    previous round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.game_description.n_players
        r = self.game_description.n_rounds
        capacity = self.game_description.capacity
        q = self.game_description.audit_prob
        F = self.game_description.fine
        t = history.round_number
        if current_stock <= 0:
            return Action.C
        if t == r - 1:
            return Action.D
        if t == 0:
            p_coop_prev = 1.0
        else:
            c_obs = int(history.opponent_cooperators[-1])
            p_coop_prev = c_obs / (n - 1) if n > 1 else 0.0
        payoff_C = current_stock / (2 * n)
        expected_fine = q * F
        payoff_D = current_stock / n - expected_fine
        delta_immediate = payoff_D - payoff_C
        coop_cons = current_stock / (2 * n)
        def_cons = current_stock / n
        exp_cons_opp = (n - 1) * (p_coop_prev * coop_cons + (1.0 - p_coop_prev) * def_cons)
        total_cons_C = exp_cons_opp + coop_cons
        total_cons_D = exp_cons_opp + def_cons
        S_rem_C = current_stock - total_cons_C
        if S_rem_C < 0.0:
            S_rem_C = 0.0
        S_rem_D = current_stock - total_cons_D
        if S_rem_D < 0.0:
            S_rem_D = 0.0

        def grow(x: float) -> float:
            return 2.0 * x * (1.0 - x / capacity)
        S_next_C = S_rem_C + grow(S_rem_C)
        if S_next_C > capacity:
            S_next_C = capacity
        S_next_D = S_rem_D + grow(S_rem_D)
        if S_next_D > capacity:
            S_next_D = capacity
        marginal = 1.0 / (2 * n)
        delta_future = marginal * (S_next_D - S_next_C)
        delta_total = delta_immediate + delta_future
        if delta_total > 0.0:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_7(BaseStrategy):
    """
    Selfish Adaptive Defection/Cooperation (SADC).

    The agent estimates opponent cooperation from past observations,
    evaluates the expected immediate payoff of cooperating vs defecting
    (including the expected fine), projects the stock evolution for the
    next round, and adds a linear approximation of future payoffs
    assuming the same action is repeated.  The action with the higher
    total expected payoff is chosen; ties are broken in favour of
    defection.  Special handling is applied for the last round and for
    zero stock.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        S = max(0.0, float(current_stock))
        if t == self.r - 1:
            return Action.D
        if S == 0.0:
            return Action.C
        R = self.r - t
        if t == 0:
            p = 0.5
        else:
            total_coops = int(np.sum(history.opponent_cooperators[:t]))
            p = total_coops / ((self.n - 1) * t)
        payoff_C = S / (2 * self.n)
        payoff_D = S / self.n - self.q * self.F
        own_cons_C = payoff_C
        own_cons_D = payoff_D + self.q * self.F
        opp_cons = p * (S / (2 * self.n)) + (1 - p) * (S / self.n)
        total_cons_C = own_cons_C + (self.n - 1) * opp_cons
        total_cons_D = own_cons_D + (self.n - 1) * opp_cons

        def project_stock(S_cur: float, total_cons: float) -> float:
            S_rem = S_cur - total_cons
            if S_rem < 0.0:
                S_rem = 0.0
            growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
            new_stock = S_rem + growth
            if new_stock > self.capacity:
                new_stock = self.capacity
            return new_stock
        S_next_C = project_stock(S, total_cons_C)
        S_next_D = project_stock(S, total_cons_D)
        if R > 1:
            future_per_round_C = S_next_C / (2 * self.n)
            future_per_round_D = S_next_D / self.n - self.q * self.F
            future_C = (R - 1) * future_per_round_C
            future_D = (R - 1) * future_per_round_D
        else:
            future_C = 0.0
            future_D = 0.0
        total_C = payoff_C + future_C
        total_D = payoff_D + future_D
        if total_D > total_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_8(BaseStrategy):
    """
    Selfish Adaptive Defection‑Cooperation (SADC) strategy.

    The agent evaluates immediate expected payoff (including expected fine)
    together with a simple myopic estimate of future payoff that depends on
    the current stock, the remaining number of rounds and the observed
    cooperation level of opponents in the previous round.
    Cooperation is chosen only when the projected total utility from cooperating
    (immediate + future) is at least as large as that from defecting.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        L = self.r - t
        if current_stock <= 0:
            return Action.C
        if t == self.r - 1:
            return Action.D if current_stock > 0 else Action.C
        if t == 0:
            p_prev = 0.5
        else:
            coop_cnt = int(history.opponent_cooperators[-1])
            opponents = self.n - 1
            p_prev = coop_cnt / opponents if opponents > 0 else 0.0
        pi_C = current_stock / (2 * self.n)
        pi_D = current_stock / self.n - self.q * self.F
        cons_coop = current_stock / (2 * self.n)
        cons_def = current_stock / self.n
        total_cons_C = pi_C + (self.n - 1) * (p_prev * cons_coop + (1 - p_prev) * cons_def)
        total_cons_D = pi_D + (self.n - 1) * (p_prev * cons_coop + (1 - p_prev) * cons_def)
        S_after_C = max(current_stock - total_cons_C, 0.0)
        S_after_D = max(current_stock - total_cons_D, 0.0)
        growth_C = 2 * S_after_C * (1 - S_after_C / self.capacity) if self.capacity != 0 else 0.0
        growth_D = 2 * S_after_D * (1 - S_after_D / self.capacity) if self.capacity != 0 else 0.0
        S_next_C = min(S_after_C + growth_C, self.capacity)
        S_next_D = min(S_after_D + growth_D, self.capacity)
        future_factor = (L - 1) / L if L > 0 else 0.0
        remaining = L - 1
        est_future_C = future_factor * (S_next_C / (2 * self.n)) * remaining
        est_future_D = future_factor * (S_next_D / self.n) * remaining
        U_C = pi_C + est_future_C
        U_D = pi_D + est_future_D
        return Action.C if U_C >= U_D else Action.D

class Strategy_SELFISH_9(BaseStrategy):
    """
    Expected‑Return Adaptive (ERA) strategy.
    The player computes the net advantage of defecting versus cooperating,
    taking into account the immediate extra payoff, the expected fine,
    and the projected impact on future stock given the observed
    cooperation level of opponents.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = float(game_description.n_players)
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n_players = self.game_description.n_players
        n_others = n_players - 1
        t = history.round_number
        r = self.game_description.n_rounds
        if current_stock <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        if t == 0:
            p_hat = 0.5
        else:
            p_hat = history.opponent_cooperators[-1] / n_others
        S = float(current_stock)
        delta_immediate = S / self.n - S / (2.0 * self.n)
        delta_fine = self.audit_prob * self.fine

        def projected_stock(my_cons: float) -> float:
            opp_cons = p_hat * (S / (2.0 * self.n)) + (1.0 - p_hat) * (S / self.n)
            total_cons = my_cons + n_others * opp_cons
            S_next = max(0.0, S - total_cons)
            growth = 2.0 * S_next * (1.0 - S_next / self.capacity)
            return min(S_next + growth, self.capacity)
        cons_C = S / (2.0 * self.n)
        cons_D = S / self.n
        S_proj_C = projected_stock(cons_C)
        S_proj_D = projected_stock(cons_D)
        T = r - t - 1

        def future_payoff_given_stock(S_future: float) -> float:
            payoff_C = S_future / (2.0 * self.n)
            payoff_D = S_future / self.n
            return p_hat * payoff_C + (1.0 - p_hat) * payoff_D
        exp_future_C = T * future_payoff_given_stock(S_proj_C)
        exp_future_D = T * future_payoff_given_stock(S_proj_D)
        delta_future = exp_future_D - exp_future_C
        net_advantage = delta_immediate - delta_fine + delta_future
        if S_proj_D <= 0.0 and S_proj_C > 0.0:
            return Action.C
        return Action.D if net_advantage > 0.0 else Action.C

class Strategy_SELFISH_10(BaseStrategy):
    """
    Selfish Adaptive Strategy for the Common Pool Resource Game.

    The agent balances immediate payoff (including expected audit fines) with a
    linear‑discounted estimate of the future stock.  It adapts to the observed
    cooperation rate of opponents over the last up to three rounds and
    incorporates the audit probability and fine when evaluating defection.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)
        self.n_opp = self.n - 1

    def _coop_rate_estimate(self, history: PlayerHistory) -> float:
        """
        Estimate the opponents' cooperation rate using the last up to three rounds.
        Returns a value in [0, 1].
        """
        t = history.round_number
        if t == 0 or self.n_opp == 0:
            return 0.0
        m = min(3, t)
        recent = history.opponent_cooperators[t - m:t]
        rates = recent / self.n_opp
        return float(np.mean(rates))

    def _forecast_stock(self, S: float, action_is_coop: bool, coop_rate: float) -> float:
        """
        Forecast the stock for the next round given the current stock `S`,
        the agent's contemplated action and the estimated opponent cooperation rate.
        """
        cons_self = S / (2 * self.n) if action_is_coop else S / self.n
        opp_cons_per = coop_rate * (S / (2 * self.n)) + (1.0 - coop_rate) * (S / self.n)
        total_cons = self.n_opp * opp_cons_per + cons_self
        S_rem = max(0.0, S - total_cons)
        growth = 2.0 * S_rem * (1.0 - S_rem / self.capacity)
        return min(S_rem + growth, self.capacity)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """
        Decide on an action for the current round.
        """
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        if t == self.r - 1:
            return Action.D
        if t == 0:
            pi_C = S / (2 * self.n)
            pi_D = S / self.n - self.audit_prob * self.fine
            delta = 0.5 * pi_C
            return Action.D if pi_D > pi_C + delta else Action.C
        coop_rate = self._coop_rate_estimate(history)
        pi_C = S / (2 * self.n)
        pi_D = S / self.n - self.audit_prob * self.fine
        S_hat_C = self._forecast_stock(S, action_is_coop=True, coop_rate=coop_rate)
        S_hat_D = self._forecast_stock(S, action_is_coop=False, coop_rate=coop_rate)
        alpha = (self.r - t - 1) / (self.r - t)
        U_C = pi_C + alpha * S_hat_C
        U_D = pi_D + alpha * S_hat_D
        return Action.C if U_C >= U_D else Action.D

class Strategy_SELFISH_11(BaseStrategy):
    """
    Selfish Adaptive Strategy – “Expected‑Payoff Threshold with Stock‑Guard”.
    Cooperates when the immediate expected payoff of cooperating is at least as good as defection,
    or when defection is better only if the stock is healthy (≥\u202fθ) and opponents have been
    sufficiently cooperative (≥\u202fτ). Defects in the last round.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.theta = 0.6 * float(game_description.capacity)
        self.tau = 0.7

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.game_description.n_players
        if n == 0:
            return Action.D
        U_C = current_stock / (2.0 * n)
        U_D = current_stock / n - self.game_description.audit_prob * self.game_description.fine
        if t == self.game_description.n_rounds - 1:
            return Action.D
        if t == 0:
            p_coop = 1.0
        else:
            coop_count = int(history.opponent_cooperators[-1])
            denom = n - 1
            p_coop = coop_count / denom if denom > 0 else 1.0
        if U_D <= U_C:
            return Action.C
        elif current_stock >= self.theta and p_coop >= self.tau:
            return Action.C
        else:
            return Action.D

class Strategy_SELFISH_12(BaseStrategy):
    """
    Selfish Adaptive Strategy – “Profit‑First with Stock‑Safety”.

    The agent estimates the opponents' cooperation probability from past
    observations, evaluates the expected immediate payoff of cooperating
    versus defecting (including the expected fine), projects the resulting
    stock for the next round, and adds a one‑step look‑ahead value based on
    the same rule applied to future rounds.  It defects when the projected
    total value of defecting exceeds that of cooperating, otherwise it
    cooperates.  In the final round it always defects; if the stock is zero
    it cooperates (harmless).
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = game_description.audit_prob
        self.F = game_description.fine

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        t = history.round_number
        n = self.n
        capacity = self.capacity
        q = self.q
        F = self.F
        if t == self.r - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        if t == 0:
            p = 0.5
        else:
            total_coop = int(history.opponent_cooperators[:t].sum())
            denom = t * (n - 1)
            p = total_coop / denom if denom > 0 else 0.5
        pay_C = current_stock / (2 * n)
        pay_D = current_stock / n - q * F
        cons_self_C = current_stock / (2 * n)
        cons_self_D = current_stock / n
        opp_cons = (n - 1) * (p * current_stock / (2 * n) + (1 - p) * current_stock / n)
        total_cons_C = cons_self_C + opp_cons
        S_rem_C = max(0.0, current_stock - total_cons_C)
        grow_C = 2 * S_rem_C * (1 - S_rem_C / capacity) if capacity != 0 else 0.0
        S_next_C = min(S_rem_C + grow_C, capacity)
        total_cons_D = cons_self_D + opp_cons
        S_rem_D = max(0.0, current_stock - total_cons_D)
        grow_D = 2 * S_rem_D * (1 - S_rem_D / capacity) if capacity != 0 else 0.0
        S_next_D = min(S_rem_D + grow_D, capacity)

        def future_value(stock: float) -> float:
            v_c = stock / (2 * n)
            v_d = stock / n - q * F
            return v_c if v_c >= v_d else v_d
        R = self.r - t
        V_C = pay_C + (R - 1) * future_value(S_next_C)
        V_D = pay_D + (R - 1) * future_value(S_next_D)
        return Action.D if V_D > V_C else Action.C

class Strategy_SELFISH_13(BaseStrategy):
    """
    Selfish Adaptive Strategy (SA‑SCP) for the Common Pool Resource game.
    The player compares immediate expected payoffs (including expected fines),
    forecasts the impact of its action on the future stock, and selects the
    action with the higher total expected return over the remaining rounds.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.q = float(game_description.audit_prob)
        self.F = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        n = self.n
        r = self.r
        capacity = self.capacity
        q = self.q
        F = self.F
        t = history.round_number
        S = float(current_stock)
        if S <= 0.0:
            return Action.C
        if t == r - 1:
            return Action.D
        E_C = S / (2.0 * n)
        E_D = S / n - q * F
        if E_D <= E_C:
            return Action.C
        if t == 0:
            p_hat = 0.5
        else:
            p_hat = history.opponent_cooperators[-1] / (n - 1)
        opp_cons_per = p_hat * (S / (2.0 * n)) + (1.0 - p_hat) * (S / n)
        total_opp_cons = (n - 1) * opp_cons_per
        cons_C = S / (2.0 * n)
        cons_D = S / n
        S_rem_C = S - (cons_C + total_opp_cons)
        S_rem_D = S - (cons_D + total_opp_cons)
        if S_rem_C < 0.0:
            S_rem_C = 0.0
        if S_rem_D < 0.0:
            S_rem_D = 0.0

        def grow(x: float) -> float:
            new_stock = x + 2.0 * x * (1.0 - x / capacity)
            return min(new_stock, capacity)
        S_next_C = grow(S_rem_C)
        S_next_D = grow(S_rem_D)
        remaining = r - t - 1
        future_C = remaining * (S_next_C / (2.0 * n))
        future_D = remaining * (S_next_D / n - q * F)
        total_E_C = E_C + future_C
        total_E_D = E_D + future_D
        if total_E_D > total_E_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_14(BaseStrategy):
    """
    Selfish Adaptive Strategy – “Risk‑Adjusted Defection”.

    The player compares the immediate expected payoff of cooperating
    versus defecting (including expected audit fines) and adds a simple
    estimate of the next‑round payoff that results from the induced stock
    level.  It defects in the final round, otherwise it chooses the action
    with the larger sum of immediate and continuation values, breaking ties
    in favour of cooperation.  The observed cooperation fraction of opponents
    from the previous round is used to forecast future consumption.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        gd = self.game_description
        n = gd.n_players
        capacity = gd.capacity
        q = gd.audit_prob
        F = gd.fine
        r = gd.n_rounds
        t = history.round_number
        if t == 0:
            p = 1.0
        else:
            opp_coops = int(history.opponent_cooperators[-1])
            opponents = n - 1
            p = opp_coops / opponents if opponents else 1.0
        if t == r - 1:
            return Action.D
        if current_stock <= 0.0:
            return Action.C
        coop_pay = current_stock / (2 * n)
        def_pay = current_stock / n
        exp_fine = q * F if t < r - 1 else 0.0
        E_C = coop_pay
        E_D = def_pay - exp_fine
        opp_coop_frac = p
        opp_def_frac = 1.0 - p
        opp_cons_per = opp_coop_frac * coop_pay + opp_def_frac * def_pay
        opp_total_cons = opp_cons_per * (n - 1)
        S_rem_C = current_stock - (opp_total_cons + coop_pay)
        S_rem_C = max(S_rem_C, 0.0)
        growth_C = 2 * S_rem_C * (1 - S_rem_C / capacity) if capacity else 0.0
        S_next_C = min(S_rem_C + growth_C, capacity)
        S_rem_D = current_stock - (opp_total_cons + def_pay)
        S_rem_D = max(S_rem_D, 0.0)
        growth_D = 2 * S_rem_D * (1 - S_rem_D / capacity) if capacity else 0.0
        S_next_D = min(S_rem_D + growth_D, capacity)
        V_C = S_next_C / (2 * n) if S_next_C > 0 else 0.0
        V_D = S_next_D / (2 * n) if S_next_D > 0 else 0.0
        if E_D + V_D > E_C + V_C:
            return Action.D
        else:
            return Action.C

class Strategy_SELFISH_15(BaseStrategy):
    """Selfish Adaptive Strategy for the Common Pool Resource Game.

    The agent defects when the immediate expected gain (after accounting for the
    expected fine) is positive, but only if doing so does not push the future
    stock below a safety margin. In the final round the agent always defects.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.game_description = game_description
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = game_description.audit_prob
        self.fine = game_description.fine
        self.safety_margin = 0.4 * self.capacity

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        """Decide the action for the current round."""
        t = history.round_number
        S = float(current_stock)
        if self.n == 0:
            coop_share = 0.0
            def_share = 0.0
        else:
            coop_share = S / (2.0 * self.n)
            def_share = S / float(self.n)
        expected_fine = self.audit_prob * self.fine
        gain_defect = def_share - coop_share - expected_fine
        if t == 0:
            return Action.D if gain_defect > 0 else Action.C
        if t == self.r - 1:
            return Action.D
        opp_coop_est = int(history.opponent_cooperators[-1])
        opp_def_est = self.n - 1 - opp_coop_est
        total_cons_C = opp_coop_est * coop_share + opp_def_est * def_share + coop_share
        stock_after_C = max(0.0, S - total_cons_C)
        growth_C = 2.0 * stock_after_C * (1.0 - stock_after_C / self.capacity) if self.capacity != 0 else 0.0
        stock_next_C = min(stock_after_C + growth_C, self.capacity)
        total_cons_D = opp_coop_est * coop_share + opp_def_est * def_share + def_share
        stock_after_D = max(0.0, S - total_cons_D)
        growth_D = 2.0 * stock_after_D * (1.0 - stock_after_D / self.capacity) if self.capacity != 0 else 0.0
        stock_next_D = min(stock_after_D + growth_D, self.capacity)
        if gain_defect > 0:
            if stock_next_D >= self.safety_margin:
                return Action.D
            else:
                return Action.C
        else:
            return Action.C

class Strategy_SELFISH_16(BaseStrategy):
    """
    Selfish Adaptive Strategy for the Common Pool Resource Game.
    The agent estimates opponents' cooperation probability from the last round,
    compares the expected utility of cooperating versus defecting (including audit
    risk and projected future stock), and chooses the action that maximizes the
    sum of current and expected future payoffs.
    """

    def __init__(self, game_description: CommonPoolDescription):
        self.n = game_description.n_players
        self.r = game_description.n_rounds
        self.capacity = float(game_description.capacity)
        self.audit_prob = float(game_description.audit_prob)
        self.fine = float(game_description.fine)

    def __call__(self, history: PlayerHistory, current_stock: float) -> Action:
        if current_stock == 0.0:
            return Action.C
        t = history.round_number
        if t == self.r - 1:
            return Action.D if current_stock > 0 else Action.C
        if t == 0:
            p_hat = 0.5
        else:
            coop_opponents = float(history.opponent_cooperators[t - 1])
            p_hat = coop_opponents / (self.n - 1) if self.n > 1 else 0.0
        pi_C = current_stock / (2.0 * self.n)
        pi_D = current_stock / self.n - self.audit_prob * self.fine
        k = p_hat * (self.n - 1)

        def next_stock(total_consumption: float) -> float:
            s_rem = max(current_stock - total_consumption, 0.0)
            if self.capacity == 0.0:
                growth = 0.0
            else:
                growth = 2.0 * s_rem * (1.0 - s_rem / self.capacity)
            return min(s_rem + growth, self.capacity)
        total_cons_C = (k + 1.0) * current_stock / (2.0 * self.n) + (self.n - 1 - k) * current_stock / self.n
        s_next_C = next_stock(total_cons_C)
        total_cons_D = k * current_stock / (2.0 * self.n) + (self.n - 1 - k + 1.0) * current_stock / self.n
        s_next_D = next_stock(total_cons_D)
        future_rounds = self.r - t - 1
        V_C = future_rounds * (s_next_C / (2.0 * self.n))
        V_D = future_rounds * (s_next_D / (2.0 * self.n))
        U_C = pi_C + V_C
        U_D = pi_D + V_D
        if U_D > U_C:
            return Action.D
        else:
            return Action.C
