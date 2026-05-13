"""
Агенты для планирования в POMDP Гилберта–Эллиотта.
Все (кроме Random) знают параметры каналов и величину штрафа,
но не знают текущее скрытое состояние.
"""

from abc import ABC, abstractmethod
import numpy as np
from .environment import ChannelConfig


class BaseAgent(ABC):
    name: str = "BaseAgent"
    def __init__(self, K: int):
        self.K = K

    @abstractmethod
    def select_action(self) -> int:
        ...

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        ...

    def reset(self) -> None:
        pass


class RandomAgent(BaseAgent):
    name = "Random"
    def __init__(self, K: int, rng: np.random.Generator):
        super().__init__(K)
        self._rng = rng

    def select_action(self) -> int:
        return int(self._rng.integers(self.K))

    def update(self, action: int, reward: float) -> None:
        pass


# ---------------------------------------------------------------------------
# Миксин для байесовского обновления веры
# ---------------------------------------------------------------------------
class _BeliefMixin:
    def _init_belief(self, configs: list[ChannelConfig]) -> None:
        self._configs = configs
        self._belief = np.array([c.stationary_good for c in configs])

    def _predict_belief(self) -> None:
        for i, c in enumerate(self._configs):
            b = self._belief[i]
            self._belief[i] = b * (1 - c.alpha) + (1 - b) * c.beta

    def _update_belief(self, action: int, reward: float) -> None:
        c = self._configs[action]
        b = self._belief[action]
        if reward > 0.5:          # успех (несмотря на возможный штраф)
            p_obs_g = c.p_good
            p_obs_b = c.p_bad
        else:
            p_obs_g = 1 - c.p_good
            p_obs_b = 1 - c.p_bad
        num = p_obs_g * b
        den = num + p_obs_b * (1 - b)
        if den > 0:
            self._belief[action] = num / den

    def _expected_rewards(self) -> np.ndarray:
        return np.array([
            self._belief[i] * c.p_good + (1 - self._belief[i]) * c.p_bad
            for i, c in enumerate(self._configs)])


# ---------------------------------------------------------------------------
# Myopic Bayes со штрафом (жадная политика)
# ---------------------------------------------------------------------------
class MyopicBayesAgent(_BeliefMixin, BaseAgent):
    name = "Myopic Bayes"
    def __init__(self, K: int, configs: list[ChannelConfig], switch_penalty: float = 0.0):
        BaseAgent.__init__(self, K)
        self._init_belief(configs)
        self.switch_penalty = switch_penalty
        self._prev_action = None

    def select_action(self) -> int:
        er = self._expected_rewards()
        if self._prev_action is not None:
            er = er - self.switch_penalty * (np.arange(self.K) != self._prev_action)
        return int(np.argmax(er))

    def update(self, action: int, reward: float) -> None:
        self._update_belief(action, reward)
        self._predict_belief()
        self._prev_action = action

    def reset(self) -> None:
        self._belief = np.array([c.stationary_good for c in self._configs])
        self._prev_action = None


# Myopic без штрафа (игнорирует стоимость переключения)
class MyopicNoPenaltyAgent(MyopicBayesAgent):
    name = "Myopic (no cost)"
    def __init__(self, K: int, configs: list[ChannelConfig]):
        super().__init__(K, configs, switch_penalty=0.0)


# ---------------------------------------------------------------------------
# ε‑Greedy на основе belief
# ---------------------------------------------------------------------------
class EpsilonGreedyBeliefAgent(_BeliefMixin, BaseAgent):
    name = "ε‑Greedy Belief"
    def __init__(self, K: int, configs: list[ChannelConfig], rng: np.random.Generator,
                 epsilon: float = 0.05, switch_penalty: float = 0.0):
        BaseAgent.__init__(self, K)
        self._init_belief(configs)
        self.epsilon = epsilon
        self.switch_penalty = switch_penalty
        self._rng = rng
        self._prev_action = None

    def select_action(self) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.K))
        er = self._expected_rewards()
        if self._prev_action is not None:
            er = er - self.switch_penalty * (np.arange(self.K) != self._prev_action)
        return int(np.argmax(er))

    def update(self, action: int, reward: float) -> None:
        self._update_belief(action, reward)
        self._predict_belief()
        self._prev_action = action

    def reset(self) -> None:
        self._belief = np.array([c.stationary_good for c in self._configs])
        self._prev_action = None


# ---------------------------------------------------------------------------
# Прямой TS по состояниям (семплируем скрытое состояние)
# ---------------------------------------------------------------------------
class DirectStateTSAgent(_BeliefMixin, BaseAgent):
    name = "TS-Direct"
    def __init__(self, K: int, configs: list[ChannelConfig], rng: np.random.Generator,
                 switch_penalty: float = 0.0):
        BaseAgent.__init__(self, K)
        self._init_belief(configs)
        self._rng = rng
        self.switch_penalty = switch_penalty
        self._prev_action = None

    def select_action(self) -> int:
        sampled_good = self._rng.random(self.K) < self._belief
        p_g = np.array([c.p_good for c in self._configs])
        p_b = np.array([c.p_bad for c in self._configs])
        p = np.where(sampled_good, p_g, p_b)
        if self._prev_action is not None:
            p = p - self.switch_penalty * (np.arange(self.K) != self._prev_action)
        return int(np.argmax(p))

    def update(self, action: int, reward: float) -> None:
        self._update_belief(action, reward)
        self._predict_belief()
        self._prev_action = action

    def reset(self) -> None:
        self._belief = np.array([c.stationary_good for c in self._configs])
        self._prev_action = None


# ---------------------------------------------------------------------------
# Предложенный BB‑TS (Beta‑TS на belief)
# ---------------------------------------------------------------------------
class BayesianBeliefTSAgent(_BeliefMixin, BaseAgent):
    name = "BB‑TS"
    def __init__(self, K: int, configs: list[ChannelConfig], rng: np.random.Generator,
                 kappa: float = 20.0, switch_penalty: float = 0.0):
        BaseAgent.__init__(self, K)
        self._init_belief(configs)
        self._rng = rng
        self.kappa = kappa
        self.switch_penalty = switch_penalty
        self._prev_action = None

    def select_action(self) -> int:
        a = np.maximum(self.kappa * self._belief, 0.01)
        b = np.maximum(self.kappa * (1.0 - self._belief), 0.01)
        theta = self._rng.beta(a, b)
        exp = np.array([theta[i] * c.p_good + (1 - theta[i]) * c.p_bad
                        for i, c in enumerate(self._configs)])
        if self._prev_action is not None:
            exp = exp - self.switch_penalty * (np.arange(self.K) != self._prev_action)
        return int(np.argmax(exp))

    def update(self, action: int, reward: float) -> None:
        self._update_belief(action, reward)
        self._predict_belief()
        self._prev_action = action

    def reset(self) -> None:
        self._belief = np.array([c.stationary_good for c in self._configs])
        self._prev_action = None