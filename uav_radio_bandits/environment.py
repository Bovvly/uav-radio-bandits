"""
Среда Гилберта–Эллиотта с возможным штрафом за переключение канала.
Все каналы независимы, переходы не зависят от действий.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional


@dataclass(frozen=True)
class ChannelConfig:
    """Параметры одного канала."""
    alpha: float       # G → B
    beta: float        # B → G
    p_good: float = 0.95
    p_bad: float = 0.10

    @property
    def stationary_good(self) -> float:
        return self.beta / (self.alpha + self.beta)

    @property
    def mean_reward(self) -> float:
        pi = self.stationary_good
        return self.p_good * pi + self.p_bad * (1 - pi)


class _GilbertElliottChannel:
    """Один канал со скрытым состоянием."""
    GOOD, BAD = 0, 1

    def __init__(self, cfg: ChannelConfig, rng: np.random.Generator):
        self.cfg = cfg
        self._rng = rng
        self.state = self.GOOD if rng.random() < cfg.stationary_good else self.BAD

    def reward_prob(self) -> float:
        return self.cfg.p_good if self.state == self.GOOD else self.cfg.p_bad

    def sample_reward(self) -> float:
        return float(self._rng.random() < self.reward_prob())

    def transition(self) -> None:
        u = self._rng.random()
        if self.state == self.GOOD:
            if u < self.cfg.alpha:
                self.state = self.BAD
        else:
            if u < self.cfg.beta:
                self.state = self.GOOD


@dataclass
class StepResult:
    """Результат одного шага среды."""
    reward: float                     # фактическая награда (уже со штрафом)
    hidden_states: np.ndarray         # состояния всех каналов до перехода
    oracle_action: int                # что выбрал бы Oracle
    oracle_expected_reward: float     # ожидаемая награда Oracle (с учётом штрафа)


class GilbertElliottMAB:
    """Среда с K каналами, штрафом за переключение и известной моделью."""

    def __init__(self, configs: list[ChannelConfig], seed: Optional[int] = None,
                 switch_penalty: float = 0.0):
        self.configs = configs
        self.K = len(configs)
        self._rng = np.random.default_rng(seed)
        self._channels = [_GilbertElliottChannel(c, self._rng) for c in configs]
        self.switch_penalty = switch_penalty
        self._prev_action: Optional[int] = None
        self.t = 0

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._channels = [_GilbertElliottChannel(c, self._rng) for c in self.configs]
        self._prev_action = None
        self.t = 0

    def step(self, action: int) -> StepResult:
        assert 0 <= action < self.K
        hidden = np.array([ch.state for ch in self._channels])

        # Oracle с учётом штрафа
        probs = np.array([ch.reward_prob() for ch in self._channels])
        if self._prev_action is None:
            oracle_a = int(np.argmax(probs))
            oracle_rew = probs[oracle_a]
        else:
            best_val = -np.inf
            best_a = 0
            for a in range(self.K):
                val = probs[a] - (self.switch_penalty if a != self._prev_action else 0.0)
                if val > best_val:
                    best_val = val
                    best_a = a
            oracle_a = best_a
            oracle_rew = best_val

        raw_reward = self._channels[action].sample_reward()
        switched = (self._prev_action is not None and action != self._prev_action)
        reward = raw_reward - (self.switch_penalty if switched else 0.0)

        for ch in self._channels:
            ch.transition()

        self._prev_action = action
        self.t += 1
        return StepResult(reward=reward, hidden_states=hidden,
                          oracle_action=oracle_a, oracle_expected_reward=oracle_rew)

    def oracle_action(self) -> int:
        """Оптимальное действие с учётом текущих состояний и штрафа."""
        probs = np.array([ch.reward_prob() for ch in self._channels])
        if self._prev_action is None:
            return int(np.argmax(probs))
        best_val = -np.inf
        best_a = 0
        for a in range(self.K):
            val = probs[a] - (self.switch_penalty if a != self._prev_action else 0.0)
            if val > best_val:
                best_val = val
                best_a = a
        return best_a


# Предустановленные сценарии
def make_base_scenario() -> list[ChannelConfig]:
    return [
        ChannelConfig(alpha=0.05, beta=0.10, p_good=0.95, p_bad=0.10),
        ChannelConfig(alpha=0.10, beta=0.05, p_good=0.90, p_bad=0.20),
        ChannelConfig(alpha=0.02, beta=0.08, p_good=0.98, p_bad=0.05),
        ChannelConfig(alpha=0.15, beta=0.15, p_good=0.85, p_bad=0.30),
        ChannelConfig(alpha=0.08, beta=0.12, p_good=0.92, p_bad=0.15),
    ]

def make_high_correlation_scenario() -> list[ChannelConfig]:
    return [
        ChannelConfig(alpha=0.01, beta=0.02, p_good=0.95, p_bad=0.05),
        ChannelConfig(alpha=0.02, beta=0.01, p_good=0.90, p_bad=0.10),
        ChannelConfig(alpha=0.005, beta=0.015, p_good=0.98, p_bad=0.02),
        ChannelConfig(alpha=0.015, beta=0.015, p_good=0.85, p_bad=0.15),
        ChannelConfig(alpha=0.01, beta=0.03, p_good=0.92, p_bad=0.08),
    ]

def make_fast_switching_scenario() -> list[ChannelConfig]:
    return [
        ChannelConfig(alpha=0.30, beta=0.30, p_good=0.95, p_bad=0.10),
        ChannelConfig(alpha=0.35, beta=0.25, p_good=0.90, p_bad=0.20),
        ChannelConfig(alpha=0.25, beta=0.35, p_good=0.98, p_bad=0.05),
        ChannelConfig(alpha=0.30, beta=0.30, p_good=0.85, p_bad=0.30),
        ChannelConfig(alpha=0.28, beta=0.32, p_good=0.92, p_bad=0.15),
    ]