"""
Эксперимент сравнения стратегий планирования в POMDP Гилберта–Эллиотта.
Агенты знают модель, но не скрытое состояние. Учитывается штраф за переключение.
"""

import os
import sys
import time
from typing import Callable
import subprocess

# Добавляем корневую директорию проекта в пути поиска, чтобы импортировать пакет
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from uav_radio_bandits import (
    GilbertElliottMAB, ChannelConfig,
    make_base_scenario, make_high_correlation_scenario, make_fast_switching_scenario,
    BaseAgent, RandomAgent, MyopicBayesAgent, MyopicNoPenaltyAgent,
    EpsilonGreedyBeliefAgent, DirectStateTSAgent, BayesianBeliefTSAgent,
)


# ---------------------------------------------------------------------------
# Рабочий процесс одного запуска
# ---------------------------------------------------------------------------
def _worker_run(run_idx: int, base_seed: int, T: int,
                scenario_fn: Callable, agent_factory: Callable,
                switch_penalty: float) -> tuple[np.ndarray, int]:
    seed = base_seed + run_idx
    rng = np.random.default_rng(seed)
    configs = scenario_fn()
    env = GilbertElliottMAB(configs, seed=seed, switch_penalty=switch_penalty)
    agent = agent_factory(configs, rng) if agent_factory is not None else None

    regrets = np.zeros(T, dtype=np.float64)
    switches_at_t = np.zeros(T, dtype=np.float64)
    prev_action = None

    for t in range(T):
        if agent is None:          # Oracle
            action = env.oracle_action()
        else:
            action = agent.select_action()

        if prev_action is not None and action != prev_action:
            switches_at_t[t] = 1.0
        prev_action = action

        result = env.step(action)
        if agent is not None:
            agent.update(action, result.reward)

        regrets[t] = result.oracle_expected_reward - result.reward

    return np.cumsum(regrets), np.cumsum(switches_at_t)


# ---------------------------------------------------------------------------
# Параллельный Монте‑Карло
# ---------------------------------------------------------------------------
def run_monte_carlo(
    scenario_fn: Callable,
    agent_factory: Callable,
    T: int = 10_000,
    n_runs: int = 500,
    base_seed: int = 42,
    switch_penalty: float = 0.5,
) -> dict:
    results = Parallel(n_jobs=-1, backend='loky', batch_size='auto')(
        delayed(_worker_run)(i, base_seed, T, scenario_fn, agent_factory, switch_penalty)
        for i in tqdm(range(n_runs), desc="  Monte Carlo Runs", leave=False)
    )
    cum_regrets, cum_switches = zip(*results)
    all_regrets = np.array(cum_regrets)
    all_switches = np.array(cum_switches)

    mean_regret = np.mean(all_regrets, axis=0)
    std_regret = np.std(all_regrets, axis=0, ddof=1)
    
    mean_switches = np.mean(all_switches, axis=0)
    
    return {
        "mean": mean_regret,
        "std": std_regret,
        "ci_low": mean_regret - 1.96 * std_regret / np.sqrt(n_runs),
        "ci_high": mean_regret + 1.96 * std_regret / np.sqrt(n_runs),
        "final_regrets": all_regrets[:, -1],
        "mean_switches": mean_switches,
        "avg_switches": mean_switches[-1],
        "std_switches": np.std(all_switches[:, -1], ddof=1),
    }


# ---------------------------------------------------------------------------
# Полный эксперимент по сценарию
# ---------------------------------------------------------------------------
def run_full_experiment(scenario_name: str, scenario_fn: Callable,
                        T: int = 10_000, n_runs: int = 500,
                        switch_penalty: float = 0.5) -> dict:
    print(f"\n{'='*60}")
    print(f" СЦЕНАРИЙ: {scenario_name.upper()} | T={T}, N={n_runs}, penalty={switch_penalty}")
    print(f"{'='*60}")

    # Фабрики агентов
    factories = [
        ("Random", lambda cfgs, rng: RandomAgent(len(cfgs), rng)),
        ("Myopic (no cost)", lambda cfgs, rng: MyopicNoPenaltyAgent(len(cfgs), cfgs)),
        ("Myopic Bayes", lambda cfgs, rng: MyopicBayesAgent(len(cfgs), cfgs, switch_penalty)),
        ("ε‑Greedy", lambda cfgs, rng: EpsilonGreedyBeliefAgent(len(cfgs), cfgs, rng,
                                                                   epsilon=0.05,
                                                                   switch_penalty=switch_penalty)),
        ("TS-Direct", lambda cfgs, rng: DirectStateTSAgent(len(cfgs), cfgs, rng,
                                                           switch_penalty=switch_penalty)),
        ("BB‑TS", lambda cfgs, rng: BayesianBeliefTSAgent(len(cfgs), cfgs, rng,
                                                          kappa=20.0,
                                                          switch_penalty=switch_penalty)),
        ("Oracle", None),   # специальный маркер для Oracle
    ]

    results = {}
    for name, factory in factories:
        print(f"Агент: {name}")
        t0 = time.time()
        res = run_monte_carlo(scenario_fn, factory, T=T, n_runs=n_runs,
                              switch_penalty=switch_penalty)
        elapsed = time.time() - t0
        print(f"  Завершено за {elapsed:.2f}s | "
              f"Final Regret: {res['mean'][-1]:.2f} ± {1.96*res['std'][-1]/np.sqrt(n_runs):.2f} | "
              f"Avg Switches: {res['avg_switches']:.1f}")
        results[name] = res
    return results


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    scenarios = [
        ("base", make_base_scenario),
        ("high_corr", make_high_correlation_scenario),
        ("fast_switch", make_fast_switching_scenario),
    ]

    T, N_RUNS, PENALTY = 10_000, 500, 0.5
    all_results = {}

    for sc_name, sc_fn in scenarios:
        all_results[sc_name] = run_full_experiment(sc_name, sc_fn, T=T,
                                                   n_runs=N_RUNS,
                                                   switch_penalty=PENALTY)
        save_dict = {}
        for agent_name, res in all_results[sc_name].items():
            for key, val in res.items():
                # Сохраняем массивы и скаляры с префиксом агента
                save_dict[f"{agent_name}__{key}"] = val
        np.savez(f"results/{sc_name}.npz", **save_dict)

    # Итоговая сводка
    print("\n" + "—"*80)
    header = f"{'Агент':<18}" + "".join(f"{s:<22}" for s, _ in scenarios)
    print(header)
    print("—" * 80)
    
    agent_names = list(all_results[scenarios[0][0]].keys())
    for name in agent_names:
        row = f"{name:<18}"
        for sc_name, _ in scenarios:
            val = all_results[sc_name][name]["mean"][-1]
            ci = 1.96 * all_results[sc_name][name]["std"][-1] / np.sqrt(N_RUNS)
            row += f"{val:6.1f}±{ci:4.1f}   "
        print(row)
    print("\nСреднее количество переключений:")
    print(f"{'Агент':<18}" + "".join(f"{s:<15}" for s, _ in scenarios))
    for name in agent_names:
        row = f"{name:<18}"
        for sc_name, _ in scenarios:
            sw = all_results[sc_name][name]["avg_switches"]
            row += f"{sw:<15.1f}"
        print(row)

    # Статистический анализ
    print("\n" + "="*80)
    print("ЗАПУСК СТАТИСТИЧЕСКОГО АНАЛИЗА...")
    print("="*80)
    subprocess.run(["python3", "scripts/stat_test.py"])