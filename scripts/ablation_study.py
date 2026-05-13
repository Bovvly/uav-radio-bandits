"""
Анализ чувствительности (ablation study) для параметра kappa алгоритма BB‑TS.
Исследуется влияние kappa на финальное накопленное сожаление в сценарии High Corr.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

# Добавляем корневую директорию проекта в пути поиска
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uav_radio_bandits import make_high_correlation_scenario, BayesianBeliefTSAgent
from scripts.experiment import run_monte_carlo

def run_ablation():
    os.makedirs("figures", exist_ok=True)
    
    T = 10_000
    N_RUNS = 100
    PENALTY = 0.5
    kappa_values = [1, 3, 5, 10, 15, 20, 25, 30, 35, 40, 100]
    
    results = {}
    
    print(f"ЗАПУСК АНАЛИЗА ЧУВСТВИТЕЛЬНОСТИ (Ablation Study)")
    print(f"Сценарий: High Correlation | T={T}, N={N_RUNS}, penalty={PENALTY}")
    print("="*70)
    
    for kappa in kappa_values:
        print(f"Тестирование kappa = {kappa}...", end=" ", flush=True)
        factory = lambda cfgs, rng: BayesianBeliefTSAgent(len(cfgs), cfgs, rng, 
                                                         kappa=float(kappa), 
                                                         switch_penalty=PENALTY)
        res = run_monte_carlo(make_high_correlation_scenario, factory, 
                              T=T, n_runs=N_RUNS, switch_penalty=PENALTY)
        results[kappa] = res
        print(f"Завершено. Final Regret: {res['mean'][-1]:.1f}")

    # Настройки для научной графики
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 12
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["legend.fontsize"] = 11
    plt.rcParams["xtick.labelsize"] = 11
    plt.rcParams["ytick.labelsize"] = 11

    # Построение графика
    plt.figure(figsize=(9, 6))
    
    x = kappa_values
    y = [results[k]['mean'][-1] for k in x]
    y_err = [1.96 * results[k]['std'][-1] / np.sqrt(N_RUNS) for k in x]
    
    plt.errorbar(x, y, yerr=y_err, fmt='-o', color='#2c3e50', markerfacecolor='#e74c3c', 
                 capsize=5, lw=2, markersize=8, label='Final Regret (mean ± 95% CI)')
    
    # Находим минимум
    best_idx = np.argmin(y)
    plt.annotate(fr'Best: $\kappa$={x[best_idx]}', 
                 xy=(x[best_idx], y[best_idx]), 
                 xytext=(x[best_idx], y[best_idx] + (max(y)-min(y))*0.15),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                 horizontalalignment='center', fontsize=11)

    # plt.xscale('log') # Убираем логарифмическую шкалу для соответствия референсу
    plt.xticks(kappa_values)
    plt.xlabel(r"Parameter $\kappa$ (Scale of Beta prior)", fontsize=12)
    plt.ylabel(r"Final Cumulative Regret $\mathbb{E}[\mathcal{L}(T)]$", fontsize=12)
    plt.title(r"Sensitivity Analysis of BB-TS to Parameter $\kappa$" + "\n(High Correlation Scenario)", fontsize=14)
    plt.grid(True, which="major", ls="-", alpha=0.15)
    plt.legend(loc='upper left')
    
    save_path = "figures/ablation_kappa_high_corr.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches='tight')
    print(f"\nГрафик сохранен: {save_path}")
    
    print("\nРЕЗЮМЕ:")
    print("-" * 30)
    for k in kappa_values:
        print(f"kappa = {k:3}: Regret = {results[k]['mean'][-1]:.1f}")
    print("-" * 30)
    print(f"Оптимальное значение kappa для данного сценария: {x[best_idx]}")

if __name__ == "__main__":
    run_ablation()
