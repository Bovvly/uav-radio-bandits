"""
Статистический анализ результатов экспериментов.
Вычисляет p‑значения Wilcoxon signed‑rank test между BB‑TS и Myopic Bayes
для каждого сценария. Загружает данные из сохранённых .npz файлов.
"""

import numpy as np
from scipy.stats import wilcoxon
from pathlib import Path

def main():
    results_dir = Path("results")
    scenarios = ["base", "high_corr", "fast_switch"]
    
    print("Статистические тесты (Wilcoxon signed‑rank) на финальном регрете")
    print("="*70)
    for sc in scenarios:
        fname = results_dir / f"{sc}.npz"
        if not fname.exists():
            print(f"Файл {fname} не найден, пропуск.")
            continue
        data = np.load(fname, allow_pickle=True)
        try:
            bb = data["BB‑TS__final_regrets"]
            mb = data["Myopic Bayes__final_regrets"]
        except KeyError:
            print(f"В сценарии {sc} отсутствуют нужные агенты.")
            continue
        
        stat, p = wilcoxon(bb, mb)
        mean_diff = np.mean(bb - mb)
        print(f"{sc:<12}: mean diff = {mean_diff:7.2f}, statistic = {stat:.1f}, p = {p:.4e}")
    print("="*70)

if __name__ == "__main__":
    main()