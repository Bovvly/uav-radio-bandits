"""
Визуализация результатов: графики накопленного сожаления с CI.
Сохраняет PDF для включения в LaTeX.
"""

import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройки шрифтов для LaTeX-совместимости
rcParams["font.family"] = "serif"
rcParams["font.size"] = 11
rcParams["axes.labelsize"] = 12
rcParams["legend.fontsize"] = 9
rcParams["figure.dpi"] = 150

# Цвета для агентов
COLORS = {
    "Oracle": "#2ecc71",
    "Random": "#95a5a6",
    "Myopic (no cost)": "#bdc3c7",
    "Myopic Bayes": "#f39c12",
    "ε‑Greedy": "#3498db",
    "TS-Direct": "#e74c3c",
    "BB‑TS": "#8e44ad",
}

LINESTYLES = {
    "Oracle": "--",
    "Random": ":",
    "Myopic (no cost)": "--",
    "Myopic Bayes": "-",
    "ε‑Greedy": "-.",
    "TS-Direct": "-",
    "BB‑TS": "-",
}

SCENARIO_TITLES_RU = {
    "base": "Базовый сценарий (K=5, гетерогенные каналы)",
    "high_corr": "Высокая корреляция (α, β ≤ 0.03)",
    "fast_switch": "Быстрые переключения (α, β ≥ 0.25)",
}

SCENARIO_TITLES_EN = {
    "base": "Base Scenario (K=5, heterogeneous channels)",
    "high_corr": "High Correlation (α, β ≤ 0.03)",
    "fast_switch": "Fast Switching (α, β ≥ 0.25)",
}


def load_results(path: str) -> dict:
    """Загрузить результаты из .npz файла."""
    data = np.load(path)
    results = {}
    for key in data.files:
        agent_name, metric = key.split("__", 1)
        if agent_name not in results:
            results[agent_name] = {}
        results[agent_name][metric] = data[key]
    return results


def plot_cumulative_regret(
    results: dict,
    title: str,
    save_path: str,
    agents_to_plot: list[str] | None = None,
    lang: str = "ru",
) -> None:
    """Построить график накопленного сожаления для всех агентов."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if agents_to_plot is None:
        agents_to_plot = list(results.keys())

    for agent_name in agents_to_plot:
        if agent_name not in results:
            continue
        res = results[agent_name]
        T = len(res["mean"])
        t = np.arange(1, T + 1)

        color = COLORS.get(agent_name, "#333333")
        ls = LINESTYLES.get(agent_name, "-")
        lw = 2.5 if agent_name == "BB-TS" else 1.5

        ax.plot(t, res["mean"], label=agent_name, color=color,
                linestyle=ls, linewidth=lw)
        ax.fill_between(t, res["ci_low"], res["ci_high"],
                        alpha=0.12, color=color)

    if lang == "ru":
        ax.set_xlabel("Шаг $t$")
        ax.set_ylabel("Накопленное сожаление $\\mathbb{E}[\\mathcal{L}(t)]$")
    else:
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel("Cumulative Regret $\\mathbb{E}[\\mathcal{L}(t)]$")
    
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_switches_over_time(
    results: dict,
    title: str,
    save_path: str,
    agents_to_plot: list[str] | None = None,
    lang: str = "ru",
) -> None:
    """Построить график количества переключений от времени."""
    fig, ax = plt.subplots(figsize=(8, 5))

    if agents_to_plot is None:
        agents_to_plot = list(results.keys())

    for agent_name in agents_to_plot:
        if agent_name not in results or "mean_switches" not in results[agent_name]:
            continue
        res = results[agent_name]
        T = len(res["mean_switches"])
        t = np.arange(1, T + 1)

        color = COLORS.get(agent_name, "#333333")
        ls = LINESTYLES.get(agent_name, "-")
        lw = 2.5 if agent_name == "BB‑TS" else 1.5

        ax.plot(t, res["mean_switches"], label=agent_name, color=color,
                linestyle=ls, linewidth=lw)

    if lang == "ru":
        ax.set_xlabel("Шаг $t$")
        ax.set_ylabel("Кумулятивное число переключений")
    else:
        ax.set_xlabel("Time step $t$")
        ax.set_ylabel("Cumulative number of switches")
    
    ax.set_title(title)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_final_regret_bar(all_results: dict, save_path: str, lang: str = "ru") -> None:
    """Столбчатая диаграмма финального сожаления по сценариям."""
    scenarios = list(all_results.keys())
    # Исключаем Oracle и Random для наглядности
    agents = [a for a in list(all_results[scenarios[0]].keys())
              if a not in ("Oracle", "Random")]
    
    titles = SCENARIO_TITLES_RU if lang == "ru" else SCENARIO_TITLES_EN

    fig, axes = plt.subplots(1, len(scenarios), figsize=(14, 5), sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for idx, sc in enumerate(scenarios):
        ax = axes[idx]
        res = all_results[sc]
        names = []
        means = []
        stds = []
        colors = []
        for a in agents:
            if a in res:
                names.append(a)
                means.append(np.mean(res[a]["final_regrets"]))
                stds.append(np.std(res[a]["final_regrets"]) / np.sqrt(len(res[a]["final_regrets"])))
                colors.append(COLORS.get(a, "#333"))

        x = np.arange(len(names))
        ax.bar(x, means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_title(titles.get(sc, sc), fontsize=10)
        
        if lang == "ru":
            ax.set_ylabel("$\\mathbb{E}[\\mathcal{L}(T)]$")
        else:
            ax.set_ylabel("Final Regret $\\mathbb{E}[\\mathcal{L}(T)]$")
        
        ax.grid(True, axis="y", alpha=0.3)

    main_title = "Финальное накопленное сожаление" if lang == "ru" else "Final Cumulative Regret"
    fig.suptitle(main_title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


if __name__ == "__main__":
    os.makedirs("figures", exist_ok=True)

    all_results = {}
    for sc in ["base", "high_corr", "fast_switch"]:
        path = f"results/{sc}.npz"
        if not os.path.exists(path):
            print(f"  SKIP {path} (not found)")
            continue
        all_results[sc] = load_results(path)

        for lang in ["ru", "en"]:
            suffix = "" if lang == "ru" else "_en"
            titles = SCENARIO_TITLES_RU if lang == "ru" else SCENARIO_TITLES_EN
            
            # Кривая сожаления — все агенты
            plot_cumulative_regret(
                all_results[sc],
                title=titles.get(sc, sc),
                save_path=f"figures/regret_{sc}{suffix}.pdf",
                lang=lang
            )

            # Кривая сожаления — без Oracle и Random (для статьи)
            plot_cumulative_regret(
                all_results[sc],
                title=titles.get(sc, sc),
                save_path=f"figures/regret_{sc}_main{suffix}.pdf",
                agents_to_plot=["Myopic (no cost)", "Myopic Bayes", "ε‑Greedy",
                                "TS-Direct", "BB‑TS"],
                lang=lang
            )

            # Переключения
            plot_switches_over_time(
                all_results[sc],
                title=titles.get(sc, sc),
                save_path=f"figures/switches_{sc}{suffix}.pdf",
                lang=lang
            )

    if all_results:
        plot_final_regret_bar(all_results, "figures/final_regret_bar.pdf", lang="ru")
        plot_final_regret_bar(all_results, "figures/final_regret_bar_en.pdf", lang="en")

    print("\nВсе графики сохранены в figures/")
