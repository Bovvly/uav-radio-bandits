"""
Скрипт для создания высококачественных иллюстраций для статьи.
Фокус на сценарии High Corr и сравнении ключевых агентов.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Настройки для научной графики
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"] + rcParams["font.serif"]
rcParams["font.size"] = 12
rcParams["axes.labelsize"] = 14
rcParams["axes.titlesize"] = 16
rcParams["legend.fontsize"] = 11
rcParams["xtick.labelsize"] = 11
rcParams["ytick.labelsize"] = 11
rcParams["figure.dpi"] = 300
rcParams["savefig.bbox"] = "tight"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3
rcParams["grid.linestyle"] = "--"

# Цвета и стили (премиальная палитра)
COLORS = {
    "Random": "#95a5a6",
    "Myopic Bayes": "#e67e22",
    "BB‑TS": "#2c3e50",
    "ε‑Greedy": "#3498db",
    "TS-Direct": "#9b59b6",
    "Myopic (no cost)": "#e74c3c",
    "Oracle": "#27ae60",
}

LINESTYLES = {
    "Random": ":",
    "Myopic Bayes": "--",
    "BB‑TS": "-",
    "ε‑Greedy": "-.",
    "TS-Direct": "-.",
    "Myopic (no cost)": ":",
    "Oracle": "--",
}

def load_results(path: str) -> dict:
    if not os.path.exists(path):
        return None
    data = np.load(path)
    results = {}
    for key in data.files:
        parts = key.split("__")
        if len(parts) != 2:
            continue
        agent_name, metric = parts
        if agent_name not in results:
            results[agent_name] = {}
        results[agent_name][metric] = data[key]
    return results

def create_central_illustration(lang: str = "en"):
    suffix = "" if lang == "ru" else "_en"
    print(f"Создание центральной иллюстрации (High Corr, {lang})...")
    results = load_results("results/high_corr.npz")
    if not results:
        print("  ОШИБКА: Данные high_corr.npz не найдены.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Агенты для отрисовки
    agents = ["Random", "Myopic Bayes", "BB‑TS", "ε‑Greedy", "TS-Direct", "Myopic (no cost)"]
    
    if lang == "ru":
        labels = {
            "Random": "Случайная стратегия",
            "Myopic Bayes": "Жадный Байес",
            "BB‑TS": "BB‑TS (Предлагаемый)",
            "ε‑Greedy": "ε-жадный",
            "TS-Direct": "TS-Direct",
            "Myopic (no cost)": "Без штрафа"
        }
        x_label = "Шаг $t$"
        y_label = "Накопленное сожаление $\mathbb{E}[\mathcal{L}(t)]$"
        zoom_title = "Увеличение: адаптивные стратегии"
        gain_text = "Прирост эффективности"
    else:
        labels = {
            "Random": "Random Policy",
            "Myopic Bayes": "Myopic Bayes",
            "BB‑TS": "BB‑TS (Proposed)",
            "ε‑Greedy": "ε-Greedy",
            "TS-Direct": "TS-Direct",
            "Myopic (no cost)": "Myopic (no cost)"
        }
        x_label = "Time step $t$"
        y_label = "Cumulative Regret $\mathbb{E}[\mathcal{L}(t)]$"
        zoom_title = "Zoom: Smart Policies"
        gain_text = "Efficiency gain"

    T = len(results["BB‑TS"]["mean"])
    t = np.arange(1, T + 1)

    ax.set_yscale("linear")
    ax.set_ylim(0, 6000) # Вмещаем Random

    for name in agents:
        if name not in results:
            continue
        
        mean = results[name]["mean"]
        ci_low = results[name]["ci_low"]
        ci_high = results[name]["ci_high"]

        color = COLORS.get(name, "#333")
        ls = LINESTYLES.get(name, "-")
        lw = 2.5 if name == "BB‑TS" else 1.8
        
        ax.plot(t, mean, label=labels.get(name, name), color=color, linestyle=ls, linewidth=lw, zorder=3 if name == "BB‑TS" else 2)
        ax.fill_between(t, ci_low, ci_high, color=color, alpha=0.15, zorder=1)

    # Создаем inset (врезку) для детального сравнения Myopic и BB-TS
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_ins = inset_axes(ax, width="45%", height="40%", loc="center right", borderpad=2)
    
    for name in ["Myopic Bayes", "BB‑TS"]:
        if name in results:
            ax_ins.plot(t, results[name]["mean"], color=COLORS[name], 
                        linestyle=LINESTYLES[name], linewidth=2)
            ax_ins.fill_between(t, results[name]["ci_low"], results[name]["ci_high"], 
                                color=COLORS[name], alpha=0.15)
    
    ax_ins.set_xlim(0, T)
    ax_ins.set_ylim(0, 1000) # Фокус на малых значениях
    ax_ins.set_title(zoom_title, fontsize=10)
    ax_ins.grid(True, alpha=0.2)
    ax_ins.tick_params(labelsize=8)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    # Добавляем аннотацию момента расхождения (в основное поле или врезку)
    if "BB‑TS" in results and "Myopic Bayes" in results:
        diff = results["Myopic Bayes"]["mean"] - results["BB‑TS"]["mean"]
        cross_idx = np.where(diff > 0)[0]
        if len(cross_idx) > 0:
            idx = cross_idx[0]
            ax_ins.annotate(gain_text, xy=(idx, results["BB‑TS"]["mean"][idx]), 
                            xytext=(idx + 2000, results["BB‑TS"]["mean"][idx] - 200),
                            arrowprops=dict(facecolor='black', arrowstyle='->', alpha=0.5),
                            fontsize=9, alpha=0.7)

    ax.legend(loc="upper left", frameon=True, shadow=False, framealpha=0.9)
    
    os.makedirs("figures", exist_ok=True)
    save_path = f"figures/article_high_corr_regret{suffix}.pdf"
    fig.savefig(save_path)
    # Сохраним также в PNG для превью, если нужно
    fig.savefig(f"figures/article_high_corr_regret{suffix}.png")
    plt.close(fig)
    print(f"  Сохранено: {save_path}")

def create_summary_bar_chart(lang: str = "en"):
    suffix = "" if lang == "ru" else "_en"
    print(f"Создание итоговой диаграммы по всем сценариям ({lang})...")
    scenarios = ["base", "high_corr", "fast_switch"]
    
    if lang == "ru":
        sc_labels = {
            "base": "Базовый",
            "high_corr": "Высок. корр.",
            "fast_switch": "Быстр. перекл."
        }
        y_label = "Финальное сожаление $\mathbb{E}[\mathcal{L}(T)]$"
        chart_title = "Сводные результаты по сценариям"
        agent_labels = {
            "Myopic Bayes": "Жадный",
            "ε‑Greedy": "ε-жадный",
            "BB‑TS": "BB‑TS"
        }
    else:
        sc_labels = {
            "base": "Base",
            "high_corr": "High Corr",
            "fast_switch": "Fast Switch"
        }
        y_label = "Final Cumulative Regret $\mathbb{E}[\mathcal{L}(T)]$"
        chart_title = "Performance Summary across Scenarios"
        agent_labels = {
            "Myopic Bayes": "Myopic",
            "ε‑Greedy": "ε-greedy",
            "BB‑TS": "BB‑TS"
        }
    
    agents = ["Myopic Bayes", "ε‑Greedy", "BB‑TS", "TS-Direct", "Myopic (no cost)"]

    data_to_plot = {a: [] for a in agents}
    err_to_plot = {a: [] for a in agents}

    for sc in scenarios:
        res = load_results(f"results/{sc}.npz")
        if not res:
            continue
        for a in agents:
            if a in res:
                final_mean = res[a]["mean"][-1]
                # Оценка стандартной ошибки среднего
                # std_err = np.std(res[a]["final_regrets"]) / np.sqrt(len(res[a]["final_regrets"]))
                # Но мы можем взять размах CI
                std_err = (res[a]["ci_high"][-1] - res[a]["ci_low"][-1]) / 2
                data_to_plot[a].append(final_mean)
                err_to_plot[a].append(std_err)

    x = np.arange(len(scenarios))
    width = 0.15
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter out agents with no data
    valid_agents = [a for a in agents if len(data_to_plot[a]) == len(scenarios)]
    print(f"  Valid agents for summary: {valid_agents}")
    
    for i, a in enumerate(valid_agents):
        ax.bar(x + (i - len(valid_agents)/2 + 0.5) * width, data_to_plot[a], width, 
               yerr=err_to_plot[a], label=agent_labels.get(a, a), 
               color=COLORS.get(a, "#333"), alpha=0.85, capsize=5)

    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels([sc_labels[s] for s in scenarios])
    ax.legend()
    ax.set_title(chart_title)
    
    save_path = f"figures/article_summary_bar{suffix}.pdf"
    fig.savefig(save_path)
    fig.savefig(f"figures/article_summary_bar{suffix}.png")
    plt.close(fig)
    print(f"  Сохранено: {save_path}")

def create_switching_bar_chart(lang: str = "en"):
    suffix = "" if lang == "ru" else "_en"
    print(f"Создание диаграммы частоты переключений ({lang})...")
    scenarios = ["base", "high_corr", "fast_switch"]
    
    if lang == "ru":
        sc_labels = {"base": "Базовый", "high_corr": "Высок. корр.", "fast_switch": "Быстр. перекл."}
        y_label = "Среднее число переключений"
        chart_title = "Частота переключений агентов"
        agent_labels = {"Myopic Bayes": "Жадный", "ε‑Greedy": "ε-жадный", "BB‑TS": "BB‑TS", "Random": "Случайный"}
    else:
        sc_labels = {"base": "Base", "high_corr": "High Corr", "fast_switch": "Fast Switch"}
        y_label = "Average Number of Switches"
        chart_title = "Agent Switching Frequency"
        agent_labels = {"Myopic Bayes": "Myopic", "ε‑Greedy": "ε-greedy", "BB‑TS": "BB‑TS", "Random": "Random"}
    
    agents = ["Random", "Myopic Bayes", "BB‑TS", "ε‑Greedy", "TS-Direct"]
    data_to_plot = {a: [] for a in agents}

    for sc in scenarios:
        res = load_results(f"results/{sc}.npz")
        if not res: continue
        for a in agents:
            if a in res:
                val = res[a].get("avg_switches", 0)
                data_to_plot[a].append(val)

    x = np.arange(len(scenarios))
    width = 0.15
    fig, ax = plt.subplots(figsize=(10, 6))
    
    valid_agents = [a for a in agents if len(data_to_plot[a]) == len(scenarios)]
    print(f"  Valid agents for switching: {valid_agents}")
    
    for i, a in enumerate(valid_agents):
        ax.bar(x + (i - len(valid_agents)/2 + 0.5) * width, data_to_plot[a], width, 
               label=agent_labels.get(a, a), color=COLORS.get(a, "#333"), alpha=0.8)

    ax.set_ylabel(y_label)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([sc_labels[s] for s in scenarios])
    ax.legend()
    ax.set_title(chart_title)
    
    save_path = f"figures/article_switching_bar{suffix}.pdf"
    fig.savefig(save_path)
    fig.savefig(save_path.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Сохранено: {save_path}")


def create_fast_switch_illustration(lang: str = "en"):
    suffix = "_fs" if lang == "ru" else "_fs_en"
    print(f"Создание иллюстрации для Fast Switch ({lang})...")
    results = load_results("results/fast_switch.npz")
    if not results: return

    fig, ax = plt.subplots(figsize=(8, 5))
    agents = ["Random", "Myopic Bayes", "BB‑TS", "ε‑Greedy", "TS-Direct"]
    
    if lang == "ru":
        labels = {"Random": "Случайная", "Myopic Bayes": "Жадный", "BB‑TS": "BB‑TS (Предл.)"}
        x_label, y_label = "Шаг $t$", "Накопленное сожаление"
    else:
        labels = {"Random": "Random", "Myopic Bayes": "Myopic", "BB‑TS": "BB‑TS (Proposed)"}
        x_label, y_label = "Time step $t$", "Cumulative Regret"

    T = len(results["BB‑TS"]["mean"])
    t = np.arange(1, T + 1)

    for name in agents:
        if name not in results: continue
        ax.plot(t, results[name]["mean"], label=labels.get(name, name), 
                color=COLORS.get(name, "#333"), linestyle=LINESTYLES.get(name, "-"),
                linewidth=2.5 if name == "BB‑TS" else 1.8)
        ax.fill_between(t, results[name]["ci_low"], results[name]["ci_high"], 
                        color=COLORS.get(name, "#333"), alpha=0.15)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend(loc="upper left")
    ax.set_title("Fast Switching Scenario" if lang == "en" else "Сценарий быстрых переключений")
    
    save_name = f"figures/article_fast_switch_regret{'_en' if lang=='en' else ''}.pdf"
    fig.savefig(save_name)
    fig.savefig(save_name.replace(".pdf", ".png"))
    plt.close(fig)
    print(f"  Сохранено: {save_name}")


if __name__ == "__main__":
    for lang in ["ru", "en"]:
        create_central_illustration(lang)
        create_summary_bar_chart(lang)
        create_switching_bar_chart(lang)
        create_fast_switch_illustration(lang)
