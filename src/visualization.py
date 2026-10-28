from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.fuzzy_systems import MamdaniSystem, SugenoSystem
from src.membership import MembershipFunctions, INPUT_RANGES


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"
RESULTS_DIR = PROJECT_ROOT / "results"


def prepare_directories():
    """Создаёт папки для результатов и графиков."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_membership_functions():
    """Строит графики функций принадлежности входов и выхода."""
    prepare_directories()
    mf = MembershipFunctions()

    variables = list(INPUT_RANGES.keys()) + ["risk_score"]

    for variable_name in variables:
        universe = mf.universes[variable_name]
        terms = mf.terms[variable_name]

        fig, ax = plt.subplots(figsize=(9, 5))

        for term_name, membership_values in terms.items():
            ax.plot(
                universe,
                membership_values,
                label=term_name,
                linewidth=2,
            )

        ax.set_title(f"Функции принадлежности: {variable_name}")
        ax.set_xlabel("Значение переменной")
        ax.set_ylabel("Степень принадлежности")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        filename = PLOTS_DIR / f"membership_{variable_name}.png"
        fig.savefig(filename, dpi=150)
        plt.close(fig)

        print(f"Сохранён график: {filename}")


def build_scenarios():
    """Возвращает набор типовых производственных сценариев."""
    return [
        {
            "scenario": "Благоприятный",
            "utilization": 80,
            "defect_rate": 1,
            "downtime_pct": 2,
            "overdue_orders": 2,
            "staff_availability": 98,
            "energy_overrun": -5,
        },
        {
            "scenario": "Обычный",
            "utilization": 85,
            "defect_rate": 4,
            "downtime_pct": 7,
            "overdue_orders": 8,
            "staff_availability": 90,
            "energy_overrun": 5,
        },
        {
            "scenario": "Напряжённый",
            "utilization": 105,
            "defect_rate": 9,
            "downtime_pct": 18,
            "overdue_orders": 24,
            "staff_availability": 78,
            "energy_overrun": 18,
        },
        {
            "scenario": "Критический",
            "utilization": 118,
            "defect_rate": 14,
            "downtime_pct": 28,
            "overdue_orders": 38,
            "staff_availability": 63,
            "energy_overrun": 28,
        },
    ]


def analyze_scenarios():
    """Сравнивает результаты двух систем для типовых сценариев."""
    prepare_directories()

    mamdani = MamdaniSystem()
    sugeno = SugenoSystem()
    results = []

    input_names = list(INPUT_RANGES.keys())

    for scenario in build_scenarios():
        name = scenario["scenario"]
        inputs = {
            key: value
            for key, value in scenario.items()
            if key != "scenario"
        }

        result = {"scenario": name}

        for key in input_names:
            result[key] = inputs[key]

        for system_name, system in (
            ("mamdani", mamdani),
            ("sugeno", sugeno),
        ):
            try:
                score = system.predict(inputs)
                result[f"{system_name}_risk"] = round(score, 3)
                result[f"{system_name}_category"] = system.classify(score)
            except (ValueError, TypeError) as error:
                result[f"{system_name}_risk"] = np.nan
                result[f"{system_name}_category"] = f"Ошибка: {error}"

        if (
            pd.notna(result["mamdani_risk"])
            and pd.notna(result["sugeno_risk"])
        ):
            result["absolute_difference"] = round(
                abs(result["mamdani_risk"] - result["sugeno_risk"]),
                3,
            )
        else:
            result["absolute_difference"] = np.nan

        results.append(result)

    dataframe = pd.DataFrame(results)
    output_file = RESULTS_DIR / "scenario_comparison.csv"
    dataframe.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("\nСравнение сценариев:")
    print(
        dataframe[
            [
                "scenario",
                "mamdani_risk",
                "mamdani_category",
                "sugeno_risk",
                "sugeno_category",
                "absolute_difference",
            ]
        ].to_string(index=False)
    )
    print(f"\nТаблица сохранена: {output_file}")

    return dataframe


def plot_model_comparison(dataframe=None):
    """Строит диаграмму сравнения результатов Мамдани и Сугено."""
    prepare_directories()

    if dataframe is None:
        dataframe = analyze_scenarios()

    valid = dataframe.dropna(
        subset=["mamdani_risk", "sugeno_risk"]
    )

    if valid.empty:
        raise ValueError("Нет корректных результатов для построения графика.")

    x = np.arange(len(valid))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(
        x - width / 2,
        valid["mamdani_risk"],
        width,
        label="Мамдани",
    )
    ax.bar(
        x + width / 2,
        valid["sugeno_risk"],
        width,
        label="Сугено",
    )

    ax.set_title("Сравнение оценок производственного риска")
    ax.set_xlabel("Производственный сценарий")
    ax.set_ylabel("Риск, баллы")
    ax.set_xticks(x)
    ax.set_xticklabels(valid["scenario"], rotation=15)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    filename = PLOTS_DIR / "05_mamdani_vs_sugeno.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"Сохранён график: {filename}")


def plot_scenario_differences(dataframe=None):
    """Строит график абсолютной разницы двух оценок."""
    prepare_directories()

    if dataframe is None:
        dataframe = analyze_scenarios()

    valid = dataframe.dropna(subset=["absolute_difference"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(valid["scenario"], valid["absolute_difference"])

    ax.set_title("Разница оценок Мамдани и Сугено")
    ax.set_xlabel("Производственный сценарий")
    ax.set_ylabel("Абсолютная разница, баллы")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    filename = PLOTS_DIR / "06_scenario_differences.png"
    fig.savefig(filename, dpi=150)
    plt.close(fig)

    print(f"Сохранён график: {filename}")


def run_full_analysis():
    """Запускает полный анализ функций принадлежности и сценариев."""
    plot_membership_functions()
    dataframe = analyze_scenarios()
    plot_model_comparison(dataframe)
    plot_scenario_differences(dataframe)

    print("\nАнализ завершён.")


if __name__ == "__main__":
    run_full_analysis()



def plot_rule_graph():
    """Строит и сохраняет граф связей между нечёткими правилами."""
    import networkx as nx
    from src.rules import RULES

    prepare_directories()
    graph = nx.DiGraph()

    for rule in RULES:
        rule_id = rule.rule_id
        graph.add_node(rule_id, node_type="rule")

        for condition in rule.conditions:
            if isinstance(condition, (tuple, list)) and len(condition) >= 2:
                variable, term = condition[0], condition[1]
            else:
                continue

            condition_node = f"{variable}\n{term}"
            graph.add_node(condition_node, node_type="condition")
            graph.add_edge(condition_node, rule_id)

        output_node = f"Риск: {rule.consequent}"
        graph.add_node(output_node, node_type="output")
        graph.add_edge(rule_id, output_node)

    plt.figure(figsize=(18, 14))
    positions = nx.spring_layout(graph, seed=42, k=0.8)

    for node_type, color, size in [
        ("condition", "lightblue", 900),
        ("rule", "lightgreen", 700),
        ("output", "salmon", 1100),
    ]:
        nodes = [
            node for node, attrs in graph.nodes(data=True)
            if attrs.get("node_type") == node_type
        ]
        nx.draw_networkx_nodes(
            graph, positions, nodelist=nodes,
            node_color=color, node_size=size
        )

    nx.draw_networkx_edges(
        graph, positions, arrows=True,
        alpha=0.35, arrowsize=10
    )
    nx.draw_networkx_labels(graph, positions, font_size=6)

    plt.title("Граф связей нечётких правил")
    plt.axis("off")
    plt.tight_layout()

    output_file = RESULTS_DIR / "rule_graph.png"
    plt.savefig(output_file, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"Граф правил сохранён: {output_file}")
