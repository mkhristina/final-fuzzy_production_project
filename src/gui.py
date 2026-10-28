
from pathlib import Path
import sys

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fuzzy_systems import MamdaniSystem, SugenoSystem
from src.membership import INPUT_RANGES


st.set_page_config(
    page_title="Оценка производственного риска",
    page_icon="🏭",
    layout="wide",
)


INPUT_COLUMNS = list(INPUT_RANGES.keys())


@st.cache_resource
def load_systems():
    return MamdaniSystem(), SugenoSystem()


def show_rules(rules, sugeno_mode=False):
    if not rules:
        st.warning("Активные правила не найдены.")
        return

    rows = []

    for rule in rules:
        row = {
            "Правило": str(rule["rule_id"]),
            "Описание": str(rule["description"]),
            "Сила активации": round(float(rule["strength"]), 4),
            "Заключение": str(rule["consequent"]),
        }

        if sugeno_mode:
            row["Числовой выход"] = float(rule["consequent_value"])

        rows.append(row)

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
    )


def show_comparison_chart(mamdani_score, sugeno_score):
    fig, ax = plt.subplots(figsize=(9, 4.5))

    ax.axhspan(0, 25, alpha=0.12, color="green", label="Низкий")
    ax.axhspan(25, 50, alpha=0.12, color="gold", label="Умеренный")
    ax.axhspan(50, 75, alpha=0.12, color="darkorange", label="Высокий")
    ax.axhspan(75, 100, alpha=0.12, color="red", label="Критический")

    names = ["Мамдани", "Сугено"]
    scores = [mamdani_score, sugeno_score]

    bars = ax.bar(
        names,
        scores,
        width=0.5,
        color=["steelblue", "mediumpurple"],
        zorder=3,
    )

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 2,
            f"{score:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ax.set_ylim(0, 110)
    ax.set_ylabel("Оценка риска (0–100)")
    ax.set_title("Сравнение нечётких моделей")
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


def validate_csv(df):
    """Проверить входные колонки, числовые значения и допустимые диапазоны."""
    missing = [
        column for column in INPUT_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "В CSV отсутствуют обязательные столбцы: "
            + ", ".join(missing)
        )

    result = df.copy()
    issues = pd.Series("", index=result.index, dtype="object")

    for column in INPUT_COLUMNS:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        invalid_numeric = result[column].isna()

        for index in result.index[invalid_numeric]:
            message = (
                f"{column}: пустое или нечисловое значение"
            )
            current = issues.at[index]
            issues.at[index] = (
                f"{current}; {message}" if current else message
            )

        low, high = INPUT_RANGES[column]

        outside = result[column].notna() & (
            (result[column] < low)
            | (result[column] > high)
        )

        for index in result.index[outside]:
            message = (
                f"{column}={result.at[index, column]} "
                f"вне диапазона {low}–{high}"
            )
            current = issues.at[index]
            issues.at[index] = (
                f"{current}; {message}" if current else message
            )

    return result, issues


def analyze_csv(uploaded_file):
    """Рассчитать обе модели для каждой корректной строки CSV."""
    df = pd.read_csv(uploaded_file)
    df, issues = validate_csv(df)

    mamdani, sugeno = load_systems()

    df["mamdani_score"] = float("nan")
    df["mamdani_category"] = ""
    df["sugeno_score"] = float("nan")
    df["sugeno_category"] = ""
    df["analysis_status"] = ""
    df["analysis_message"] = ""

    for index, row in df.iterrows():
        if issues.loc[index]:
            df.at[index, "analysis_status"] = "Ошибка данных"
            df.at[index, "analysis_message"] = issues.loc[index]
            continue

        inputs = {
            column: float(row[column])
            for column in INPUT_COLUMNS
        }

        try:
            mamdani_score = mamdani.predict(inputs)
            sugeno_score = sugeno.predict(inputs)

            df.at[index, "mamdani_score"] = mamdani_score
            df.at[index, "mamdani_category"] = (
                mamdani.classify(mamdani_score)
            )
            df.at[index, "sugeno_score"] = sugeno_score
            df.at[index, "sugeno_category"] = (
                sugeno.classify(sugeno_score)
            )
            df.at[index, "analysis_status"] = "Успешно"

        except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
            df.at[index, "analysis_status"] = "Ошибка расчёта"
            df.at[index, "analysis_message"] = str(error)

    return df


def show_manual_calculation():
    st.subheader("1. Ручная оценка производственного риска")

    with st.form("production_metrics_form"):
        col1, col2 = st.columns(2)

        with col1:
            utilization = st.number_input(
                "Загрузка оборудования (%)",
                40.0, 120.0, 75.0, step=1.0,
            )
            defect_rate = st.number_input(
                "Доля дефектной продукции (%)",
                0.0, 15.0, 5.0, step=0.5,
            )
            downtime_pct = st.number_input(
                "Простои оборудования (%)",
                0.0, 30.0, 10.0, step=1.0,
            )

        with col2:
            overdue_orders = st.number_input(
                "Количество просроченных заказов",
                0, 40, 5, step=1,
            )
            staff_availability = st.number_input(
                "Доступность персонала (%)",
                60.0, 100.0, 90.0, step=1.0,
            )
            energy_overrun = st.number_input(
                "Перерасход энергии (%)",
                -10.0, 30.0, 8.0, step=1.0,
            )

        submitted = st.form_submit_button(
            "Рассчитать производственный риск",
            width="stretch",
            type="primary",
        )

    if not submitted:
        st.info("Введите показатели и нажмите кнопку расчёта.")
        return

    inputs = {
        "utilization": utilization,
        "defect_rate": defect_rate,
        "downtime_pct": downtime_pct,
        "overdue_orders": overdue_orders,
        "staff_availability": staff_availability,
        "energy_overrun": energy_overrun,
    }

    try:
        mamdani, sugeno = load_systems()

        mamdani_score = mamdani.predict(inputs)
        sugeno_score = sugeno.predict(inputs)

        mamdani_category = mamdani.classify(mamdani_score)
        sugeno_category = sugeno.classify(sugeno_score)

        mamdani_rules = mamdani.explain(inputs)
        sugeno_rules = sugeno.explain(inputs)

    except (ValueError, TypeError, KeyError, ZeroDivisionError) as error:
        st.error(f"Не удалось рассчитать риск: {error}")
        return

    st.divider()
    st.subheader("2. Результаты оценки")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown("### Модель Мамдани")
        st.metric("Оценка риска", f"{mamdani_score:.2f} / 100")
        st.progress(min(max(mamdani_score / 100, 0.0), 1.0))
        st.write(f"**Категория риска:** {mamdani_category}")

    with result_col2:
        st.markdown("### Модель Сугено")
        st.metric("Оценка риска", f"{sugeno_score:.2f} / 100")
        st.progress(min(max(sugeno_score / 100, 0.0), 1.0))
        st.write(f"**Категория риска:** {sugeno_category}")

    st.subheader("3. Графическое сравнение")
    show_comparison_chart(mamdani_score, sugeno_score)

    st.caption(
        "Пороговые значения: низкий — менее 25; умеренный — от 25 до 50; "
        "высокий — от 50 до 75; критический — 75 и выше."
    )

    st.divider()
    st.subheader("4. Активные правила")

    tab_mamdani, tab_sugeno = st.tabs(["Мамдани", "Сугено"])

    with tab_mamdani:
        show_rules(mamdani_rules)

    with tab_sugeno:
        show_rules(sugeno_rules, sugeno_mode=True)


def show_batch_analysis():
    st.subheader("5. Пакетный анализ CSV")

    st.write(
        "Загрузите CSV с шестью входными показателями. "
        "Дополнительные столбцы, например цех, смена и экспертная оценка, "
        "будут сохранены в итоговом файле."
    )

    uploaded_file = st.file_uploader(
        "Выберите CSV-файл",
        type=["csv"],
        key="batch_csv_uploader",
    )

    if uploaded_file is None:
        st.info("Можно загрузить готовый файл data/production.csv.")
        return

    if st.button("Проанализировать CSV", type="primary"):
        try:
            with st.spinner("Выполняется пакетный расчёт..."):
                results = analyze_csv(uploaded_file)

            st.session_state["batch_results"] = results

        except (
            ValueError,
            pd.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            st.error(f"Не удалось прочитать или проверить CSV: {error}")
            return

    results = st.session_state.get("batch_results")

    if results is None:
        return

    successful = int(
        (results["analysis_status"] == "Успешно").sum()
    )
    failed = len(results) - successful

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Всего строк", len(results))
    metric2.metric("Успешно рассчитано", successful)
    metric3.metric("Ошибки", failed)

    st.dataframe(
        results,
        width="stretch",
        hide_index=True,
    )

    # Отдельная таблица, чтобы причины ошибок было легко увидеть.
    errors = results[
        results["analysis_status"] != "Успешно"
    ]

    if not errors.empty:
        st.subheader("6. Строки с ошибками")

        error_columns = [
            column for column in [
                "period",
                "workshop",
                "shift",
                "utilization",
                "defect_rate",
                "downtime_pct",
                "overdue_orders",
                "staff_availability",
                "energy_overrun",
                "analysis_status",
                "analysis_message",
            ]
            if column in errors.columns
        ]

        st.warning(
            f"Найдено записей с ошибками: {len(errors)}. "
            "Исходные значения не изменялись."
        )

        st.dataframe(
            errors[error_columns],
            width="stretch",
            hide_index=True,
        )

    csv_bytes = results.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        "Скачать результаты CSV",
        data=csv_bytes,
        file_name="production_risk_results.csv",
        mime="text/csv",
        width="stretch",
    )

    if successful:
        st.subheader("7. Распределение рассчитанного риска")

        fig, ax = plt.subplots(figsize=(9, 4.5))

        ax.hist(
            results.loc[
                results["analysis_status"] == "Успешно",
                "mamdani_score",
            ],
            bins=20,
            alpha=0.7,
            label="Мамдани",
        )

        ax.hist(
            results.loc[
                results["analysis_status"] == "Успешно",
                "sugeno_score",
            ],
            bins=20,
            alpha=0.7,
            label="Сугено",
        )

        ax.set_xlabel("Оценка риска (0–100)")
        ax.set_ylabel("Количество записей")
        ax.set_title("Распределение оценок производственного риска")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def main():
    st.title(
        "🏭 Интеллектуальная система оценки производственного риска"
    )

    st.markdown(
        """
        Система оценивает производственный риск методами нечёткой логики
        Мамдани и Сугено. Доступны ручной расчёт и пакетный анализ CSV.
        """
    )

    manual_tab, batch_tab = st.tabs(
        ["Ручной расчёт", "Пакетный анализ CSV"]
    )

    with manual_tab:
        show_manual_calculation()

    with batch_tab:
        show_batch_analysis()


if __name__ == "__main__":
    main()
