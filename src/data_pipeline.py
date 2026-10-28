
from pathlib import Path

import numpy as np
import pandas as pd


# Допустимые диапазоны производственных показателей
RANGES = {
    "utilization": (40.0, 120.0),
    "defect_rate": (0.0, 15.0),
    "downtime_pct": (0.0, 30.0),
    "overdue_orders": (0, 40),
    "staff_availability": (60.0, 100.0),
    "energy_overrun": (-10.0, 30.0),
    "expert_risk": (0.0, 100.0),
}


def generate_production_data(
    n_samples: int = 2000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Генерирует воспроизводимый набор производственных данных.

    expert_risk создаётся как условная экспертная оценка
    на основе производственных показателей с добавлением шума.
    """
    if n_samples < 1:
        raise ValueError("n_samples должен быть не меньше 1.")

    rng = np.random.default_rng(random_seed)

    # Основные производственные показатели
    utilization = rng.uniform(40, 120, n_samples)
    defect_rate = rng.beta(2, 5, n_samples) * 15
    downtime_pct = rng.beta(2, 5, n_samples) * 30
    overdue_orders = rng.poisson(8, n_samples)
    overdue_orders = np.clip(overdue_orders, 0, 40).astype(int)

    staff_availability = rng.triangular(
        60, 94, 100, n_samples
    )
    energy_overrun = rng.normal(5, 8, n_samples)
    energy_overrun = np.clip(energy_overrun, -10, 30)

    # Нормализованные факторы риска.
    # Все значения здесь ограничены диапазоном 0–1.
    overload = np.clip(
        (utilization - 85) / 35, 0, 1
    )
    underload = np.clip(
        (65 - utilization) / 25, 0, 1
    )
    defects = defect_rate / 15
    downtime = downtime_pct / 30
    overdue = overdue_orders / 40
    staff_shortage = (100 - staff_availability) / 40
    energy = np.clip(
        (energy_overrun + 10) / 40, 0, 1
    )

    # Условная экспертная оценка.
    # Перегрузка, дефекты, простои, задержки,
    # дефицит персонала и перерасход энергии
    # увеличивают риск.
    raw_risk = (
        12 * overload
        + 5 * underload
        + 23 * defects
        + 21 * downtime
        + 15 * overdue
        + 16 * staff_shortage
        + 8 * energy
    )

    # Добавляем умеренный шум, имитирующий различия
    # между экспертными оценками.
    expert_risk = np.clip(
        raw_risk + rng.normal(0, 4, n_samples),
        0,
        100,
    )

    # Служебные поля для анализа.
    periods = rng.integers(1, 13, n_samples)
    workshops = rng.choice(
        ["Механический", "Сборочный", "Литейный", "Упаковочный"],
        n_samples,
    )
    shifts = rng.choice(
        ["Утренняя", "Вечерняя", "Ночная"],
        n_samples,
    )

    df = pd.DataFrame({
        "period": periods,
        "workshop": workshops,
        "shift": shifts,
        "utilization": utilization,
        "defect_rate": defect_rate,
        "downtime_pct": downtime_pct,
        "overdue_orders": overdue_orders,
        "staff_availability": staff_availability,
        "energy_overrun": energy_overrun,
        "expert_risk": expert_risk,
    })

    return df


def validate_production_data(df: pd.DataFrame) -> None:
    """Проверяет обязательные столбцы, пропуски и диапазоны."""
    required_columns = list(RANGES)

    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"Отсутствуют столбцы: {sorted(missing_columns)}"
        )

    if df[required_columns].isnull().any().any():
        raise ValueError("В данных обнаружены пропуски.")

    for column, (minimum, maximum) in RANGES.items():
        values = df[column]

        if not np.isfinite(values).all():
            raise ValueError(
                f"Столбец {column} содержит NaN или inf."
            )

        if not values.between(minimum, maximum).all():
            raise ValueError(
                f"Столбец {column} выходит за диапазон "
                f"{minimum}–{maximum}."
            )

    if not pd.api.types.is_integer_dtype(df["overdue_orders"]):
        raise ValueError(
            "overdue_orders должен содержать целые числа."
        )


def save_production_data(
    df: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Проверяет и сохраняет данные в CSV."""
    validate_production_data(df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main() -> None:
    # Корень проекта — родитель папки src.
    project_root = Path(__file__).resolve().parent.parent
    output_path = project_root / "data" / "production.csv"

    df = generate_production_data(
        n_samples=2000,
        random_seed=42,
    )

    validate_production_data(df)
    save_production_data(df, output_path)

    print("Данные успешно созданы!")
    print(f"Файл: {output_path}")
    print(f"Количество наблюдений: {len(df)}")
    print("\nПервые 5 строк:")
    print(df.head().round(2).to_string(index=False))

    print("\nРазмер набора данных:")
    print(df.shape)

    print("\nОписательная статистика:")
    print(df[list(RANGES)].describe().round(2))

    print("\nПропуски:")
    print(df.isnull().sum())


if __name__ == "__main__":
    main()
