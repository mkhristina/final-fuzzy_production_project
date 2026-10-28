
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "production.csv"
RESULTS_DIR = ROOT / "results"

TARGET = "expert_risk"
FEATURES = [
    "utilization",
    "defect_rate",
    "downtime_pct",
    "overdue_orders",
    "staff_availability",
    "energy_overrun",
]


def triangular_membership(x, a, b, c):
    """Треугольная функция принадлежности с параметрами a < b < c."""
    x = np.asarray(x, dtype=float)
    left = np.divide(
        x - a, b - a,
        out=np.zeros_like(x),
        where=(b != a),
    )
    right = np.divide(
        c - x, c - b,
        out=np.zeros_like(x),
        where=(c != b),
    )
    return np.maximum(0.0, np.minimum(left, right))


def risk_proxy(data, params):
    """
    Учебная непрерывная оценка на основе входных показателей.
    Оптимизируемая функция принадлежности используется как нелинейный
    признак. Это не заменяет модели Мамдани и Сугено.
    """
    utilization = data["utilization"].to_numpy(dtype=float)
    defects = data["defect_rate"].to_numpy(dtype=float)
    downtime = data["downtime_pct"].to_numpy(dtype=float)
    overdue = data["overdue_orders"].to_numpy(dtype=float)
    staff = data["staff_availability"].to_numpy(dtype=float)
    energy = data["energy_overrun"].to_numpy(dtype=float)

    a, b, c = params
    defect_membership = triangular_membership(defects, a, b, c)

    # Базовый набор признаков и нелинейный вклад дефектов.
    return np.clip(
        12 * np.clip((utilization - 85) / 35, 0, 1)
        + 5 * np.clip((65 - utilization) / 25, 0, 1)
        + 23 * defects / 15
        + 21 * downtime / 30
        + 15 * overdue / 40
        + 16 * (100 - staff) / 40
        + 8 * np.clip((energy + 10) / 40, 0, 1)
        + 5 * defect_membership,
        0,
        100,
    )


def _metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def optimize_membership():
    """Подбирает параметры на обучающей части и оценивает на отложенной."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Не найден файл данных: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"В CSV отсутствуют столбцы: {missing}")

    data = data[required].replace([np.inf, -np.inf], np.nan).dropna()
    if len(data) < 20:
        raise ValueError("Для оптимизации требуется не менее 20 строк данных.")

    train, test = train_test_split(
        data,
        test_size=0.2,
        random_state=42,
    )

    # Начальная треугольная функция для доли брака, диапазон 0–15.
    initial = np.array([0.0, 5.0, 15.0])
    bounds = [(0.0, 15.0), (0.0, 15.0), (0.0, 15.0)]

    def objective(params):
        a, b, c = params
        if not (0 <= a < b < c <= 15):
            return 1e6 + abs(b - a) + abs(c - b)

        prediction = risk_proxy(train, params)
        actual = train[TARGET].to_numpy(dtype=float)
        return mean_absolute_error(actual, prediction)

    result = minimize(
        objective,
        x0=initial,
        method="SLSQP",
        bounds=bounds,
        constraints=[
            {"type": "ineq", "fun": lambda p: p[1] - p[0] - 0.1},
            {"type": "ineq", "fun": lambda p: p[2] - p[1] - 0.1},
        ],
        options={"maxiter": 500, "ftol": 1e-8},
    )

    if not result.success:
        raise RuntimeError(f"Оптимизация не завершилась успешно: {result.message}")

    optimized = result.x
    actual_test = test[TARGET].to_numpy(dtype=float)

    before = risk_proxy(test, initial)
    after = risk_proxy(test, optimized)

    rows = []
    for stage, params, predictions in [
        ("До оптимизации", initial, before),
        ("После оптимизации", optimized, after),
    ]:
        metrics = _metrics(actual_test, predictions)
        rows.append({
            "stage": stage,
            "parameter_a": float(params[0]),
            "parameter_b": float(params[1]),
            "parameter_c": float(params[2]),
            **metrics,
            "test_records": int(len(test)),
        })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "optimization_results.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    results = optimize_membership()
    print(results.to_string(index=False))
    print(f"\nРезультаты сохранены в: {RESULTS_DIR / 'optimization_results.csv'}")
