from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "production.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

FEATURES = [
    "utilization",
    "defect_rate",
    "downtime_pct",
    "overdue_orders",
    "staff_availability",
    "energy_overrun",
]


def prepare_directories():
    """Создать каталоги для результатов."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_scale_data():
    """Загрузить производственные данные и стандартизировать признаки."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Файл данных не найден: {DATA_FILE}\n"
            "Сначала запустите генерацию данных."
        )

    dataframe = pd.read_csv(DATA_FILE)

    missing_columns = [
        column for column in FEATURES if column not in dataframe.columns
    ]
    if missing_columns:
        raise ValueError(
            f"В CSV отсутствуют необходимые столбцы: {missing_columns}"
        )

    if len(dataframe) < 6:
        raise ValueError("Для кластеризации необходимо минимум 6 строк.")

    values = dataframe[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("В данных есть пропуски или бесконечные значения.")

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(values)

    return dataframe, scaled_values, scaler


def evaluate_clusters(scaled_values, cluster_counts=range(2, 7)):
    """Оценить варианты FCM с разным количеством кластеров."""
    models = {}
    metrics = []

    # scikit-fuzzy ожидает данные в формате:
    # количество признаков × количество наблюдений.
    fcm_data = scaled_values.T

    for cluster_count in cluster_counts:
        (
            centers,
            membership,
            initial_membership,
            distances,
            objective,
            iterations,
            fpc,
        ) = fuzz.cluster.cmeans(
            data=fcm_data,
            c=cluster_count,
            m=2.0,
            error=0.005,
            maxiter=1000,
            seed=42,
        )

        labels = np.argmax(membership, axis=0)

        if len(np.unique(labels)) > 1:
            silhouette = silhouette_score(scaled_values, labels)
        else:
            silhouette = -1.0

        models[cluster_count] = {
            "centers": centers,
            "membership": membership,
            "labels": labels,
            "iterations": iterations,
            "fpc": fpc,
        }

        metrics.append(
            {
                "clusters": cluster_count,
                "fpc": float(fpc),
                "silhouette": float(silhouette),
                "iterations": int(iterations),
                "objective": float(objective[-1]),
            }
        )

        print(
            f"Кластеров: {cluster_count} | "
            f"FPC: {fpc:.4f} | "
            f"Silhouette: {silhouette:.4f} | "
            f"Итераций: {iterations}"
        )

    return pd.DataFrame(metrics), models


def plot_cluster_metrics(metrics):
    """Построить графики качества кластеризации."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        metrics["clusters"],
        metrics["fpc"],
        marker="o",
    )
    ax.set_title("Индекс нечёткого разбиения FPC")
    ax.set_xlabel("Количество кластеров")
    ax.set_ylabel("FPC")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "07_fcm_fpc.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        metrics["clusters"],
        metrics["silhouette"],
        marker="o",
    )
    ax.set_title("Силуэтная оценка кластеризации FCM")
    ax.set_xlabel("Количество кластеров")
    ax.set_ylabel("Silhouette score")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "08_fcm_silhouette.png", dpi=150)
    plt.close(fig)


def save_cluster_assignments(dataframe, model):
    """Сохранить принадлежность каждого наблюдения к кластерам."""
    result = dataframe.copy()
    membership = model["membership"]

    result["cluster"] = np.argmax(membership, axis=0) + 1
    result["max_membership"] = np.max(membership, axis=0)

    for index in range(membership.shape[0]):
        result[f"membership_cluster_{index + 1}"] = membership[index]

    output_file = RESULTS_DIR / "fcm_assignments.csv"
    result.to_csv(output_file, index=False)
    print(f"Назначения кластеров сохранены: {output_file}")


def save_cluster_profiles(dataframe, model, scaler):
    """Сохранить профили кластеров и их центры в исходных единицах."""
    result = dataframe[FEATURES].copy()
    result["cluster"] = model["labels"] + 1

    profiles = result.groupby("cluster")[FEATURES].mean()
    profiles.to_csv(RESULTS_DIR / "fcm_cluster_profiles.csv")

    original_centers = scaler.inverse_transform(model["centers"])
    centers_frame = pd.DataFrame(
        original_centers,
        columns=FEATURES,
    )
    centers_frame.index = np.arange(1, len(centers_frame) + 1)
    centers_frame.index.name = "cluster"
    centers_frame.to_csv(RESULTS_DIR / "fcm_centers.csv")

    print("Профили кластеров сохранены.")
    print("\nСредние характеристики кластеров:")
    print(profiles.round(2).to_string())


def run_clustering():
    """Выполнить полный анализ методом нечётких C-средних."""
    prepare_directories()

    print("Загрузка и стандартизация данных...")
    dataframe, scaled_values, scaler = load_and_scale_data()
    print(f"Загружено наблюдений: {len(dataframe)}")

    print("\nСравнение количества кластеров...")
    metrics, models = evaluate_clusters(scaled_values)

    metrics.to_csv(RESULTS_DIR / "fcm_metrics.csv", index=False)
    plot_cluster_metrics(metrics)

    # Выбираем число кластеров с наибольшей силуэтной оценкой.
    best_cluster_count = int(
        metrics.loc[metrics["silhouette"].idxmax(), "clusters"]
    )
    best_model = models[best_cluster_count]

    print(f"\nВыбрано кластеров: {best_cluster_count}")
    save_cluster_assignments(dataframe, best_model)
    save_cluster_profiles(dataframe, best_model, scaler)

    print(f"\nГрафики сохранены в: {PLOTS_DIR}")
    print(f"Результаты сохранены в: {RESULTS_DIR}")


if __name__ == "__main__":
    run_clustering()
