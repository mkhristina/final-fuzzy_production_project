from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent


def run_step(module_name, description):
    """Запустить модуль проекта и остановиться при ошибке."""
    print(f"\n{'=' * 60}", flush=True)
    print(description, flush=True)
    print("=" * 60, flush=True)

    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=PROJECT_ROOT,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Этап '{description}' завершился с ошибкой "
            f"(код {result.returncode})."
        )


def main():
    print(
        "ИНТЕЛЛЕКТУАЛЬНАЯ НЕЧЁТКАЯ СИСТЕМА ОЦЕНКИ",
        flush=True,
    )
    print(
        "ПРОИЗВОДСТВЕННОГО РИСКА ЦЕХА",
        flush=True,
    )

    steps = [
        ("src.data_pipeline", "1. Генерация производственных данных"),
        ("src.fuzzy_systems", "2. Проверка нечётких систем Мамдани и Сугено"),
        ("src.visualization", "3. Построение графиков и анализ сценариев"),
        ("src.clustering", "4. Кластеризация Fuzzy C-Means"),
    ]

    for module_name, description in steps:
        run_step(module_name, description)

    print(f"\n{'=' * 60}", flush=True)
    print("ВСЕ ЭТАПЫ ПРОЕКТА УСПЕШНО ЗАВЕРШЕНЫ", flush=True)
    print(f"Результаты: {PROJECT_ROOT / 'results'}", flush=True)
    print(
        f"Графики: {PROJECT_ROOT / 'results' / 'plots'}",
        flush=True,
    )
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
