
import tkinter as tk
from tkinter import ttk, messagebox

from src.fuzzy_systems import MamdaniSystem, SugenoSystem


INPUTS = {
    "Загрузка оборудования (%)": ("utilization", 40, 120, 80),
    "Доля брака (%)": ("defect_rate", 0, 15, 3),
    "Простой (%)": ("downtime_pct", 0, 30, 5),
    "Просроченные заказы": ("overdue_orders", 0, 40, 5),
    "Доступность персонала (%)": ("staff_availability", 60, 100, 95),
    "Превышение энергопотребления (%)": ("energy_overrun", -10, 30, 0),
}


class ProductionRiskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Оценка производственного риска")
        self.root.geometry("780x720")
        self.root.minsize(680, 600)

        try:
            self.mamdani = MamdaniSystem()
            self.sugeno = SugenoSystem()
        except Exception as exc:
            messagebox.showerror(
                "Ошибка инициализации",
                f"Не удалось создать нечёткие модели:\n{exc}",
                parent=self.root,
            )
            raise

        self.entries = {}
        self._build_ui()

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=18)
        main.pack(fill="both", expand=True)

        ttk.Label(
            main,
            text="Интеллектуальная система оценки риска",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", pady=(0, 4))

        ttk.Label(
            main,
            text="Введите показатели работы производственного цеха.",
        ).pack(anchor="w", pady=(0, 14))

        input_frame = ttk.LabelFrame(
            main, text="Входные показатели", padding=12
        )
        input_frame.pack(fill="x")

        for row, (label, (_, minimum, maximum, default)) in enumerate(
            INPUTS.items()
        ):
            ttk.Label(input_frame, text=label).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=5
            )

            entry = ttk.Entry(input_frame, width=18)
            entry.insert(0, str(default))
            entry.grid(row=row, column=1, sticky="ew", pady=5)

            ttk.Label(
                input_frame,
                text=f"Допустимо: {minimum} – {maximum}",
            ).grid(row=row, column=2, sticky="w", padx=12, pady=5)

            self.entries[label] = entry

        input_frame.columnconfigure(1, weight=1)

        ttk.Button(
            main,
            text="Рассчитать риск",
            command=self.calculate,
        ).pack(anchor="w", pady=14)

        result_frame = ttk.LabelFrame(
            main, text="Результаты оценки", padding=12
        )
        result_frame.pack(fill="x")

        self.mamdani_label = ttk.Label(
            result_frame, text="Мамдани: —", font=("Segoe UI", 11)
        )
        self.mamdani_label.pack(anchor="w", pady=4)

        self.sugeno_label = ttk.Label(
            result_frame, text="Сугено: —", font=("Segoe UI", 11)
        )
        self.sugeno_label.pack(anchor="w", pady=4)

        self.difference_label = ttk.Label(
            result_frame, text="Абсолютное расхождение: —"
        )
        self.difference_label.pack(anchor="w", pady=4)

        self.category_label = ttk.Label(
            result_frame,
            text="Итоговая категория: —",
            font=("Segoe UI", 12, "bold"),
        )
        self.category_label.pack(anchor="w", pady=4)

        ttk.Label(
            main,
            text="Объяснение по базе правил",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w", pady=(16, 6))

        explanation_frame = ttk.Frame(main)
        explanation_frame.pack(fill="both", expand=True)

        self.explanation = tk.Text(
            explanation_frame,
            height=10,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        scrollbar = ttk.Scrollbar(
            explanation_frame,
            orient="vertical",
            command=self.explanation.yview,
        )
        self.explanation.configure(yscrollcommand=scrollbar.set)

        self.explanation.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    @staticmethod
    def _get_score(result):
        if isinstance(result, dict):
            for key in ("risk_score", "score", "risk", "output"):
                if key in result:
                    return float(result[key])
        return float(result)

    @staticmethod
    def _format_explanation(result):
        if result is None:
            return "Модель не вернула объяснение."

        if isinstance(result, str):
            return result

        if isinstance(result, dict):
            lines = []
            for key, value in result.items():
                lines.append(f"{key}: {value}")
            return "\n".join(lines) if lines else "Активные правила не найдены."

        if isinstance(result, (list, tuple)):
            if not result:
                return "Активные правила не найдены."
            return "\n".join(str(item) for item in result)

        return str(result)

    def calculate(self):
        values = {}

        try:
            for label, (key, minimum, maximum, _) in INPUTS.items():
                raw_value = self.entries[label].get().strip().replace(",", ".")

                if not raw_value:
                    raise ValueError(f"Заполните поле «{label}».")

                value = float(raw_value)

                if not minimum <= value <= maximum:
                    raise ValueError(
                        f"Для поля «{label}» допустим диапазон "
                        f"от {minimum} до {maximum}."
                    )

                values[key] = value

            mamdani_result = self.mamdani.predict(values)
            sugeno_result = self.sugeno.predict(values)

            mamdani_score = self._get_score(mamdani_result)
            sugeno_score = self._get_score(sugeno_result)

            difference = abs(mamdani_score - sugeno_score)
            category = self.mamdani.classify(mamdani_score)

            self.mamdani_label.config(
                text=f"Мамдани: {mamdani_score:.2f} из 100"
            )
            self.sugeno_label.config(
                text=f"Сугено: {sugeno_score:.2f} из 100"
            )
            self.difference_label.config(
                text=f"Абсолютное расхождение: {difference:.2f}"
            )
            self.category_label.config(
                text=f"Итоговая категория (Мамдани): {category}"
            )

            try:
                mamdani_explanation = self.mamdani.explain(values)
            except Exception as exc:
                mamdani_explanation = (
                    f"Не удалось получить объяснение Мамдани: {exc}"
                )

            try:
                sugeno_explanation = self.sugeno.explain(values)
            except Exception as exc:
                sugeno_explanation = (
                    f"Не удалось получить объяснение Сугено: {exc}"
                )

            explanation_text = (
                "МОДЕЛЬ МАМДАНИ\n"
                f"{self._format_explanation(mamdani_explanation)}\n\n"
                "МОДЕЛЬ СУГЕНО\n"
                f"{self._format_explanation(sugeno_explanation)}\n\n"
                "Примечание: итоговая категория определяется по оценке Мамдани."
            )

            self.explanation.configure(state="normal")
            self.explanation.delete("1.0", "end")
            self.explanation.insert("1.0", explanation_text)
            self.explanation.configure(state="disabled")

        except (ValueError, TypeError) as exc:
            messagebox.showwarning(
                "Проверьте входные данные",
                str(exc),
                parent=self.root,
            )
        except Exception as exc:
            messagebox.showerror(
                "Ошибка расчёта",
                f"Не удалось рассчитать риск:\n{exc}",
                parent=self.root,
            )


def main():
    root = tk.Tk()
    ProductionRiskApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
