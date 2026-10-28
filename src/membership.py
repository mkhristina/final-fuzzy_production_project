
import numpy as np
import skfuzzy as fuzz


# Диапазоны входных переменных
INPUT_RANGES = {
    "utilization": (40, 120),
    "defect_rate": (0, 15),
    "downtime_pct": (0, 30),
    "overdue_orders": (0, 40),
    "staff_availability": (60, 100),
    "energy_overrun": (-10, 30),
}

# Диапазон выходной переменной
RISK_RANGE = (0, 100)


class MembershipFunctions:
    """
    Компонент функций принадлежности.
    Использует треугольные и трапециевидные функции.
    """

    def __init__(self, resolution=1001):
        self.resolution = resolution

        self.universes = {
            name: np.linspace(low, high, resolution)
            for name, (low, high) in INPUT_RANGES.items()
        }

        self.universes["risk_score"] = np.linspace(
            RISK_RANGE[0],
            RISK_RANGE[1],
            resolution,
        )

        self.terms = self._build_terms()

    @staticmethod
    def _triangle(x, points):
        return fuzz.trimf(x, points)

    @staticmethod
    def _trapezoid(x, points):
        return fuzz.trapmf(x, points)

    def _build_terms(self):
        u = self.universes
        terms = {}

        # 1. Загрузка производственной мощности
        x = u["utilization"]
        terms["utilization"] = {
            "low": self._trapezoid(x, [40, 40, 55, 70]),
            "normal": self._triangle(x, [60, 80, 100]),
            "high": self._triangle(x, [85, 100, 115]),
            "critical": self._trapezoid(
                x, [105, 115, 120, 120]
            ),
        }

        # 2. Доля дефектной продукции
        x = u["defect_rate"]
        terms["defect_rate"] = {
            "low": self._trapezoid(x, [0, 0, 1, 3]),
            "medium": self._triangle(x, [2, 5, 8]),
            "high": self._triangle(x, [6, 9, 12]),
            "critical": self._trapezoid(
                x, [10, 13, 15, 15]
            ),
        }

        # 3. Доля простоев
        x = u["downtime_pct"]
        terms["downtime_pct"] = {
            "low": self._trapezoid(x, [0, 0, 2, 5]),
            "medium": self._triangle(x, [3, 8, 13]),
            "high": self._triangle(x, [10, 16, 22]),
            "critical": self._trapezoid(
                x, [18, 24, 30, 30]
            ),
        }

        # 4. Просроченные заказы
        x = u["overdue_orders"]
        terms["overdue_orders"] = {
            "low": self._trapezoid(x, [0, 0, 2, 6]),
            "medium": self._triangle(x, [4, 10, 16]),
            "high": self._triangle(x, [12, 20, 28]),
            "critical": self._trapezoid(
                x, [24, 32, 40, 40]
            ),
        }

        # 5. Доступность персонала:
        # чем ниже доступность, тем выше риск
        x = u["staff_availability"]
        terms["staff_availability"] = {
            "critical": self._trapezoid(
                x, [60, 60, 65, 72]
            ),
            "low": self._triangle(x, [68, 78, 88]),
            "medium": self._triangle(x, [82, 90, 96]),
            "high": self._trapezoid(
                x, [93, 97, 100, 100]
            ),
        }

        # 6. Отклонение энергопотребления
        x = u["energy_overrun"]
        terms["energy_overrun"] = {
            "low": self._trapezoid(
                x, [-10, -10, -5, 2]
            ),
            "normal": self._triangle(x, [-3, 3, 9]),
            "high": self._triangle(x, [6, 14, 22]),
            "critical": self._trapezoid(
                x, [18, 24, 30, 30]
            ),
        }

        # 7. Выходная переменная — риск
        x = u["risk_score"]
        terms["risk_score"] = {
            "low": self._trapezoid(x, [0, 0, 10, 25]),
            "moderate": self._triangle(x, [15, 35, 55]),
            "high": self._triangle(x, [45, 65, 82]),
            "critical": self._trapezoid(
                x, [70, 85, 100, 100]
            ),
        }

        return terms

    def membership(self, variable, term, value):
        """Возвращает степень принадлежности одного значения."""
        if variable not in self.terms:
            raise KeyError(f"Неизвестная переменная: {variable}")

        if term not in self.terms[variable]:
            raise KeyError(
                f"Неизвестный терм {term} переменной {variable}"
            )

        low, high = (
            RISK_RANGE
            if variable == "risk_score"
            else INPUT_RANGES[variable]
        )

        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(
                f"{variable} должен находиться в диапазоне "
                f"{low}–{high}"
            )

        universe = self.universes[variable]
        curve = self.terms[variable][term]

        return float(
            np.interp(value, universe, curve)
        )

    def fuzzify(self, variable, value):
        """Вычисляет принадлежность значения всем термам."""
        return {
            term: self.membership(variable, term, value)
            for term in self.terms[variable]
        }


if __name__ == "__main__":
    mf = MembershipFunctions()

    example = 85
    print(f"Загрузка цеха: {example}%")
    print(mf.fuzzify("utilization", example))

    print("\nРиск 60:")
    print(mf.fuzzify("risk_score", 60))
