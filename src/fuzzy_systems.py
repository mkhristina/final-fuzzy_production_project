
from abc import ABC, abstractmethod

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from src.membership import MembershipFunctions, INPUT_RANGES
from src.rules import RULES


class FuzzySystem(ABC):
    """Базовый интерфейс нечёткой системы оценки риска."""

    @abstractmethod
    def predict(self, inputs):
        raise NotImplementedError

    @abstractmethod
    def explain(self, inputs):
        raise NotImplementedError

    @staticmethod
    def classify(score):
        if isinstance(score, (bool, np.bool_)):
            raise TypeError("Оценка риска должна быть числом.")

        if not isinstance(score, (int, float, np.integer, np.floating)):
            raise TypeError("Оценка риска должна быть числом.")

        score = float(score)

        if not np.isfinite(score) or not 0 <= score <= 100:
            raise ValueError(
                "Оценка риска должна находиться в диапазоне 0–100."
            )

        if score < 25:
            return "Низкий"
        if score < 50:
            return "Умеренный"
        if score < 75:
            return "Высокий"
        return "Критический"


class InputValidationMixin:
    """Общая проверка входных данных."""

    INPUT_NAMES = list(INPUT_RANGES.keys())

    def _validate_inputs(self, inputs):
        if not isinstance(inputs, dict):
            raise TypeError("Входные данные должны быть словарём.")

        missing = set(self.INPUT_NAMES) - set(inputs)

        if missing:
            raise ValueError(
                f"Отсутствуют обязательные параметры: {sorted(missing)}"
            )

        validated = {}

        for name in self.INPUT_NAMES:
            value = inputs[name]
            low, high = INPUT_RANGES[name]

            if isinstance(value, (bool, np.bool_)):
                raise TypeError(
                    f"Параметр {name} не может быть логическим."
                )

            if not isinstance(
                value,
                (int, float, np.integer, np.floating),
            ):
                raise TypeError(f"Параметр {name} должен быть числом.")

            value = float(value)

            if not np.isfinite(value):
                raise ValueError(
                    f"Параметр {name} должен быть конечным числом."
                )

            if not low <= value <= high:
                raise ValueError(
                    f"Параметр {name} должен находиться в диапазоне "
                    f"от {low} до {high}; получено {value}."
                )

            validated[name] = value

        return validated


class MamdaniSystem(FuzzySystem, InputValidationMixin):
    """Система Мамдани с дефаззификацией методом центроида."""

    FALLBACK_CONSEQUENTS = {
        "utilization": {
            "low": "low",
            "normal": "low",
            "high": "high",
            "critical": "critical",
        },
        "defect_rate": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "downtime_pct": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "overdue_orders": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "staff_availability": {
            "critical": "critical",
            "low": "high",
            "medium": "moderate",
            "high": "low",
        },
        "energy_overrun": {
            "low": "low",
            "normal": "moderate",
            "high": "high",
            "critical": "critical",
        },
    }

    def __init__(self):
        self.mf = MembershipFunctions()
        self.antecedents = {}

        for name in self.INPUT_NAMES:
            low, high = INPUT_RANGES[name]
            variable = ctrl.Antecedent(
                np.linspace(low, high, 1001),
                name,
            )

            for term, curve in self.mf.terms[name].items():
                variable[term] = curve

            self.antecedents[name] = variable

        self.risk = ctrl.Consequent(
            np.linspace(0, 100, 1001),
            "risk_score",
        )

        for term, curve in self.mf.terms["risk_score"].items():
            self.risk[term] = curve

        self.risk.defuzzify_method = "centroid"
        self.control_rules = []

        for rule in RULES:
            expression = None

            for variable_name, term in rule.conditions:
                current = self.antecedents[variable_name][term]

                if expression is None:
                    expression = current
                elif rule.operator == "AND":
                    expression = expression & current
                else:
                    expression = expression | current

            consequent = self.risk[rule.consequent] % rule.weight

            self.control_rules.append(
                ctrl.Rule(
                    expression,
                    consequent,
                    label=rule.rule_id,
                )
            )

        self.control_system = ctrl.ControlSystem(self.control_rules)

    def _calculate_rule_strengths(self, inputs):
        activations = []

        for rule in RULES:
            degrees = [
                self.mf.membership(variable, term, inputs[variable])
                for variable, term in rule.conditions
            ]

            strength = (
                min(degrees)
                if rule.operator == "AND"
                else max(degrees)
            )
            strength *= rule.weight

            if strength > 0:
                activations.append({
                    "rule_id": rule.rule_id,
                    "strength": float(strength),
                    "consequent": rule.consequent,
                    "description": rule.description,
                })

        activations.sort(
            key=lambda item: item["strength"],
            reverse=True,
        )
        return activations

    def _predict_fallback(self, inputs):
        """Резервный вывод Мамдани при отсутствии основного результата."""
        fallback_rules = []

        for name in self.INPUT_NAMES:
            memberships = self.mf.fuzzify(name, inputs[name])
            best_term = max(memberships, key=memberships.get)
            best_strength = memberships[best_term]

            if best_strength <= 0:
                raise ValueError(
                    f"Не удалось подобрать терм для переменной {name}."
                )

            consequent_term = self.FALLBACK_CONSEQUENTS[name][best_term]

            fallback_rules.append(
                ctrl.Rule(
                    self.antecedents[name][best_term],
                    self.risk[consequent_term],
                    label=f"FALLBACK_{name}_{best_term}",
                )
            )

        fallback_system = ctrl.ControlSystem(fallback_rules)
        simulation = ctrl.ControlSystemSimulation(
            fallback_system,
            clip_to_bounds=False,
        )

        for name, value in inputs.items():
            simulation.input[name] = value

        try:
            simulation.compute()
        except (ValueError, KeyError) as exc:
            raise ValueError(
                "Не удалось сформировать резервную оценку Мамдани."
            ) from exc

        if "risk_score" not in simulation.output:
            raise ValueError(
                "Основная и резервная модели Мамдани не сформировали выход."
            )

        score = float(simulation.output["risk_score"])

        if not np.isfinite(score):
            raise ValueError(
                "Резервная модель Мамдани вернула некорректный результат."
            )

        return float(np.clip(score, 0, 100))

    def predict(self, inputs):
        validated = self._validate_inputs(inputs)

        simulation = ctrl.ControlSystemSimulation(
            self.control_system,
            clip_to_bounds=False,
        )

        for name, value in validated.items():
            simulation.input[name] = value

        try:
            simulation.compute()
        except (ValueError, KeyError):
            return self._predict_fallback(validated)

        if "risk_score" not in simulation.output:
            return self._predict_fallback(validated)

        score = float(simulation.output["risk_score"])

        if not np.isfinite(score):
            return self._predict_fallback(validated)

        return float(np.clip(score, 0, 100))

    def explain(self, inputs):
        validated = self._validate_inputs(inputs)
        activations = self._calculate_rule_strengths(validated)

        if not activations:
            return [{
                "rule_id": "FALLBACK",
                "strength": 1.0,
                "consequent": "резервный вывод",
                "description": (
                    "Основная база правил не активировалась; "
                    "для оценки использован резервный вывод Мамдани."
                ),
            }]

        return activations[:5]


class SugenoSystem(FuzzySystem, InputValidationMixin):
    """
    Система Сугено нулевого порядка.

    При отсутствии активных правил используется резервная оценка,
    основанная на степенях принадлежности входных термов.
    """

    CONSEQUENT_VALUES = {
        "low": 12.5,
        "moderate": 37.5,
        "high": 62.5,
        "critical": 87.5,
    }

    # Соответствие термов входных показателей уровням риска.
    FALLBACK_CONSEQUENTS = {
        "utilization": {
            "low": "low",
            "normal": "low",
            "high": "high",
            "critical": "critical",
        },
        "defect_rate": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "downtime_pct": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "overdue_orders": {
            "low": "low",
            "medium": "moderate",
            "high": "high",
            "critical": "critical",
        },
        "staff_availability": {
            "critical": "critical",
            "low": "high",
            "medium": "moderate",
            "high": "low",
        },
        "energy_overrun": {
            "low": "low",
            "normal": "moderate",
            "high": "high",
            "critical": "critical",
        },
    }

    def __init__(self):
        self.mf = MembershipFunctions()

    def _calculate_rule_strengths(self, inputs):
        activations = []

        for rule in RULES:
            degrees = [
                self.mf.membership(variable, term, inputs[variable])
                for variable, term in rule.conditions
            ]

            strength = (
                min(degrees)
                if rule.operator == "AND"
                else max(degrees)
            )
            strength *= rule.weight

            if strength > 0:
                activations.append({
                    "rule_id": rule.rule_id,
                    "strength": float(strength),
                    "consequent": rule.consequent,
                    "consequent_value": self.CONSEQUENT_VALUES[
                        rule.consequent
                    ],
                    "description": rule.description,
                })

        activations.sort(
            key=lambda item: item["strength"],
            reverse=True,
        )
        return activations

    def _predict_fallback(self, inputs):
        """
        Резервная оценка Сугено.

        Для каждого показателя выбирается терм с максимальной
        степенью принадлежности. Затем вычисляется взвешенное
        среднее постоянных значений категорий риска.
        """
        fallback_activations = []

        for name in self.INPUT_NAMES:
            memberships = self.mf.fuzzify(name, inputs[name])
            best_term = max(memberships, key=memberships.get)
            strength = float(memberships[best_term])

            if strength <= 0:
                raise ValueError(
                    f"Не удалось подобрать терм для переменной {name}."
                )

            consequent = self.FALLBACK_CONSEQUENTS[name][best_term]
            consequent_value = self.CONSEQUENT_VALUES[consequent]

            fallback_activations.append({
                "rule_id": f"FALLBACK_{name}_{best_term}",
                "strength": strength,
                "consequent": consequent,
                "consequent_value": consequent_value,
                "description": (
                    f"Резервное правило для {name}: "
                    f"терм «{best_term}» → риск «{consequent}»."
                ),
            })

        denominator = sum(
            item["strength"] for item in fallback_activations
        )

        if denominator <= 0:
            raise ValueError(
                "Не удалось рассчитать резервную оценку Сугено."
            )

        numerator = sum(
            item["strength"] * item["consequent_value"]
            for item in fallback_activations
        )

        score = numerator / denominator

        if not np.isfinite(score):
            raise ValueError(
                "Резервная модель Сугено вернула некорректный результат."
            )

        return float(np.clip(score, 0, 100))

    def predict(self, inputs):
        validated = self._validate_inputs(inputs)
        activations = self._calculate_rule_strengths(validated)

        if not activations:
            return self._predict_fallback(validated)

        numerator = sum(
            item["strength"] * item["consequent_value"]
            for item in activations
        )

        denominator = sum(
            item["strength"] for item in activations
        )

        if denominator <= 0:
            return self._predict_fallback(validated)

        score = numerator / denominator

        if not np.isfinite(score):
            raise ValueError(
                "Система Сугено вернула некорректный результат."
            )

        return float(np.clip(score, 0, 100))

    def explain(self, inputs):
        validated = self._validate_inputs(inputs)
        activations = self._calculate_rule_strengths(validated)

        if not activations:
            return self._build_fallback_explanation(validated)

        return activations[:5]

    def _build_fallback_explanation(self, inputs):
        """Возвращает объяснение резервной оценки Сугено."""
        result = []

        for name in self.INPUT_NAMES:
            memberships = self.mf.fuzzify(name, inputs[name])
            best_term = max(memberships, key=memberships.get)
            strength = float(memberships[best_term])
            consequent = self.FALLBACK_CONSEQUENTS[name][best_term]

            result.append({
                "rule_id": f"FALLBACK_{name}_{best_term}",
                "strength": strength,
                "consequent": consequent,
                "consequent_value": self.CONSEQUENT_VALUES[consequent],
                "description": (
                    f"Резервное правило для {name}: "
                    f"терм «{best_term}» → риск «{consequent}»."
                ),
            })

        result.sort(
            key=lambda item: item["strength"],
            reverse=True,
        )
        return result[:5]


if __name__ == "__main__":
    example = {
        "utilization": 56.19,
        "defect_rate": 5.34,
        "downtime_pct": 2.76,
        "overdue_orders": 3,
        "staff_availability": 72.25,
        "energy_overrun": -1.79,
    }

    systems = [
        ("Мамдани", MamdaniSystem()),
        ("Сугено", SugenoSystem()),
    ]

    print("Проверка нечётких систем оценки риска")
    print("=" * 50)

    for name, system in systems:
        try:
            score = system.predict(example)

            print(f"\nСистема: {name}")
            print(f"Оценка риска: {score:.2f} из 100")
            print(f"Категория: {system.classify(score)}")
            print("Объяснение:")

            for item in system.explain(example)[:3]:
                print(
                    f'  {item["rule_id"]}: {item["description"]}; '
                    f'сила = {item["strength"]:.3f}'
                )

        except (ValueError, TypeError, KeyError) as error:
            print(f"\n{name}: ошибка — {error}")
