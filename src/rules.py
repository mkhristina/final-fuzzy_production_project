from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    rule_id: str
    conditions: tuple
    operator: str
    consequent: str
    description: str
    weight: float = 1.0


RULES = [
    Rule("R01", (("utilization", "critical"),), "AND", "critical",
         "Критическая загрузка цеха"),
    Rule("R02", (("defect_rate", "critical"),), "AND", "critical",
         "Критический уровень брака"),
    Rule("R03", (("downtime_pct", "critical"),), "AND", "critical",
         "Критический уровень простоев"),
    Rule("R04", (("overdue_orders", "critical"),), "AND", "critical",
         "Критическое количество просроченных заказов"),
    Rule("R05", (("staff_availability", "critical"),), "AND", "high",
         "Критическая нехватка персонала"),
    Rule("R06", (("energy_overrun", "critical"),), "AND", "high",
         "Критический перерасход энергии"),

    Rule("R07", (("utilization", "high"), ("downtime_pct", "high")),
         "AND", "critical", "Высокая загрузка и высокий уровень простоев"),
    Rule("R08", (("defect_rate", "high"), ("downtime_pct", "high")),
         "AND", "high", "Высокий уровень брака и простоев"),
    Rule("R09", (("overdue_orders", "high"), ("staff_availability", "critical")),
         "AND", "critical", "Просроченные заказы и нехватка персонала"),
    Rule("R10", (("utilization", "high"), ("energy_overrun", "high")),
         "AND", "high", "Высокая загрузка и перерасход энергии"),
    Rule("R11", (("defect_rate", "high"), ("overdue_orders", "high")),
         "AND", "high", "Высокий уровень брака и просроченных заказов"),
    Rule("R12", (("utilization", "low"), ("defect_rate", "low"),
                 ("downtime_pct", "low")),
         "AND", "moderate", "Низкая загрузка при малом количестве брака и простоев"),
    Rule("R13", (("utilization", "normal"), ("defect_rate", "low"),
                 ("downtime_pct", "low")),
         "AND", "low", "Нормальная загрузка, низкий брак и малые простои"),
    Rule("R14", (("utilization", "normal"), ("staff_availability", "high"),
                 ("energy_overrun", "low")),
         "AND", "low", "Нормальная работа с достаточным персоналом"),
    Rule("R15", (("utilization", "low"), ("staff_availability", "high")),
         "AND", "moderate", "Низкая загрузка при достаточном персонале"),
    Rule("R16", (("utilization", "high"), ("defect_rate", "high")),
         "AND", "critical", "Высокая загрузка и высокий уровень брака"),
    Rule("R17", (("downtime_pct", "high"), ("overdue_orders", "high")),
         "AND", "high", "Высокие простои и просроченные заказы"),
    Rule("R18", (("staff_availability", "critical"), ("energy_overrun", "high")),
         "AND", "high", "Нехватка персонала и перерасход энергии"),
    Rule("R19", (("defect_rate", "critical"), ("overdue_orders", "high")),
         "OR", "critical", "Критический брак или много просроченных заказов"),
    Rule("R20", (("utilization", "critical"), ("downtime_pct", "critical")),
         "OR", "critical", "Критическая загрузка или критические простои"),

    # Дополнительные правила для средних условий производства
    Rule("R21", (("defect_rate", "medium"), ("downtime_pct", "medium")),
         "AND", "moderate", "Средний уровень брака и простоев"),
    Rule("R22", (("defect_rate", "medium"), ("overdue_orders", "medium")),
         "AND", "moderate", "Средний уровень брака и просроченных заказов"),
    Rule("R23", (("downtime_pct", "medium"), ("overdue_orders", "medium")),
         "AND", "moderate", "Средние простои и просроченные заказы"),
    Rule("R24", (("utilization", "normal"), ("defect_rate", "medium")),
         "AND", "moderate", "Нормальная загрузка при среднем уровне брака"),
    Rule("R25", (("utilization", "normal"), ("downtime_pct", "medium")),
         "AND", "moderate", "Нормальная загрузка при среднем уровне простоев"),
    Rule("R26", (("staff_availability", "medium"), ("energy_overrun", "normal")),
         "AND", "moderate", "Средняя доступность персонала при нормальной энергии"),

    # Правила для дополнительных сочетаний входных факторов
    Rule("R27", (("utilization", "normal"), ("staff_availability", "medium")),
         "AND", "moderate", "Нормальная загрузка при средней доступности персонала"),
    Rule("R28", (("defect_rate", "low"), ("downtime_pct", "medium")),
         "AND", "low", "Низкий брак при среднем уровне простоев"),
    Rule("R29", (("defect_rate", "medium"), ("staff_availability", "high")),
         "AND", "moderate", "Средний уровень брака при достаточном персонале"),
    Rule("R30", (("overdue_orders", "low"), ("staff_availability", "high")),
         "AND", "low", "Мало просроченных заказов и достаточно персонала"),
]


def get_rules():
    """Возвращает список правил нечёткого вывода."""
    return list(RULES)


def validate_rules():
    """Проверяет структуру и корректность базы правил."""
    if len(RULES) < 18:
        raise ValueError("База должна содержать не менее 18 правил.")

    rule_ids = [rule.rule_id for rule in RULES]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError("Идентификаторы правил должны быть уникальными.")

    valid_variables = {
        "utilization",
        "defect_rate",
        "downtime_pct",
        "overdue_orders",
        "staff_availability",
        "energy_overrun",
    }
    valid_terms = {"low", "normal", "medium", "high", "critical"}
    valid_consequents = {"low", "moderate", "high", "critical"}

    for rule in RULES:
        if rule.operator not in {"AND", "OR"}:
            raise ValueError(f"Недопустимый оператор в {rule.rule_id}.")

        if not rule.conditions:
            raise ValueError(f"У правила {rule.rule_id} нет условий.")

        if rule.consequent not in valid_consequents:
            raise ValueError(f"Недопустимый выход в {rule.rule_id}.")

        if not 0 < rule.weight <= 1:
            raise ValueError(f"Вес правила {rule.rule_id} должен быть в (0, 1].")

        for variable, term in rule.conditions:
            if variable not in valid_variables:
                raise ValueError(
                    f"Неизвестная входная переменная {variable} в {rule.rule_id}."
                )
            if term not in valid_terms:
                raise ValueError(
                    f"Неизвестный терм {term} в правиле {rule.rule_id}."
                )

    return True


if __name__ == "__main__":
    validate_rules()
    print(f"Количество правил: {len(RULES)}")
    print("Проверка базы правил пройдена.")

    for rule in RULES:
        print(
            f"{rule.rule_id}: {rule.description} "
            f"-> {rule.consequent} ({rule.operator})"
        )
