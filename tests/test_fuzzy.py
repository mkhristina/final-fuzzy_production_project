
import numpy as np
import pytest

from src.membership import MembershipFunctions


@pytest.fixture
def mf():
    return MembershipFunctions()


def test_membership_is_between_zero_and_one(mf):
    result = mf.membership("utilization", "normal", 80)

    assert 0 <= result <= 1


def test_triangle_peak(mf):
    result = mf.membership("utilization", "normal", 80)

    assert result == pytest.approx(1.0)


def test_trapezoid_plateau(mf):
    result = mf.membership("utilization", "low", 40)

    assert result == pytest.approx(1.0)


def test_all_input_variables_have_at_least_three_terms(mf):
    input_variables = [
        "utilization",
        "defect_rate",
        "downtime_pct",
        "overdue_orders",
        "staff_availability",
        "energy_overrun",
    ]

    for variable in input_variables:
        assert len(mf.terms[variable]) >= 3


def test_output_has_four_risk_categories(mf):
    assert len(mf.terms["risk_score"]) >= 4


def test_invalid_input_is_rejected(mf):
    with pytest.raises(ValueError):
        mf.membership("utilization", "normal", 150)


def test_nan_is_rejected(mf):
    with pytest.raises(ValueError):
        mf.membership("defect_rate", "low", np.nan)


def test_invalid_term_is_rejected(mf):
    with pytest.raises(KeyError):
        mf.membership("utilization", "unknown", 80)



from src.rules import RULES, validate_rules


def test_rule_count():
    assert len(RULES) >= 18


def test_rule_ids_are_unique():
    ids = [rule.rule_id for rule in RULES]

    assert len(ids) == len(set(ids))


def test_rule_ids_have_expected_format():
    for rule in RULES:
        assert rule.rule_id.startswith("R")
        assert len(rule.rule_id) == 3
        assert rule.rule_id[1:].isdigit()


def test_rules_are_valid():
    assert validate_rules() is True


def test_rules_cover_all_risk_categories():
    categories = {rule.consequent for rule in RULES}

    assert categories == {"low", "moderate", "high", "critical"}


def test_rules_use_and_and_or():
    operators = {rule.operator for rule in RULES}

    assert operators == {"AND", "OR"}



from src.fuzzy_systems import FuzzySystem, MamdaniSystem


@pytest.fixture
def mamdani():
    return MamdaniSystem()


def test_mamdani_returns_valid_score(mamdani):
    inputs = {
        "utilization": 85,
        "defect_rate": 4,
        "downtime_pct": 7,
        "overdue_orders": 8,
        "staff_availability": 90,
        "energy_overrun": 5,
    }

    score = mamdani.predict(inputs)

    assert np.isfinite(score)
    assert 0 <= score <= 100


def test_classification_boundaries():
    assert FuzzySystem.classify(10) == "Низкий"
    assert FuzzySystem.classify(30) == "Умеренный"
    assert FuzzySystem.classify(60) == "Высокий"
    assert FuzzySystem.classify(90) == "Критический"


def test_mamdani_rejects_out_of_range_input(mamdani):
    inputs = {
        "utilization": 150,
        "defect_rate": 4,
        "downtime_pct": 7,
        "overdue_orders": 8,
        "staff_availability": 90,
        "energy_overrun": 5,
    }

    with pytest.raises(ValueError):
        mamdani.predict(inputs)


def test_mamdani_explanation_contains_rule_ids(mamdani):
    inputs = {
        "utilization": 110,
        "defect_rate": 12,
        "downtime_pct": 25,
        "overdue_orders": 30,
        "staff_availability": 65,
        "energy_overrun": 25,
    }

    explanation = mamdani.explain(inputs)

    assert isinstance(explanation, list)
    assert len(explanation) > 0

    for item in explanation:
        assert item["rule_id"].startswith("R")
        assert 0 <= item["strength"] <= 1
