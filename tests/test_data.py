
import numpy as np
import pandas as pd
import pytest

from src.data_pipeline import (
    RANGES,
    generate_production_data,
    validate_production_data,
)


def test_data_generation_is_reproducible():
    """Одинаковый seed должен давать одинаковые данные."""
    df1 = generate_production_data(2000, random_seed=42)
    df2 = generate_production_data(2000, random_seed=42)

    pd.testing.assert_frame_equal(df1, df2)


def test_minimum_number_of_observations():
    """Набор содержит не менее 1500 наблюдений."""
    df = generate_production_data(2000, random_seed=42)

    assert len(df) >= 1500


def test_all_values_are_in_allowed_ranges():
    """Все числовые показатели находятся в заданных диапазонах."""
    df = generate_production_data(2000, random_seed=42)

    validate_production_data(df)

    for column, (minimum, maximum) in RANGES.items():
        assert df[column].between(minimum, maximum).all()


def test_no_missing_values():
    """В созданных данных нет пропусков."""
    df = generate_production_data(2000, random_seed=42)

    assert not df.isnull().any().any()


def test_overdue_orders_are_integers():
    """Число просроченных заказов должно быть целым."""
    df = generate_production_data(2000, random_seed=42)

    assert pd.api.types.is_integer_dtype(df["overdue_orders"])


def test_invalid_sample_count():
    """Нельзя создать ноль или отрицательное количество строк."""
    with pytest.raises(ValueError):
        generate_production_data(0)

    with pytest.raises(ValueError):
        generate_production_data(-10)


def test_validator_rejects_out_of_range_values():
    """Валидатор должен отклонять недопустимые значения."""
    df = generate_production_data(10, random_seed=42)
    df.loc[0, "utilization"] = 150

    with pytest.raises(ValueError):
        validate_production_data(df)


def test_validator_rejects_missing_values():
    """Валидатор должен обнаруживать пропуски."""
    df = generate_production_data(10, random_seed=42)
    df.loc[0, "defect_rate"] = np.nan

    with pytest.raises(ValueError):
        validate_production_data(df)
