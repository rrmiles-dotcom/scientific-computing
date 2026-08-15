import numpy as np
import pytest

from simpson_rule import(
    simpsons_rule,
    simpsons_from_samples,
    simpsons_error,
)

def test_integrates_constant_exactly():
    result = simpsons_rule(
        lambda x: 5.0,
        0.0,
        3.0,
        intervals = 4,
    )

    assert result == pytest.approx(15.0)

def test_integrates_linear_exactly():
    result = simpsons_rule(
        lambda x: 2.0 * x + 1.0,
        0.0,
        2.0,
        intervals = 4,
    )

    assert result == pytest.approx(6.0)

def test_integrates_quadratic_exactly():
    result = simpsons_rule(
        lambda x: x**2,
        0.0,
        1.0,
        intervals = 4, 
    )

    assert result == pytest.approx(
        1.0 / 3.0,
        abs = 1e-12,
    )

def test_integrates_cubic_exactly():
    result = simpsons_rule(
        lambda x: x**3,
        0.0,
        2.0,
        intervals = 4,
    )

    assert result == pytest.approx(
        4.0,
        abs = 1e-12,
    )

def test_integrates_sine():
    result = simpsons_rule(
        np.sin,
        0.0,
        np.pi,
        intervals = 200,
    )

    assert result == pytest.approx(
        2.0,
        abs = 1e-8,
    )

def test_fourth_order_error_reduction():
    exact = np.e - 1.0

    coarse = simpsons_error(
        np.exp,
        exact,
        0.0,
        1.0,
        intervals = 10,
    )

    fine = simpsons_error(
        np.exp,
        exact,
        0.0,
        1.0,
        intervals = 20,
    )

    # Halving h should reduce O(h^4) error by approximately sixteen.
    assert fine < coarse / 14.0

def test_from_samples_quadratic():
    x_values = np.linspace(
        0.0,
        2.0,
        5,
    )

    y_values = x_values**2

    result = simpsons_from_samples(
        x_values,
        y_values,
    )

    assert result == pytest.approx(
        8.0 / 3.0,
        abs = 1e-12,
    )

def test_from_samples_cubic():
    x_values = np.linspace(
        0.0,
        2.0,
        5,
    )

    y_values = x_values**3

    result = simpsons_from_samples(
        x_values,
        y_values,
    )

    assert result == pytest.approx(
        4.0,
        abs = 1e-12,
    )

def test_error_is_nonnegative():
    result = simpsons_error(
        np.exp,
        np.e - 1.0,
        0.0,
        1.0,
        intervals = 10,
    )

    assert result >= 0.0

def test_rejects_odd_number_of_intervals():
    with pytest.raises(ValueError):
        simpsons_rule(
            lambda x: x**2,
            0.0,
            1.0,
            intervals = 3,
        )

def test_rejects_nonpositive_intervals():
    with pytest.raises(ValueError):
        simpsons_rule(
            lambda x: x,
            0.0,
            1.0,
            intervals = 0,
        )

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        simpsons_rule(
            lambda x: x,
            2.0,
            1.0,
            intervals = 4,
        )

def test_samples_reject_non_vector_input():
    x_values = np.ones((2, 2))
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        simpsons_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_length_mismatch():
    x_values = np.ones(5)
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        simpsons_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_too_few_points():
    x_values = np.array([
        0.0,
        1.0,
    ])

    y_values = np.array([
        0.0,
        1.0,
    ])

    with pytest.raises(ValueError):
        simpsons_from_samples(
            x_values,
            y_values,
        )

def test_rejects_reject_odd_number_of_intervals():
    x_values = np.linspace(
        0.0,
        1.0,
        4,
    )

    y_values = x_values**2

    with pytest.raises(ValueError):
        simpsons_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_nonuniform_spacing():
    x_values = np.array([
        0.0,
        0.5,
        1.5,
        2.0,
        3.0,
    ])

    y_values = x_values**2

    with pytest.raises(ValueError):
        simpsons_from_samples(
            x_values,
            y_values,
        )