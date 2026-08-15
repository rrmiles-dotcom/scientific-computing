import numpy as np
import pytest

from trapezodial_rule import(
    trapezoidal_rule,
    trapezoidal_from_samples,
    trapezoidal_error,
)

def test_integrates_constant_function_exactly():
    result = trapezoidal_rule(
        lambda x: 5.0,
        0.0,
        3.0,
        intervals = 4,
    )

    assert result == pytest.approx(15.0)

def test_integrates_linear_function_exactly():
    result = trapezoidal_rule(
        lambda x: 2.0 * x + 1.0,
        0.0,
        2.0,
        intervals = 5,
    )

    assert result == pytest.approx(6.0)

def test_quadratic_function_approximates_integral():
    result = trapezoidal_rule(
        lambda x: x**2,
        0.0,
        1.0,
        intervals = 100,
    )

    assert result == pytest.approx(
        1.0 / 3.0,
        abs = 1e-4,
    )

def test_more_intervals_improve_accuracy():
    exact = 1.0 / 3.0

    coarse = trapezoidal_rule(
        lambda x: x**2,
        0.0,
        1.0,
        intervals = 10,
    )

    fine = trapezoidal_rule(
        lambda x: x**2,
        0.0,
        1.0,
        intervals = 100,
    )

    assert abs(fine - exact) < abs(coarse - exact)

def test_expected_second_order_error_reduction():
    exact = 1.0 / 3.0

    coarse_error = trapezoidal_error(
        lambda x: x**2,
        exact,
        0.0,
        1.0,
        intervals = 10,
    )

    fine_error = trapezoidal_error(
        lambda x: x**2,
        exact,
        0.0,
        1.0,
        intervals = 20,
    )

    # Halving h should reduce O(h^2) error by approximately four.
    assert fine_error < coarse_error / 3.5

def test_integrates_sine():
    result = trapezoidal_rule(
        np.sin,
        0.0,
        np.pi,
        intervals = 1000,
    )

    assert result == pytest.approx(
        2.0,
        abs = 1e-5,
    )

def test_from_samples_linear_data():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        3.0,
        5.0,
    ])

    result = trapezoidal_from_samples(
        x_values,
        y_values,
    )

    assert result == pytest.approx(6.0)

def test_from_samples_supports_nonunifrom_spacing():
    x_values = np.array([
        0.0,
        1.0,
        3.0,
    ])

    y_values = np.array([
        0.0,
        1.0,
        3.0,
    ])

    result = trapezoidal_from_samples(
        x_values,
        y_values,
    )

    assert result == pytest.approx(4.5)

def test_trapezoidal_error():
    result = trapezoidal_error(
        lambda x: x**2,
        exact_integral = 1.0 / 3.0,
        left = 0.0,
        right = 1.0,
        intervals = 10,
    )

    assert result > 0.0

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        trapezoidal_rule(
            lambda x: x,
            2.0,
            1.0,
            intervals = 10,
        )

def test_rejects_equal_endpoints():
    with pytest.raises(ValueError):
        trapezoidal_rule(
            lambda x: x,
            1.0,
            1.0,
            intervals = 10,
        )

def test_rejects_nonpositive_intervals():
    with pytest.raises(ValueError):
        trapezoidal_rule(
            lambda x: x,
            0.0,
            1.0,
            intervals = 0,
        )

def test_samples_reject_non_vector_input():
    x_values = np.ones((2, 2))
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        trapezoidal_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_too_few_points():
    x_values = np.array([0.0])
    y_values = np.array([1.0])

    with pytest.raises(ValueError):
        trapezoidal_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_length_mismatch():
    x_values = np.ones(3)
    y_values = np.ones(2)

    with pytest.raises(ValueError):
        trapezoidal_from_samples(
            x_values,
            y_values,
        )

def test_samples_reject_unsorted_points():
    x_values = np.array([
        0.0,
        2.0,
        1.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        trapezoidal_from_samples(
            x_values,
            y_values,
        )