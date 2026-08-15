import numpy as np
import pytest

from least_squares_approximation import(
    design_matrix,
    polynomial_least_squares,
    evaluate_polynomial,
    residuals,
    sum_squared_errors,
)

def test_design_matrix_linear():
    x_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = design_matrix(
        x_values,
        degree = 1,
    )

    expected = np.array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_design_matrix_quadratic():
    x_values = np.array([
        0.0,
        2.0,
    ])

    result = design_matrix(
        x_values,
        degree = 2,
    )

    expected = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 2.0, 4.0],
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_exact_linear_fit():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = 2.0 + 3.0 * x_values

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 1,
    )

    np.testing.assert_allclose(
        coefficients,
        np.array([2.0, 3.0]),
        atol = 1e-12,
    )

def test_exact_quadratic_fit():
    x_values = np.array([
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ])

    y_values = (
        1.0
        - 2.0 * x_values
        + 0.5 * x_values**2
    )

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 2,
    )

    np.testing.assert_allclose(
        coefficients,
        np.array([1.0, -2.0, 0.5]),
        atol = 1e-12,
    )

def test_noisy_linear_fit_matches_numpy():
    rng = np.random.default_rng(42)

    x_values = np.linspace(
        -2.0,
        2.0,
        20,
    )

    y_values = (
        1.5
        + 2.0 * x_values
        + rng.normal(scale = 0.1, size = x_values.size)
    )

    result = polynomial_least_squares(
        x_values,
        y_values,
        degree = 1,
    )

    matrix = design_matrix(
        x_values,
        degree = 1,
    )

    expected, _, _, _ = np.linalg.lstsq(
        matrix,
        y_values,
        rcond = None,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_constant_fit_returns_mean():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        2.0,
        4.0,
        6.0,
        8.0,
    ])

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 0,
    )

    assert coefficients[0] == pytest.approx(
        np.mean(y_values)
    )

def test_evaluate_polynomial_scalar():
    coefficients = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = evaluate_polynomial(
        coefficients,
        2.0,
    )

    assert result == pytest.approx(17.0)

def test_evaluate_polynomial_vector():
    coefficients = np.array([
        1.0,
        2.0,
    ])

    points = np.array([
        0.0,
        1.0,
        2.0,
    ])

    result = evaluate_polynomial(
        coefficients,
        points,
    )

    expected = np.array([
        1.0,
        3.0,
        5.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_residuals_exact_fit_are_zero():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = 1.0 + 2.0 * x_values

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 1,
    )

    result = residuals(
        x_values,
        y_values,
        coefficients,
    )

    np.testing.assert_allclose(
        result,
        np.zeros(x_values.size),
        atol = 1e-12,
    )

def test_sum_squared_errors_exact_fit_is_zero():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = 4.0 - x_values

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 1,
    )

    result = sum_squared_errors(
        x_values,
        y_values,
        coefficients,
    )

    assert result == pytest.approx(
        0.0,
        abs = 1e-12,
    )

def test_sum_squared_errors_is_nonnegative():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        2.0,
        4.0,
    ])

    coefficients = polynomial_least_squares(
        x_values,
        y_values,
        degree = 1,
    )

    result = sum_squared_errors(
        x_values,
        y_values,
        coefficients,
    )

    assert result >= 0.0

def test_rejects_length_mismatch():
    x_values = np.ones(4)
    y_values = np.ones(3)

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = 1,
        )

def test_rejects_non_vector_data():
    x_values = np.ones((2, 2))
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = 1,
        )

def test_rejects_empty_data():
    x_values = np.array([])
    y_values = np.array([])

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = 0,
        )

def test_rejects_negative_degree():
    x_values = np.ones(3)
    y_values = np.ones(3)

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = -1,
        )

def test_rejects_degree_too_large():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = 3,
        )

def test_rejects_rank_deficient_design_matrix():
    x_values = np.array([
        1.0,
        1.0,
        1.0,
        1.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    with pytest.raises(ValueError):
        polynomial_least_squares(
            x_values,
            y_values,
            degree = 1,
        )

def test_evaluate_rejects_non_vector_coefficients():
    coefficients = np.ones((2, 2))

    with pytest.raises(ValueError):
        evaluate_polynomial(
            coefficients,
            1.0,
        )

def test_evaluate_rejects_empty_coefficients():
    coefficients = np.array([])

    with pytest.raises(ValueError):
        evaluate_polynomial(
            coefficients,
            1.0,
        )