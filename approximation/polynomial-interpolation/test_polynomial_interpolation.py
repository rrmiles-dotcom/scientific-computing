import numpy as np
import pytest

from polynomial_interpolation import(
    vandermonde_matrix,
    polynomial_coefficients,
    evaluate_polynomial,
    interpolate,
)

def test_vandermonde_matrix():
    x_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = vandermonde_matrix(x_values)

    expected = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 2.0, 4.0],
        [1.0, 3.0, 9.0],
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_coefficients_for_known_quadratic():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        6.0,
        17.0,
    ])

    coefficients = polynomial_coefficients(
        x_values,
        y_values,
    )

    expected = np.array([
        1.0,
        2.0,
        3.0,
    ])

    np.testing.assert_allclose(
        coefficients,
        expected,
        atol = 1e-12,
    )

def test_interpolation_reproduces_nodes():
    x_values = np.array([
        -1.0,
        0.0,
        2.0,
    ])

    y_values = np.array([
        4.0,
        1.0,
        3.0,
    ])

    result = interpolate(
        x_values,
        y_values,
        x_values,
    )

    np.testing.assert_allclose(
        result,
        y_values,
        atol = 1e-12,
    )

def test_evaluate_scalar():
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

def test_evaluate_vector():
    coefficients = np.array([
        1.0,
        1.0,
    ])

    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    result = evaluate_polynomial(
        coefficients,
        x_values,
    )

    expected = np.array([
        1.0,
        2.0,
        3.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_linear_interpolation():
    x_values = np.array([
        0.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        5.0,
    ])

    result = interpolate(
        x_values,
        y_values,
        1.0,
    )

    assert result == pytest.approx(3.0)

def test_constant_polynomial():
    x_values = np.array([
        4.0,
    ])

    y_values = np.array([
        7.0,
    ])

    result = interpolate(
        x_values,
        y_values,
        100.0,
    )

    assert result == pytest.approx(7.0)

def test_matches_numpy_polynomial_evaluation():
    x_values = np.array([
        -2.0,
        -1.0,
        1.0,
        3.0,
    ])

    y_values = np.array([
        5.0,
        2.0,
        4.0,
        10.0,
    ])

    coefficients = polynomial_coefficients(
        x_values,
        y_values,
    )

    points = np.linspace(
        -2.0,
        3.0,
        20,
    )

    expected = np.polynomial.polynomial.polyval(
        points,
        coefficients,
    )

    result = evaluate_polynomial(
        coefficients,
        points,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_rejects_duplicate_nodes():
    x_values = np.array([
        1.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        2.0,
        3.0,
        4.0,
    ])

    with pytest.raises(ValueError):
        polynomial_coefficients(
            x_values,
            y_values,
        )

def test_rejects_length_mismatch():
    x_values = np.ones(3)
    y_values = np.ones(2)

    with pytest.raises(ValueError):
        polynomial_coefficients(
            x_values,
            y_values,
        )

def test_rejects_non_vector_nodes():
    x_values = np.ones((2, 2))

    with pytest.raises(ValueError):
        vandermonde_matrix(
            x_values,
        )

def test_rejects_empty_nodes():
    x_values = np.array([])

    with pytest.raises(ValueError):
        vandermonde_matrix(
            x_values,
        )

def test_rejects_non_vector_coefficients():
    coefficients = np.ones((2, 2))

    with pytest.raises(ValueError):
        evaluate_polynomial(
            coefficients,
            1.0,
        )