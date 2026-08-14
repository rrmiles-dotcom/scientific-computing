import numpy as np
import pytest

from convergence_analysis import(
    absolute_errors,
    convergence_ratios,
    estimate_convergence_order,
    observed_reduction_factors,
)

def test_absolute_errors():
    approximations = np.array([
        1.0,
        1.4,
        1.41,
    ])

    result = absolute_errors(
        approximations,
        exact_value = np.sqrt(2.0),
    )

    expected = np.abs(
        approximations - np.sqrt(2.0)
    )

    np.testing.assert_allclose(result, expected)

def test_absolute_errors_rejects_non_vector():
    approximations = np.ones((2, 2))

    with pytest.raises(ValueError):
        absolute_errors(
            approximations,
            exact_value = 1.0,
        )

def test_absolute_errors_rejects_empty_input():
    approximations = np.array([])

    with pytest.raises(ValueError):
        absolute_errors(
            approximations,
            exact_value = 1.0,
        )

def test_linear_convergence_ratios():
    errors = np.array([
        0.5,
        0.25,
        0.125,
        0.0625,
    ])

    result = convergence_ratios(
        errors,
        order = 1.0,
    )

    expected = np.array([
        0.5,
        0.5,
        0.5,
    ])

    np.testing.assert_allclose(result, expected)

def test_quadratic_convergence_ratios():
    errors = np.array([
        0.1,
        0.01,
        0.0001,
    ])

    result = convergence_ratios(
        errors,
        order = 2.0,
    )

    expected = np.array([
        1.0,
        1.0,
    ])

    np.testing.assert_allclose(result, expected)

def test_convergence_ratios_rejects_too_few_errors():
    errors = np.array([0.1])

    with pytest.raises(ValueError):
        convergence_ratios(errors)

def test_convergence_ratios_rejects_nonpositive_order():
    errors = np.array([
        0.1,
        0.01,
    ])

    with pytest.raises(ValueError):
        convergence_ratios(
            errors,
            order = 0.0,
        )

def test_convergence_ratios_rejects_negative_errors():
    errors = np.array([
        0.1,
        -0.01,
    ])

    with pytest.raises(ValueError):
        convergence_ratios(errors)

def test_estimate_linear_convergence_order():
    errors = np.array([
        0.8,
        0.4,
        0.2,
        0.1,
    ])

    result = estimate_convergence_order(errors)

    np.testing.assert_allclose(
        result,
        np.ones(2),
        atol = 1e-12,
    )

def test_estimate_quadratic_convergence_order():
    errors = np.array([
        0.1,
        0.01,
        0.0001,
        1e-8,
    ])

    result = estimate_convergence_order(errors)

    np.testing.assert_allclose(
        result,
        np.full(2, 2.0),
        atol = 1e-12, 
    )

def test_estimate_order_rejects_too_few_errors():
    errors = np.array([
        0.1,
        0.01,
    ])

    with pytest.raises(ValueError):
        estimate_convergence_order(errors)

def test_estimate_order_rejects_zero_error():
    errors = np.array([
        0.1,
        0.01,
        0.0,
    ])

    with pytest.raises(ValueError):
        estimate_convergence_order(errors)

def test_estimate_order_rejects_unchanged_errors():
    errors = np.array([
        0.1,
        0.1,
        0.05,
    ])

    with pytest.raises(ValueError):
        estimate_convergence_order(errors)

def test_observed_reduction_factors():
    errors = np.array([
        1.0,
        0.5,
        0.25,
        0.125,
    ])

    result = observed_reduction_factors(errors)

    expected = np.array([
        0.5,
        0.5,
        0.5,
    ])

    np.testing.assert_allclose(result, expected)

def test_observed_reduction_rejects_zero_previous_error():
    errors = np.array([
        0.1,
        0.0,
        0.0,
    ])

    with pytest.raises(ValueError):
        observed_reduction_factors(errors)

def test_observed_reduction_rejects_negative_errors():
    errors = np.array([
        0.1,
        -0.01,
    ])

    with pytest.raises(ValueError):
        observed_reduction_factors(errors)