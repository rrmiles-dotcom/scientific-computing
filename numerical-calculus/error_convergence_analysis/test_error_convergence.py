import numpy as np
import pytest

from error_convergence import(
    absolute_errors,
    estimate_order,
    error_reduction_factors,
    expected_reduction_factor,
    asymptotic_constant,
)

def test_absolute_errors():
    approximations = np.array([
        1.1,
        0.9,
        1.01,
    ])

    result = absolute_errors(
        approximations,
        exact_value = 1.0,
    )

    expected = np.array([
        0.1,
        0.1,
        0.01,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_estimate_first_order_convergence():
    step_sizes = np.array([
        0.4,
        0.2,
        0.1,
        0.05,
    ])

    errors = 3.0 * step_sizes

    result = estimate_order(
        step_sizes,
        errors,
    )

    np.testing.assert_allclose(
        result,
        np.ones(3),
        atol = 1e-12,
    )

def test_estimate_second_order_convergence():
    step_sizes = np.array([
        0.4,
        0.2,
        0.1,
        0.05,
    ])

    errors = 2.0 * step_sizes**2

    result = estimate_order(
        step_sizes,
        errors,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, 2.0),
        atol = 1e-12,
    )

def test_estimate_fourth_order_convergence():
    step_sizes = np.array([
        0.4,
        0.2,
        0.1,
        0.05,
    ])

    errors = 0.5 * step_sizes**4

    result = estimate_order(
        step_sizes,
        errors,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, 4.0),
        atol = 1e-12,
    )

def test_error_reduction_factors():
    errors = np.array([
        0.16,
        0.04,
        0.01,
    ])

    result = error_reduction_factors(errors)

    np.testing.assert_allclose(
        result,
        np.array([4.0, 4.0])
    )

def test_expected_reduction_factor_first_order():
    result = expected_reduction_factor(
        refinement_factor = 2.0,
        order = 1.0,
    )

    assert result == pytest.approx(2.0)

def test_expected_reduction_factor_second_order():
    result = expected_reduction_factor(
        refinement_factor = 2.0,
        order = 2.0,
    )

    assert result == pytest.approx(4.0)

def test_expected_reduction_factor_fourth_order():
    result = expected_reduction_factor(
        refinement_factor = 2.0,
        order = 4.0,
    )

    assert result == pytest.approx(16.0)

def test_asymptotic_constant():
    step_sizes = np.array([
        0.4,
        0.2,
        0.1,
    ])

    errors = 3.0 * step_sizes**2

    result = asymptotic_constant(
        step_sizes,
        errors,
        order = 2.0,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, 3.0),
        atol = 1e-12,
    )


def test_absolute_errors_reject_non_vector_input():
    approximations = np.ones((2, 2))

    with pytest.raises(ValueError):
        absolute_errors(
            approximations,
            exact_value = 1.0,
        )

def test_absolute_errors_reject_empty_input():
    approximations = np.array([])

    with pytest.raises(ValueError):
        absolute_errors(
            approximations,
            exact_value = 1.0,
        )

def test_estimate_order_rejects_length_mismatch():
    step_sizes = np.ones(4)
    errors = np.ones(3)

    with pytest.raises(ValueError):
        estimate_order(
            step_sizes,
            errors,
        )

def test_estimate_order_rejects_nonpositive_steps():
    step_sizes = np.array([
        0.5,
        0.0,
        0.1,
    ])

    errors = np.array([
        0.25,
        0.04,
        0.01,
    ])

    with pytest.raises(ValueError):
        estimate_order(
            step_sizes,
            errors,
        )

def test_estimate_order_rejects_nonpositive_errors():
    step_sizes = np.array([
        0.5,
        0.25,
        0.125,
    ])

    errors = np.array([
        0.25,
        0.0,
        0.01,
    ])

    with pytest.raises(ValueError):
        estimate_order(
            step_sizes,
            errors,
        )

def test_error_reduction_rejects_zero_previous_error():
    errors = np.array([
        0.1,
        0.0,
        0.0,
    ])

    with pytest.raises(ValueError):
        error_reduction_factors(errors)

def test_expected_reduction_rejects_invalid_refinement():
    with pytest.raises(ValueError):
        expected_reduction_factor(
            refinement_factor = 1.0,
            order = 2.0,
        )

def test_expected_reduction_rejects_nonpositive_order():
    with pytest.raises(ValueError):
        expected_reduction_factor(
            refinement_factor = 2.0,
            order = 0.0,
        )

def test_asymptotic_constant_rejects_length_mismatch():
    step_sizes = np.ones(3)
    errors = np.ones(2)

    with pytest.raises(ValueError):
        asymptotic_constant(
            step_sizes,
            errors,
            order = 2.0,
        )

def test_asymptotic_constant_rejects_nonpositive_order():
    step_sizes = np.array([
        0.5,
        0.25,
    ])

    errors = np.array([
        0.1,
        0.025,
    ])

    with pytest.raises(ValueError):
        asymptotic_constant(
            step_sizes,
            errors,
            order = 0.0,
        )