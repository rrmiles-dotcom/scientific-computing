import numpy as np
import pytest

from euler_method import(
    euler_method,
    euler_final_value,
    euler_error,
)

def test_constant_derivative():
    times, values = euler_method(
        lambda t, y: 2.0,
        initial_time = 0.0,
        initial_value = 1.0,
        final_time = 1.0,
        step = 0.1,
    )

    expected = 1.0 + 2.0 * times

    np.testing.assert_allclose(
        values,
        expected,
        atol = 1e-12,
    )

def test_zero_derivative_keeps_solution_constant():
    _, values = euler_method(
        lambda t, y: 0.0,
        initial_time = 0.0,
        initial_value = 5.0,
        final_time = 1.0,
        step = 0.2,
    )

    np.testing.assert_allclose(
        values,
        np.full(values.shape, 5.0),
        atol = 1e-12,
    )

def test_exponential_growth_approximation():
    result = euler_final_value(
        lambda t, y: y,
        initial_time = 0.0,
        initial_value = 1.0,
        final_time = 1.0,
        step = 0.001,
    )

    assert result == pytest.approx(
        np.e,
        abs = 2e-3,
    )

def test_smaller_step_improves_accuracy():
    exact = np.e

    coarse = euler_final_value(
        lambda t, y: y,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    fine = euler_final_value(
        lambda t, y: y,
        0.0,
        1.0,
        1.0,
        0.01,
    )

    assert abs(fine - exact) < abs(coarse - exact)

def test_first_order_error_reduction():
    coarse = euler_error(
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    fine = euler_error(
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        0.05,
    )

    # Halving h should reduce first-order global error by about two.

    ratio = coarse / fine

    assert ratio == pytest.approx(
        2.0,
        rel = 0.2,
    )

def test_time_grid_is_correct():
    times, _ = euler_method(
        lambda t, y: y,
        initial_time = 0.0,
        initial_value = 1.0,
        final_time = 1.0,
        step = 0.25,
    )

    expected = np.array([
        0.0,
        0.25,
        0.5,
        0.75,
        1.0,
    ])

    np.testing.assert_allclose(
        times,
        expected,
    )

def test_initial_value_is_preserved():
    _, values = euler_method(
        lambda t, y: y,
        initial_time = 2.0,
        initial_value = 7.0,
        final_time = 3.0,
        step = 0.25, 
    )

    assert values[0] == pytest.approx(7.0)

def test_final_value_matches_full_solution():
    function = lambda t, y: -2.0 * y

    _, values = euler_method(
        function,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    result = euler_final_value(
        function,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    assert result == pytest.approx(values[-1])

def test_euler_error_is_nonnegative():
    result = euler_error(
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    assert result >= 0.0

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        euler_method(
            lambda t, y: y,
            initial_time = 1.0,
            initial_value = 1.0,
            final_time = 0.0,
            step = 0.1,
        )

def test_rejects_equal_time():
    with pytest.raises(ValueError):
        euler_method(
            lambda t, y: y,
            initial_time = 1.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.1,
        )

def test_rejects_zero_step():
    with pytest.raises(ValueError):
        euler_method(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.0,
        )

def test_rejects_negative_step():
    with pytest.raises(ValueError):
        euler_method(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = -0.1,
        )

def test_rejects_step_that_does_not_divide_interval():
    with pytest.raises(ValueError):
        euler_method(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.3,
        )