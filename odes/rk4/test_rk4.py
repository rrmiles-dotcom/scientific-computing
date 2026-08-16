import numpy as np
import pytest

from rk4 import(
    rk4,
    rk4_final_value,
    rk4_error,
)

def test_constant_derivative():
    times, values = rk4(
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
    _, values = rk4(
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
    result = rk4_final_value(
        lambda t, y: y,
        initial_time = 0.0,
        initial_value = 1.0,
        final_time = 1.0,
        step = 0.1,
    )

    assert result == pytest.approx(
        np.e,
        abs = 3e-6,
    )

def test_smaller_step_improves_accuracy():
    exact = np.e

    coarse = rk4_final_value(
        lambda t, y: y,
        0.0,
        1.0,
        1.0,
        0.2,
    )

    fine = rk4_final_value(
        lambda t, y: y,
        0.0,
        1.0,
        1.0,
        0.05,
    )

    assert abs(fine - exact) < abs(coarse - exact)

def test_fourth_order_error_reduction():
    coarse = rk4_error(
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    fine = rk4_error(
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        0.05,
    )

    # Halving h should reduce fourth-order global error by about sixteen.

    ratio = coarse / fine

    assert ratio == pytest.approx(
        16.0,
        rel = 0.25,
    )

def test_rk4_is_more_accurate_than_rk2_for_same_step():
    step = 0.1

    rk4_result = rk4_final_value(
        lambda t, y: y,
        0.0,
        1.0,
        1.0,
        step,
    )

    value = 1.0

    for _ in range(10):
        k1 = value
        k2 = value + step * k1 / 2.0
        value += step * k2

    rk2_result = value
    exact = np.e

    assert (
        abs(rk4_result - exact)
        < abs(rk2_result - exact)
    )

def test_time_grid_is_correct():
    times, _ = rk4(
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
    _, values = rk4(
        lambda t, y: y,
        initial_time = 2.0,
        initial_value = 7.0,
        final_time = 3.0,
        step = 0.25,
    )

    assert values[0] == pytest.approx(7.0)

def test_final_value_matches_full_solution():
    function = lambda t, y: 2.0 * y

    _, values = rk4(
        function,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    result = rk4_final_value(
        function,
        0.0,
        1.0,
        1.0,
        0.1,
    )

    assert result == pytest.approx(values[-1])

def test_rk4_error_is_nonnegative():
    result = rk4_error(
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
        rk4(
            lambda t, y: y,
            initial_time = 1.0,
            initial_value = 1.0,
            final_time = 0.0,
            step = 0.1, 
        )

def test_rejects_equal_times():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            initial_time = 1.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.1,
        )

def test_rejects_zero_step():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.0,
        )

def test_rejects_negative_step():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = -0.1, 
        )

def test_rejects_step_that_does_not_divide_interval():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            initial_time = 0.0,
            initial_value = 1.0,
            final_time = 1.0,
            step = 0.3,
        )