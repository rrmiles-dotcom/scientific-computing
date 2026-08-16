import numpy as np
import pytest

from convergence import (
    final_errors,
    estimate_orders,
    reduction_factors,
    expected_reduction,
    asymptotic_constants,
)

def euler_solver(
    function,
    initial_time,
    initial_value,
    final_time,
    step,
):
    intervals = int(round(
        (final_time - initial_time) / step
    ))

    times = np.linspace(
        initial_time,
        final_time,
        intervals + 1,
    )

    values = np.zeros(
        intervals + 1,
        dtype = float,
    )

    values[0] = initial_value

    for i in range(intervals):
        values[i + 1] = (
            values[i]
            + step
            * function(
                times[i],
                values[i]
            )
        )

    return times, values

def rk2_solver(
    function,
    initial_time,
    initial_value,
    final_time,
    step,
):
    intervals = int(round(
        (final_time - initial_time) / step
    ))

    times = np.linspace(
        initial_time,
        final_time,
        intervals + 1,
    )

    values = np.zeros(
        intervals + 1,
        dtype = float
    )

    values[0] = initial_value

    for i in range(intervals):
        time = times[i]
        value = values[i]

        k1 = function(
            time,
            value,
        )

        k2 = function(
            time + step / 2.0,
            value + step * k1 / 2.0,
        )

        values[i + 1] = (
            value
            + step * k2
        )

    return times, values

def rk4_solver(
    function,
    initial_time,
    initial_value,
    final_time,
    step,
):
    intervals = int(round(
        (final_time - initial_time) / step
    ))

    times = np.linspace(
        initial_time,
        final_time,
        intervals + 1,
    )

    values = np.zeros(
        intervals + 1,
        dtype = float
    )

    values[0] = initial_value

    for i in range(intervals):
        time = times[i]
        value = values[i]

        k1 = function(
            time, 
            value,
        )

        k2 = function(
            time + step / 2.0,
            value + step * k1 / 2.0,
        )

        k3 = function(
            time + step / 2.0,
            value + step * k2 / 2.0,
        )

        k4 = function(
            time + step,
            value + step * k3,
        )

        values[i + 1] = (
            value
            + step
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )

    return times, values

def test_final_errors_are_positive():
    step_sizes = np.array([
        0.2,
        0.1,
        0.05,
    ])

    errors = final_errors(
        euler_solver,
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        step_sizes,
    )

    assert np.all(errors > 0.0)

def test_euler_first_order_convergence():
    step_sizes = np.array([
        0.1,
        0.05,
        0.025,
        0.0125,
    ])

    errors = final_errors(
        euler_solver,
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        step_sizes,
    )

    orders = estimate_orders(
        step_sizes,
        errors,
    )

    assert orders[-1] == pytest.approx(
        1.0,
        rel = 0.08,
    )

def test_rk2_second_order_convergence():
    step_sizes = np.array([
        0.1,
        0.05,
        0.025,
        0.0125,
    ])

    errors = final_errors(
        rk2_solver,
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        step_sizes,
    )

    orders = estimate_orders(
        step_sizes,
        errors,
    )

    assert orders[-1] == pytest.approx(
        2.0,
        rel = 0.08,
    )

def test_rk4_fourth_order_convergence():
    step_sizes = np.array([
        0.2,
        0.1,
        0.05,
        0.025,
    ])

    errors = final_errors(
        rk4_solver,
        lambda t, y: y,
        np.exp,
        0.0,
        1.0,
        1.0,
        step_sizes,
    )

    orders = estimate_orders(
        step_sizes,
        errors,
    )

    assert orders[-1] == pytest.approx(
        4.0,
        rel = 0.08,
    )

def test_error_reduction_factors():
    errors = np.array([
        0.16,
        0.04,
        0.01,
    ])

    result = reduction_factors(errors)

    np.testing.assert_allclose(
        result,
        np.array([
            4.0,
            4.0,
        ])
    )

def test_expected_reduction_euler():
    result = expected_reduction(
        refinement_factor = 2.0,
        order = 1.0,
    )

    assert result == pytest.approx(2.0)

def test_expected_reduction_rk2():
    result = expected_reduction(
        refinement_factor = 2.0,
        order = 2.0,
    )

    assert result == pytest.approx(4.0)

def test_expected_reduction_rk4():
    result = expected_reduction(
        refinement_factor = 2.0,
        order = 4.0,
    )

    assert result == pytest.approx(16.0)

def test_asymptotic_constants_second_order():
    step_sizes = np.array([
        0.4,
        0.2,
        0.1,
    ])

    errors = 3.0 * step_sizes**2

    result = asymptotic_constants(
        step_sizes,
        errors,
        order = 2.0,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, 3.0),
        atol = 1e-12,
    )

def test_final_errors_reject_non_vector_steps():
    step_sizes = np.ones((2, 2))

    with pytest.raises(ValueError):
        final_errors(
            euler_solver,
            lambda t, y: y,
            np.exp,
            0.0,
            1.0,
            1.0,
            step_sizes,
        )

def test_final_errors_reject_empty_steps():
    step_sizes = np.array([])

    with pytest.raises(ValueError):
        final_errors(
            euler_solver,
            lambda t, y: y,
            np.exp,
            0.0,
            1.0,
            1.0,
            step_sizes,
        )

def test_final_errors_reject_nonpositive_steps():
    step_sizes = np.array([
        0.1,
        0.0,
    ])

    with pytest.raises(ValueError):
        final_errors(
            euler_solver,
            lambda t, y: y,
            np.exp,
            0.0,
            1.0,
            1.0,
            step_sizes,
        )

def test_estimate_orders_reject_length_mismatch():
    step_sizes = np.ones(3)
    errors = np.ones(2)

    with pytest.raises(ValueError):
        estimate_orders(
            step_sizes,
            errors,
        )

def test_estimate_orders_reject_nonpositive_errors():
    step_sizes = np.array([
        0.2,
        0.1,
    ])

    errors = np.array([
        0.01,
        0.0,
    ])

    with pytest.raises(ValueError):
        estimate_orders(
            step_sizes,
            errors,
        )

def test_reduction_factors_reject_too_few_errors():
    errors = np.array([
        0.1,
    ])

    with pytest.raises(ValueError):
        reduction_factors(errors)

def test_expected_reduction_rejects_invalid_refinement():
    with pytest.raises(ValueError):
        expected_reduction(
            refinement_factor=1.0,
            order = 2.0,
        )

def test_expected_reduction_rejects_nonpositive_order():
    with pytest.raises(ValueError):
        expected_reduction(
            refinement_factor = 2.0,
            order = 0.0,
        )

def test_asymptotic_constants_reject_length_mismatch():
    step_sizes = np.ones(3)
    errors = np.ones(2)

    with pytest.raises(ValueError):
        asymptotic_constants(
            step_sizes,
            errors,
            order = 2.0,
        )

def test_asymptotic_constants_reject_nonpositive_order():
    step_sizes = np.array([
        0.2,
        0.1,
    ])

    errors = np.array([
        0.01,
        0.0025,
    ])

    with pytest.raises(ValueError):
        asymptotic_constants(
            step_sizes,
            errors,
            order = 0.0,
        )