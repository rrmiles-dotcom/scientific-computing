import numpy as np
import pytest

from stability import (
    euler_amplification_factor,
    rk2_amplification_factor,
    rk4_amplification_factor,
    is_stable,
    euler_is_stable,
    rk2_is_stable,
    rk4_is_stable,
    simulate_test_equation,
)

def test_euler_amplification_factor():
    result = euler_amplification_factor(
        eigenvalue = -2.0,
        step = 0.25,
    )

    assert result == pytest.approx(0.5)

def test_rk2_amplification_factor():
    result = rk2_amplification_factor(
        eigenvalue = -1.0,
        step = 0.5,
    )

    expected = (
        1.0
        -0.5
        +0.5 * 0.5**2
    )

    assert result == pytest.approx(expected)

def test_rk4_amplification_factor():
    result = rk4_amplification_factor(
        eigenvalue = -1.0,
        step = 0.5,
    )

    z = -0.5

    expected = (
        1.0
        + z
        + z**2 / 2.0
        + z**3 / 6.0
        + z**4 / 24.0
    )

    assert result == pytest.approx(expected)

def test_is_stable_when_factor_magnitude_below_one():
    assert is_stable(0.5)

def test_is_unstable_when_factor_magnitude_above_one():
    assert not is_stable(1.2)

def test_is_unstable_on_boundary():
    assert not is_stable(1.0)

def test_euler_stable_for_small_negative_step_product():
    assert euler_is_stable(
        eigenvalue = -1.0,
        step = 1.0,
    )

def test_euler_unstable_for_large_step():
    assert not euler_is_stable(
        eigenvalue = -1.0,
        step = 3.0,
    )

def test_euler_boundary_is_not_stable():
    assert not euler_is_stable(
        eigenvalue = -1.0,
        step = 2.0,
    )

def test_rk2_stability_check():
    assert rk2_is_stable(
        eigenvalue = -1.0,
        step = 1.0,
    )

def test_rk4_allows_larger_stable_step_than_euler():
    eigenvalue = -1.0
    step = 2.5

    assert not euler_is_stable(
        eigenvalue,
        step,
    )

    assert rk4_is_stable(
        eigenvalue,
        step,
    )

def test_simulation_stable_case_decays():
    values = simulate_test_equation(
        eigenvalue = -1.0,
        initial_value = 1.0,
        steps = 10,
        step = 0.5,
    )

    assert abs(values[-1]) < abs(values[0])

def test_simulation_unstable_case_grows():
    values = simulate_test_equation(
        eigenvalue = -1.0,
        initial_value = 1.0,
        steps = 10,
        step = 3.0,
    )

    assert abs(values[-1]) > abs(values[0])

def test_simulation_matches_euler_factor():
    eigenvalue = -2.0
    step = 0.25
    steps = 4

    values = simulate_test_equation(
        eigenvalue = eigenvalue,
        initial_value = 1.0,
        steps = steps,
        step = step,
    )

    factor = euler_amplification_factor(
        eigenvalue,
        step,
    )

    expected = np.array([
        factor**i
        for i in range(steps + 1)
    ])

    np.testing.assert_allclose(
        values,
        expected,
        atol = 1e-12,
    )

def test_zero_steps_returns_initial_value_only():
    values = simulate_test_equation(
        eigenvalue = -1.0,
        initial_value = 3.0,
        steps = 0,
        step = 0.1, 
    )

    np.testing.assert_allclose(
        values,
        np.array([3.0]),
    )

def test_euler_rejects_zero_step():
    with pytest.raises(ValueError):
        euler_amplification_factor(
            eigenvalue = -1.0,
            step = 0.0,
        )

def test_rk2_rejects_negative_step():
    with pytest.raises(ValueError):
        rk2_amplification_factor(
            eigenvalue = -1.0,
            step = -0.1,
        )

def test_rk4_rejects_zero_step():
    with pytest.raises(ValueError):
        rk4_amplification_factor(
            eigenvalue = -1.0,
            step = 0.0,
        )

def test_simulation_rejects_negative_steps():
    with pytest.raises(ValueError):
        simulate_test_equation(
            eigenvalue = -1.0,
            initial_value = 1.0,
            steps = -1,
            step = 0.1,
        )

def test_simulation_rejects_nonpositive_step():
    with pytest.raises(ValueError):
        simulate_test_equation(
            eigenvalue = -1.0,
            initial_value = 1.0,
            steps = 10,
            step = 0.0,
        )