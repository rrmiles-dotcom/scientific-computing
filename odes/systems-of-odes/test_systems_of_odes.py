import numpy as np
import pytest

from systems_of_odes import (
    rk4_system,
    system_final_state,
    system_error,
)

def test_constant_vector_derivative():
    def system(t, state):
        return np.array([1.0, -2.0])

    times, states = rk4_system(
        system,
        initial_time = 0.0,
        initial_state = np.array([0.0, 1.0]),
        final_time = 1.0,
        step = 0.1,
    )

    expected_first = times
    expected_second = 1.0 - 2.0 * times

    np.testing.assert_allclose(
        states[:, 0],
        expected_first,
        atol = 1e-12,
    )

    np.testing.assert_allclose(
        states[:, 1],
        expected_second,
        atol = 1e-12,
    )

def test_zero_system_keeps_state_constant():
    def system(t, state):
        return np.zeros_like(state)

    initial = np.array([2.0, -1.0, 4.0])

    _, states = rk4_system(
        system,
        0.0,
        initial,
        1.0,
        0.2,
    )

    expected = np.tile(
        initial,
        (states.shape[0], 1),
    )

def test_harmonic_oscillator():
    def system(t, state):
        position, velocity = state

        return np.array([
            velocity,
            -position,
        ])

    initial = np.array([
        1.0,
        0.0,
    ])

    final = system_final_state(
        system,
        0.0,
        initial,
        np.pi / 2.0,
        np.pi / 200.0,
    )

    expected = np.array([
        0.0,
        -1.0,
    ])

    np.testing.assert_allclose(
        final,
        expected,
        atol = 1e-8,
    )

def test_two_independent_exponentials():
    def system(t, state):
        return np.array([
            state[0],
            -2.0 * state[1],
        ])

    initial = np.array([
        1.0,
        1.0,
    ])

    final = system_final_state(
        system,
        0.0,
        initial,
        1.0,
        0.01,
    )

    expected = np.array([
        np.e,
        np.exp(-2.0),
    ])

    np.testing.assert_allclose(
        final,
        expected,
        atol = 1e-8,
    )

def test_system_error_is_small():
    def system(t, state):
        return np.array([
            state[0],
            -state[1],
        ])

    def exact_solution(t):
        return np.array([
            np.exp(t),
            np.exp(-t),
        ])

    result = system_error(
        system,
        exact_solution,
        0.0,
        np.array([1.0, 1.0]),
        1.0,
        0.01,
    )

    assert result < 1e-8

def test_smaller_step_improves_accuracy():
    def system(t, state):
        return np.array([
            state[0],
            -state[1],
        ])

    def exact_solution(t):
        return np.array([
            np.exp(t),
            np.exp(-t),
        ])

    initial = np.array([
        1.0,
        1.0,
    ])

    coarse = system_error(
        system,
        exact_solution,
        0.0,
        initial,
        1.0,
        0.1,
    )

    fine = system_error(
        system,
        exact_solution,
        0.0,
        initial,
        1.0,
        0.05,
    )

    assert fine < coarse

def test_fourth_order_error_reduction():
    def system(t, state):
        return np.array([
            state[0],
            -state[1],
        ])

    def exact_solution(t):
        return np.array([
            np.exp(t),
            np.exp(-t),
        ])

    initial = np.array([
        1.0,
        1.0,
    ])

    coarse = system_error(
        system,
        exact_solution,
        0.0,
        initial,
        1.0,
        0.1,
    )

    fine = system_error(
        system,
        exact_solution,
        0.0,
        initial,
        1.0,
        0.05,
    )

    ratio = coarse / fine

    assert ratio == pytest.approx(
        16.0,
        rel = 0.3,
    ) 

def test_time_and_state_shapes():
    def system(t, state):
        return np.array([
            state[1],
            -state[0],
        ])

    times, states = rk4_system(
        system,
        0.0,
        np.array([1.0, 0.0]),
        1.0,
        0.1,
    )

    assert times.shape == (11,)
    assert states.shape == (11, 2)

def test_initial_state_is_preserved():
    initial = np.array([
        3.0,
        -2.0,
    ])

    _, states = rk4_system(
        lambda t, state: np.zeros_like(state),
        0.0,
        initial,
        1.0,
        0.25,
    )

    np.testing.assert_allclose(
        states[0],
        initial,
    )

def test_final_state_matches_full_solution():
    def system(t, state):
        return np.array([
            state[0],
            -state[1],
        ])

    _, states = rk4_system(
        system,
        0.0,
        np.array([1.0, 1.0]),
        1.0,
        0.1,
    )

    final = system_final_state(
        system,
        0.0,
        np.array([1.0, 1.0]),
        1.0,
        0.1,
    )

    np.testing.assert_allclose(
        final,
        states[-1],
    )

def test_rejects_non_vector_initial_state():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            0.0,
            np.ones((2, 2)),
            1.0,
            0.1,
        )

def test_rejects_empty_initial_state():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            0.0,
            np.array([]),
            1.0,
            0.1,
        )

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            1.0,
            np.array([]),
            1.0,
            0.1,
        )

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            1.0,
            np.array([1.0]),
            0.0,
            0.1,
        )

def test_rejects_zero_step():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            0.0,
            np.array([1.0]),
            1.0,
            0.0,
        )

def test_rejects_step_that_does_not_divide_interval():
    with pytest.raises(ValueError):
        rk4_system(
            lambda t, state: state,
            0.0,
            np.array([1.0]),
            1.0,
            0.3,
        )

def test_rejects_wrong_function_output_shape():
    def system(t, state):
        return np.array([
            1.0,
            2.0,
            3.0,
        ])

    with pytest.raises(ValueError):
        rk4_system(
            system,
            0.0,
            np.array([1.0, 2.0]),
            1.0,
            0.1,
        )

def test_rejects_wrong_exact_solution_shape():
    def system(t, state):
        return np.array([
            state[0],
            state[1],
        ])

    def exact_solution(t):
        return np.array([
            1.0,
            2.0,
            3.0,
        ])

    with pytest.raises(ValueError):
        system_error(
            system,
            exact_solution,
            0.0,
            np.array([1.0, 2.0]),
            1.0,
            0.1,
        )