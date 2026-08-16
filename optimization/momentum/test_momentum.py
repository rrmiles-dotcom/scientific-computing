import numpy as np
import pytest

from momentum import(
    momentum_descent,
    momentum_final_point,
    gradient_norm,
)

def test_quadratic_minimum():
    def objective(point):
        return (point[0] - 3.0) ** 2

    def gradient(point):
        return np.array([
            2.0 * (point[0] - 3.0)
        ])

    result = momentum_final_point(
        objective,
        gradient,
        initial_point = np.array([0.0]),
        learning_rate = 0.05,
        momentum = 0.8,
    )

    np.testing.assert_allclose(
        result,
        np.array([3.0]),
        atol = 1e-7,
    )

def test_two_dimensional_quadratic():
    def objective(point):
        return (
            (point[0] - 2.0)**2
            + (point[1] + 1.0)**2
        )

    def gradient(point):
        return np.array([
            2.0 * (point[0] - 2.0),
            2.0 * (point[1] + 1.0),
        ])

    result = momentum_final_point(
        objective,
        gradient,
        initial_point = np.array([
            10.0,
            5.0,
        ]),
        learning_rate = 0.05,
        momentum = 0.8,
    )

    np.testing.assert_allclose(
        result,
        np.array([
            2.0,
            -1.0,
        ]),
        atol = 1e-7,
    )

def test_starts_at_minimum():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point, history = momentum_descent(
        objective,
        gradient,
        initial_point = np.array([
            0.0,
            0.0,
        ]),
    )

    np.testing.assert_allclose(
        point,
        np.zeros(2),
    )

    assert history.size == 1

def test_gradient_norm_decreases():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    initial = np.array([
        4.0,
        -3.0,
    ])

    final = momentum_final_point(
        objective,
        gradient,
        initial,
        learning_rate = 0.05,
        momentum = 0.8,
    )

    assert (
        gradient_norm(gradient, final)
        < gradient_norm(gradient, initial)
    )

def test_history_contains_initial_objectives():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    initial = np.array([
        2.0,
        1.0,
    ])

    _, history = momentum_descent(
        objective,
        gradient,
        initial,
        learning_rate = 0.05,
        momentum = 0.8,
    )

    assert history[0] == pytest.approx(
        objective(initial)
    )

def test_initial_point_not_modified():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    initial = np.array([
        2.0,
        3.0,
    ])

    original = initial.copy()

    momentum_descent(
        objective,
        gradient,
        initial,
        learning_rate = 0.05,
        momentum = 0.8,
    )

    np.testing.assert_array_equal(
        initial,
        original,
    )

def test_zero_momentum_behaves_like_gradient_descent():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point, _ = momentum_descent(
        objective,
        gradient,
        initial_point = np.array([1.0]),
        learning_rate = 0.1,
        momentum = 0.0,
    )

    assert abs(point[0]) < 1e-7

def test_smaller_tolerance_produces_smaller_gradient():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    coarse = momentum_final_point(
        objective,
        gradient,
        np.array([1.0]),
        learning_rate = 0.05,
        momentum = 0.5,
        tolerance = 1e-3,
    )

    fine = momentum_final_point(
        objective,
        gradient,
        np.array([1.0]),
        learning_rate = 0.05,
        momentum = 0.5,
        tolerance = 1e-8,
    )

    assert (
        gradient_norm(gradient, fine)
        < gradient_norm(gradient, coarse)
    )

def test_gradient_norm_known_result():
    def gradient(point):
        return np.array([
            3.0,
            4.0,
        ])

    result = gradient_norm(
        gradient,
        np.array([
            1.0,
            2.0,
        ]),
    )

    assert result == pytest.approx(5.0)

def test_rejects_non_vector_initial_point():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(np.sum(point**2)),
            lambda point: 2.0 * point,
            initial_point = np.array([]),
        )

def test_rejects_zero_learning_rate():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            initial_point = np.array([1.0]),
            learning_rate = 0.0,
        )

def test_rejects_negative_momentum():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            initial_point = np.array([1.0]),
            momentum = -0.1,
        )

def test_rejects_momentum_equal_to_one():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            initial_point = np.array([1.0]),
            momentum = 1.0,
        )

def test_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            initial_point = np.array([1.0]),
            tolerance = 0.0,
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        momentum_descent(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            initial_point = np.array([1.0]),
            max_iterations = 0,
        )

def test_rejects_wrong_gradient_shape():
    def objective(point):
        return point @ point

    def gradient(point):
        return np.array([
            1.0,
            2.0,
        ])

    with pytest.raises(ValueError):
        momentum_descent(
            objective,
            gradient,
            initial_point = np.array([1.0]), 
        )

def test_raises_when_iteration_limit_is_too_small():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    with pytest.raises(RuntimeError):
        momentum_descent(
            objective,
            gradient,
            initial_point = np.array([10.0]),
            learning_rate = 0.001,
            momentum = 0.9,
            tolerance = 1e-12,
            max_iterations = 1,
        )