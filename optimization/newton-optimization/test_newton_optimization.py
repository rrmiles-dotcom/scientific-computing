import numpy as np
import pytest

from newton_optimization import (
    newton_optimization,
    newton_final_point,
    newton_step,
    gradient_norm,
)

def test_one_dimensional_quadratic():
    def objective(point):
        return (point[0] - 3.0) ** 2

    def gradient(point):
        return np.array([
            2.0 * (point[0] - 3.0)
        ])

    def hessian(point):
        return np.array([
            [2.0],
        ])

    result = newton_final_point(
        objective,
        gradient,
        hessian,
        initial_point = np.array([10.0]),
    )

    np.testing.assert_allclose(
        result,
        np.array([3.0]),
        atol = 1e-12,
    )

def test_quadratic_converges_in_one_update():
    def objective(point):
        return (point[0] - 4.0) ** 2

    def gradient(point):
        return np.array([
            2.0 * (point[0] - 4.0)
        ])

    def hessian(point):
        return np.array([
            [2.0]
        ])

    _, history = newton_optimization(
        objective,
        gradient,
        hessian,
        initial_point = np.array([10.0]),
    )

    # A quadratic with constant nonsingular Hessian requires one Newton update.
    assert history.size == 2
    assert history[-1] == pytest.approx(0.0)

def test_two_dimensional_quadratic():
    def objective(point):
        x, y = point

        return (
            (x - 2.0) ** 2
            + 2.0 * (y + 1.0) ** 2
        )

    def gradient(point):
        x, y = point

        return np.array([
            2.0 * (x - 2.0),
            4.0 * (y + 1.0),
        ])

    def hessian(point):
        return np.array([
            [2.0, 0.0],
            [0.0, 4.0],
        ])

    result = newton_final_point(
        objective,
        gradient,
        hessian,
        initial_point = np.array([
            10.0,
            5.0,
        ]),
    )

    np.testing.assert_allclose(
        result, 
        np.array([
            2.0,
            -1.0,
        ]),
        atol = 1e-12,
    )

def test_coupled_quadratic():
    matrix = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    vector = np.array([
        1.0,
        2.0,
    ])

    def objective(point):
        return (
            0.5 * point @ matrix @ point
            - vector @ point
        )

    def gradient(point):
        return matrix @ point - vector

    def hessian(point):
        return matrix

    result = newton_final_point(
        objective,
        gradient,
        hessian,
        initial_point = np.array([
            5.0,
            -3.0,
        ]),
    )

    expected = np.linalg.solve(
        matrix,
        vector,
    )

    np.testing.assert_allclose(
        result,
        expected,
        atol = 1e-12,
    )

def test_starts_at_minimum():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    def hessian(point):
        return 2.0 * np.eye(point.size)

    point, history = newton_optimization(
        objective,
        gradient,
        hessian,
        initial_point = np.zeros(2),
    )

    np.testing.assert_allclose(
        point,
        np.zeros(2)
    )

    assert history.size == 1

def test_history_contains_initial_objective():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    def hessian(point):
        return 2.0 * np.eye(point.size)

    initial = np.array([
        2.0,
        -1.0,
    ])

    _, history = newton_optimization(
        objective,
        gradient,
        hessian,
        initial,
    )

    assert history[0] == pytest.approx(
        objective(initial)
    )

def test_initial_point_not_modified():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    def hessian(point):
        return 2.0 * np.eye(point.size)

    initial = np.array([
        4.0,
        -2.0,
    ])

    original = initial.copy()

    newton_optimization(
        objective,
        gradient,
        hessian,
        initial,
    )

    np.testing.assert_array_equal(
        initial,
        original,
    )

def test_newton_step_known_result():
    gradient_value = np.array([
        2.0,
        8.0,
    ])

    hessian_value = np.array([
        [2.0, 0.0],
        [0.0, 4.0],
    ])

    result = newton_step(
        gradient_value,
        hessian_value,
    )

    expected = np.array([
        1.0,
        2.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
        atol = 1e-12,
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
        ])
    )

    assert result == pytest.approx(5.0)

def test_rejects_non_vector_initial_point():
    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(np.sum(point**2)),
            lambda point: 2.0 * point,
            lambda point: np.eye(point.size),
            initial_point = np.ones((2, 2)),
        )

def test_rejects_empty_initial_point():
    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: 0.0,
            lambda point: point,
            lambda point: np.empty((0, 0)),
            initial_point = np.array([]),
        )

def test_rejects_nonpositive_tolerance():
    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            lambda point: 2.0 * np.eye(point.size),
            initial_point = np.array([1.0]),
            tolerance = 0.0, 
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            lambda point: 2.0 * np.eye(point.size),
            initial_point = np.array([1.0]),
            max_iterations = 0,
        )

def test_rejects_wrong_gradient_shape():
    def gradient(point):
        return np.array([
            1.0,
            2.0,
        ])

    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            gradient,
            lambda point: np.array([[2.0]]),
            initial_point = np.array([1.0]),
        )

def test_rejectes_wrong_hessian_shape():
    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            lambda point: np.eye(3),
            initial_point = np.array([
                1.0,
                2.0,
            ]),
        )

def test_rejects_nonsymmetric_hessian():
    def hessian(point):
        return np.array([
            [2.0, 1.0],
            [0.0, 2.0],
        ])

    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            hessian,
            initial_point = np.array([
                1.0,
                2.0,
            ]),
        )

def test_rejects_singular_hessian():
    def hessian(point):
        return np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])

    with pytest.raises(ValueError):
        newton_optimization(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            hessian,
            initial_point = np.array([
                1.0,
                2.0,
            ]),
        )

def test_newton_step_rejects_non_vector_gradient():
    with pytest.raises(ValueError):
        newton_step(
            np.ones((2, 2)),
            np.eye(4),
        )

def test_newton_step_rejects_wrong_hessian_dimensions():
    with pytest.raises(ValueError):
        newton_step(
            np.array([
                1.0,
                2.0,
            ]),
            np.ones(2),
        )

def test_newton_step_rejects_singular_hessian():
    with pytest.raises(ValueError):
        newton_step(
            np.array([
                1.0,
                2.0,
            ]),
            np.array([
                [1.0, 1.0],
                [1.0, 1.0],
            ]),
        )

def test_raises_when_iteration_limit_is_too_small():
    def objective(point):
        return point[0] ** 4

    def gradient(point):
        return np.array([
            4.0 * point[0] ** 3
        ])

    def hessian(point):
        return np.array([
            [12.0 * point[0] ** 2]
        ])

    with pytest.raises(RuntimeError):
        newton_optimization(
            objective,
            gradient,
            hessian,
            initial_point = np.array([2.0]),
            tolerance = 1e-12,
            max_iterations = 1,
        )