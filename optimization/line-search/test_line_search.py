import numpy as np
import pytest

from line_search import(
    backtracking_line_search,
    gradient_descent_direction,
    newton_direction,
)

def test_gradient_descent_direction():
    def gradient(point):
        return np.array([
            2.0,
            -3.0,
        ])

    result = gradient_descent_direction(
        gradient,
        np.array([
            1.0,
            2.0,
        ]),
    )

    np.testing.assert_allclose(
        result,
        np.array([
            -2.0,
            3.0,
        ]),
    )

def test_newton_direction_known_result():
    def gradient(point):
        return np.array([
            2.0,
            8.0,
        ])

    def hessian(point):
        return np.array([
            [2.0, 0.0],
            [0.0, 4.0],
        ])

    result = newton_direction(
        gradient,
        hessian,
        np.array([
            1.0,
            1.0,
        ])
    )

    np.testing.assert_allclose(
        result,
        np.array([
            -1.0,
            -2.0,
        ]),
        atol = 1e-12,
    )

def test_backtracking_accepts_full_step_when_sufficient():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point = np.array([
        1.0,
    ])

    direction = gradient_descent_direction(
        gradient,
        point,
    )

    step = backtracking_line_search(
        objective,
        gradient,
        point,
        direction,
        initial_step = 0.5,
    )

    assert step == pytest.approx(0.5)

def test_backtracking_reduces_large_step():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point = np.array([
        1.0,
    ])

    direction = gradient_descent_direction(
        gradient,
        point,
    )

    step = backtracking_line_search(
        objective,
        gradient,
        point,
        direction,
        initial_step = 10.0,
        reduction_factor = 0.5,
    )

    assert step < 10.0

def test_returned_step_decreases_objective():
    def objective(point):
        return (
            (point[0] - 3.0) ** 2
            + (point[1] + 2.0) ** 2
        )

    def gradient(point):
        return np.array([
            2.0 * (point[0] - 3.0),
            2.0 * (point[1] + 2.0),
        ])

    point = np.array([
        10.0,
        5.0,
    ])

    direction = gradient_descent_direction(
        gradient,
        point,
    )

    step = backtracking_line_search(
        objective,
        gradient,
        point,
        direction,
    )

    new_point = (
        point
        + step * direction
    )

    assert objective(new_point) < objective(point)

def test_returned_step_satisfies_armijo_condition():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point = np.array([
        2.0,
        -1.0,
    ])

    direction = gradient_descent_direction(
        gradient,
        point,
    )

    armijo_constant = 1e-4

    step = backtracking_line_search(
        objective,
        gradient,
        point,
        direction,
        armijo_constant = armijo_constant,
    )

    left = objective(
        point + step * direction
    )

    right = (
        objective(point)
        + armijo_constant
        * step
        * gradient(point) @ direction
    )

    assert left <= right

def test_newton_direction_is_descent_for_positive_definite_hessian():
    matrix = np.array([
        [4.0, 1.0],
        [1.0, 3.0],
    ])

    def gradient(point):
        return matrix @ point

    def hessian(point):
        return matrix

    point = np.array([
        2.0,
        -1.0,
    ])

    direction = newton_direction(
        gradient,
        hessian,
        point,
    )

    assert gradient(point) @ direction < 0.0

def test_rejects_non_vector_point():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(np.sum(point**2)),
            lambda point: 2.0 * point,
            point = np.ones((2, 2)),
            direction = np.ones(4),
        )

def test_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            point = np.ones(2),
            direction = np.ones(3),
        )

def test_rejects_zero_initial_step():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            point = np.array([1.0]),
            direction = np.array([-1.0]),
            initial_step = 0.0,
        )

def test_rejects_invalid_reduction_factor():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            point = np.array([1.0]),
            direction = np.array([-1.0]),
            reduction_factor = 1.0,
        )

def test_rejects_invalid_armijo_constant():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            point = np.array([1.0]),
            direction = np.array([-1.0]),
            armijo_constant = 0.0,
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            lambda point: 2.0 * point,
            point = np.array([1.0]),
            direction = np.array([-1.0]),
            max_iterations = 0,
        )

def test_rejects_non_descent_direction():
    def objective(point):
        return point @ point

    def gradient(point):
        return 2.0 * point

    point = np.array([
        1.0,
    ])

    direction = np.array([
        1.0,
    ])

    with pytest.raises(ValueError):
        backtracking_line_search(
            objective,
            gradient,
            point,
            direction,
        )

def test_rejects_wrong_gradient_shape():
    def gradient(point):
        return np.array([
            1.0,
            2.0,
        ])

    with pytest.raises(ValueError):
        backtracking_line_search(
            lambda point: float(point @ point),
            gradient,
            point = np.array([1.0]),
            direction = np.array([-1.0]),
        )

def test_newton_direction_rejects_wrong_hessian_shape():
    with pytest.raises(ValueError):
        newton_direction(
            lambda point: 2.0 * point,
            lambda point: np.eye(3),
            np.array([
                1.0,
                2.0,
            ]),
        )

def test_newton_direction_rejects_nonsymmetric_hessian():
    def hessian(point):
        return np.array([
            [2.0, 1.0],
            [0.0, 2.0],
        ])

    with pytest.raises(ValueError):
        newton_direction(
            lambda point: 2.0 * point,
            hessian,
            np.array([
                1.0,
                2.0,
            ]),
        )

def test_newton_direction_rejects_singular_hessian():
    def hessian(point):
        return np.array([
            [1.0, 1.0],
            [1.0, 1.0],
        ])

    with pytest.raises(ValueError):
        newton_direction(
            lambda point: 2.0 * point,
            hessian,
            np.array([
                1.0,
                2.0,
            ]),
        )