from collections.abc import Callable

import numpy as np

def backtracking_line_search(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
    direction: np.ndarray,
    initial_step: float = 1.0,
    reduction_factor: float = 0.5,
    armijo_constant: float = 1e-4,
    max_iterations: int = 100,
) -> float:
    """
    Choose a step length using Armijo backtracking line search.
    """

    if point.ndim != 1 or direction.ndim != 1:
        raise ValueError("point and direction must be one-dimensional")

    if point.shape != direction.shape:
        raise ValueError("point and direction must have matching shapes")

    if initial_step <= 0:
        raise ValueError("initial_step must be positive")

    if not 0.0 < reduction_factor < 1.0:
        raise ValueError("reduction_factor must satisfy 0 < factor < 1")

    if not 0.0 < armijo_constant < 1.0:
        raise ValueError("armijo_constant must satisfy 0 < constant < 1")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    grad = np.asarray(
        gradient(point),
        dtype = float,
    )

    if grad.shape != point.shape:
        raise ValueError(
            "gradient output must match point shape"
        )

    directional_derivative = float(
        grad @ direction
    )

    if directional_derivative >= 0:
        raise ValueError(
            "direction must be a descent direction"
        )

    step = float(initial_step)
    current_value = float(objective(point))

    for _ in range(max_iterations):
        candidate = (
            point
            + step * direction
        )

        candidate_value = float(
            objective(candidate)
        )

        # Accept once the Armijo sufficient-decrease condition is satisfied.
        if candidate_value <= (
            current_value
            + armijo_constant
            * step
            * directional_derivative
        ):
            return step

        step *= reduction_factor

    raise RuntimeError(
        "line search did not find an acceptable step"
    ) 

def gradient_descent_direction(
    gradient: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
) -> np.ndarray:
    """
    Return the steepest-descent direction.
    """

    if point.ndim != 1:
        raise ValueError("point must be one-dimensional")

    grad = np.asarray(
        gradient(point),
        dtype = float,
    )

    if grad.shape != point.shape:
        raise ValueError(
            "gradient output must match point shape"
        )

    return -grad

def newton_direction(
    gradient: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
) -> np.ndarray:
    """
    Return the Newton search direction by solving H p = -grad.
    """

    if point.ndim != 1:
        raise ValueError("point must be one-dimensional")

    grad = np.asarray(
        gradient(point),
        dtype = float,
    )

    hess = np.asarray(
        hessian(point),
        dtype = float,
    )

    if grad.shape != point.shape:
        raise ValueError(
            "gradient output must match point shape"
        )

    if hess.shape != (point.size, point.size):
        raise ValueError(
            "hessian dimensions must match point size"
        )

    if not np.allclose(
        hess,
        hess.T,
    ):
        raise ValueError("hessian must be symmetric")

    try:
        direction = np.linalg.solve(
            hess,
            -grad,
        )

    except np.linalg.LinAlgError as error:
        raise ValueError(
            "hessian must be nonsingular"
        ) from error

    return direction