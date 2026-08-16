from collections.abc import Callable

import numpy as np

def newton_optimization(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize a twice-differentiable objective using Newton's method.

    Returns the final parameter vector and the objective-value history.
    """

    if initial_point.ndim != 1:
        raise ValueError("initial_point must be one-dimensional")

    if initial_point.size == 0:
        raise ValueError("initial_point must not be empty")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    point = initial_point.astype(float).copy()

    history = [
        float(objective(point))
    ]

    for _ in range(max_iterations):
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
                "gradient output must match initial_point shape"
            )

        expected_hessian_shape = (
            point.size,
            point.size,
        )

        if hess.shape != expected_hessian_shape:
            raise ValueError(
                "hessian output must be a square matrix matching point size"
            )

        if not np.allclose(
            hess,
            hess.T,
        ):
            raise ValueError("hessian must be symmetric")

        if np.linalg.norm(grad) <= tolerance:
            return point, np.array(history)

        # Solve H p = grad instead of explicitly forming H^{-1}.
        try:
            step_direction = np.linalg.solve(
                hess,
                grad,
            )
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "hessian must be nonsingular"
            ) from error

        new_point = point - step_direction

        history.append(
            float(objective(new_point))
        )

        point = new_point

    raise RuntimeError(
        "Newton optimization did not converge within max_iterations"
    )

def newton_final_point(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    hessian: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> np.ndarray:
    """Return only the final Newton iterate."""

    point, _ = newton_optimization(
        objective,
        gradient,
        hessian,
        initial_point,
        tolerance,
        max_iterations,
    )

    return point

def newton_step(
    gradient_value: np.ndarray,
    hessian_value: np.ndarray,
) -> np.ndarray:
    """
    Compute the Newton step p from H p = grad.
    """

    if gradient_value.ndim != 1:
        raise ValueError("gradient_value must be one-dimensional")

    if hessian_value.ndim != 2:
        raise ValueError("hessian_value must be two-dimensional")

    n = gradient_value.size

    if hessian_value.shape != (n, n):
        raise ValueError(
            "hessian dimensions must match gradient size"
        )

    if not np.allclose(
        hessian_value,
        hessian_value.T,
    ):
        raise ValueError("hessian must be symmetric")

    try:
        return np.linalg.solve(
            hessian_value.astype(float),
            gradient_value.astype(float),
        )

    except np.linalg.LinAlgError as error:
        raise ValueError(
            "hessian must be nonsingular"
        ) from error

def gradient_norm(
    gradient: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
) -> float:
    """Return the Euclidean norm of the gradient."""

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

    return float(
        np.linalg.norm(grad)
    )