from collections.abc import Callable

import numpy as np

def momentum_descent(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Minimize a differentiable objective using gradient descent with momentum.

    Returns the final parameter vector and objective-value history.
    """

    if initial_point.ndim != 1:
        raise ValueError("initial_point must be one-dimensional")

    if initial_point.size == 0:
        raise ValueError("initial_point must not be empty")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    if not 0.0 <= momentum < 1.0:
        raise ValueError("momentum must satisfy 0 <= momentum < 1")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    point = initial_point.astype(float).copy()
    velocity = np.zeros_like(point)

    history = [
        float(objective(point))
    ]

    for _ in range(max_iterations):
        grad = np.asarray(
            gradient(point),
            dtype = float,
        )

        if grad.shape != point.shape:
            raise ValueError(
                "gradient output must match initial_point shape"
            )

        if np.linalg.norm(grad) <= tolerance:
            return point, np.array(history)

        # Retain part of the previous update while incorporating the new gradient.
        velocity = (
            momentum * velocity
            - learning_rate * grad
        )

        point = point + velocity

        history.append(
            float(objective(point))
        )

    raise RuntimeError(
        "momentum descent did not converge within max_iterations"
    )

def momentum_final_point(
    objective: Callable[[np.ndarray], float],
    gradient: Callable[[np.ndarray], np.ndarray],
    initial_point: np.ndarray,
    learning_rate: float = 0.01,
    momentum: float = 0.9,
    tolerance: float = 1e-8,
    max_iterations: int = 10000,
) -> np.ndarray:
    """
    Return only the final parameter vector.
    """

    point, _ = momentum_descent(
        objective,
        gradient,
        initial_point,
        learning_rate,
        momentum,
        tolerance,
        max_iterations,
    )

    return point

def gradient_norm(
    gradient: Callable[[np.ndarray], np.ndarray],
    point: np.ndarray,
) -> float:
    """
    Return the Euclidean norm of the gradient at a point.
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

    return float(
        np.linalg.norm(grad)
    )