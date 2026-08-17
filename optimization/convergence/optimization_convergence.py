from collections.abc import Callable

import numpy as np


def parameter_errors(
    iterates: np.ndarray,
    optimum: np.ndarray,
) -> np.ndarray:
    """
    Return Euclidean distances between iterates and the known optimum.
    """
    if iterates.ndim != 2:
        raise ValueError("iterates must be two-dimensional")

    if optimum.ndim != 1:
        raise ValueError("optimum must be one-dimensional")

    if iterates.shape[1] != optimum.size:
        raise ValueError("iterate dimension must match optimum size")

    if iterates.shape[0] == 0:
        raise ValueError("iterates must not be empty")

    return np.linalg.norm(
        iterates - optimum,
        axis = 1,
    )

def objective_gaps(
    objective_values: np.ndarray,
    optimum_value: float,
) -> np.ndarray:
    """
    Return absolute objective-value gaps from the optimum.
    """

    if objective_values.ndim != 1:
        raise ValueError("objective_values must be one-dimensional")

    if objective_values.size == 0:
        raise ValueError("objective values must not be empty")

    return np.abs(
        objective_values.astype(float) - optimum_value
    )

def gradient_norms(
    gradient: Callable[[np.ndarray], np.ndarray],
    iterates: np.ndarray,
) -> np.ndarray:
    """
    Return the gradient norm at each optimization iterate.
    """

    if iterates.ndim != 2:
        raise ValueError("iterates must be two-dimensional")

    if iterates.shape[0] == 0:
        raise ValueError("iterates must not be empty")

    norms = np.zeros(
        iterates.shape[0],
        dtype = float,
    )

    for i, point in enumerate(iterates):
        grad = np.asarray(
            gradient(point),
            dtype = float,
        )

        if grad.shape != point.shape:
            raise ValueError(
                "gradient output must match iterate shape"
            )

        norms[i] = np.linalg.norm(grad)

    return norms

def convergence_ratios(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Return successive ratios e_(k+1) / e(k).
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 2:
        raise ValueError("at least two errors are required")

    if np.any(errors < 0):
        raise ValueError("errprs must be nonnegative")

    if np.any(errors[:-1] == 0):
        raise ValueError(
            "zero error cannot precede another convergence step"
        )

    return errors[1:] / errors[:-1]

def estimate_convergence_order(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Estimate local convergence order from three successive errrors.
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 3:
        raise ValueError("at least three errors are required")

    if np.any(errors <= 0):
        raise ValueError("errors must be strictly positive")

    orders = []

    for i in range(2, errors.size):
        previous = errors[i - 2]
        current = errors[i - 1]
        following = errors[i]

        denominator = np.log(
            current / previous
        )

        if np.isclose(denominator, 0.0):
            raise ValueError(
                "consecutive errors must change"
            )

        # For e_(k+1) = C e_k^p, successive errors reveal the local order p.
        order = (
            np.log(following / current)
            / denominator
        )

        orders.append(order)

    return np.array(orders)

def asymptotic_constants(
    errors: np.ndarray,
    order: float,
) -> np.ndarray:
    """
    Estimate C in the model e_(k + 1) = C e_k^p.
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 2:
        raise ValueError("at least two errors are required")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    if np.any(errors[:-1] == 0):
        raise ValueError(
            "zero error cannot precede another convergence step"
        ) 

    if order <= 0:
        raise ValueError("order must be positive")

    return (
        errors[1:]
        / errors[: -1]**order
    )