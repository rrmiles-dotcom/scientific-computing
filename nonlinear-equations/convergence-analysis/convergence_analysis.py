import numpy as np

def absolute_errors(
    approximations: np.ndarray,
    exact_value: float,
) -> np.ndarray:
    """
    Compute absolute errors | x_n - x* | for a sequence of approximations. 
    """

    if approximations.ndim != 1:
        raise ValueError("approximations must be one-dimensional")

    if approximations.size == 0:
        raise ValueError("approximations must not be empty")

    return np.abs(
        approximations.astype(float) - exact_value
    )

def convergence_ratios(
    errors: np.ndarray,
    order: float = 1.0,
) -> np.ndarray:
    "Compute ratios e_(n+1) / e_n^p for an assumed convergence order p."

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 2:
        raise ValueError("at least two errors are required")

    if order <= 0:
        raise ValueError("order must be positive")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    previous = errors[:-1]
    following = errors[1:]

    if np.any(previous == 0):
        raise ValueError(
            "zero error cannot precede another convergence step"
        )

    return following / previous**order

def estimate_convergence_order(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Estimate convergence order from consecutive numerical errors.
    
    Three successive errors are required for each local estimate.
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 3:
        raise ValueError("at least three errors are required")

    if np.any(errors <= 0):
        raise ValueError("errors must be strictly positive")

    estimates = []

    for i in range(2, errors.size):
        e_previous = errors[i - 2]
        e_current = errors[i - 1]
        e_next = errors[i]

        denominator = np.log(e_current / e_previous)

        # A zero logarithmic ratio provides no information about the order.
        if np.isclose(denominator, 0.0):
            raise ValueError(
                "consecutive errors must change"
            )

        order = (
            np.log(e_next / e_current)
            / denominator
        )

        estimates.append(order)

    return np.array(estimates)

def observed_reduction_factors(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Return e_(n+1) / e_n to measure successive error reduction.
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

    return errors[1:] / errors[:-1]