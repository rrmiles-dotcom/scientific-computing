import numpy as np

def absolute_errors(
    approximations: np.ndarray,
    exact_value: float,
) -> np.ndarray:
    """Return absolute errors for a sequence of approximations."""

    if approximations.ndim != 1:
        raise ValueError("approximations must be one-dimensional")

    if approximations.size == 0:
        raise ValueError("approximations must not be empty")

    return np.abs(
        approximations.astype(float) - exact_value
    )

def estimate_order(
    step_sizes: np.ndarray,
    errors: np.ndarray,
) -> np.ndarray:
    """
    Estimate local convergence orders from successive step sizes and numerical errors.
    """

    if step_sizes.ndim != 1 or errors.ndim != 1:
        raise ValueError("step_sizes and errors must be one-dimensional")

    if step_sizes.size != errors.size:
        raise ValueError("step_sizes and errors must have equal length")

    if step_sizes.size < 2:
        raise ValueError("at least two data points are required")

    if np.any(step_sizes <= 0):
        raise ValueError("step sizes must be positive")

    if np.any(errors <= 0):
        raise ValueError("errors must be positive")

    orders = []

    for i in range(step_sizes.size - 1):
        # From E(h) = C h^p, eliminate C using two successive errors.
        order = (
            np.log(errors[i] / errors[i + 1])
            / np.log(step_sizes[i] / step_sizes[i + 1])
        )

        orders.append(order)

    return np.array(orders)

def error_reduction_factors(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Return ratios of successive numerical errors.
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 2:
        raise ValueError("at least two errors are required")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    if np.any(errors[:-1] == 0):
        raise ValueError("zero error cannot precede another step")

    return errors[:-1] / errors[1:]

def expected_reduction_factor(
    refinement_factor: float,
    order: float,
) -> float:
    """
    Return the theoretical error reduction for a refinement factor.

    For example, having h corresponds to refinement_factor = 2.
    """

    if refinement_factor <= 1:
        raise ValueError("refinement_factor must be greater than one")

    if order <= 0:
        raise ValueError("order must be positive")

    return float(refinement_factor**order)

def asymptotic_constant(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    order: float,
) -> np.ndarray:
    """
    Estimate the constant C in the asymptotic model E(h) = C h^p.
    """

    if step_sizes.size != errors.size:
        raise ValueError("step_sizes and errors must have equal length")

    if step_sizes.size == 0:
        raise ValueError("data must not be empty")

    if np.any(step_sizes <= 0):
        raise ValueError("step sizes must be positive")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    if order <= 0:
        raise ValueError("order must be positive")

    return errors / step_sizes**order