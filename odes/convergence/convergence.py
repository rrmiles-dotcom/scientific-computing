from collections.abc import Callable

import numpy as np

def final_errors(
    solver: Callable[
        [Callable[[float, float], float], float, float, float, float],
        tuple[np.ndarray, np.ndarray],
    ],
    function: Callable[[float, float], float],
    exact_solution: Callable[[float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step_sizes: np.ndarray,
) -> np.ndarray:
    """
    Compute final-time errors for an ODE solver over several step sizes.
    """

    if step_sizes.ndim != 1:
        raise ValueError("step_sizes must be one-dimensional")

    if step_sizes.size == 0:
        raise ValueError("step_sizes must not be empty")

    if np.any(step_sizes <= 0):
        raise ValueError("step_sizes must be positive")

    exact = exact_solution(final_time)
    errors = np.zeros(step_sizes.size, dtype = float)

    for i, step in enumerate(step_sizes):
        _, values = solver(
            function,
            initial_time,
            initial_value,
            final_time,
            float(step)
        )

        errors[i] = abs(values[-1] - exact)

    return errors

def estimate_orders(
    step_sizes: np.ndarray,
    errors: np.ndarray,
) -> np.ndarray:
    """
    Estimate local convergence orders from successive ODE errors.
    """

    if step_sizes.ndim != 1 or errors.ndim != 1:
        raise ValueError("step_sizes and errors must be one-dimensional")

    if step_sizes.size != errors.size:
        raise ValueError("step_sizes and errors must have equal length")

    if step_sizes.size < 2:
        raise ValueError("at least two data points are required")

    if np.any(step_sizes <= 0):
        raise ValueError("step_sizes must be positive")

    if np.any(errors <= 0):
        raise ValueError("errors must be positive")

    # Eliminate the unknown constant C from E(h) = C h^p.
    return (
        np.log(errors[:-1] / errors[1:])
        / np.log(step_sizes[:-1] / step_sizes[1:])
    )

def reduction_factors(
    errors: np.ndarray,
) -> np.ndarray:
    """
    Return successive error reduction factors.
    """

    if errors.ndim != 1:
        raise ValueError("errors must be one-dimensional")

    if errors.size < 2:
        raise ValueError("at least two errors are required")

    if np.any(errors <= 0):
        raise ValueError("errors must be positive")

    return errors[:-1] / errors[1:]

def expected_reduction(
    refinement_factor: float,
    order: float,
) -> float:
    """
    Return the theoretical error reduction r^p.
    """

    if refinement_factor <= 1:
        raise ValueError("refinement_factor must be greater than one.")

    if order <= 0:
        raise ValueError("order must be positive")

    return float(
        refinement_factor**order
    )

def asymptotic_constants(
    step_sizes: np.ndarray,
    errors: np.ndarray,
    order: float,
) -> np.ndarray:
    """
    Estimate C in the asymptotic error model E(h) = C h^p.
    """

    if step_sizes.ndim != 1 or errors.ndim != 1:
        raise ValueError("step_sizes and errors must be one-dimensional")

    if step_sizes.size != errors.size:
        raise ValueError("step_sizes and errors must have equal length")

    if step_sizes.size == 0:
        raise ValueError("data must not be empty")

    if np.any(step_sizes <= 0):
        raise ValueError("step_sizes must be positive")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    if order <= 0:
        raise ValueError("order must be positive")

    return errors / step_sizes**order