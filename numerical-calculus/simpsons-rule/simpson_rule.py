from collections.abc import Callable

import numpy as np

def simpsons_rule(
    function: Callable[[float], float],
    left: float,
    right: float,
    intervals: int,
) -> float:
    """Approximate a definite integral using composite Simpson's rule."""

    if left >= right:
        raise ValueError("left endpoint must be smaller than right endpoint")

    if intervals <= 0:
        raise ValueError("intervals must be positive")

    if intervals % 2 != 0:
        raise ValueError("intervals must be even")

    step = (right - left) / intervals

    x_values = np.linspace(
        left,
        right,
        intervals + 1,
    )

    y_values = np.array([
        function(x)
        for x in x_values   
    ], dtype = float)

    # Interior samples alternate between Simpson weights 4 and 2.
    odd_sum = np.sum(y_values[1:-1:2])
    even_sum = np.sum(y_values[2:-1:2])

    integral = (
        step / 3.0
        * (
            y_values[0]
            + 4.0 * odd_sum
            + 2.0 * even_sum
            + y_values[-1]
        )
    )

    return float(integral)

def simpsons_from_samples(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Approximate an integral from equally spaced samples using composite Simpson's rule.
    """

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size < 3:
        raise ValueError("at least three sample points are required")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    intervals = x_values.size - 1

    if intervals % 2 != 0:
        raise ValueError("number of intervals must be even")

    differences = np.diff(x_values)

    if np.any(differences <= 0):
        raise ValueError("x_values must be strictly increasing")

    # Classical composite Simpson's rule assumes uniform spacing.
    if not np.allclose(differences, differences[0]):
        raise ValueError("x_values must be equally spaced")

    step = differences[0]

    odd_sum = np.sum(y_values[1:-1:2])
    even_sum = np.sum(y_values[2:-1:2])

    integral = (
        step / 3.0
        * (
            y_values[0]
            + 4.0 * odd_sum
            + 2.0 * even_sum
            + y_values[-1]
        )
    )

    return float(integral)

def simpsons_error(
    function: Callable[[float], float],
    exact_integral: float,
    left: float,
    right: float,
    intervals: int,
) -> float:
    """Return the absolute error of a Simpson approximation."""

    approximation = simpsons_rule(
        function,
        left,
        right,
        intervals,
    )

    return float(
        abs(approximation - exact_integral)
    )