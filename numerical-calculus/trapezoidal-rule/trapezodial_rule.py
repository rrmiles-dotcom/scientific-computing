from collections.abc import Callable

import numpy as np

def trapezoidal_rule(
    function: Callable[[float], float],
    left: float,
    right: float,
    intervals: int,
) -> float:
    """
    Approximate a definite integral using the composite trapezoidal rule.
    """

    if left >= right:
        raise ValueError("left endpoint must be smaller than right endpoint")

    if intervals <= 0:
        raise ValueError("intervals must be positive")

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

    # Interior samples contribute fully; endpoints contribute half.
    integral = step * (
        0.5 * y_values[0]
        + np.sum(y_values[1:-1])
        + 0.5 * y_values[-1]
    )

    return float(integral)

def trapezoidal_from_samples(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> float:
    """
    Approximate an integral from sampled data using trapezoidal areas.
    """

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size < 2:
        raise ValueError("at least two sample points are required")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if np.any(np.diff(x_values) <= 0):
        raise ValueError("x_values must be strictly increasing")

    widths = np.diff(x_values)

    # Each interval uses the average endpoint height times its width.

    areas = (
        widths
        * (y_values[:-1] + y_values[1:])
        / 2.0
    )

    return float(np.sum(areas))

def trapezoidal_error(
    function: Callable[[float], float],
    exact_integral: float,
    left: float,
    right: float,
    intervals: int,
) -> float:
    """Return the absolute error of a trapezoidal approximation"""

    approximation = trapezoidal_rule(
        function,
        left,
        right,
        intervals,
    )

    return float(
        abs(approximation - exact_integral)
    )