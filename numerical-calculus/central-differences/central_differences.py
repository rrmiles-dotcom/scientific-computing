from collections.abc import Callable

import numpy as np

def central_difference(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-5,
) -> float:
    """
    Approximate the first derivative of a scalar function using the central-difference formula.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    derivative = (
        function(x + step)
        - function(x - step)
    ) / (2.0 * step)

    return float(derivative)


def central_difference_array(
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    """
    Approximate first derivatives at interior points of equally spaced sampled data.
    """

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if values.size < 3:
        raise ValueError("at least three values are required")

    if step <= 0:
        raise ValueError("step must be positive")

    # Interior derivatives use one neighboring sample on each side.
    return (
        values[2:].astype(float)
        - values[:-2].astype(float)
    ) / (2.0 * step)

def central_difference_error(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x: float,
    step: float = 1e-5, 
) -> float:
    """
    Compute the absolute error of a central-difference approximation.
    """

    approximation = central_difference(
        function,
        x,
        step,
    )

    exact = derivative(x)

    return float(abs(approximation - exact))

def second_central_difference(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-4,
) -> float:
    """
    Approximate the second derivative using a central difference.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    # Symmetric sampling gives a second-order approximation of f''(x).
    derivative = (
        function(x + step)
        - 2.0 * function(x)
        + function(x - step)
    ) / step**2

    return float(derivative)