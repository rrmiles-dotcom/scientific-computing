from collections.abc import Callable
import numpy as np

def backward_difference(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-6,
) -> float:
    """
    Approximate the first derivative of a scalar function using the backward-difference formula.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    derivative = (
        function(x) - function(x - step)
    ) / step

    return float(derivative)

def backward_difference_array(
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    """
    Approximate first derivatives from equally spaced function values.
    """

    if values.ndim != 1:
        raise ValueError("values must be one-dimensional")

    if values.size < 2:
        raise ValueError("at least two values are required")

    if step <= 0:
        raise ValueError("step must be positive")

    # Each derivative uses the current value and its previous neighbor.
    return np.diff(values.astype(float)) / step

def backward_difference_error(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x: float,
    step: float = 1e-6,
) -> float:
    """
    Compute the absolute error of a backward-difference approximation.
    """

    approximation = backward_difference(
        function,
        x,
        step,
    )

    exact = derivative(x)

    return float(abs(approximation - exact))