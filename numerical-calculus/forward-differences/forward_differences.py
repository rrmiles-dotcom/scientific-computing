from collections.abc import Callable

import numpy as np

def forward_difference(
    function: Callable[[float], float],
    x: float,
    step: float = 1e-6,
) -> float:
    """
    Approximate the first derivative of a scalar function using 
    the forward-difference formula.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    derivative = (
        function(x + step) - function(x)
    ) / step

    return float(derivative)

def forward_difference_array(
    values: np.ndarray,
    step: float,
) -> np.ndarray:
    """
    Approximate first derivatives from equally spaced function values.
    """

    if values.ndim != 1:
        raise ValueError("values must be one dimensional")

    if values.size < 2:
        raise ValueError("at least two values are required")

    if step <= 0:
        raise ValueError("step must be positive")

    # Each derivative uses the current value and its forward neighbor
    return np.diff(values.astype(float)) / step

def forward_difference_error(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    x: float,
    step: float = 1e-6,
) -> float:
    """
    Compute the absolute error of a forward-difference approximation.
    """

    approximation = forward_difference(
        function,
        x,
        step,
    )

    exact = derivative(x)

    return float(abs(approximation - exact))