import numpy as np

def machine_epsilon() -> float:
    """Return machine epsilon for Python/NumPy float64."""
    return float(np.finfo(np.float64).eps)

def next_float(x: float) -> float:
    """Return the next representable float greater than x."""
    return float(np.nextafter(x, np.inf))

def absolute_error(approximation: float, exact: float) -> float:
    """Return the absolute error.""" 
    return abs(approximation - exact)

def relative_error(approximation: float, exact: float) -> float:
    """Return the relative error"""

    if exact == 0:
        raise ValueError("relative error is undefined when exact value is zero")

    return abs(approximation - exact) / abs(exact)

def approximately_equal(
    a: float,
    b: float,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 0.0,
) -> bool:
    """Return whether two floats are equal within numerical tolerances"""

    if relative_tolerance < 0 or absolute_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    
    return bool(
        np.isclose(
            a,
            b,
            rtol = relative_tolerance,
            atol = absolute_tolerance,
        )
    )

def cancellation_example(x: float) -> tuple[float, float]:
    """
    Compare two mathematically equivalent expressions.
    
    Direct:
        sqrt(x + 1) - sqrt(x)

    Stable:
        1 / (sqrt(x + 1) + sqrt(x))
    """

    if x < 0:
        raise ValueError("x must be non-negative")

    direct = np.sqrt(x + 1.0) - np.sqrt(x)

    stable = 1.0 / (
        np.sqrt(x + 1.0) + np.sqrt(x)
    )

    return float(direct), float(stable)

def summation_forward(values: np.ndarray) -> float:
    """Sum floating-point values from left to right"""

    total = 0.0

    for value in values:
        total += float(value)

    return total

def summation_reverse(values: np.ndarray) -> float:
    """Sum floating-point values from right to left"""

    total = 0.0

    for value in values[::-1]:
        total += float(value)

    return total
