from collections.abc import Callable

def bisection(
    function: Callable[[float], float],
    left: float,
    right: float,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> float:

    """
    Approximate a root of a continuous scalar function using bisection.

    The initial interval must bracket a root, meaning that the function has opposite signs at its endpoints.
    """

    if left >= right:
        raise ValueError("left endpoint must be smaller than right endpoint")

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    f_left = function(left)
    f_right = function(right)

    if f_left == 0:
        return float(left)

    if f_right == 0:
        return float(right)

    # A sign change guarantees a root in the interval for continuous functions.
    if f_left * f_right > 0:
        raise ValueError("initial interval must bracket a root")

    for _ in range(max_iterations):
        midpoint = (left + right) / 2.0
        f_midpoint = function(midpoint)

        if abs(f_midpoint) <= tolerance:
            return float(midpoint)

        # Preserve the half-interval that still brackets a root.
        if f_left * f_midpoint < 0:
            right = midpoint
        else:
            left = midpoint
            f_left = f_midpoint

        # The midpoint error is bounded by half the current interval width.
        if (right - left) / 2.0 <= tolerance:
            return float((left + right) / 2.0)

    raise RuntimeError("bisection did not converge within max_iterations")