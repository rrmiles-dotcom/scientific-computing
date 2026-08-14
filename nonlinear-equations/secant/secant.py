from collections.abc import Callable

def secant(
    function: Callable[[float], float],
    first_guess: float,
    second_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:

    """
    Approximate a root of f(x) = 0 using the secant method.

    The method estimates derivative information from two successive iterates and therefore does not require an explicit derivative.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    previous = float(first_guess)
    current = float(second_guess)

    if previous == current:
        raise ValueError("initial guesses must be distinct")

    f_previous = function(previous)
    f_current = function(current)

    if abs(f_previous) <= tolerance:
        return previous

    if abs(f_current) <= tolerance:
        return current

    for _ in range(max_iterations):
        denominator = f_current - f_previous

        # Nearly equal function values make the secant slope unreliable.
        if abs(denominator) <= tolerance:
            raise ValueError("secant slope is too close to zero")

        next_value = (
            current
            - f_current
            * (current - previous)
            / denominator
        )

        f_next = function(next_value)

        if abs(f_next) <= tolerance:
            return float(next_value)

        # Successive iterates provide a second convergence criterion.
        if abs(next_value - current) <= tolerance:
            return float(next_value)

        previous = current
        current = next_value

        f_previous = f_current
        f_current = f_next

    raise RuntimeError(
        "secant method did not converge within max_iterations"
    )