from collections.abc import Callable

def newton_raphson(
    function: Callable[[float], float],
    derivative: Callable[[float], float],
    initial_guess: float,
    tolerance: float = 1e-10,
    max_iterations: int = 100,
) -> float:

    """
    Approximate a root of f(x) = 0 using Newton-Raphson iteration.

    The method requires the function derivative and a sufficiently good initial guess near the desired root.
    """

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    current = float(initial_guess)

    for _ in range(max_iterations):
        function_value = function(current)
        derivative_value = derivative(current)

        if abs(function_value) <= tolerance:
            return current

        # A near-zero derivative would make the Newton step unstable.
        if abs(derivative_value) <= tolerance:
            raise ValueError("derivative is too close to zero")

        next_value = (
            current
            - function_value / derivative_value
        )

        # Stop once successive iterates are numerically indistinguishable.
        if abs(next_value - current) <= tolerance:
            return float(next_value)

        current = next_value

    raise RuntimeError(
        "Newton-Raphson did not converge within max_iterations"
    )