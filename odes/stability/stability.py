import numpy as np

def euler_amplification_factor(
    eigenvalue: complex,
    step: float,
) -> complex:
    """
    Return the amplification factor for explicit Euler applied to y' = lambda y.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    return 1.0 + step * eigenvalue

def rk2_amplification_factor(
    eigenvalue: complex,
    step: float,
) -> complex:
    """
    Return the amplification factor for point RK2.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    z = step * eigenvalue

    return 1.0 + z + 0.5 * z**2

def rk4_amplification_factor(
    eigenvalue: complex,
    step: float,
) -> complex:
    """
    Return the amplification factor for classical RK4.
    """

    if step <= 0:
        raise ValueError("step must be positive")

    z = step * eigenvalue

    return (
        1.0
        + z
        + z**2 / 2.0
        + z**3 / 6.0
        + z**4 / 24.0
    )

def is_stable(
    amplification_factor: complex,
) -> bool:
    """
    Return whether a one-step method is absolutely stable.
    """

    return bool(
        abs(amplification_factor) < 1.0
    )

def euler_is_stable(
    eigenvalue: complex,
    step: float,
) -> bool:
    """
    Check absolute stability of explicit Euler.
    """

    factor = euler_amplification_factor(
        eigenvalue,
        step,
    )

    return is_stable(factor)

def rk2_is_stable(
    eigenvalue: complex,
    step: float,
) -> bool:
    """
    Check absolute stability of midpoint RK2.
    """

    factor = rk2_amplification_factor(
        eigenvalue,
        step,
    )

    return is_stable(factor)

def rk4_is_stable(
    eigenvalue: complex,
    step: float,
) -> bool:
    """
    Check absolute stability of classical RK4.
    """

    factor = rk4_amplification_factor(
        eigenvalue,
        step,
    )

    return is_stable(factor)

def simulate_test_equation(
    eigenvalue: float,
    initial_value: float,
    steps: int,
    step: float,
) -> np.ndarray:
    """
    Simulate y' = lambda y using explicit Euler.
    """

    if steps < 0:
        raise ValueError("steps must be nonnegative")

    if step <= 0:
        raise ValueError("step must be positive")

    values = np.zeros(
        steps + 1,
        dtype = float,
    )

    values[0] = initial_value

    factor = euler_amplification_factor(
        eigenvalue,
        step,
    )

    # Repeated multiplication exposes stable decay or numerical growth directly.
    for i in range(steps):
        values[i + 1] = factor * values[i]

    return values