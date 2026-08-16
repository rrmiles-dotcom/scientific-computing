from collections.abc import Callable

import numpy as np

def rk2(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a scalar first-order ODE using the midpoint RK2 method.
    """

    if final_time <= initial_time:
        raise ValueError("final_time must be greater than initial_time")

    if step <= 0:
        raise ValueError("step must be positive")

    duration = final_time - initial_time
    intervals = int(round(duration / step))

    if not np.isclose(
        intervals * step,
        duration,
    ):
        raise ValueError(
            "step must divide the integration interval exactly"
        )

    times = np.linspace(
        initial_time,
        final_time,
        intervals + 1,
    )

    values = np.zeros(
        intervals + 1,
        dtype = float,
    )

    values[0] = initial_value

    for i in range(intervals):
        time = times[i]
        value = values[i]

        k1 = function(
            time,
            value,
        )

        # Estimate the slope at the midpoint using an Euler half-step.
        k2 = function(
            time + step / 2.0,
            value + step * k1 / 2.0,
        )

        values[i + 1] = (
            value
            + step * k2
        )

    return times, values

def rk2_final_value(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """
    Return only the RK2 approximation at the final time.
    """

    _, values = rk2(
        function,
        initial_time,
        initial_value,
        final_time,
        step,
    )

    return float(values[-1])

def rk2_error(
    function: Callable[[float, float], float],
    exact_solution: Callable[[float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """Return the absolute RK2 error at the final time."""

    approximation = rk2_final_value(
        function,
        initial_time,
        initial_value,
        final_time,
        step,
    )

    exact = exact_solution(final_time)

    return float(
        abs(approximation - exact)
    )