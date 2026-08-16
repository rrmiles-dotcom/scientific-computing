from collections.abc import Callable

import numpy as np

def rk4(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a scalar first-order ODE using the classical RK4 method.
    """

    if final_time <= initial_time:
        raise ValueError("final_time must be rgeater than initial_time")

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

        k2 = function(
            time + step / 2.0,
            value + step * k1 / 2.0,
        )

        k3 = function(
            time + step / 2.0,
            value + step * k2 / 2.0,
        )

        k4 = function(
            time + step,
            value + step * k3,
        )

        # RK4 combines four slope estimates to cancel lower-order error terms.
        values[i + 1] = (
            value
            + step
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )

    return times, values

def rk4_final_value(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """
    Return only the RK4 approximation at the final time.
    """

    _, values = rk4(
        function,
        initial_time,
        initial_value,
        final_time,
        step,
    )

    return float(values[-1])

def rk4_error(
    function: Callable[[float, float], float],
    exact_solution: Callable[[float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """
    Return the absolute RK4 error at the final time.
    """

    approximation = rk4_final_value(
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