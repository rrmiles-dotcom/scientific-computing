from collections.abc import Callable

import numpy as np

def euler_method(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a scalar first-order ODE using the explicit Euler method.
    
    The problem is assumed to have the form y' = f(t, y)
    with a specified initial value y(t0).
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
        # Advance along the local tangent using the current ODE slope.
        values[i + 1] = (
            values[i]
            + step
            * function(
                times[i],
                values[i],
            )
        )

    return times, values

def euler_final_value(
    function: Callable[[float, float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """
    Return only the Euler approximation at the final time.
    """

    _, values = euler_method(
        function,
        initial_time,
        initial_value,
        final_time,
        step,
    )

    return float(values[-1])


def euler_error(
    function: Callable[[float, float], float],
    exact_solution: Callable[[float], float],
    initial_time: float,
    initial_value: float,
    final_time: float,
    step: float,
) -> float:
    """
    Return the absolute error of Euler's method at the final time.
    """

    approximation = euler_final_value(
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