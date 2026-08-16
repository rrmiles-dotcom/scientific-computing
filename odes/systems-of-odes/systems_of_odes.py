from collections.abc import Callable

import numpy as np

def rk4_system(
    function: Callable[[float, np.ndarray], np.ndarray],
    initial_time: float,
    initial_state: np.ndarray,
    final_time: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve a first-order system of ODEs using the classical RK4 method.
    """

    if initial_state.ndim != 1:
        raise ValueError("initial_state must be one-dimensional")

    if initial_state.size == 0:
        raise ValueError("initial_state must not be empty")

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

    states = np.zeros(
        (intervals + 1, initial_state.size),
        dtype = float,
    )

    states[0] = initial_state.astype(float)

    for i in range(intervals):
        time = times[i]
        state = states[i]

        k1 = np.asarray(
            function(time, state),
            dtype = float,
        )

        k2 = np.asarray(
            function(
                time + step / 2.0,
                state + step * k1 / 2.0,
            ),
            dtype = float,
        )

        k3 = np.asarray(
            function(
                time + step / 2.0,
                state + step * k2 / 2.0,
            ),
            dtype = float,
        )

        k4 = np.asarray(
            function(
                time + step,
                state + step * k3,
            ),
            dtype = float,
        )

        # Every slope vector must describe the same state dimensions.
        for slope in (k1, k2, k3, k4):
            if slope.shape != initial_state.shape:
                raise ValueError(
                    "function output must match initial_state shape"
                )

        states[i + 1] = (
            state
            + step 
            * (
                k1
                + 2.0 * k2
                + 2.0 * k3
                + k4
            )
            / 6.0
        )

    return times, states

def system_final_state(
    function: Callable[[float, np.ndarray], np.ndarray],
    initial_time: float,
    initial_state: np.ndarray,
    final_time: float,
    step: float,
) -> np.ndarray:
    """
    Return only the final state of an RK4 system solution.
    """
    _, states = rk4_system(
        function,
        initial_time,
        initial_state,
        final_time,
        step,
    )

    return states[-1].copy()

def system_error(
    function: Callable[[float, np.ndarray], np.ndarray],
    exact_solution: Callable[[float], np.ndarray],
    initial_time: float,
    initial_state: np.ndarray,
    final_time: float,
    step: float,
) -> float:
    """
    Return the Euclidean error of the final numerical state.
    """

    approximation = system_final_state(
        function,
        initial_time,
        initial_state,
        final_time,
        step,
    )

    exact = np.asarray(
        exact_solution(final_time),
        dtype = float,
    )

    if exact.shape != initial_state.shape:
        raise ValueError(
            "exact_solution output must match initial_state shape"
        )

    return float(
        np.linalg.norm(
            approximation - exact
        )
    )