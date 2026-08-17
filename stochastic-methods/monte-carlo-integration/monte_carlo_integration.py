from collections.abc import Callable

import numpy as np

def monte_carlo_integral(
    function: Callable[[np.ndarray], np.ndarray],
    left: float,
    right: float,
    samples: int,
    seed: int | None = None,
) -> float:
    """
    Estimate a one-dimensional definite integral using Monte Carlo sampling.
    """

    if left >= right:
        raise ValueError("left endpoint must be smaller than right endpoint")

    if not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be positive")

    if samples <= 0:
        raise ValueError("samples must be positive")

    rng = np.random.default_rng(seed)

    x_values = rng.uniform(
        low = left,
        high = right,
        size = samples,
    )

    function_values = np.asarray(
        function(x_values),
        dtype = float,
    )

    if function_values.shape != x_values.shape:
        raise ValueError(
            "function output must match sampled input shape"
        )

    # Uniform sampling converts the integral into interval width times E[f(X)].
    estimate = (
        (right - left)
        * np.mean(function_values)
    )

    return float(estimate)

def monte_carlo_integral_statistics(
    function: Callable[[np.ndarray], np.ndarray],
    left: float,
    right: float,
    samples: int,
    seed: int | None = None,
) -> tuple[float, float]:
    """
    Return the Monte Carlo integral estimate and its estimated standard error.
    """

    if left >= right:
        raise ValueError("left endpoint must be smaller than right endpoint")

    if not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be an integer")

    if samples < 2:
        raise ValueError(
            "atleast two samples are required"
        )

    rng = np.random.default_rng(seed)

    x_values = rng.uniform(
        left,
        right,
        size = samples,
    )

    function_values = np.asarray(
        function(x_values),
        dtype = float,
    )

    if function_values.shape != x_values.shape:
        raise ValueError(
            "function output must match sampled input shape"
        )

    width = right - left

    estimate = (
        width
        * np.mean(function_values)
    ) 

    # Standard error follows from the sample variance of f(X).
    standard_error = (
        width
        * np.std(
            function_values,
            ddof = 1,
        )
        / np.sqrt(samples)
    )

    return (
        float(estimate),
        float(standard_error),
    )

def monte_carlo_absolute_error(
    function: Callable[[np.ndarray], np.ndarray],
    exact_integral: float,
    left: float,
    right: float,
    samples: int,
    seed: int | None = None,
) -> float:
    """
    Return the absolute error of a Monte Carlo integral estimate.
    """

    estimate = monte_carlo_integral(
        function,
        left,
        right,
        samples,
        seed,
    )

    return float(
        abs(estimate - exact_integral)
    )