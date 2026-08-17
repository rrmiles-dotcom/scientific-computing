from collections.abc import Callable

import numpy as np

def simulate_sample_means(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    sample_size: int,
    repetitions: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate the sampling distribution of the sample mean.
    """

    if not isinstance(sample_size, (int, np.integer)):
        raise TypeError("sample_size must be an integer")

    if not isinstance(repetitions, (int, np.integer)):
        raise TypeError("repetitions must be an integer")

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rng = np.random.default_rng(seed)

    means = np.zeros(
        repetitions,
        dtype = float,
    )

    for i in range(repetitions):
        samples = np.asarray(
            sampler(sample_size, rng),
            dtype = float,
        )

        if samples.shape != (sample_size,):
            raise ValueError(
                "sampler output must match sample_size"
            )

        means[i] = np.mean(samples)

    return means

def simulate_estimator(
    sampler: Callable[[int, np.random.Generator], np.ndarray],
    estimator: Callable[[np.ndarray], float],
    sample_size: int,
    repetitions: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Apply an estimator repeatedly to independently simulated datasets.
    """

    if not isinstance(sample_size, (int, np.integer)):
        raise TypeError("sample_size must be an integer")

    if not isinstance(repetitions, (int, np.integer)):
        raise TypeError("repetitions must be an integer")

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")

    if repetitions <= 0:
        raise ValueError("repetitions must be positive")

    rng = np.random.default_rng(seed)

    estimates = np.zeros(
        repetitions,
        dtype = float, 
    )

    for i in range(repetitions):
        samples = np.asarray(
            sampler(sample_size, rng),
            dtype = float,
        )

        if samples.shape != (sample_size,):
            raise ValueError(
                "sampler output must match sample_size"
            )

        estimates[i] = float(
            estimator(samples)
        )

    return estimates

def estimate_bias(
    estimates: np.ndarray,
    true_value: float,
) -> float:
    """
    Estimate estimator bias from repeated simzlation results.
    """

    estimates = np.asarray(
        estimates,
        dtype = float,
    )

    if estimates.ndim != 1:
        raise ValueError("estimates must be one-dimensional")

    if estimates.size == 0:
        raise ValueError("estimates must not be empty")

    return float(
        np.mean(estimates) - true_value
    )

def estimate_variance(
    estimates: np.ndarray,
) -> float:
    """
    Estimate the variance of an estimator across repeated simulations.
    """

    estimates = np.asarray(
        estimates,
        dtype = float,
    )

    if estimates.ndim != 1:
        raise ValueError("estimates must be one-dimensional")

    if estimates.size < 2:
        raise ValueError(
            "at least two estimates are required"
        )

    return float(
        np.var(
            estimates,
            ddof = 1,
        )
    )

def mean_squared_error(
    estimates: np.ndarray,
    true_value: float,
) -> float:
    """
    Estimates mean sqaured error relative to the true parameter value.
    """

    estimates = np.asarray(
        estimates,
        dtype = float,
    )

    if estimates.ndim != 1:
        raise ValueError("estimates must be one-dimensional")

    if estimates.size == 0:
        raise ValueError("estimates must not be empty")

    errors = estimates - true_value

    return float(
        np.mean(errors**2)
    )

def confidence_interval_coverage(
    intervals: np.ndarray,
    true_value: float,
) -> float:
    """
    Estimate the proportion of confidence intervals a true value. 
    """

    intervals = np.asarray(
        intervals,
        dtype = float,
    )

    if intervals.ndim != 2:
        raise ValueError("intervals must be two-dimensional")

    if intervals.shape[1] != 2:
        raise ValueError(
            "intervals must contain lower and upper bounds"
        )

    if intervals.shape[0] == 0:
        raise ValueError("intervals must not be empty")

    if np.any(
        intervals[:, 0] > intervals[:, 1]
    ):
        raise ValueError(
            "lower bounds must not exceed upper bounds"
        )

    # Coverage is the empirical probability that an interval contains the truth.
    contains = (
        (intervals[:, 0] <= true_value)
        & (true_value <= intervals[:, 1])
    )

    return float(
        np.mean(contains)
    )    