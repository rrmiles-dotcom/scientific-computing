import math
import numpy as np

def sample_variance(
    samples: np.ndarray,
) -> float:
    """Return the unbiased sample variance."""

    samples = np.asarray(
        samples,
        dtype = float,
    )

    if samples.ndim != 1:
        raise ValueError(
            "samples must be one-dimensional"
        )

    if samples.size < 2:
        raise ValueError(
            "atleast two samples are required"
        )

    return float(
        np.var(
            samples,
            ddof = 1,
        )
    )

def sample_standard_deviation(
    samples: np.ndarray,
) -> float:
    """Return the unbiased sample standard deviation."""

    return float(
        math.sqrt(
            sample_variance(samples)
        )
    )

def standard_error_mean(
    samples: np.ndarray,
) -> float:
    """Return the estimated standard error of the sample mean."""

    samples = np.asarray(
        samples,
        dtype = float,
    )

    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")

    if samples.size < 2:
        raise ValueError(
            "at least two samples are required"
        )

    standard_deviation = (
        sample_standard_deviation(samples)
    )

    # Uncertainty in the sample mean decreases at the root-N rate.
    return float(
        standard_deviation
        / math.sqrt(samples.size)
    )

def normal_confidence_interval(
    samples: np.ndarray,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """
    Return a normal-approximation confidence interval for the mean.
    """

    samples = np.asarray(
        samples,
        dtype = float,
    )

    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")

    if samples.size < 2:
        raise ValueError(
            "at least two samples are required"
        )

    if z_value <= 0:
        raise ValueError("z_value must be positive")

    mean = float(
        np.mean(samples)
    )

    standard_error = standard_error_mean(
        samples
    )

    margin = (
        z_value
        * standard_error
    )

    return (
        mean - margin,
        mean + margin,
    )

def confidence_interval_width(
    samples: np.ndarray,
    z_value: float = 1.96,
) -> float:
    """
    Return the width of a normal confidence interval.
    """

    lower, upper = normal_confidence_interval(
        samples,
        z_value,
    )

    return float(
        upper - lower
    )

def confidence_interval_from_statistics(
    mean: float,
    standard_deviation: float,
    samples: int,
    z_value: float = 1.96,
) -> tuple[float, float]:
    """
    Construct a normal confidence interval from summary statistics.
    """

    if standard_deviation < 0:
        raise ValueError(
            "standard_deviation must be nonnegative"
        )

    if not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be an integer")

    if samples <= 0:
        raise ValueError("samples must be positive")

    if z_value <= 0:
        raise ValueError("z_value must be positive")

    standard_error = (
        standard_deviation
        / math.sqrt(samples)
    )

    margin = (
        z_value
        * standard_error
    )

    return (
        float(mean - margin),
        float(mean + margin),
    )