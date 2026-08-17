import numpy as np

def _validate_size(size: int) -> None:
    """Validate the requested sample size."""

    if not isinstance(size, (int, np.integer)):
        raise TypeError("size must be an integer")

    if size <= 0:
        raise ValueError("size must be positive")

def uniform_samples(
    size: int,
    low: float = 0.0,
    high: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples from a uniform distribution.
    """

    _validate_size(size)

    if high <= low:
        raise ValueError("high must be greater than low")

    rng = np.random.default_rng(seed)

    return rng.uniform(
        low = low,
        high = high,
        size = size,
    )

def normal_samples(
    size: int,
    mean: float = 0.0,
    standard_deviation: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples from a normal distribution.
    """

    _validate_size(size)

    if standard_deviation <= 0:
        raise ValueError(
            "standard_deviation must be positive"
        )

    rng = np.random.default_rng(seed)

    return rng.normal(
        loc = mean,
        scale = standard_deviation,
        size = size,
    )

def bernoulli_samples(
    size: int,
    probability: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate Bernoulli samples taking values 0 or 1.
    """

    _validate_size(size)

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must satisfy 0 <= p <= 1"
        )

    rng = np.random.default_rng(seed)

    return rng.binomial(
        n = 1,
        p = probability,
        size = size,
    )

def binomial_samples(
    size: int,
    trials: int,
    probability: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate samples from a binomial distribution.
    """

    _validate_size(size)

    if not isinstance(trials, (int, np.integer)):
        raise TypeError("trials must be an integer")

    if trials <= 0:
        raise ValueError("trials must be positive")

    if not 0.0 <= probability <= 1.0:
        raise ValueError(
            "probability must satisfy 0 <= p <= 1"
        )

    rng = np.random.default_rng(seed)

    return rng.binomial(
        n = trials,
        p = probability,
        size = size,
    )

def sample_mean(samples: np.ndarray) -> float:
    """Return the empirical sample mean."""

    samples = np.asarray(
        samples,
        dtype = float,
    )

    if samples.ndim != 1:
        raise ValueError("samples must be one-dimensional")

    if samples.size == 0:
        raise ValueError("samples must not be empty")

    return float(
        np.mean(samples)
    )

def sample_variance(samples: np.ndarray) -> float:
    """
    Return the unbiased sample variance.
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

    # ddof = 1 gives the usual unbiased estimator of poulation variance.
    return float(
        np.var(samples, ddof = 1)
    )