import numpy as np

def theoretical_standard_error(
    standard_deviation: float,
    samples: int,
) -> float:
    """
    Return the theoretical standard error sigma / sqrt(N).
    """

    if standard_deviation < 0:
        raise ValueError("standard_deviation must be nonnegative")

    if not isinstance(samples, (int, np.integer)):
        raise TypeError("samples must be an integer")

    if samples <= 0:
        raise ValueError("samples must be positive")

    return float(
        standard_deviation / np.sqrt(samples)
    )

def expected_error_ratio(
    first_samples: int,
    second_samples: int,
) -> float:
    """
    Return the theoretical Monte Carlo error ratio
    error_2 / error_1
    """

    if not isinstance(first_samples, (int, np.integer)):
        raise TypeError("first_samples must be an integer")

    if not isinstance(second_samples, (int, np.integer)):
        raise TypeError("second_samples must be an integer")

    if first_samples <= 0 or second_samples <= 0:
        raise ValueError("sample sizes must be positive")

    return float(
        np.sqrt(first_samples / second_samples)
    )

def scaled_errors(
    sample_sizes: np.ndarray,
    constant: float = 1.0,
) -> np.ndarray:
    """
    Return theoretical errors C / sqrt(N) for several sample sizes.
    """

    if sample_sizes.ndim != 1:
        raise ValueError("sample_sizes must be one-dimensional")

    if sample_sizes.size == 0:
        raise ValueError("sample_sizes must not be empty")

    if np.any(sample_sizes <= 0):
        raise ValueError("sample_sizes must be positive")

    if constant < 0:
        raise ValueError("constant must be nonnegative")

    return (
        constant
        / np.sqrt(sample_sizes.astype(float))
    )

def estimate_scaling_exponent(
    sample_sizes: np.ndarray,
    errors: np.ndarray,
) -> np.ndarray:
    """
    Estimate the exponent p in the model error approximately C N^p.
    """

    if sample_sizes.ndim != 1 or errors.ndim != 1:
        raise ValueError("sample_sizes and errors must be one-dimensional")

    if sample_sizes.size != errors.size:
        raise ValueError("sample_sizes and errors must have equal length")

    if sample_sizes.size < 2:
        raise ValueError("at least two data points are required")

    if np.any(sample_sizes <= 0):
        raise ValueError("sample sizes must be positive")

    if np.any(errors <= 0):
        raise ValueError("errors must be positive")

    # Log ratios eliminate the unknown constant in error = C N^p.
    return (
        np.log(errors[1:] / errors[:-1])
        / np.log(sample_sizes[1:] / sample_sizes[:-1])
    )

def root_n_scaled_errors(
    sample_sizes: np.ndarray,
    errors: np.ndarray,
) -> np.ndarray:
    """
    Return sqrt(N) * error to test 1/sqrt(N) scaling.
    """

    if sample_sizes.ndim != 1 or errors.ndim != 1:
        raise ValueError("sample_sizes and errors must be one-dimensional")

    if sample_sizes.size != errors.size:
        raise ValueError("sample_sizes and errors must have equal length")

    if sample_sizes.size == 0:
        raise ValueError("data must not be empty")

    if np.any(sample_sizes <= 0):
        raise ValueError("sample sizes must be positive")

    if np.any(errors < 0):
        raise ValueError("errors must be nonnegative")

    return (
        np.sqrt(sample_sizes.astype(float))
        * errors
    )