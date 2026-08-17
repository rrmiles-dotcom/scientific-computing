import numpy as np
import pytest

from monte_carlo_integration import (
    monte_carlo_integral,
    monte_carlo_integral_statistics,
    monte_carlo_absolute_error,
)

def test_integrates_constant_function():
    result = monte_carlo_integral(
        lambda x: np.full_like(x, 3.0),
        left = 0.0,
        right = 2.0,
        samples = 1000,
        seed = 42,
    )

    assert result == pytest.approx(6.0)

def test_integrates_linear_function_approximately():
    result = monte_carlo_integral(
        lambda x: x,
        left = 0.0,
        right = 1.0,
        samples = 100000,
        seed = 42, 
    )

    assert result == pytest.approx(
        0.5,
        abs = 0.005,
    )

def test_integrates_quadratic_function_approximately():
    result = monte_carlo_integral(
        lambda x: x**2,
        left = 0.0,
        right = 1.0,
        samples = 100000,
        seed = 42, 
    )

    assert result == pytest.approx(
        1.0 / 3.0,
        abs = 0.005,
    )

def test_integrates_sine_approximately():
    result = monte_carlo_integral(
        np.sin,
        left = 0.0,
        right = np.pi,
        samples = 100000,
        seed = 42,
    )

    assert result == pytest.approx(
        2.0,
        abs = 0.01,
    )

def test_same_seed_produces_same_estimates():
    first = monte_carlo_integral(
        lambda x: x**2,
        0.0,
        1.0,
        samples = 1000,
        seed = 42,
    )

    second = monte_carlo_integral(
        lambda x: x**2,
        0.0,
        1.0,
        samples = 1000,
        seed = 42,
    )

    assert first == second

def test_statistics_returns_estimate_and_standard_error():
    estimate, standard_error = (
        monte_carlo_integral_statistics(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = 10000,
            seed = 42,
        )
    )

    assert estimate == pytest.approx(
        0.5,
        abs = 0.02,
    )

    assert standard_error > 0.0

def test_constant_function_has_zero_standard_error():
    estimate, standard_error = (
        monte_carlo_integral_statistics(
            lambda x: np.full_like(x, 4.0),
            left = 0.0,
            right = 2.0,
            samples = 1000,
            seed = 42,
        )
    )

    assert estimate == pytest.approx(8.0)
    assert standard_error == pytest.approx(0.0)

def test_standard_error_matches_formula():
    samples = 1000
    seed = 42

    rng = np.random.default_rng(seed)

    x_values = rng.uniform(
        0.0,
        1.0,
        size = samples,
    )

    expected_standard_error = (
        np.std(
            x_values,
            ddof = 1,
        )
        / np.sqrt(samples)
    )

    _, standard_error = (
        monte_carlo_integral_statistics(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = samples,
            seed = seed,
        )
    )

    assert standard_error == pytest.approx(
        expected_standard_error
    )

def test_absolute_error_known_constant():
    result = monte_carlo_absolute_error(
        lambda x: np.full_like(x, 2.0),
        exact_integral = 4.0,
        left = 0.0,
        right = 2.0,
        samples = 100,
        seed = 42,
    )

    assert result == pytest.approx(0.0)

def test_absolute_error_is_nonnegative():
    result = monte_carlo_absolute_error(
        lambda x: x**2,
        exact_integral = 1.0 / 3.0,
        left = 0.0,
        right = 1.0,
        samples = 1000,
        seed = 42,
    )

    assert result >= 0.0

def test_rejects_equal_endpoints():
    with pytest.raises(ValueError):
        monte_carlo_integral(
            lambda x: x,
            left = 1.0,
            right = 1.0,
            samples = 100,
        )

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        monte_carlo_integral(
            lambda x: x,
            left = 2.0,
            right = 1.0,
            samples = 100,
        )

def test_rejects_zero_samples():
    with pytest.raises(ValueError):
        monte_carlo_integral(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = 0,
        )

def test_rejects_negative_samples():
    with pytest.raises(ValueError):
        monte_carlo_integral(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = -10,
        )

def test_rejects_noninteger_samples():
    with pytest.raises(TypeError):
        monte_carlo_integral(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = 10.5,
        )

def test_statistics_requires_two_samples():
    with pytest.raises(ValueError):
        monte_carlo_integral_statistics(
            lambda x: x,
            left = 0.0,
            right = 1.0,
            samples = 1,
        )

def test_rejects_wrong_function_output_shape():
    def function(x):
        return np.array([1.0])

    with pytest.raises(ValueError):
        monte_carlo_integral(
            function,
            left = 0.0,
            right = 1.0,
            samples = 100,
            seed = 42,
        )

def test_statistics_rejects_wrong_function_output_shape():
    def function(x):
        return np.array([1.0])

    with pytest.raises(ValueError):
        monte_carlo_integral_statistics(
            function,
            left = 0.0,
            right = 1.0,
            samples = 100,
            seed = 42,
        )