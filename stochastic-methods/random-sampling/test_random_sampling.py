import numpy as np
import pytest

from random_sampling import (
    uniform_samples,
    normal_samples,
    bernoulli_samples,
    binomial_samples,
    sample_mean,
    sample_variance,
)

def test_uniform_sample_size():
    result = uniform_samples(
        size = 100,
        seed = 42,
    )

    assert result.shape == (100,)

def test_uniform_samples_within_bounds():
    result = uniform_samples(
        size = 1000,
        low = -2.0,
        high = 3.0,
        seed = 42,
    )

    assert np.all(result >= -2.0)
    assert np.all(result < 3.0)

def test_normal_sample_size():
    result = normal_samples(
        size = 100,
        seed = 42,
    )

    assert result.shape == (100,)

def test_normal_mean_approximately_correct():
    result = normal_samples(
        size = 100000,
        mean = 5.0,
        standard_deviation = 2.0,
        seed = 42,
    )

    assert np.mean(result) == pytest.approx(
        5.0,
        abs = 0.03,
    )

def test_normal_standard_deviation_approximately_correct():
    result = normal_samples(
        size = 100000,
        mean = 0.0,
        standard_deviation = 3.0,
        seed = 42,
    )

    assert np.std(result) == pytest.approx(
        3.0,
        abs = 0.03,
    )

def test_bernoulli_contains_only_zero_and_one():
    result = bernoulli_samples(
        size = 1000,
        probability = 0.4,
        seed = 42,
    )

    assert set(np.unique(result)).issubset({
        0,
        1,
    })

def test_bernoulli_probability_approximately_correct():
    result = bernoulli_samples(
        size = 100000,
        probability = 0.7,
        seed = 42,
    )

    assert np.mean(result) == pytest.approx(
        0.7,
        abs = 0.01,
    )

def test_binomial_values_within_bonds():
    result = binomial_samples(
        size = 1000,
        trials = 10,
        probability = 0.5,
        seed = 42,
    )

    assert np.all(result >= 0)
    assert np.all(result <= 10)


def test_binomial_mean_approximately_correct():
    result = binomial_samples(
        size = 100000,
        trials = 10,
        probability = 0.3,
        seed = 42,
    )

    expected_mean = 10 * 0.3

    assert np.mean(result) == pytest.approx(
        expected_mean,
        abs = 0.03,
    )

def test_same_seed_produces_same_samples():
    first = normal_samples(
        size = 100,
        seed = 42,
    )

    second = normal_samples(
        size = 100,
        seed = 42,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

def test_different_seeds_produce_different_samples():
    first = normal_samples(
        size = 100,
        seed = 1,
    )

    second = normal_samples(
        size = 100,
        seed = 2,
    )

    assert not np.array_equal(
        first,
        second,
    )

def test_sample_mean_known_values():
    samples = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    result = sample_mean(samples)

    assert result == pytest.approx(2.5)

def test_sample_variance_known_values():
    samples = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = sample_variance(samples)

    assert result == pytest.approx(1.0)

def test_rejects_zero_sample_size():
    with pytest.raises(ValueError):
        uniform_samples(
            size = 0,
        )

def test_rejects_noninteger_sample_size():
    with pytest.raises(TypeError):
        uniform_samples(
            size = 10.5,
        )

def test_uniform_rejects_invalid_bounds():
    with pytest.raises(ValueError):
        uniform_samples(
            size = 10,
            low = 2.0,
            high = 1.0,
        )

def test_normal_rejects_nonpositive_standard_deviation():
    with pytest.raises(ValueError):
        normal_samples(
            size = 10,
            standard_deviation = 0.0,
        )

def test_bernoulli_rejects_invalid_probability():
    with pytest.raises(ValueError):
        bernoulli_samples(
            size = 10,
            probability = 1.5,
        )

def test_binomial_rejects_nonpositive_trials():
    with pytest.raises(ValueError):
        binomial_samples(
            size = 10,
            trials = 0,
            probability = 0.5,
        )

def test_binomial_rejects_noninteger_trials():
    with pytest.raises(TypeError):
        binomial_samples(
            size = 10,
            trials = 2.5,
            probability = 0.5,
        )

def test_sample_mean_rejects_empty_array():
    with pytest.raises(ValueError):
        sample_mean(
            np.array([])
        )

def test_sample_mean_rejects_non_vector():
    with pytest.raises(ValueError):
        sample_mean(
            np.ones((2, 2))
        )

def test_sample_variance_requires_two_samples():
    with pytest.raises(ValueError):
        sample_variance(
            np.array([1.0])
        )