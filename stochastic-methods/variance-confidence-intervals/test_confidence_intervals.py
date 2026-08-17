import numpy as np
import pytest

from confidence_intervals import (
    sample_variance,
    sample_standard_deviation,
    standard_error_mean,
    normal_confidence_interval,
    confidence_interval_width,
    confidence_interval_from_statistics,
)

def test_sample_variance_known_values():
    samples = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = sample_variance(samples)

    assert result == pytest.approx(1.0)

def test_sample_standard_deviation_known_values():
    samples = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = sample_standard_deviation(samples)

    assert result == pytest.approx(1.0)

def test_standard_error_known_values():
    samples = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    expected = (
        np.std(samples, ddof = 1)
        / np.sqrt(samples.size)
    )

    result = standard_error_mean(samples)

    assert result == pytest.approx(expected)

def test_standard_error_decreases_with_more_samples():
    small = np.array([
        1.0,
        3.0,
    ])

    large = np.array([
        1.0,
        3.0,
        1.0,
        3.0,
        1.0,
        3.0,
        1.0,
        3.0,
    ])

    assert (
        standard_error_mean(large)
        < standard_error_mean(small)
    )

def test_normal_confidence_interval_known_result():
    samples = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    mean = np.mean(samples)
    se = np.std(samples, ddof = 1) / np.sqrt(samples.size)

    expected_lower = mean - 1.96 * se
    expected_upper = mean + 1.96 * se

    lower, upper = normal_confidence_interval(samples)

    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)

def test_confidence_interval_is_centered_on_mean():
    samples = np.array([
        2.0,
        4.0,
        6.0,
        8.0,
    ])

    lower, upper = normal_confidence_interval(samples)

    midpoint = (lower + upper) / 2.0

    assert midpoint == pytest.approx(
        np.mean(samples)
    )

def test_confidence_interval_width():
    samples = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    lower, upper = normal_confidence_interval(samples)

    result = confidence_interval_width(samples)

    assert result == pytest.approx(
        upper - lower
    )

def test_larger_z_value_gives_wider_interval():
    samples = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    narrow = confidence_interval_width(
        samples,
        z_value = 1.0,
    )

    wide = confidence_interval_width(
        samples,
        z_value = 2.0,
    )

    assert wide > narrow

def test_confidence_interval_from_statistics():
    lower, upper = confidence_interval_from_statistics(
        mean = 100.0,
        standard_deviation = 10.0,
        samples = 100,
        z_value = 1.96,
    )

    expected_margin = 1.96

    assert lower == pytest.approx(
        100.0 - expected_margin
    )

    assert upper == pytest.approx(
        100.0 + expected_margin
    )

def test_zero_standard_deviation_gives_zero_width():
    lower, upper = confidence_interval_from_statistics(
        mean = 5.0,
        standard_deviation = 0.0,
        samples = 100,
    )

    assert lower == pytest.approx(5.0)
    assert upper == pytest.approx(5.0)

def test_rejects_non_vector_samples():
    with pytest.raises(ValueError):
        sample_variance(
            np.ones((2, 2))
        )

def test_variance_requires_two_samples():
    with pytest.raises(ValueError):
        sample_variance(
            np.array([1.0])
        )

def test_standard_error_requires_two_samples():
    with pytest.raises(ValueError):
        standard_error_mean(
            np.array([1.0])
        )

def test_confidence_interval_requires_two_samples():
    with pytest.raises(ValueError):
        normal_confidence_interval(
            np.array([1.0])
        )

def test_rejects_nonpositive_z_value():
    with pytest.raises(ValueError):
        normal_confidence_interval(
            np.array([
                1.0,
                2.0,
            ]),
            z_value = 0.0,
        )

def test_statistics_reject_negative_standard_deviation():
    with pytest.raises(ValueError):
        confidence_interval_from_statistics(
            mean = 0.0,
            standard_deviation = -1.0,
            samples = 100,
        )

def test_statistics_reject_noninteger_sample_size():
    with pytest.raises(TypeError):
        confidence_interval_from_statistics(
            mean = 0.0,
            standard_deviation = 1.0,
            samples = 10.5,
        )

def test_statistics_reject_nonpositive_sample_size():
    with pytest.raises(ValueError):
        confidence_interval_from_statistics(
            mean = 0.0,
            standard_deviation = 1.0,
            samples = 0,
        )

def test_statistics_reject_nonpositive_z_value():
    with pytest.raises(ValueError):
        confidence_interval_from_statistics(
            mean = 0.0,
            standard_deviation = 1.0,
            samples = 100,
            z_value = 0.0,
        )