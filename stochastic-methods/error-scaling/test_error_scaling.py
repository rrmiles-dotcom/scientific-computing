import numpy as np
import pytest

from error_scaling import (
    theoretical_standard_error,
    expected_error_ratio,
    scaled_errors,
    estimate_scaling_exponent,
    root_n_scaled_errors,
)

def test_theoretical_standard_error():
    result = theoretical_standard_error(
        standard_deviation = 2.0,
        samples = 100,
    )

    assert result == pytest.approx(0.2)

def test_standard_error_decreases_with_sample_size():
    small = theoretical_standard_error(
        standard_deviation = 1.0,
        samples = 100,
    )

    large = theoretical_standard_error(
        standard_deviation = 1.0,
        samples = 400,
    )

    assert large < small

def test_quadrupling_samples_halves_standard_error():
    small = theoretical_standard_error(
        standard_deviation = 3.0,
        samples = 100,
    )

    large = theoretical_standard_error(
        standard_deviation = 3.0,
        samples = 400,
    )

    assert large == pytest.approx(
        small / 2.0
    )

def test_expected_error_ratio():
    result = expected_error_ratio(
        first_samples = 100,
        second_samples = 400,
    )

    assert result == pytest.approx(0.5)

def test_scaled_errors_follow_inverse_square_root():
    sample_sizes = np.array([
        100,
        400,
        1600,
    ])

    result = scaled_errors(
        sample_sizes,
        constant = 2.0,
    )

    expected = np.array([
        0.2,
        0.1,
        0.05,
    ])

    np.testing.assert_allclose(
        result,
        expected,
        atol = 1e-12,
    )

def test_estimate_scaling_exponent_is_minus_half():
    sample_sizes = np.array([
        100,
        400,
        1600,
        6400,
    ])

    errors = (
        3.0
        / np.sqrt(sample_sizes)
    )

    result = estimate_scaling_exponent(
        sample_sizes,
        errors,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, -0.5),
        atol = 1e-12,
    )

def test_root_n_scaled_errors_are_constant():
    sample_sizes = np.array([
        100,
        400,
        1600,
        6400,
    ])

    errors = (
        5.0
        / np.sqrt(sample_sizes)
    )

    result = root_n_scaled_errors(
        sample_sizes,
        errors,
    )

    np.testing.assert_allclose(
        result,
        np.full(4, 5.0),
        atol = 1e-12,
    )

def test_zero_standard_deviation_gives_zero_error():
    result = theoretical_standard_error(
        standard_deviation = 0.0,
        samples = 100,
    )

    assert result == pytest.approx(0.0)

def test_rejects_negative_standard_deviation():
    with pytest.raises(ValueError):
        theoretical_standard_error(
            standard_deviation = -1.0,
            samples = 100,
        )

def test_rejects_noninteger_sample_size():
    with pytest.raises(TypeError):
        theoretical_standard_error(
            standard_deviation = 1.0,
            samples = 10.5,
        )

def test_rejects_nonpositive_sample_size():
    with pytest.raises(ValueError):
        theoretical_standard_error(
            standard_deviation = 1.0,
            samples = 0,
        )

def test_expected_ratio_rejects_noninteger_sample_size():
    with pytest.raises(TypeError):
        expected_error_ratio(
            first_samples = 100.5,
            second_samples = 400,
        )

def test_expected_ratio_rejects_nonpositive_sample_size():
    with pytest.raises(ValueError):
        expected_error_ratio(
            first_samples = 100,
            second_samples = 0,
        )

def test_scaled_errors_reject_non_vector_input():
    with pytest.raises(ValueError):
        scaled_errors(
            np.ones((2, 2))
        )

def test_scaled_errors_reject_empty_input():
    with pytest.raises(ValueError):
        scaled_errors(
            np.array([])
        )

def test_scaled_errors_reject_nonpositive_samples():
    with pytest.raises(ValueError):
        scaled_errors(
            np.array([
                100,
                0,
                400,
            ])
        )

def test_scaled_errors_reject_negative_constant():
    with pytest.raises(ValueError):
        scaled_errors(
            np.array([
                100,
                400,
            ]),
            constant = -1.0,
        )

def test_estimate_scaling_rejects_length_mismatch():
    with pytest.raises(ValueError):
        estimate_scaling_exponent(
            np.array([
                100,
                400,
                1600,
            ]),
            np.array([
                0.1,
                0.05,
            ]),
        )

def test_estimate_scaling_rejects_nonpositive_errors():
    with pytest.raises(ValueError):
        estimate_scaling_exponent(
            np.array([
                100,
                400,
            ]),
            np.array([
                0.1,
                0.0,
            ]),
        )

def test_root_n_scaled_errors_reject_length_mismatch():
    with pytest.raises(ValueError):
        root_n_scaled_errors(
            np.array([
                100,
                400,
            ]),
            np.array([
                0.1,
            ]),
        )