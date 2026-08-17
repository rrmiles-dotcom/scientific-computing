import numpy as np
import pytest

from simulation import(
    simulate_sample_means,
    simulate_estimator,
    estimate_bias,
    estimate_variance,
    mean_squared_error,
    confidence_interval_coverage,
)

def normal_sampler(
    size,
    rng,
):
    return rng.normal(
        loc = 5.0,
        scale = 2.0,
        size = size,
    )

def test_simulate_sample_means_shape():
    result = simulate_sample_means(
        normal_sampler,
        sample_size = 20,
        repetitions = 100,
        seed = 42,
    )

    assert result.shape == (100,)

def test_simulated_sample_means_center_near_population_mean():
    result = simulate_sample_means(
        normal_sampler,
        sample_size = 50,
        repetitions = 5000,
        seed = 42,
    )

    assert np.mean(result) == pytest.approx(
        5.0,
        abs = 0.05,
    )

def test_same_seed_reproduces_sample_means():
    first  = simulate_sample_means(
        normal_sampler,
        sample_size = 20,
        repetitions = 100,
        seed = 42,
    )

    second = simulate_sample_means(
        normal_sampler,
        sample_size = 20,
        repetitions = 100,
        seed = 42,
    )

    np.testing.assert_array_equal(
        first,
        second,
    )

def test_simulate_estimator_mean():
    result = simulate_estimator(
        normal_sampler,
        np.mean,
        sample_size = 20,
        repetitions = 5000,
        seed = 42,
    )

    assert np.mean(result) == pytest.approx(
        5.0,
        abs = 0.05,
    )

def test_simulate_estimator_variance():
    def sampler(size, rng):
        return rng.normal(
            0.0,
            1.0,
            size = size,
        )

    result = simulate_estimator(
        sampler,
        lambda samples: np.var(
            samples,
            ddof = 1,
        ),
        sample_size = 50,
        repetitions = 3000,
        seed = 42,
    )

    assert np.mean(result) == pytest.approx(
        1.0,
        abs = 0.05,
    )

def test_estimate_bias_unbiased_sequence():
    estimates = np.array([
        4.0,
        5.0,
        6.0,
    ])

    result = estimate_bias(
        estimates,
        true_value = 5.0,
    )

    assert result == pytest.approx(0.0)

def test_estimate_bias_positive():
    estimates = np.array([
        6.0,
        7.0,
        8.0,
    ])

    result = estimate_bias(
        estimates,
        true_value = 5.0,
    )

    assert result == pytest.approx(2.0)

def test_estimate_variance_known_values():
    estimates = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = estimate_variance(
        estimates
    )

    assert result == pytest.approx(1.0)

def test_mean_squared_error_known_values():
    estimates = np.array([
        4.0,
        5.0,
        6.0,
    ])

    result = mean_squared_error(
        estimates,
        true_value = 5.0,
    )

    expected = (
        1.0 + 0.0 + 1.0
    ) / 3.0

    assert result == pytest.approx(expected)

def test_mse_equals_variance_plus_bias_squared_approximately():
    estimates = np.array([
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ])

    true_value = 4.0

    mse = mean_squared_error(
        estimates,
        true_value,
    )

    bias = estimate_bias(
        estimates,
        true_value,
    )

    population_variance = np.var(
        estimates,
        ddof = 0,
    )

    assert mse == pytest.approx(
        population_variance + bias**2
    )

def test_confidence_interval_coverage_known_result():
    intervals = np.array([
        [0.0, 2.0],
        [1.0, 3.0],
        [3.0, 4.0],
        [-1.0, 1.0],
    ])

    result = confidence_interval_coverage(
        intervals,
        true_value = 1.5,
    )

    assert result == pytest.approx(0.5)

def test_coverage_includes_interval_boundaries():
    intervals = np.array([
        [1.0, 2.0],
        [0.0, 1.0],
    ])

    result = confidence_interval_coverage(
        intervals,
        true_value = 1.0,
    )

    assert result == pytest.approx(1.0)

def test_simulation_rejects_noninteger_sample_size():
    with pytest.raises(TypeError):
        simulate_sample_means(
            normal_sampler,
            sample_size = 10.5,
            repetitions = 100,
        )

def test_simulation_rejects_noninteger_repetitions():
    with pytest.raises(TypeError):
        simulate_sample_means(
            normal_sampler,
            sample_size = 10,
            repetitions = 100.5,
        )

def test_simulation_rejects_nonpositive_sample_size():
    with pytest.raises(ValueError):
        simulate_sample_means(
            normal_sampler,
            sample_size = 0,
            repetitions = 100,
        )

def test_simulation_rejects_nonpositive_repetitions():
    with pytest.raises(ValueError):
        simulate_sample_means(
            normal_sampler,
            sample_size = 10,
            repetitions = 0,
        )

def test_simulation_rejects_wrong_sampler_shape():
    def bad_sampler(size, rng):
        return np.ones(size + 1)

    with pytest.raises(ValueError):
        simulate_sample_means(
            bad_sampler,
            sample_size = 10,
            repetitions = 5,
            seed = 42,
        )

def test_estimator_simulation_rejects_wrong_sampler_shape():
    def bad_sampler(size, rng):
        return np.ones(size + 1)

    with pytest.raises(ValueError):
        simulate_estimator(
            bad_sampler,
            np.mean,
            sample_size = 10,
            repetitions = 5,
            seed = 42,
        )

def test_bias_rejects_non_vector_input():
    with pytest.raises(ValueError):
        estimate_bias(
            np.ones((2, 2)),
            true_value = 0.0,
        )

def test_bias_rejects_empty_input():
    with pytest.raises(ValueError):
        estimate_bias(
            np.array([]),
            true_value = 0.0,
        )

def test_variance_requires_two_estimates():
    with pytest.raises(ValueError):
        estimate_variance(
            np.array([1.0])
        )

def test_mse_rejects_empty_input():
    with pytest.raises(ValueError):
        mean_squared_error(
            np.array([]),
            true_value = 0.0,
        )

def test_coverage_rejects_non_matrix_input():
    with pytest.raises(ValueError):
        confidence_interval_coverage(
            np.array([
                1.0,
                2.0,
            ]),
            true_value = 1.5,
        )


def test_coverage_rejects_wrong_number_of_columns():
    with pytest.raises(ValueError):
        confidence_interval_coverage(
            np.ones((4, 3)),
            true_value = 0.0,
        )

def test_coverage_rejects_empty_intervals():
    with pytest.raises(ValueError):
        confidence_interval_coverage(
            np.empty((0, 2)),
            true_value = 0.0,
        )

def test_coverage_rejects_reversed_bounds():
    intervals = np.array([
        [2.0, 1.0],
        [0.0, 3.0],
    ])

    with pytest.raises(ValueError):
        confidence_interval_coverage(
            intervals,
            true_value = 1.0,
        )