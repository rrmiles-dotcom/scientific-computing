import numpy as np
import pytest

from optimization_convergence import(
    parameter_errors,
    objective_gaps,
    gradient_norms,
    convergence_ratios,
    estimate_convergence_order,
    asymptotic_constants,
)

def test_parameter_errors_known_values():
    iterates = np.array([
        [3.0, 4.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ])

    optimum = np.array([
        0.0,
        0.0,
    ])

    result = parameter_errors(
        iterates,
        optimum,
    )

    expected = np.array([
        5.0,
        2.0,
        0.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_parameter_errors_at_optimum_are_zero():
    optimum = np.array([
        2.0,
        -1.0,
    ])

    iterates = np.array([
        [2.0, -1.0],
        [2.0, -1.0],
    ])

    result = parameter_errors(
        iterates,
        optimum,
    )

    np.testing.assert_allclose(
        result,
        np.zeros(2),
    )

def test_objective_gaps_known_values():
    values = np.array([
        10.0,
        5.0,
        2.0,
    ])

    result = objective_gaps(
        values,
        optimum_value = 2.0,
    )

    np.testing.assert_allclose(
        result,
        np.array([
            8.0,
            3.0,
            0.0,
        ]),
    )

def test_objective_gaps_use_absolute_difference():
    values = np.array([
        1.0,
        3.0,
    ])

    result = objective_gaps(
        values,
        optimum_value = 2.0,
    )

    np.testing.assert_allclose(
        result,
        np.array([
            1.0,
            1.0,
        ]),
    )

def test_gradient_norms_known_values():
    def gradient(point):
        return 2.0 * point

    iterates = np.array([
        [3.0, 4.0],
        [0.0, 2.0],
        [0.0, 0.0],
    ])

    result = gradient_norms(
        gradient,
        iterates,
    )

    expected = np.array([
        10.0,
        4.0,
        0.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_convergence_ratios_linear_sequence():
    errors = np.array([
        1.0,
        0.5,
        0.25,
        0.125,
    ])

    result = convergence_ratios(errors)

    np.testing.assert_allclose(
        result,
        np.full(3, 0.5),
    )

def test_estimate_linear_convergence_order():
    errors = np.array([
        0.8,
        0.4,
        0.2,
        0.1,
    ])

    result = estimate_convergence_order(
        errors
    )

    np.testing.assert_allclose(
        result,
        np.ones(2),
        atol = 1e-12,
    )

def test_estimate_quadratic_convergence_order():
    errors = np.array([
        0.5,
        0.25,
        0.0625,
        0.00390625,
    ])

    result = estimate_convergence_order(
        errors
    )

    np.testing.assert_allclose(
        result,
        np.array([
            2.0,
            2.0,
        ]),
        atol = 1e-12,
    )

def test_asymptotic_constants_linear():
    errors = np.array([
        1.0,
        0.5,
        0.25,
        0.125,
    ])

    result = asymptotic_constants(
        errors,
        order = 1.0,
    )

    np.testing.assert_allclose(
        result,
        np.full(3, 0.5)
    )

def test_asymptotic_constants_quadratic():
    errors = np.array([
        0.5,
        0.25,
        0.0625,
    ])

    result = asymptotic_constants(
        errors,
        order = 2.0,
    )

    np.testing.assert_allclose(
        result,
        np.ones(2)
    )

def test_parameter_errors_reject_non_matrix_iterates():
    with pytest.raises(ValueError):
        parameter_errors(
            np.ones(3),
            np.ones(3),
        )

def test_parameter_errors_reject_non_vector_optimum():
    with pytest.raises(ValueError):
        parameter_errors(
            np.ones((3, 2)),
            np.ones((1, 2)),
        )

def test_parameter_errors_reject_dimension_mismatch():
    with pytest.raises(ValueError):
        parameter_errors(
            np.ones((3, 2)),
            np.ones(3),
        )

def test_parameter_errors_reject_empty_iterates():
    with pytest.raises(ValueError):
        parameter_errors(
            np.empty((0, 2)),
            np.ones(2),
        )

def test_objective_gaps_reject_non_vector_values():
    with pytest.raises(ValueError):
        objective_gaps(
            np.ones((2, 2)),
            optimum_value = 0.0,
        )

def test_objective_gaps_reject_empty_values():
    with pytest.raises(ValueError):
        objective_gaps(
            np.array([]),
            optimum_value = 0.0,
        )

def test_gradient_norms_reject_wrong_gradient_shape():
    def gradient(point):
        return np.ones(3)

    with pytest.raises(ValueError):
        gradient_norms(
            gradient,
            np.ones((4, 2)),
        )

def test_convergence_ratios_reject_too_few_errors():
    with pytest.raises(ValueError):
        convergence_ratios(
            np.array([1.0])
        )

def test_convergence_ratios_reject_negative_errors():
    with pytest.raises(ValueError):
        convergence_ratios(
            np.array([
                1.0,
                -0.5,
            ])
        )

def test_convergence_ratios_reject_zero_preceding_error():
    with pytest.raises(ValueError):
        convergence_ratios(
            np.array([
                1.0,
                0.0,
                0.1,
            ])
        )

def test_estimate_order_rejects_too_few_errors():
    with pytest.raises(ValueError):
        estimate_convergence_order(
            np.array([
                1.0,
                0.5,
            ])
        )

def test_estimate_order_rejects_nonpositive_errors():
    with pytest.raises(ValueError):
        estimate_convergence_order(
            np.array([
                1.0,
                0.5,
                0.0,
            ])
        )

def test_estimate_order_rejects_unchanged_errors():
    with pytest.raises(ValueError):
        estimate_convergence_order(
            np.array([
                1.0,
                1.0,
                0.5,
            ])
        )

def test_asymptotic_constants_reject_nonpositive_order():
    with pytest.raises(ValueError):
        asymptotic_constants(
            np.array([
                1.0,
                0.5,
            ]),
            order = 0.0,
        )