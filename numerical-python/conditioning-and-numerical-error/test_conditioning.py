import numpy as np
import pytest

from conditioning import(
    absolute_error,
    relative_error,
    residual,
    relative_residual,
    condition_number,
    perturbation_amplification,
)

def test_absolute_error():
    approximation = np.array([1.0, 2.0])
    exact = np.array([1.0, 3.0])

    assert absolute_error(approximation, exact) == pytest.approx(1.0)

def test_absolute_error_requires_same_shape():
    approximation = np.array([1.0, 2.0])
    exact = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        absolute_error(approximation, exact)

def test_relative_error():
    approximation = np.array([1.0, 2.0])
    exact = np.array([1.0, 2.0])

    expected = np.linalg.norm(
        approximation - exact
    ) / np.linalg.norm(exact)

    assert relative_error(approximation, exact) == pytest.approx(expected)

def test_relative_error_zero_exact_vector():
    approximation = np.array([1.0, 2.0])
    exact = np.zeros(2)

    with pytest.raises(ValueError):
        relative_error(approximation, exact)

def test_residual_exact_solution_is_zero():
    matrix = np.array([
        [2.0, 0.0],
        [0.0, 3.0],
    ])

    solution = np.array([2.0, 2.0])
    rhs = np.array([4.0, 6.0])

    result = residual(matrix, solution, rhs)

    np.testing.assert_allclose(result, np.zeros(2))

def test_residual_nonzero():
    matrix = np.eye(2)
    solution = np.array([1.0, 2.0])
    rhs = np.array([2.0, 4.0])

    expected = np.array([1.0, 2.0])

    np.testing.assert_allclose(
        residual(matrix, solution, rhs),
        expected,
    )

def test_relative_residual_exact_solution():
    matrix = np.array([
        [2.0, 0.0],
        [0.0, 4.0],
    ])

    solution = np.array([1.0, 2.0])
    rhs = np.array([2.0, 8.0])

    assert relative_residual(
        matrix,
        solution,
        rhs,
    ) == pytest.approx(0.0)

def test_relative_residual_zero_rhs():
    matrix = np.eye(2)
    solution = np.ones(2)
    rhs = np.zeros(2)

    with pytest.raises(ValueError):
        relative_residual(matrix, solution, rhs)

def test_identity_condition_number_is_one():
    matrix = np.eye(4)

    assert condition_number(matrix) == pytest.approx(1.0)

def test_condition_number_large_for_ill_conditioned_matrix():
    matrix = np.array([
        [1.0, 1.0],
        [1.0, 1.0000001],
    ])

    assert condition_number(matrix) > 1e6

def test_condition_number_requires_square_matrix():
    matrix = np.ones((2, 3))

    with pytest.raises(ValueError):
        condition_number(matrix)

def test_perturbation_amplification_identity():
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0])
    perturbation = np.array([0.01, -0.02])

    amplification = perturbation_amplification(
        matrix,
        rhs,
        perturbation,
    )

    assert amplification == pytest.approx(1.0)

def test_ill_conditioned_matrix_can_amplify_error():
    matrix = np.array([
        [1.0, 1.0],
        [1.0, 1.000001],
    ])

    rhs = np.array([2.0, 2.000001])
    perturbation = np.array([0.0, 1e-7])

    amplification = perturbation_amplification(
        matrix,
        rhs,
        perturbation,
    )

    assert amplification > 1.0

def test_zero_perturbation_rejected():
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0])
    perturbation = np.zeros(2)

    with pytest.raises(ValueError):
        perturbation_amplification(
            matrix,
            rhs,
            perturbation,
        )

def test_perturbation_shape_mismatch():
    matrix = np.eye(2)
    rhs = np.array([1.0, 2.0])
    perturbation = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        perturbation_amplification(
            matrix,
            rhs,
            perturbation,
        )