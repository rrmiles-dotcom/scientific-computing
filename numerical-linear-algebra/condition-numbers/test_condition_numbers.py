import numpy as np
import pytest

from condition_numbers import(
    matrix_condition_number,
    relative_input_error,
    relative_solution_error,
    solve_with_perturbation,
    error_amplification,
)

def test_identity_condition_number_is_one():
    matrix = np.eye(4)

    result = matrix_condition_number(matrix)

    assert result == pytest.approx(1.0)

def test_diagonal_condition_number():
    matrix = np.diag([
        1.0,
        2.0,
        10.0,

    ])

    result = matrix_condition_number(matrix)

    assert result == pytest.approx(10.0)

def test_condition_number_matches_numpy():
    matrix = np.array([
        [4.0, 2.0],
        [1.0, 3.0],
    ])

    result = matrix_condition_number(matrix)
    expected = np.linalg.cond(matrix, 2)

    assert result == pytest.approx(
        expected,
        rel = 1e-10,
    )

def test_ill_conditioned_matrix_has_large_condition_number():
    matrix = np.array([
        [1.0, 1.0],
        [1.0, 1.000001],
    ])

    result = matrix_condition_number(matrix)

    assert result > 1e6

def test_singular_matrix_has_infinite_conditon_number():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    result = matrix_condition_number(matrix)

    assert np.isinf(result)

def test_condition_number_requires_matrix():
    values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        matrix_condition_number(values)

def test_condition_number_requires_square_matrix():
    matrix = np.ones((3, 2))

    with pytest.raises(ValueError):
        matrix_condition_number(matrix)

def test_relative_input_error_known_result():
    original = np.array([
        3.0,
        4.0,
    ])

    perturbed = np.array([
        6.0,
        8.0,
    ])

    result = relative_input_error(
        original,
        perturbed,
    )

    assert result == pytest.approx(1.0)


def test_relative_input_error_zero_for_identical_arrays():
    values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    result = relative_input_error(
        values,
        values.copy(),
    )

    assert result == pytest.approx(0.0)

def test_relative_input_error_rejects_shape_mismatch():
    original = np.ones(3)
    perturbed = np.ones(4)

    with pytest.raises(ValueError):
        relative_input_error(
            original,
            perturbed,
    )

def test_relative_input_error_rejects_zero_original():
    original = np.zeros(3)
    perturbed = np.ones(3)

    with pytest.raises(ValueError):
        relative_input_error(
            original,
            perturbed,
        )

def test_relative_solution_error():
    original = np.array([
        2.0,
        4.0,
    ])

    perturbed = np.array([
        2.2,
        4.4,
    ])

    result = relative_solution_error(
        original,
        perturbed,
    )

    assert result == pytest.approx(0.1)

def test_solve_with_zero_perturbation_gives_same_solution():
    matrix = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])

    rhs = np.array([
        7.0,
        5.0,
    ])

    perturbation = np.zeros(2)

    original, perturbed = solve_with_perturbation(
        matrix,
        rhs,
        perturbation,
    )

    np.testing.assert_allclose(
        original,
        perturbed,
        atol = 1e-12,
    )

def test_solve_with_perturbation_matches_numpy():
    matrix = np.array([
        [4.0, 1.0],
        [2.0, 3.0],
    ])

    rhs = np.array([
        5.0,
        7.0,
    ])

    perturbation = np.array([
        0.01,
        -0.02,
    ])

    original, perturbed = solve_with_perturbation(
        matrix,
        rhs,
        perturbation,
    )

    np.testing.assert_allclose(
        original,
        np.linalg.solve(matrix, rhs)
    )

    np.testing.assert_allclose(
        perturbed,
        np.linalg.solve(
            matrix,
            rhs + perturbation,
        ),
    )

def test_solve_rejects_singular_matrix():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    rhs = np.array([
        1.0,
        2.0,
    ])

    perturbation = np.array([
        0.01,
        0.01,
    ])

    with pytest.raises(ValueError):
        solve_with_perturbation(
            matrix,
            rhs,
            perturbation,
        )

def test_solve_rejects_rhs_dimension_mismatch():
    matrix = np.eye(3)
    rhs = np.ones(2)
    perturbation = np.ones(2)

    with pytest.raises(ValueError):
        solve_with_perturbation(
            matrix,
            rhs,
            perturbation,
        )

def test_solve_rejects_perturbation_shape_mismatch():
    matrix = np.eye(3)
    rhs = np.ones(3)
    perturbation = np.ones(2)

    with pytest.raises(ValueError):
        solve_with_perturbation(
            matrix,
            rhs,
            perturbation,
        )

def test_error_amplification_zero_perturbation():
    matrix = np.eye(2)
    rhs = np.array([
        1.0,
        2.0,
    ])

    perturbation = np.zeros(2)

    result = error_amplification(
        matrix,
        rhs,
        perturbation,
    )

    assert result == pytest.approx(0.0)

def test_error_amplification_is_nonnegative():
    matrix = np.array([
        [1.0, 0.99],
        [0.99, 0.98],
    ])

    rhs = np.array([
        2.0,
        1.97,
    ])

    perturbation = np.array([
        0.001,
        0.0,
    ])

    result = error_amplification(
        matrix,
        rhs,
        perturbation,
    )

    assert result >= 0.0