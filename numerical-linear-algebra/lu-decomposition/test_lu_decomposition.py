import numpy as np
import pytest

from lu_decomposition import (
    lu_decomposition,
    forward_substitution,
    backward_substitution,
    solve_lu,
)

def test_lu_reconstructs_permuted_matrix():
    matrix = np.array([
        [2.0, 1.0, 1.0],
        [4.0, -6.0, 0.0],
        [-2.0, 7.0, 2.0],
    ])

    permutation, lower, upper = lu_decomposition(matrix)

    np.testing.assert_allclose(
        permutation @ matrix,
        lower @ upper,
        atol = 1e-12,
    )

def test_lower_is_lower_triangular():
    matrix = np.array([
        [4.0, 3.0],
        [6.0, 3.0],
    ])

    _, lower, _ = lu_decomposition(matrix)

    np.testing.assert_allclose(
        lower,
        np.tril(lower),
    )

def test_lower_has_unit_diagonal():
    matrix = np.array([
        [4.0, 3.0],
        [6.0, 3.0],
    ])

    _, lower, _ = lu_decomposition(matrix)

    np.testing.assert_allclose(
        np.diag(lower),
        np.ones(2),
    )

def test_upper_is_upper_triangular():
    matrix = np.array([
        [4.0, 3.0],
        [6.0, 3.0],
    ])

    _, _, upper = lu_decomposition(matrix)

    np.testing.assert_allclose(
        upper,
        np.triu(upper),
        atol = 1e-12,
    )

def test_partial_pivoting():
    matrix = np.array([
        [0.0, 2.0],
        [3.0, 4.0],
    ])

    permutation, lower, upper = lu_decomposition(matrix)

    np.testing.assert_allclose(
        permutation @ matrix,
        lower @ upper,
    )

    assert permutation[0, 1] == pytest.approx(1.0)

def test_solve_lu_known_system():
    matrix = np.array([
        [3.0, 1.0],
        [1.0, 2.0],
    ])

    rhs = np.array([9.0, 8.0])
    result = solve_lu(matrix, rhs)
    expected = np.array([2.0, 3.0])

    np.testing.assert_allclose(result, expected)

def test_solution_satisfies_system():
    matrix = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 4.0],
    ])

    rhs = np.array([7.0, 4.0, 5.0])
    solution = solve_lu(matrix, rhs)

    np.testing.assert_allclose(
        matrix @ solution,
        rhs,
        atol = 1e-12,
    )

def test_matches_numpy_solve():
    matrix = np.array([
        [2.0, -1.0, 1.0],
        [3.0, 3.0, 9.0],
        [3.0, 3.0, 5.0],
    ])

    rhs = np.array([2.0, -1.0, 4.0])

    result = solve_lu(matrix, rhs)
    expected = np.linalg.solve(matrix, rhs)

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12
    )

def test_forward_substitution():
    lower = np.array([
        [1.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [3.0, -1.0, 1.0],
    ])

    rhs = np.array([1.0, 4.0, 4.0])
    result = forward_substitution(lower, rhs)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)

def test_backward_substitution():
    upper = np.array([
        [2.0, 1.0, -1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 4.0]
    ])

    rhs = np.array([1.0, 12.0, 12.0])
    result = backward_substitution(upper, rhs)
    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(result, expected)

def test_integer_input_not_modified():
    matrix = np.array([
        [2, 1],
        [4, 3],
    ])

    original = matrix.copy()
    lu_decomposition(matrix)
    np.testing.assert_array_equal(matrix, original)

def test_non_square_matrix_rejected():
    matrix = np.ones((2, 3))

    with pytest.raises(ValueError):
        lu_decomposition(matrix)

def test_singular_matrix_rejected():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(ValueError):
        lu_decomposition(matrix)

def test_rhs_must_be_vector():
    matrix = np.eye(2)
    rhs = np.ones((2, 1))

    with pytest.raises(ValueError):
        solve_lu(matrix, rhs)

def test_rhs_dimension_must_match():
    matrix = np.eye(3)
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        solve_lu(matrix, rhs)

def test_random_system_matches_numpy():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size = (8, 8))
    matrix += 8.0 * np.eye(8)

    rhs = rng.normal(size = 8)

    result = solve_lu(matrix, rhs)
    expected = np.linalg.solve(matrix, rhs)

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12 
    )