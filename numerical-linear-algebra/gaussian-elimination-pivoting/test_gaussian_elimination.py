import numpy as np
import pytest

from gaussian_elimination import(
    gaussian_elimination,
    backward_substitution,
    solve_gaussian,
)

def test_gaussian_elimination_produces_upper_triangular_matrix():
    matrix = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ])

    rhs = np.array([8.0, -11.0, -3.0])

    upper, transformed_rhs = gaussian_elimination(matrix, rhs)

    np.testing.assert_allclose(
        upper,
        np.triu(upper),
        atol = 1e-12,
    )

    assert transformed_rhs.shape == rhs.shape

def test_solve_gaussian_known_system():
    matrix = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ])

    rhs = np.array([8.0, -11.0, -3.0])

    result = solve_gaussian(matrix, rhs)

    expected = np.array([2.0, 3.0, -1.0])

    np.testing.assert_allclose(result, expected)


def test_solution_satisfies_original_system():
    matrix = np.array([
        [4.0, 2.0, 1.0],
        [2.0, 5.0, 2.0],
        [1.0, 2.0, 4.0],
    ])

    rhs = np.array([7.0, 4.0, 5.0])

    solution = solve_gaussian(matrix, rhs)

    np.testing.assert_allclose(
        matrix @ solution,
        rhs,
        atol = 1e-12
    )

def test_matches_numpy_solve():
    matrix = np.array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 4.0],
        [-1.0, 0.5, -1.0],
    ])

    rhs = np.array([1.0, -2.0, 0.0])
    result = solve_gaussian(matrix, rhs)
    expected = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(result, expected)

def test_partial_pivoting_handles_zero_initial_pivot():
    matrix = np.array([
        [0.0, 2.0],
        [1.0, 3.0],
    ])

    rhs = np.array([4.0, 7.0])
    result = solve_gaussian(matrix, rhs)
    expected = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(result, expected)

def test_partial_pivoting_uses_largest_available_pivot():
    matrix = np.array([
        [1e-10, 1.0],
        [2.0, 3.0],
    ])

    rhs = np.array([1.0, 5.0])
    upper, _ = gaussian_elimination(matrix, rhs)
    assert upper[0, 0] == pytest.approx(2.0)

def test_backward_substitution():
    upper = np.array([
        [2.0, 1.0, -1.0],
        [0.0, 3.0, 2.0],
        [0.0, 0.0, 4.0],
    ])

    rhs = np.array([1.0, 12.0, 12.0])
    result = backward_substitution(upper, rhs)
    expected = np.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(result, expected)

def test_integer_inputs_are_not_modified():
    matrix = np.array([
        [2, 1],
        [1, 3],
    ])

    rhs = np.array([5, 7])

    matrix_original = matrix.copy()
    rhs_original = rhs.copy()
    solve_gaussian(matrix, rhs)

    np.testing.assert_array_equal(matrix, matrix_original)
    np.testing.assert_array_equal(rhs, rhs_original)

def test_requires_square_matrix():
    matrix = np.ones((2, 3))
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_elimination(matrix, rhs)

def test_requires_vector_rhs():
    matrix = np.eye(2)
    rhs = np.ones((2, 1))

    with pytest.raises(ValueError):
        gaussian_elimination(matrix, rhs)

def test_rhs_dimension_must_match():
    matrix = np.eye(3)
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_elimination(matrix, rhs)

def test_singular_matrix_rejected():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    rhs = np.array([3.0, 6.0])

    with pytest.raises(ValueError):
        solve_gaussian(matrix, rhs)

def test_larger_random_system_matches_numpy():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size = (8, 8))
    matrix += 8.0 * np.eye(8)

    rhs = rng.normal(size = 8)

    result = solve_gaussian(matrix, rhs)
    expected = np.linalg.solve(matrix, rhs)

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
    )
        