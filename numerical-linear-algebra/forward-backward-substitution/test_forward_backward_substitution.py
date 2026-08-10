import numpy as np
import pytest

from forward_backward_substitution import(
    forward_substitution,
    backward_substitution,
)

def test_forward_substitution():
    lower = np.array([
        [2.0, 0.0, 0.0],
        [3.0, 1.0, 0.0],
        [1.0, -1.0, 2.0],
    ])

    rhs = np.array([2.0, 5.0, 5.0])

    result = forward_substitution(lower, rhs)

    expected = np.array([1.0, 2.0, 3.0])

    np.testing.assert_allclose(result, expected)

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

def test_forward_solution_satisfies_system():
    lower = np.array([
        [4.0, 0.0, 0.0],
        [2.0, 5.0, 0.0],
        [-1.0, 3.0, 2.0],
    ])

    rhs = np.array([8.0, 14.0, 9.0])
    solution = forward_substitution(lower, rhs)

    np.testing.assert_allclose(
        lower @ solution,
        rhs,
    )

def test_backward_solution_satisfies_system():
    upper = np.array([
        [3.0, -1.0, 2.0],
        [0.0, 4.0, 1.0],
        [0.0, 0.0, 5.0],
    ])

    rhs = np.array([7.0, 9.0, 10.0])

    solution = backward_substitution(upper, rhs)

    np.testing.assert_allclose(
        upper @ solution,
        rhs,
    )

def test_forward_matches_numpy_solve():
    lower = np.array([
        [2.0, 0.0, 0.0],
        [1.0, 3.0, 0.0],
        [4.0, -2.0, 5.0],
    ])

    rhs = np.array([4.0, 7.0, 3.0])

    result = forward_substitution(lower, rhs)
    expected = np.linalg.solve(lower, rhs)

    np.testing.assert_allclose(result, expected)

def test_backward_matches_numpy_solve():
    upper = np.array([
        [2.0, -1.0, 3.0],
        [0.0, 4.0, 2.0],
        [0.0, 0.0, 6.0],
    ])

    rhs = np.array([5.0, 8.0, 12.0])

    result = backward_substitution(upper, rhs)
    expected = np.linalg.solve(upper, rhs)

    np.testing.assert_allclose(result, expected)

def test_forward_requires_square_matrix():
    lower = np.ones((2, 3))
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        forward_substitution(lower, rhs)

def test_backward_requires_square_matrix():
    upper = np.ones((2, 3))
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        backward_substitution(upper, rhs)

def test_forward_rejects_non_lower_triangular_matrix():
    matrix = np.array([
        [1.0, 1.0],
        [2.0, 3.0],
    ])

    rhs = np.ones(2)

    with pytest.raises(ValueError):
        forward_substitution(matrix, rhs)

def test_backward_rejects_non_upper_triangular_matrix():
    matrix = np.array([
        [1.0, 2.0],
        [1.0, 3.0],
    ])

    rhs = np.ones(2)

    with pytest.raises(ValueError):
        backward_substitution(matrix, rhs)

def test_forward_rejects_zero_diagonal():
    lower = np.array([
        [1.0, 0.0],
        [2.0, 0.0],
    ])

    rhs = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        forward_substitution(lower, rhs)

def test_backward_rejects_zero_diagonal():
    upper = np.array([
        [0.0, 2.0],
        [0.0, 3.0],
    ])

    rhs = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        backward_substitution(upper, rhs)

def test_forward_rejects_wrong_rhs_size():
    lower = np.eye(3)
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        forward_substitution(lower, rhs)

def test_backward_rejects_non_vector_rhs():
    upper = np.eye(2)
    rhs = np.ones((2, 1))

    with pytest.raises(ValueError):
        backward_substitution(upper, rhs)