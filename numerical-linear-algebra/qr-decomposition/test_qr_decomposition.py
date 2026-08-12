import numpy as np
import pytest

from qr_decomposition import (
    qr_decomposition,
    solve_least_squares_qr,
)

def test_qr_reconstructs_matrix():
    matrix = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [1.0, 2.0],
    ])

    q, r = qr_decomposition(matrix)

    np.testing.assert_allclose(
        q @ r,
        matrix,
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_q_has_orthogonal_columns():
    matrix = np.array([
        [1.0, 1.0],
        [1.0, -1.0],
        [1.0, 2.0],
    ])

    q, _ = qr_decomposition(matrix)

    np.testing.assert_allclose(
        q.T @ q,
        np.eye(q.shape[1]),
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_r_is_upper_triangular():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 7.0],
    ])

    _, r = qr_decomposition(matrix)

    np.testing.assert_allclose(
        r,
        np.triu(r),
        atol = 1e-12,
    )

def test_square_matrix_qr():
    matrix = np.array([
        [2.0, 1.0],
        [1.0, 3.0],
    ])

    q, r = qr_decomposition(matrix)

    np.testing.assert_allclose(
    q.T @ q,
    np.eye(2),
    rtol=1e-10,
    atol=1e-12,
    )   
   

def test_tall_matrix_shapes():
    matrix = np.ones((5, 3))
    matrix[:, 1] = np.arange(5)
    matrix[:, 2] = np.arange(5) ** 2

    q, r = qr_decomposition(matrix)

    assert q.shape == (5, 3)
    assert r.shape == (3, 3)

def test_qr_matches_numpy_reconstruction():
    matrix = np.array([
        [12.0, -51.0, 4.0],
        [6.0, 167.0, -68.0],
        [-4.0, 24.0, -41.0],
    ])

    q, r = qr_decomposition(matrix)

    np.testing.assert_allclose(
        q @ r,
        matrix,
        rtol = 1e-10,
        atol = 1e-12,
    )

    np.testing.assert_allclose(
        q.T @ q,
        np.eye(3),
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_least_squares_known_problem():
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
    ])

    rhs = np.array([1.0, 2.0, 2.0])
    result = solve_least_squares_qr(matrix, rhs)

    expected, _, _, _ = np.linalg.lstsq(
        matrix,
        rhs,
        rcond = None,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_least_squares_matches_numpy_random_system():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size = (10, 4))
    rhs = rng.normal(size = 10)

    result = solve_least_squares_qr(matrix, rhs)

    expected, _, _, _ = np.linalg.lstsq(
        matrix,
        rhs,
        rcond = None
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
    )

def test_least_squares_residual_is_orthogonal_to_columns():
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])

    rhs = np.array([1.0, 2.0, 2.0, 4.0])

    solution = solve_least_squares_qr(matrix, rhs)

    residual = rhs - matrix @ solution

    np.testing.assert_allclose(
        matrix.T @ residual,
        np.zeros(matrix.shape[1]),
        atol = 1e-10,
    )

def test_linearly_dependent_columns_rejected():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    with pytest.raises(ValueError):
        qr_decomposition(matrix)

def test_wide_matrix_rejected():
    matrix = np.ones((2, 3))

    with pytest.raises(ValueError):
        qr_decomposition(matrix)

def test_non_matrix_input_rejected():
    values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        qr_decomposition(values)

def test_least_squares_rhs_must_be_vector():
    matrix = np.eye(3)
    rhs = np.ones((3, 1))

    with pytest.raises(ValueError):
        solve_least_squares_qr(matrix, rhs)

def test_least_squares_rhs_dimension_must_match():
    matrix = np.ones((4, 2))
    rhs = np.ones(3)

    with pytest.raises(ValueError):
        solve_least_squares_qr(matrix, rhs)