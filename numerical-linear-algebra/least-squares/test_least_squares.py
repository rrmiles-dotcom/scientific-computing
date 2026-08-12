import numpy as np
import pytest

from least_squares import(
    solve_normal_equations,
    solve_least_squares_qr,
    residual,
    residual_norm,
)

def test_normal_equations_matches_numpy():
    rng = np.random.default_rng(42)

    matrix = rng.normal(size = (10, 3))
    rhs = rng.normal(size = 10)

    normal_result = solve_normal_equations(
        matrix,
        rhs,
    )

    qr_result = solve_least_squares_qr(
        matrix,
        rhs,
    )

    np.testing.assert_allclose(
        normal_result,
        qr_result,
        rtol = 1e-9,
        atol = 1e-11,
    )

def test_exact_system_has_zero_residual():
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])

    solution = np.array([2.0, 3.0])
    rhs = matrix @ solution

    result = solve_least_squares_qr(
        matrix,
        rhs,
    )

    assert residual_norm(
        matrix,
        result,
        rhs,
    ) == pytest.approx(0.0, abs = 1e-12)

def test_residual_is_orthogonal_to_column():
    matrix = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
    ])

    rhs = np.array([1.0, 2.0, 2.0, 4.0])
    solution = solve_least_squares_qr(
        matrix,
        rhs,
    )

    r = residual(
        matrix,
        solution,
        rhs,
    )

    np.testing.assert_allclose(
        matrix.T @ r,
        np.zeros(matrix.shape[1]),
        atol = 1e-10,
    )

def test_residual_function():
    matrix = np.eye(2)
    solution = np.array([1.0, 2.0])
    rhs = np.array([2.0, 4.0])

    expected = np.array([1.0, 2.0])

    np.testing.assert_allclose(
        residual(matrix, solution, rhs),
        expected,
    )

def test_residual_norm():
    matrix = np.eye(2)
    solution = np.array([1.0, 2.0])
    rhs = np.array([4.0, 6.0])

    expected = 5.0

    assert residual_norm(
        matrix,
        solution,
        rhs,
    ) == pytest.approx(expected)

def test_normal_equations_rejects_rank_deficiency():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])

    rhs = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        solve_normal_equations(matrix, rhs)

def test_qr_rejects_rank_deficiency():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ]) 

    rhs = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        solve_least_squares_qr(matrix, rhs)

def test_wide_matrix_rejected():
    matrix = np.ones((2, 3))
    rhs = np.ones(2)

    with pytest.raises(ValueError):
        solve_least_squares_qr(matrix, rhs)

def test_rhs_dimension_mismatch_rejected():
    matrix = np.ones([4, 2])
    rhs = np.ones(3)

    with pytest.raises(ValueError):
        solve_least_squares_qr(matrix, rhs)

def test_rhs_must_be_vector():
    matrix = np.ones((4, 2))
    rhs = np.ones((4, 1))

    with pytest.raises(ValueError):
        solve_normal_equations(matrix, rhs)

def test_random_problem_matches_numpy():
    rng = np.random.default_rng(123)

    matrix = rng.normal(size = (20, 5))
    rhs = rng.normal(size = 20)

    result = solve_least_squares_qr(
        matrix,
        rhs,
    )

    expected, _, _, _ = np.linalg.lstsq(
        matrix,
        rhs,
        rcond = None,
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-9,
        atol = 1e-11,
    )