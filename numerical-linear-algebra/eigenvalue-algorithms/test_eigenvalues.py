import numpy as np
import pytest

from eigenvalues import(
    rayleigh_quotient,
    power_iteration,
    inverse_iteration,
    qr_eigenvalues,
)

def test_rayleigh_quotient_known_eigenvector():
    matrix = np.array([
        [2.0, 0.0],
        [0.0, 5.0],
    ])

    vector = np.array([0.0, 1.0])

    result = rayleigh_quotient(matrix, vector)

    assert result == pytest.approx(5.0)

def test_power_iteration_dominant_eigenvalue():
    matrix = np.array([
        [4.0, 1.0],
        [1.0, 2.0],
    ])

    eigenvalue, vector = power_iteration(matrix)
    expected = np.max(
        np.linalg.eigvalsh(matrix)
    )

    assert eigenvalue == pytest.approx(
        expected,
        rel = 1e-8,
    )

    np.testing.assert_allclose(
        matrix @ vector,
        eigenvalue * vector,
        atol = 1e-8,
    )

def test_power_iteration_diagonal_matrix():
    matrix = np.diag([
        1.0,
        3.0,
        7.0,
    ])

    eigenvalue, vector = power_iteration(matrix)

    assert eigenvalue == pytest.approx(
        7.0,
        rel = 1e-8,
    )

    np.testing.assert_allclose(
        matrix @ vector,
        eigenvalue * vector,
        atol = 1e-8,
    )

def test_inverse_iteration_smallest_eigenvalue():
    matrix = np.array([
        [4.0, 1.0],
        [1.0, 2.0],
    ])

    eigenvalue, vector = inverse_iteration(matrix)

    expected = np.min(
        np.linalg.eigvalsh(matrix)
    )

    assert eigenvalue == pytest.approx(
        expected,
        rel = 1e-8,
    )

    np.testing.assert_allclose(
        matrix @ vector,
        eigenvalue * vector,
        atol = 1e-8,
    )

def test_inverse_iteration_diagonal_matrix():
    matrix = np.diag([
        1.0,
        4.0,
        9.0,
    ])

    eigenvalue, vector = inverse_iteration(matrix)

    assert eigenvalue == pytest.approx(
        1.0,
        rel = 1e-8,
    )

    np.testing.assert_allclose(
        matrix @ vector,
        eigenvalue * vector,
        atol = 1e-8,
    )

def test_qr_eigenvalues_matches_numpy():
    matrix = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 1.0],
        [0.0, 1.0, 2.0],
    ])

    result = qr_eigenvalues(matrix)
    expected = np.linalg.eigvalsh(matrix)

    np.testing.assert_allclose(
        np.sort(result),
        np.sort(expected),
        rtol = 1e-8,
        atol = 1e-8,
    )

def test_qr_eigenvalues_diagonal_matrix():
    matrix = np.diag([
        2.0,
        5.0,
        8.0,
    ])

    result = qr_eigenvalues(matrix)

    np.testing.assert_allclose(
        np.sort(result),
        np.array([2.0, 5.0, 8.0]),
        atol = 1e-12,
    )

def test_power_iteration_requires_square_matrix():
    matrix = np.ones((3, 2))

    with pytest.raises(ValueError):
        power_iteration(matrix)

def test_inverse_iteration_requires_square_matrix():
    matrix = np.ones((3, 2))

    with pytest.raises(ValueError):
        inverse_iteration(matrix)

def test_qr_eigenvalues_requires_square_matrix():
    matrix = np.ones((3, 2))

    with pytest.raises(ValueError):
        qr_eigenvalues(matrix)

def test_power_iteration_rejects_bad_tolerance():
    matrix = np.eye(2)

    with pytest.raises(ValueError):
        power_iteration(
            matrix,
            tolerance = 0.0,
        )

def test_power_iteration_rejects_bad_max_iterations():
    matrix = np.eye(2)

    with pytest.raises(ValueError):
        power_iteration(
            matrix,
            max_iterations = 0,
        )

def test_inverse_iteration_rejects_singular_matrix():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(ValueError):
        inverse_iteration(matrix)

def test_qr_iteration_requires_symmetric_matrix():
    matrix = np.array([
        [1.0, 2.0],
        [0.0, 3.0],
    ])

    with pytest.raises(ValueError):
        qr_eigenvalues(matrix)

def test_rayleigh_quotient_rejects_zero_vector():
    matrix = np.eye(2)
    vector = np.zeros(2)

    with pytest.raises(ValueError):
        rayleigh_quotient(
            matrix,
            vector,
        )

def test_rayleigh_quotient_dimension_mismatch():
    matrix = np.eye(3)
    vector = np.ones(2)

    with pytest.raises(ValueError):
        rayleigh_quotient(
            matrix,
            vector,
        )

def test_rayleigh_quotient_requires_vector():
    matrix = np.eye(2)
    vector = np.ones((2, 1))

    with pytest.raises(ValueError):
        rayleigh_quotient(
            matrix,
            vector,
        )