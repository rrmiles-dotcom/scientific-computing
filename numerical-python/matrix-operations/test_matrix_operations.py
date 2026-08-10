import numpy as np
import pytest

from matrix_operations import (
    dot_product,
    vector_norm,
    transpose,
    matrix_vector_product,
    matrix_matrix_product,
    trace,
    frobenius_norm,
)

def test_dot_product():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])

    result = dot_product(x, y)

    assert result == pytest.approx(32.0)

def test_dot_product_orthogonal_vectors():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])

    assert dot_product(x, y) == pytest.approx(0.0)

def test_dot_product_requires_same_shape():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        dot_product(x, y)

def test_vector_norm():
    x = np.array([3.0, 4.0])
    
    assert vector_norm(x) == pytest.approx(5.0)

def test_transpose():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    expected = np.array([
        [1.0, 4.0],
        [2.0, 5.0],
        [3.0, 6.0],
    ])

    np.testing.assert_allclose(transpose(matrix), expected)

def test_double_transpose_returns_original():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    np.testing.assert_allclose(
        transpose(transpose(matrix)),
        matrix,
    )

def test_matrix_vector_product():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0]
    ])
    vector = np.array([5.0, 6.0])

    expected = np.array([17.0, 39.0])

    np.testing.assert_allclose(
        matrix_vector_product(matrix, vector),
        expected,
    )

def test_matrix_vector_incompatible_dimensions():
    matrix = np.ones((2, 3))
    vector = np.ones(2)

    with pytest.raises(ValueError):
        matrix_vector_product(matrix, vector)

def test_matrix_matrix_product():
    a = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
    ])

    b = np.array([
        [5.0, 6.0],
        [7.0, 8.0],
    ])

    expected = np.array([
        [19.0, 22.0],
        [43.0, 50.0],
    ])

    np.testing.assert_allclose(
        matrix_matrix_product(a, b),
        expected,
    )

def test_matrix_product_with_identity():
    matrix = np.array([
        [2.0, 3.0],
        [4.0, 5.0],
    ])

    identity = np.eye(2)

    np.testing.assert_allclose(
        matrix_matrix_product(matrix, identity),
        matrix,
    )

def test_matrix_matrix_incompatible_dimensions():
    a = np.ones((2, 3))
    b = np.ones((2, 2))

    with pytest.raises(ValueError):
        matrix_matrix_product(a, b)

def test_trace():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])

    assert trace(matrix) == pytest.approx(15.0)

def test_trace_requires_square_matrix():
    matrix = np.ones((2, 3))

    with pytest.raises(ValueError):
        trace(matrix)

def test_frobenius_norm():
    matrix = np.ones((2, 3))

    with pytest.raises(ValueError):
        trace(matrix)

def test_frobenius_norm():
    matrix = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    assert frobenius_norm(matrix) == pytest.approx(5.0)

def test_frobenius_norm_matches_flattened_euclidean_norm():
    matrix = np.array([
        [1.0, -2.0],
        [3.0, 4.0],
    ])

    expected = np.linalg.norm(matrix.ravel())

    assert frobenius_norm(matrix) == pytest.approx(expected)