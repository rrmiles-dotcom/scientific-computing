import numpy as np
import pytest

from broadcasting import (
    add_scalar,
    add_row_vector,
    add_column_vector,
    center_columns,
    normalize_rows,
)

def test_add_scalar():
    values = np.array([1.0, 2.0, 3.0])

    result = add_scalar(values, 2.0)

    np.testing.assert_allclose(result, [3.0, 4.0, 5.0])

def test_add_row_vector():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    row_vector = np.array([10.0, 20.0, 30.0])

    result = add_row_vector(matrix, row_vector)

    expected = np.array([
        [11.0, 22.0, 33.0],
        [14.0, 25.0, 36.0],
    ])

    np.testing.assert_allclose(result, expected)

def test_add_column_vector():
    matrix = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])

    column_vector = np.array([10.0, 20.0, 30.0])

    result = add_column_vector(matrix, column_vector)

    expected = np.array([
        [11.0, 12.0],
        [23.0, 24.0],
        [35.0, 36.0],
    ])

    np.testing.assert_allclose(result, expected)

def test_center_columns():
    matrix = np.array([
        [1.0, 4.0],
        [3.0, 6.0],
        [5.0, 8.0],
    ])

    result = center_columns(matrix)

    expected = np.array([
        [-2.0, -2.0],
        [0.0, 0.0],
        [2.0, 2.0],
    ])

    np.testing.assert_allclose(result, expected)

def test_centered_columns_have_zero_mean():
    matrix = np.array([
        [1.0, 10.0, 4.0],
        [2.0, 20.0, 8.0],
        [3.0, 30.0, 12.0],
    ])

    result = center_columns(matrix)

    np.testing.assert_allclose(
        np.mean(result, axis = 0),
        np.zeros(3),
        atol = 1e-12
    )

def test_normalize_rows():
    matrix = np.array([
        [3.0, 4.0],
        [5.0, 12.0],
    ])

    result = normalize_rows(matrix)

    expected = np.array([
        [3.0 / 5.0, 4.0 / 5.0],
        [5.0 / 13.0, 12.0 / 13.0],
    ])

    np.testing.assert_allclose(result, expected)

def test_normalized_rows_have_unit_norm():
    matrix = np.array([
        [3.0, 4.0],
        [1.0, 2.0],
        [5.0, 12.0],
    ])

    result = normalize_rows(matrix)

    np.testing.assert_allclose(
        np.linalg.norm(result, axis = 1),
        np.ones(3),
    )


def test_row_vector_wrong_length():
    matrix = np.ones((2, 3))
    row_vector = np.ones(2)

    with pytest.raises(ValueError):
        add_row_vector(matrix, row_vector)

def test_column_vector_wrong_length():
    matrix = np.ones((2, 3))
    column_vector = np.ones(3)

    with pytest.raises(ValueError):
        add_column_vector(matrix, column_vector)

def test_zero_row_cannot_be_normalized():
    matrix = np.array([
        [3.0, 4.0],
        [0.0, 0.0],
    ])

    with pytest.raises(ValueError):
        normalize_rows(matrix)

def test_matrix_input_must_be_two_dimensional():
    values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        center_columns(values)