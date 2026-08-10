import numpy as np
import pytest

from vectorization import (
    scale_loop,
    scale_vectorized,
    squared_distance_loop,
    squared_distance_vectorized,
    row_means_loop,
    row_means_vectorized,
)

def test_scale_loop():
    values = np.array([1.0, 2.0, 3.0])

    result = scale_loop(values, 2.0)

    np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

def test_scale_vectorized():
    values = np.array([1.0, 2.0, 3.0])

    result = scale_vectorized(values, 2.0)

    np.testing.assert_allclose(result, [2.0, 4.0, 6.0])

def test_scaling_methods_agree():
    values = np.array([-2.0, 0.0, 1.5, 4.0])

    loop_result = scale_loop(values, 3.5)
    vectorized_result = scale_vectorized(values, 3.5)

    np.testing.assert_allclose(loop_result, vectorized_result)

def test_squared_distance_loop():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 2.0, 1.0])
    result = squared_distance_loop(x, y)

    assert result == pytest.approx(13.0)

def test_squared_distance_vectorized():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 2.0, 1.0])

    result = squared_distance_vectorized(x, y)

    assert result == pytest.approx(13.0)

def test_squared_distance_methods_agree():
    x = np.array([-1.0, 2.5, 7.0])
    y = np.array([3.0, -2.0, 4.0])

    assert squared_distance_loop(x, y) == pytest.approx(
        squared_distance_vectorized(x, y)
    )

def test_squared_distance_rejects_different_shapes():
    x = np.array([1.0, 2.0])
    y = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        squared_distance_vectorized(x, y)

def test_row_means_loop():
    matrix = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    result = row_means_vectorized(matrix)
    np.testing.assert_allclose(result, [2.0, 5.0])

def test_row_mean_methods_agree():
    matrix = np.array([
        [2.0, 4.0],
        [-1.0, 3.0],
        [10.0, 20.0],
    ])

    np.testing.assert_allclose(
        row_means_loop(matrix),
        row_means_vectorized(matrix),
    )

def test_row_means_requires_matrix():
    values = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        row_means_vectorized(values)