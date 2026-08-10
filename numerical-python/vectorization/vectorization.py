from multiprocessing import Value

import numpy as np

def scale_loop(values: np.ndarray, factor: float) -> np.ndarray:
    """Scale every element using an explicit Python loop"""

    result = np.empty_like(values, dtype = float)

    for i in range(len(values)):
        result[i] = values[i] * factor

    return result

def scale_vectorized(values: np.ndarray, factor: float) -> np.ndarray:
    """Scale every element using NumPy vectorization"""

    return values * factor

def squared_distance_loop(x: np.ndarray, y: np.ndarray) -> float:
    """Return squared Euclidean distance using a Python loop"""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    total = 0.0

    for i in range(len(x)):
        difference = x[i] - y[i]
        total += difference * difference

    return float(total)

def squared_distance_vectorized(x: np.ndarray, y: np.ndarray) -> float:
    """Return squared Euclidean distance using NumPy vectorization"""

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")
    
    difference = x - y

    return float(np.dot(difference, difference))

def row_means_loop(matrix: np.ndarray) -> np.ndarray:
    """Compute row means using explicit Python loops"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape
    result = np.empty(rows, dtype = float)

    for i in range(rows):
        total = 0.0

        for j in range(cols):
            total += matrix[i, j]
        
        result[i] = total / cols

    return result

def row_means_vectorized(matrix: np.ndarray) -> np.ndarray:
    """Compute row means using NumPy vectorization"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    return np.mean(matrix, axis = 1)