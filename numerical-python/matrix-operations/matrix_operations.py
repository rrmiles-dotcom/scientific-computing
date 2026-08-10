from multiprocessing import Value

import numpy as np

def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Return the dot product of two vectors"""

    if x.ndim != 1 or y.ndim != 1:
        raise ValueError("x and y must be one-dimensional")

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    return float(np.dot(x, y))


def vector_norm(x: np.ndarray) -> float:
    """Return the Euclidean norm of a vector"""

    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")

    return float(np.linalg.norm(x))

def transpose(matrix: np.ndarray) -> np.ndarray:
    """Return the transpose of a two-dimensional matrix"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    return matrix.T

def matrix_vector_product(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> np.ndarray:
    """Return the matrix-vector product"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")

    if matrix.shape[1] != vector.shape[0]:
        raise ValueError("matrix and vector dimensions are incompatible")

    return matrix @ vector

def matrix_matrix_product(
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Return the matrix product A @ B."""

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be two-dimensional")

    if a.shape[1] != b.shape[0]:
        raise ValueError("matrix dimensions are incompatible")
    
    return a @ b

def trace(matrix: np.ndarray) -> float:
    """Return the trace of a square matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    return float(np.trace(matrix))

def frobenius_norm(matrix: np.ndarray) -> float:
    """Return the Frobenius norm of a matrix"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    return float(np.linalg.norm(matrix, ord = "fro"))