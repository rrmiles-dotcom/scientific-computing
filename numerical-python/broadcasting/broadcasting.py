from multiprocessing import Value

import numpy as np

def add_scalar(values: np.ndarray, scalar: float) -> np.ndarray:
    """Add a scalar to every element of an array using broadcasting"""

    return values + scalar

def add_row_vector(
    matrix: np.ndarray,
    row_vector: np.ndarray,
) -> np.ndarray:
    """Add a row vector to every row of a matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if row_vector.ndim != 1:
        raise ValueError("row_vector must be one-dimensional")

    if matrix.shape[1] != row_vector.shape[0]:
        raise ValueError("row_vector length must match matrix columns")
    
    return matrix + row_vector

def add_column_vector(
    matrix: np.ndarray,
    column_vector: np.ndarray,
) -> np.ndarray:
    """Add a column vector to every column of a matrix"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if column_vector.ndim != 1:
        raise ValueError("column_vector must be one-dimensional")

    if matrix.shape[0] != column_vector.shape[0]:
        raise ValueError("column_vector length must match matrix rows")
    
    return matrix + column_vector[:, np.newaxis]

def center_columns(matrix: np.ndarray) -> np.ndarray:
    """Center each column by subtracting its mean"""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    column_means = np.mean(matrix, axis = 0)

    return matrix - column_means

def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Normalize each row to unit Euclidean norm."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    norms = np.linalg.norm(matrix, axis = 1)

    if np.any(norms == 0):
        raise ValueError("cannot normalize rows with zero norm")

    return matrix / norms[:, np.newaxis]