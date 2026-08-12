import numpy as np 

def gaussian_elimination(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reduce Ax = b to an upper-triangular system  Ux = y
    using Gaussian elimination with partial pivoting.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("matrix must be square")

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimensions must match matrix size")

    upper = matrix.astype(float).copy()
    transformed_rhs = rhs.astype(float).copy()

    n = rows

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(upper[k:, k]))

        if np.isclose(upper[pivot_row, k], 0.0):
            raise ValueError("matrix is singular")

        if pivot_row != k:
            upper[[k, pivot_row]] = upper[[pivot_row, k]]
            transformed_rhs[[k, pivot_row]] = (
                transformed_rhs[[pivot_row, k]]
            )

        for i in range(k + 1, n):
            factor = upper[i, k] / upper[k, k]

            upper[i, k:] -= factor * upper[k, k:]
            transformed_rhs[i] -= factor * transformed_rhs[k]

    if np.isclose(upper[-1, -1], 0.0):
        raise ValueError("matrix is singular")

    return upper, transformed_rhs

def backward_substitution(
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve Ux = b for an upper-triangular matrix U."""

    n = upper.shape[0]
    solution = np.zeros(n, dtype = float)

    for i in range(n - 1, -1, -1):
        known_sum = np.dot(
            upper[i, i + 1:],
            solution[i + 1:],
        )

        solution[i] = (
            rhs[i] - known_sum
        ) / upper[i, i]

    return solution

def solve_gaussian(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.
    """

    upper, transformed_rhs = gaussian_elimination(
        matrix,
        rhs,
    )

    return backward_substitution(
        upper,
        transformed_rhs,
    )