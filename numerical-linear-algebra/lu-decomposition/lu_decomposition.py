import numpy as np

def lu_decomposition(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute PA = LU using LU decomposition with partial pivoting.

    Returns:
        P: permutation matrix
        L: lower-triangular matrix with unit diagonal
        U: upper-triangular matrix
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("matrix must be square")

    n = rows

    upper = matrix.astype(float).copy()
    lower = np.eye(n, dtype = float)
    permutation = np.eye(n, dtype = float)

    for k in range(n - 1):
        pivot_row = k + np.argmax(np.abs(upper[k:, k]))

        if np.isclose(upper[pivot_row, k], 0):
            raise ValueError("matrix is singular")

        if pivot_row != k:
            upper[[k, pivot_row]] = upper[[pivot_row, k]]
            permutation[[k, pivot_row]] = permutation[[pivot_row, k]]

            if k > 0:
                lower[[k, pivot_row], :k] = lower[
                    [pivot_row, k], :k
                ]

        for i in range(k + 1, n):
            factor = upper[i, k] / upper[k, k]

            lower[i, k] = factor
            upper[i, k:] -= factor * upper[k, k:]

    if np.isclose(upper[-1, -1], 0.0):
        raise ValueError("matrix is singular")

    return permutation, lower, upper

def forward_substitution(
    lower: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve Ly = b for lower-triangular L."""

    n = lower.shape[0]
    solution = np.zeros(n, dtype = float)

    for i in range(n):
        known_sum = np.dot(
            lower[i, :i],
            solution[:i],
        )

        solution[i] = (
            rhs[i] - known_sum
        ) / lower[i, i]

    return solution

def backward_substitution(
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve Ux = y for upper-triangular U."""

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

def solve_lu(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve Ax = b using LU decomposition with partial pivoting."""

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    if rhs.shape[0] != matrix.shape[0]:
        raise ValueError("rhs dimension must match matrix size")

    permutation, lower, upper = lu_decomposition(matrix)

    permuted_rhs = permutation @ rhs

    y = forward_substitution(
        lower,
        permuted_rhs,
    )        

    return backward_substitution(
        upper,
        y,
    )