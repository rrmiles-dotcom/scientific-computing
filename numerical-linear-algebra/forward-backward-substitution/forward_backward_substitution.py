import numpy as np

def forward_substitution(
    lower: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve Lx = b for a lower-triangular matrix L"""

    if lower.ndim != 2:
        raise ValueError("lower must be two-dimensional")

    rows, cols = lower.shape

    if rows != cols:
        raise ValueError("lower must be square")

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimension must match matrix size")

    if not np.allclose(lower, np.tril(lower)):
        raise ValueError("matrix must be lower triangular")

    if np.any(np.isclose(np.diag(lower), 0.0)):
        raise ValueError("matrix must have non-zero diagnoal entries")

    solution = np.zeros(rows, dtype = float)

    for i in range(rows):
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
    """Solve Ux = b for an upper-triangular matrix U."""

    if upper.ndim != 2:
        raise ValueError("upper must be two-dimensional")

    rows, cols = upper.shape

    if rows != cols:
        raise ValueError("upper must be square")

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimension must match matrix size")

    if not np.allclose(upper, np.triu(upper)):
        raise ValueError("matrix must be upper triangular")

    if np.any(np.isclose(np.diag(upper), 0.0)):
        raise ValueError("matrix must have non-zero diagonal entries")

    solution = np.zeros(rows, dtype = float)

    for i in range(rows - 1, -1, -1):
        known_sum = np.dot(
            upper[i, i + 1:],
            solution[i + 1:],
        )

        solution[i] = (
            rhs[i] - known_sum
        ) / upper[i, i]

    return solution