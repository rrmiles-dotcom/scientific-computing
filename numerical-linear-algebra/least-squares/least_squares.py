import numpy as np

def solve_normal_equations(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve a least-squares problem using the normal equations:
    
        A^T A x = A^T b
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if matrix.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    rows, cols = matrix.shape

    if rows < cols:
        raise ValueError("matrix must have atleast as many rows as columns")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimension must match matrix rows")

    gram_matrix = matrix.T @ matrix
    transformed_rhs = matrix.T @ rhs

    if np.linalg.matrix_rank(gram_matrix) < cols:
        raise ValueError("matrix must have full column rank")

    return np.linalg.solve(
        gram_matrix,
        transformed_rhs,
    )

def qr_decomposition(
        matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute reduced QR decomposition using modified Gram-Schmidt.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows < cols:
        raise ValueError("matrix must have atleast as many rows as columns")

    q = np.zeros((rows, cols), dtype = float)
    r = np.zeros((cols, cols), dtype = float)

    vectors = matrix.astype(float).copy()

    for i in range(cols):
        norm = np.linalg.norm(vectors[:, i])

        if np.isclose(norm, 0.0):
            raise ValueError("matrix must have full column rank")

        r[i, i] = norm
        q[:, i] = vectors[:, i] / norm

        for j in range(i + 1, cols):
            r[i, j] = np.dot(
                q[i:, i],
                vectors[:, j],
            )

            vectors[:, j] -= (
                r[i, j] * q[:, i]
            )

    return q, r

def backward_substitution(
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve an upper-triangular system."""

    n = upper.shape[0]
    solution = np.zeros(n, dtype = float)

    for i in range(n - 1, -1, -1):
        known_sum = np.dot(
            upper[i, i + 1:],
            solution[i + 1:],
        )

        if np.isclose(upper[i, i], 0.0):
            raise ValueError("matrix is singular")

        solution[i] = (
            rhs[i] - known_sum
        ) / upper[i, i]

    return solution

def solve_least_squares_qr(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve min || Ax - b ||_2 using QR decomposition.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    rows, cols = matrix.shape

    if rows < cols:
        raise ValueError("matrix must have atleast as many rows as columns")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimension must match matrix rows")

    q, r = qr_decomposition(matrix)

    transformed_rhs = q.T @ rhs

    return backward_substitution(
        r,
        transformed_rhs,
    )

def residual(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Return least-squares residual b - Ax."""

    return rhs - matrix @ solution

def residual_norm(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return || b - Ax ||_2."""

    return float(
        np.linalg.norm(
            residual(matrix, solution, rhs)
        )
    )