import numpy as np

def qr_decomposition(
    matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the reduced QR decomposition A = QR
    using modified Gram-Schmidt.

    Returns:
        Q: matrix with orthonormal columns
        R: upper-triangular matrix
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows < cols:
        raise ValueError("matrix must have at least as many rows as columns")

    q = np.zeros((rows, cols), dtype = float)
    r = np.zeros((cols, cols), dtype = float)

    vectors = matrix.astype(float).copy()

    for i in range(cols):
        norm = np.linalg.norm(vectors[:, i])

        if np.isclose(norm, 0.0):
            raise ValueError("matrix columns must be linearly independent")

        r[i, i] = norm
        q[:, i] = vectors[:, i] / norm

        for j in range(i + 1, cols):
            r[i, j] = np.dot(
                q[:, i],
                vectors[:, j],
            )

            vectors[:, j] -= (
                r[i, j] * q[:, i]
            )

    return q, r

def solve_least_squares_qr(
    matrix: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """
    Solve min || Ax - b ||_2 using QR decomposition.

    Assumes A has full column rank.
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

    solution = np.zeros(cols, dtype = float)

    for i in range(cols - 1, -1, -1):
        known_sum = np.dot(
            r[i, i + 1:],
            solution[i + 1:],
        )

        solution[i] = (
            transformed_rhs[i] - known_sum
        ) / r[i, i]

    return solution