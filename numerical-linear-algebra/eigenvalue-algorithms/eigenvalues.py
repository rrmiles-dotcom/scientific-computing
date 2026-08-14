import numpy as np

def _validate_square_matrix(matrix: np.ndarray) -> None:
    """Validate that matrix is a non-empty square matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("matrix must be square")

    if rows == 0:
        raise ValueError("matrix must not be empty")

def rayleigh_quotient(
    matrix: np.ndarray,
    vector: np.ndarray,
) -> float:
    """Return the Rayleigh quotient of a vector"""

    _validate_square_matrix(matrix)

    if vector.ndim != 1:
        raise ValueError("vector must be one-dimensional")

    if vector.shape[0] != matrix.shape[0]:
        raise ValueError("vector dimension must match matrix size")

    denominator = np.dot(vector, vector)

    if np.isclose(denominator, 0.0):
        raise ValueError("vector must be non-zero")

    numerator  = np.dot(
        vector,
        matrix @ vector,
    )

    return float(numerator / denominator)

def power_iteration(
    matrix: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> tuple[float, np.ndarray]:
    """
    Approximate the dominant eigenvalue and eigenvector using power iteration.
    """

    _validate_square_matrix(matrix)

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    matrix = matrix.astype(float)

    n = matrix.shape[0]

    vector = np.ones(n, dtype = float)
    vector /= np.linalg.norm(vector)

    eigenvalue = rayleigh_quotient(
        matrix,
        vector,
    )

    for _ in range(max_iterations):
        next_vector = matrix @ vector

        norm = np.linalg.norm(next_vector)

        if np.isclose(norm, 0.0):
            raise ValueError("power iteration encountered a zero vector")

        next_vector /= norm

        next_eigenvalue = rayleigh_quotient(
            matrix,
            vector,
        )

        residual = np.linalg.norm(
            matrix @ next_vector
            - next_eigenvalue * next_vector
        )

        vector = next_vector
        eigenvalue = next_eigenvalue

        if residual < tolerance:
            return eigenvalue, vector

    raise RuntimeError("power iteration did not converge")

def inverse_iteration(
    matrix: np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> tuple[float, np.ndarray]:
    """
    Approximate the smallest-magnitude eigenvalue using inverse iteration.
    """

    _validate_square_matrix(matrix)

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    if np.linalg.matrix_rank(matrix) < matrix.shape[0]:
        raise ValueError("matrix must be non-singular")

    matrix = matrix.astype(float)

    n = matrix.shape[0]

    vector = np.ones(n, dtype = float)
    vector /= np.linalg.norm(vector)

    for _ in range(max_iterations):
        next_vector = np.linalg.solve(
            matrix, 
            vector,
        )

        norm = np.linalg.norm(next_vector)

        if np.isclose(norm, 0.0):
            raise ValueError("inverse iteration encountered a zero vector")

        next_vector /= norm

        eigenvalue = rayleigh_quotient(
            matrix,
            next_vector,
        )

        residual = np.linalg.norm(
            matrix @ next_vector
            - eigenvalue * next_vector
        )

        vector= next_vector

        if residual < tolerance:
            return eigenvalue, vector

    raise RuntimeError("inverse iteration did not converge")

def qr_eigenvalues(
    matrix = np.ndarray,
    tolerance: float = 1e-10,
    max_iterations: int = 2000,
) -> np.ndarray:

    """
    Approximate all eigenvalues of a real symmetric matrix using unshifted QR iteration.
    """

    _validate_square_matrix(matrix)

    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")

    if not np.allclose(matrix, matrix.T):
        raise ValueError("matrix must be symmetric")

    current = matrix.astype(float).copy()

    for _ in range(max_iterations):
        q, r = np.linalg.qr(current)

        current = r @ q

        off_diagonal = current - np.diag(
            np.diag(current)
        )

        if np.linalg.norm(off_diagonal) < tolerance:
            return np.diag(current).copy()

    raise RuntimeError("QR iteration did not converge")