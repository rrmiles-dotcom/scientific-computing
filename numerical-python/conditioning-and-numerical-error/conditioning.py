import numpy as np

def absolute_error(
    approximation: np.ndarray,
    exact: np.ndarray,
) -> float:
    """Return the Euclidean absolute error between two vectors."""

    if approximation.shape != exact.shape:
        raise ValueError("approximation and exact must have the same shape")

    return float(np.linalg.norm(approximation - exact))

def relative_error(
    approximation: np.ndarray,
    exact: np.ndarray,
) -> float:
    """Return the relative Euclidean error."""

    if approximation.shape != exact.shape:
        raise ValueError("approximation and exact must have the same shape")

    exact_norm = np.linalg.norm(exact)

    if exact_norm == 0:
        raise ValueError("relative error is undefined for zero exact vector")

    return float(
        np.linalg.norm(approximation - exact) / exact_norm
    )

def residual(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Return the residual r = b - Ax."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if solution.ndim != 1 or rhs.ndim != 1:
        raise ValueError("solution and rhs must be one-dimensional")

    if matrix.shape[1] != solution.shape[0]:
        raise ValueError("matrix and solution dimensions are incompatible")

    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError("matrix and rhs dimensions are incompatible")

    return rhs - matrix @ solution

def relative_residual(
    matrix: np.ndarray,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    """Return ||b - Ax|| / ||b||."""

    rhs_norm = np.linalg.norm(rhs)

    if rhs_norm == 0:
        raise ValueError("relative residual is undefined for zero rhs")

    r = residual(matrix, solution, rhs)

    return float(np.linalg.norm(r) / rhs_norm)

def condition_number(matrix: np.ndarray) -> float:
    """Return the 2-norm condition number of a square matrix."""

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    return float(np.linalg.cond(matrix))

def perturbation_amplification(
    matrix: np.ndarray,
    rhs: np.ndarray,
    perturbation: np.ndarray,
) -> float:
    """
    Measure how much a perturbation in b is amplified in the solution.

    Returns:
        relative change in solution / relative change in rhs
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    if rhs.ndim != 1 or perturbation.ndim != 1:
        raise ValueError("rhs and perturbation must be one-dimensional")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")

    if matrix.shape[0] != rhs.shape[0]:
        raise ValueError("matrix and rhs dimensions are incompatible")

    if rhs.shape != perturbation.shape:
        raise ValueError("rhs and perturbation must have the same shape")

    if np.linalg.norm(rhs) == 0:
        raise ValueError("rhs must be non-zero")

    if np.linalg.norm(perturbation) == 0:
        raise ValueError("perturbation must be non-zero")

    x = np.linalg.solve(matrix, rhs)
    x_perturbed = np.linalg.solve(
        matrix,
        rhs + perturbation,
    )

    relative_rhs_change = (
        np.linalg.norm(perturbation)
        / np.linalg.norm(rhs)
    )

    relative_solution_change = (
        np.linalg.norm(x_perturbed - x)
        / np.linalg.norm(x)
    )

    return float(
        relative_solution_change / relative_rhs_change
    )