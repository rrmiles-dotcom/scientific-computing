import numpy as np

def matrix_condition_number(
    matrix: np.ndarray,
) -> float:
    """
    Compute the 2-norm condition number of a matrix.

    kappa(A) = sigma_max(A) / sigma_min(A)
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("matrix must be square")

    singular_values = np.linalg.svd(
        matrix.astype(float),
        compute_uv = False
    )

    largest = singular_values[0]
    smallest = singular_values[-1]

    if np.isclose(smallest, 0.0):
        return float("inf")

    return float(largest / smallest)

def relative_input_error(
    original: np.ndarray,
    perturbed: np.ndarray,
) -> float:
    """
    Compute relative perturbation:

    || perturbed - original || / || original ||
    """

    if original.shape != perturbed.shape:
        raise ValueError("arrays must have matching shapes")

    original_norm = np.linalg.norm(original)

    if np.isclose(original_norm, 0.0):
        raise ValueError("original array must be nonzero")

    return float(
        np.linalg.norm(perturbed - original)
        / original_norm
    )

def relative_solution_error(
    original_solution: np.ndarray,
    perturbed_solution: np.ndarray,
) -> float:
    """
    Compute relative error between two solutions.
    """

    return relative_input_error(
        original_solution,
        perturbed_solution,
    )

def solve_with_perturbation(
    matrix: np.ndarray,
    rhs: np.ndarray,
    perturbation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve 
    
        Ax = b 
    
    and 

        Ax_perturbed = b + perturbation

    to demonstrate sensitivity to perturbations.
    """

    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")

    rows, cols = matrix.shape

    if rows != cols:
        raise ValueError("matrix must be square")

    if rhs.ndim != 1:
        raise ValueError("rhs must be one-dimensional")

    if perturbation.ndim != 1:
        raise ValueError("perturbation must be one-dimensional")

    if rhs.shape[0] != rows:
        raise ValueError("rhs dimension must match matrix")

    if perturbation.shape != rhs.shape:
        raise ValueError(
            "perturbation must have same shape as rhs"
        )

    if np.linalg.matrix_rank(matrix) < rows:
        raise ValueError("matrix must be nonsingular")

    original_solution = np.linalg.solve(
        matrix,
        rhs,
    )

    perturbed_solution = np.linalg.solve(
        matrix,
        rhs + perturbation,
    )

    return original_solution, perturbed_solution

def error_amplification(
    matrix: np.ndarray,
    rhs: np.ndarray,
    perturbation: np.ndarray,
) -> float:
    """
    Measure how much a relative perturbation in b 
    is amplified in the solution x.
    """

    original_solution, perturbed_solution = (
        solve_with_perturbation(
            matrix,
            rhs,
            perturbation,
        )
    )

    input_error = relative_input_error(
        rhs,
        rhs + perturbation,
    )

    if np.isclose(input_error, 0.0):
        return 0.0

    solution_error = relative_solution_error(
        original_solution,
        perturbed_solution,
    )

    return float(solution_error / input_error)
