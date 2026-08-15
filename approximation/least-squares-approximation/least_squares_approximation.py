import numpy as np

def _validate_data(
    x_values: np.ndarray,
    y_values: np.ndarray,
    degree: int,
) -> None:
    """Validate data for polynomial least-squares approximation"""

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size == 0:
        raise ValueError("data must not be empty")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if degree < 0:
        raise ValueError("degree must be nonnegative")

    if degree >= x_values.size:
        raise ValueError("degree must be smaller than number of data points")


def design_matrix(
    x_values: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Construct the polynomial design matrix.

    Column j contains x^j.
    """

    if x_values.ndim != 1:
        raise ValueError("x_values must be one-dimensional")

    if x_values.size == 0:
        raise ValueError("x_values must not be empty")

    if degree < 0:
        raise ValueError("degree must be nonnegative")

    return np.vander(
        x_values,
        N = degree + 1,
        increasing = True,
    )

def polynomial_least_squares(
    x_values: np.ndarray,
    y_values: np.ndarray,
    degree: int,
) -> np.ndarray:
    """
    Fit a polynomial to data by minimizing the squared residual error.

    Coefficients are returned in ascending polynomial order.
    """

    _validate_data(
        x_values,
        y_values,
        degree,
    )

    matrix = design_matrix(
        x_values.astype(float),
        degree,
    )

    # Solve the overdetermined system directly rather than forming A^T A.
    coefficients, _, rank, _ = np.linalg.lstsq(
        matrix,
        y_values.astype(float),
        rcond = None
    )

    if rank < degree + 1:
        raise ValueError("design matrix does not have full column rank")

    return coefficients

def evaluate_polynomial(
    coefficients: np.ndarray,
    x: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate a polynomial whose coefficients are in ascending order."""

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    if coefficients.size == 0:
        raise ValueError("coefficients must not be empty")

    values = np.asarray(x, dtype = float)

    # Horner's method avoids explicitly computing every power of x.
    result = np.zeros_like(values, dtype = float)

    for coefficient in coefficients[::-1]:
        result = result * values + coefficient

    if values.ndim == 0:
        return float(result)

    return result

def residuals(
    x_values: np.ndarray,
    y_values: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Return observed minus fitted values."""

    predictions = evaluate_polynomial(
        coefficients,
        x_values,
    )

    return y_values - predictions


def sum_squared_errors(
    x_values: np.ndarray,
    y_values: np.ndarray,
    coefficients: np.ndarray,
) -> float:
    """Return the sum of squared residual errors."""

    errors = residuals(
        x_values,
        y_values,
        coefficients,
    )

    return float(errors @ errors)