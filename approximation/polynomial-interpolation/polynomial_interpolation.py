import numpy as np

def vandermonde_matrix(
    x_values: np.ndarray,
) -> np.ndarray:
    """
    Construct the Vandermonde matrix for polynomial interpolation.

    Columns correspond to increasing polynomial powers:
    1, x, x^2, ..., x^(n - 1).
    """

    if x_values.ndim != 1:
        raise ValueError("x_values must be one-dimensional")

    if x_values.size == 0:
        raise ValueError("x_values must not be empty")

    return np.vander(
        x_values.astype(float),
        N = x_values.size,
        increasing = True,
    )

def polynomial_coefficients(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """
    Compute coefficients of the unique interpolating polynomial.
    
    Coefficients are returned in increasing powers:

    c[0] + c[1]x + c[2]x^2 + ...
    """

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if x_values.size == 0:
        raise ValueError("input arrays must not be empty")

    # Repeated interpolation nodes make the Vandermonde system singular.
    if np.unique(x_values).size != x_values.size:
        raise ValueError("x_values must contain distinct nodes")

    matrix = vandermonde_matrix(x_values)

    return np.linalg.solve(
        matrix,
        y_values.astype(float)
    )

def evaluate_polynomial(
    coefficients: np.ndarray,
    x: float | np.ndarray,
) -> float | np.ndarray:

    """
    Evaluate a polynomial whose coefficients use increasing powers.
    """

    if coefficients.ndim != 1:
        raise ValueError("coefficients must be one-dimensional")

    if coefficients.size == 0:
        raise ValueError("coeffcients must not be empty")

    values = np.asarray(x, dtype = float)

    # Horner evaluation avoids explicitly forming high polynomial powers.
    result = np.zeros_like(values, dtype = float)

    for coefficient in coefficients[::-1]:
        result = result * values + coefficient

    if np.ndim(x) == 0:
        return float(result)

    return result

def interpolate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x: float| np.ndarray,
) -> float | np.ndarray:
    """
    Construct the interpolating polynomial and evaluate it at x.
    """

    coefficients = polynomial_coefficients(
        x_values,
        y_values,
    )

    return evaluate_polynomial(
        coefficients,
        x,
    )