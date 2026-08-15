import numpy as np

def _validate_nodes(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Validate interpolation nodes and corresponding values"""

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size == 0:
        raise ValueError("interpolation data must not be empty")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if np.unique(x_values).size != x_values.size:
        raise ValueError("interpolation nodes must be distinct")

def divided_differences(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """
    Compute Newton divided-difference coefficients.
    """

    _validate_nodes(x_values, y_values)

    n = x_values.size
    coefficients = y_values.astype(float).copy()

    # Update coefficients in reverse to preserve values from the previous order.
    for order in range(1, n):
        for i in range(n - 1, order - 1, -1):
            coefficients[i] = (
                coefficients[i]
                - coefficients[i - 1]
            ) / (
                x_values[i]
                - x_values[i - order]
            )

    return coefficients

def newton_interpolate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x: float | np.ndarray,
) -> float | np.ndarray:
    """
    Evaluate the Newton interpolating polynomial.
    """

    _validate_nodes(x_values, y_values)

    coefficients = divided_differences(
        x_values,
        y_values,
    )

    values = np.asarray(x, dtype = float)

    # Nested multiplication evaluates the Newton form efficiently.
    result = np.full_like(
        values,
        coefficients[-1],
        dtype = float,
    )

    for i in range(x_values.size - 2, -1, -1):
        result = (
            coefficients[i]
            + (values - x_values[i]) * result
        )

    if values.ndim == 0:
        return float(result)

    return result