import numpy as np

def _validate_nodes(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """
    Validate interpolation nodes and values.
    """

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if x_values.size == 0:
        raise ValueError("input arrays msut not be empty")

    if np.unique(x_values).size != x_values.size:
        raise ValueError("x_values must contain distinct nodes")

def langrange_basis(
    x_values: np.ndarray,
    index: int,
    x: float | np.ndarray,
) -> float | np.ndarray:
    """
    Evaluate the Lagrange basis polynomial L_i(x).
    """

    if x_values.ndim != 1:
        raise ValueError("x_values must be one-dimensional")

    if x_values.size == 0:
        raise ValueError("x_values must not be empty")

    if np.unique(x_values).size != x_values.size:
        raise ValueError("x_values must contain distinct nodes")

    if index < 0 or index >= x_values.size:
        raise ValueError("basis index is out of range")

    values = np.asarray(x, dtype = float)
    result = np.ones_like(values, dtype = float)

    x_i = x_values[index]

    for j in range(x_values.size):
        if j == index:
            continue

        # Each factor L_i to vanish at every node except x_i.
        result *= (
            values - x_values[j]
        ) / (
            x_i - x_values[j]
        )

    if np.ndim(x) == 0:
        return float(result)

    return result

def lagrange_interpolate(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x: float | np.ndarray,
) -> float | np.ndarray:
    """
    Evaluate the interpolating polynomial in Lagrange form.
    """

    _validate_nodes(
        x_values,
        y_values,
    )

    values = np.asarray(x, dtype = float)
    result = np.zeros_like(values, dtype = float)

    for i in range(x_values.size):
        result += (
            y_values[i]
            * langrange_basis(
                x_values,
                i,
                values,
            )
        )

    if np.ndim(x) == 0:
        return float(result)

    return result

