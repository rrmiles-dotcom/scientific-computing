import numpy as np

def _validate_nodes(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Validate spline interpolation nodes and values."""

    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x_values and y_values must be one-dimensional")

    if x_values.size < 2:
        raise ValueError("at least two interpolation nodes are required")

    if x_values.size != y_values.size:
        raise ValueError("x_values and y_values must have equal length")

    if np.any(np.diff(x_values) <= 0):
        raise ValueError("x_values must be strictly increasing")

def natural_cubic_spline_coefficients(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute coefficients of the natural cubic spline.

    Each interval is represented as:
        S_i(x) = a_i + b_i dx + c_i dx^2 + d_i dx^3
    where dx = x - x_i.
    """

    _validate_nodes(x_values, y_values)

    x_values = x_values.astype(float)
    y_values = y_values.astype(float)

    n = x_values.size
    h = np.diff(x_values)

    a = y_values[:-1].copy()

    # Natural boundary conditions impose zero second derivative at both ends.
    system = np.zeros((n, n), dtype = float)
    rhs = np.zeros(n, dtype = float)

    system[0, 0] = 1.0
    system[-1, -1] = 1.0

    for i in range(1, n - 1):
        system[i, i - 1] = h[i - 1]
        system[i, i] = 2.0 * (h[i - 1] + h[i])
        system[i, i + 1] = h[i]

        rhs[i] = 3.0 * (
            (y_values[i + 1] - y_values[i]) / h[i]
            - (y_values[i] - y_values[i - 1]) / h[i - 1]             
        )

    c_full = np.linalg.solve(system, rhs)

    b = np.zeros(n - 1, dtype = float)
    c = c_full[:-1].copy()
    d = np.zeros(n - 1, dtype = float)

    for i in range(n - 1):
        b[i] = (
            (y_values[i + 1] - y_values[i]) / h[i]
            - h[i] * (2.0 * c_full[i] + c_full[i + 1]) / 3.0
        )

        d[i] = (
            c_full[i + 1] - c_full[i]
        ) / (3.0 * h[i])

    return a, b, c, d

def natural_cubic_spline(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x: float | np.ndarray,
) -> float | np.ndarray:
    """Evaluate the natural cubic spline at one or more points."""

    _validate_nodes(x_values, y_values)

    a, b, c, d = natural_cubic_spline_coefficients(
        x_values,
        y_values,
    )

    values = np.asarray(x, dtype = float)

    if np.any(values < x_values[0]) or np.any(values > x_values[-1]):
        raise ValueError("evaluation points must lie inside the interpolation range")

    # Select the spline segment immediately to the left of each evaluation point.
    indices = np.searchsorted(
        x_values,
        values,
        side = "right",
    ) - 1

    indices = np.clip(
        indices,
        0,
        x_values.size - 2,
    )

    dx = values - x_values[indices]

    result = (
        a[indices]
        + b[indices] * dx
        + c[indices] * dx**2
        + d[indices] * dx**3
    )

    if values.ndim == 0:
        return float(result)

    return result