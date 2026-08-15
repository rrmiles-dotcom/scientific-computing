import numpy as np
import pytest

from splines import(
    natural_cubic_spline_coefficients,
    natural_cubic_spline,
)

def test_spline_reproduces_nodes():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        0.0,
        2.0,
    ])

    result = natural_cubic_spline(
        x_values,
        y_values,
        x_values,
    )

    np.testing.assert_allclose(
        result,
        y_values,
        atol = 1e-12,
    )

def test_linear_data_remains_linear():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = 2.0 * x_values + 1.0

    points = np.linspace(
        0.0,
        3.0,
        20,
    )

    result = natural_cubic_spline(
        x_values,
        y_values,
        points
    )

    expected = 2.0 * points + 1.0

    np.testing.assert_allclose(
        result,
        expected,
        atol = 1e-12,
    )

def test_constant_data_remains_constant():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        5.0,
        5.0,
        5.0,
    ])

    points = np.linspace(
        0.0,
        2.0,
        10,
    )

    result = natural_cubic_spline(
        x_values,
        y_values,
        points,
    )

    np.testing.assert_allclose(
        result,
        np.full(points.shape, 5.0),
        atol = 1e-12,
    )

def test_two_nodes_reduce_to_linear_interpolation():
    x_values = np.array([
        0.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        5.0,
    ])

    result = natural_cubic_spline(
        x_values,
        y_values,
        1.0,
    )

    assert result == pytest.approx(3.0)

def test_scalar_evaluation_returns_float():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        0.0,
        1.0,
        0.0,
    ])

    result = natural_cubic_spline(
        x_values,
        y_values,
        0.5,
    )

    assert isinstance(result, float)

def test_coefficient_shapes():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        1.0,
        3.0,
        2.0,
        4.0,
    ])

    a, b, c, d = natural_cubic_spline_coefficients(
        x_values,
        y_values,
    )

    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)
    assert d.shape == (3,)

def test_natural_boundary_conditions():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        0.0,
        1.0,
        0.0,
        1.0,
    ])

    a, b, c, d = natural_cubic_spline_coefficients(
        x_values,
        y_values,
    )

    left_second_derivative = 2.0 * c[0]

    h_last = x_values[-1] - x_values[-2]
    right_second_derivative = (
        2.0 * c[-1]
        + 6.0 * d[-1] * h_last
    )

    assert left_second_derivative == pytest.approx(
        0.0,
        abs = 1e-12,
    )

    assert right_second_derivative == pytest.approx(
        0.0,
        abs = 1e-12,
    )

def test_first_derivative_is_continuous():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        1.0,
        0.0,
        2.0,
        1.0,
    ])

    a, b, c, d = natural_cubic_spline_coefficients(
        x_values,
        y_values,
    )

    for i in range(len(b) - 1):
        h = x_values[i + 1] - x_values[i]

        derivative_left = (
            b[i]
            + 2.0 * c[i] * h
            + 3.0 * d[i] * h**2
        )

        derivative_right = b[i + 1]

        assert derivative_left == pytest.approx(
            derivative_right,
            abs = 1e-10,
        )

def test_second_derivative_is_continous():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
        3.0,
    ])

    y_values = np.array([
        2.0,
        -1.0,
        3.0,
        0.0,
    ])

    a, b, c, d = natural_cubic_spline_coefficients(
        x_values,
        y_values,
    )

    for i in range(len(c) - 1):
        h = x_values[i + 1] - x_values[i]

        second_left = (
            2.0 * c[i]
            + 6.0 * d[i] * h
        )

        second_right = 2.0 * c[i + 1]

        assert second_left == pytest.approx(
            second_right,
            abs = 1e-10,
        )  

def test_rejects_unsorted_nodes():
    x_values = np.array([
        0.0,
        2.0,
        1.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        natural_cubic_spline(
            x_values,
            y_values,
            1.0,
        )

def test_rejects_duplicate_nodes():
    x_values = np.array([
        0.0,
        1.0,
        1.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        natural_cubic_spline(
            x_values,
            y_values,
            0.5,
        )

def test_rejects_length_mismatch():
    x_values = np.ones(3)
    y_values = np.ones(2)

    with pytest.raises(ValueError):
        natural_cubic_spline_coefficients(
            x_values,
            y_values,
        )

def test_rejects_too_few_nodes():
    x_values = np.array([1.0])
    y_values = np.array([2.0])

    with pytest.raises(ValueError):
        natural_cubic_spline(
            x_values,
            y_values,
            1.0,
        )

def test_rejects_non_vector_input():
    x_values = np.ones((2, 2))
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        natural_cubic_spline(
            x_values,
            y_values,
            0.5,
        )

def test_rejects_evaluation_outside_range():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        natural_cubic_spline(
            x_values,
            y_values,
            3.0,
        )