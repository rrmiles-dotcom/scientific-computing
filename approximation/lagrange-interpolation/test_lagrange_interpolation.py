import numpy as np
import pytest

from lagrange_interpolation import(
    langrange_basis,
    lagrange_interpolate,
)

def test_basis_is_one_at_own_node():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    result = langrange_basis(
        x_values,
        index = 1,
        x = 1.0,
    )

    assert result == pytest.approx(1.0)


def test_basis_zero_at_other_nodes():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    result = langrange_basis(
        x_values,
        index = 1,
        x = np.array([0.0, 2.0])
    )

    np.testing.assert_allclose(
        result,
        np.array([0.0, 0.0]),
        atol = 1e-12,
    )

def test_basis_partition_of_unity():
    x_values = np.array([
        -1.0,
        0.0,
        2.0,
    ])

    point = 0.5

    total = sum(
        langrange_basis(
            x_values,
            i,
            point,
        )

        for i in range(x_values.size)
    )

    assert total == pytest.approx(
        1.0,
        abs = 1e-12,
    )

def test_interpolation_reproduces_node():
    x_values = np.array([
        -1.0,
        0.0,
        2.0,
    ])

    y_values = np.array([
        4.0,
        1.0,
        3.0,
    ])

    result = lagrange_interpolate(
        x_values,
        y_values,
        x_values,
    )

    np.testing.assert_allclose(
        result,
        y_values,
        atol = 1e-12,
    )

def test_known_quadratic():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        6.0,
        17.0,
    ])

    result = lagrange_interpolate(
        x_values,
        y_values,
        3.0,
    )

    assert result == pytest.approx(34.0)

def test_linear_interpolation():
    x_values = np.array([
        0.0,
        2.0,
    ])

    y_values = np.array([
        1.0,
        5.0,
    ])

    result = lagrange_interpolate(
        x_values,
        y_values,
        1.0,
    )

    assert result == pytest.approx(3.0)

def test_constant_interpolation():
    x_values = np.array([
        2.0,
    ])

    y_values = np.array([
        7.0,
    ])

    result = lagrange_interpolate(
        x_values,
        y_values,
        100.0,
    )

    assert result == pytest.approx(7.0)

def test_vector_evaluation():
    x_values = np.array([
        0.0,
        1.0,
    ])

    y_values = np.array([
        1.0,
        3.0,
    ])

    points = np.array([
        0.0,
        0.5,
        1.0,
    ])

    result = lagrange_interpolate(
        x_values,
        y_values,
        points,
    )

    expected = np.array([
        1.0,
        2.0,
        3.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_matches_polynomial_defined_by_nodes():
    x_values = np.array([
        -2.0,
        -1.0,
        1.0,
        3.0,
    ])

    y_values = (
    2.0
    + 3.0 * x_values
    - x_values**2
    + 0.5 * x_values**3
    )

    points = np.linspace(
        -2.0,
        3.0,
        20,
    )

    result = lagrange_interpolate(
        x_values,
        y_values,
        points,
    )

    expected = (
        2.0
        + 3.0 * points
        - points**2
        + 0.5 * points**3 
    )

    np.testing.assert_allclose(
        result,
        expected,
        rtol = 1e-10,
        atol = 1e-12,
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
        lagrange_interpolate(
            x_values,
            y_values,
            0.5,
        )

def test_rejects_length_mismatch():
    x_values = np.ones(3)
    y_values = np.ones(2)

    with pytest.raises(ValueError):
        lagrange_interpolate(
            x_values,
            y_values,
            0.5,
        )

def test_rejects_non_vector_nodes():
    x_values = np.ones((2, 2))
    y_values = np.ones(4)

    with pytest.raises(ValueError):
        lagrange_interpolate(
            x_values,
            y_values,
            0.5,
        )

def test_rejects_empty_input():
    x_values = np.array([])
    y_values = np.array([])

    with pytest.raises(ValueError):
        lagrange_interpolate(
            x_values,
            y_values,
            0.5,
        )

def test_basis_rejects_invalid_index():
    x_values = np.array([
        0.0,
        1.0,
        2.0,
    ])

    with pytest.raises(ValueError):
        langrange_basis(
            x_values,
            index = 3,
            x = 1.0,
        )