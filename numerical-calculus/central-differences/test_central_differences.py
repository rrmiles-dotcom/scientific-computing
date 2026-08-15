import numpy as np
import pytest

from central_differences import(
    central_difference,
    central_difference_array,
    central_difference_error,
    second_central_difference,
)

def test_central_difference_linear():
    result = central_difference(
        lambda x: 3.0 * x + 2.0,
        4.0,
    )

    assert result == pytest.approx(
        3.0,
        abs = 1e-10,
    )

def test_central_difference_quadratic():
    result = central_difference(
        lambda x: x**2,
        2.0,
        step = 1e-5,
    )

    assert result == pytest.approx(
        4.0,
        abs = 1e-9,
    )

def test_central_difference_sine():
    result = central_difference(
        np.sin,
        0.0,
        step = 1e-5,
    )

    assert result == pytest.approx(
        1.0,
        abs = 1e-9,
    )

def test_central_difference_is_more_accurate_than_forward():
    function = np.exp
    exact = 1.0
    step = 1e-2

    central = central_difference(
        function,
        0.0,
        step = step,
    )

    forward = (
        function(step) - function(0.0)
    ) / step

    assert (
        abs(central - exact)
        < abs(forward - exact)
    )

def test_error_decreases_quadratically():
    function = np.exp
    derivative = np.exp

    coarse = central_difference_error(
        function,
        derivative,
        0.0,
        step = 0.1,
    )

    fine = central_difference_error(
        function,
        derivative,
        0.0,
        step = 0.05,
    )

    # Halving h should reduce the O(h^2) error by approximately four.

    assert fine < coarse / 3.5

def test_central_difference_array_linear():
    values = np.array([
        1.0,
        3.0,
        5.0,
        7.0,
        9.0,
    ])

    result = central_difference_array(
        values,
        step = 0.5,
    )

    expected = np.array([
        4.0,
        4.0,
        4.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_central_difference_array_quadratic():
    values = np.array([
        0.0,
        1.0,
        4.0,
        9.0,
        16.0,
    ])

    result = central_difference_array(
        values,
        step = 1.0,
    )

    expected = np.array([
        2.0,
        4.0,
        6.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
        atol = 1e-12,
    )

def test_central_difference_error():
    result = central_difference_error(
        np.sin,
        np.cos,
        0.5,
        step = 1e-4,
    )

    assert result < 1e-8

def test_second_difference_quadratic():
    result = second_central_difference(
        lambda x: x**2,
        3.0,
        step = 1e-3,
    )

    assert result == pytest.approx(
        2.0,
        abs = 1e-8,
    )

def test_second_difference_sine():
    x = 0.5

    result = second_central_difference(
        np.sin,
        x,
        step = 1e-4,
    )

    assert result == pytest.approx(
        -np.sin(x),
        abs = 1e-7,
    )

def test_rejects_zero_step():
    with pytest.raises(ValueError):
        central_difference(
            lambda x: x**2,
            1.0,
            step = 0.0,
        )

def test_rejects_negative_step():
    with pytest.raises(ValueError):
        central_difference(
            lambda x: x**2,
            1.0,
            step = -0.1,
        )

def test_array_rejects_non_vector_input():
    values = np.ones((2, 2))

    with pytest.raises(ValueError):
        central_difference_array(
            values,
            step = 1.0,
        )

def test_array_rejects_too_few_values():
    values = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(ValueError):
        central_difference_array(
            values,
            step = 1.0,
        )

def test_array_rejects_zero_step():
    values = np.array([
        1.0,
        2.0,
        3.0,
    ])

    with pytest.raises(ValueError):
        central_difference_array(
            values,
            step = 0.0,
        )

def test_second_difference_rejects_zero_step():
    with pytest.raises(ValueError):
        second_central_difference(
            lambda x: x**2,
            1.0,
            step = 0.0,
        )