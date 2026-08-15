import numpy as np
import pytest

from forward_differences import(
    forward_difference,
    forward_difference_array,
    forward_difference_error,
)

def test_forward_difference_linear():
    result = forward_difference(
        lambda x: 3.0 * x + 2.0,
        4.0,
    )

    assert result == pytest.approx(
        3.0,
        abs = 1e-8,
    )

def test_forward_difference_quadratic():
    result = forward_difference(
        lambda x: x**2,
        2.0,
        step = 1e-6,
    )

    assert result == pytest.approx(
        4.0,
        abs = 1e-5,
    )

def test_forward_difference_exponential():
    result = forward_difference(
        np.exp,
        0.0,
        step = 1e-6,
    )

    assert result == pytest.approx(
        1.0,
        abs = 1e-5,
    )

def test_smaller_step_improves_quadratic_approximation():
    exact = 4.0

    coarse = forward_difference(
        lambda x: x**2,
        2.0,
        step = 1e-2,
    )

    fine = forward_difference(
        lambda x: x**2,
        2.0,
        step = 1e-5,
    )

    assert abs(fine - exact) < abs(coarse - exact)

def test_forward_difference_array_linear():
    values = np.array([
        1.0,
        3.0,
        5.0,
        7.0,
    ])

    result = forward_difference_array(
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

def test_forward_difference_array_quadratic():
    values = np.array([
        0.0,
        1.0,
        4.0,
        9.0,
    ])

    result = forward_difference_array(
        values,
        step = 1.0,
    )

    expected = np.array([
        1.0,
        3.0,
        5.0,
    ])

    np.testing.assert_allclose(
        result,
        expected,
    )

def test_forward_difference_error():
    result = forward_difference_error(
        lambda x: x**2,
        lambda x: 2.0 * x,
        2.0,
        step = 0.01,
    )

    # For x^2, the forward-difference error is exactly h.
    assert result == pytest.approx(
        0.01,
        abs = 1e-12,
    )

def test_error_decreases_with_step_size():
    coarse = forward_difference_error(
        lambda x: x**2,
        lambda x: 2.0 * x,
        2.0,
        step = 0.1,
    )

    fine = forward_difference_error(
        lambda x: x**2,
        lambda x: 2.0 * x,
        2.0,
        step = 0.01,  
    )

    assert fine < coarse

def test_rejects_zero_step():
    with pytest.raises(ValueError):
        forward_difference(
            lambda x: x**2,
            1.0,
            step = 0.0,
        )

def test_rejects_negative_step():
    with pytest.raises(ValueError):
        forward_difference(
            lambda x: x**2,
            1.0,
            step = -0.1,
        )

def test_array_rejects_non_vector_input():
    values = np.ones((2, 2))

    with pytest.raises(ValueError):
        forward_difference_array(
            values,
            step = 1.0,
        )

def test_array_rejects_single_value():
    values = np.array([
        1.0,
    ])

    with pytest.raises(ValueError):
        forward_difference_array(
            values,
            step = 1.0,
        )

def test_array_rejects_zero_step():
    values = np.array([
        1.0,
        2.0,
    ])

    with pytest.raises(ValueError):
        forward_difference_array(
            values,
            step = 0.0,
        )