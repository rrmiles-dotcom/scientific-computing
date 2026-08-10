import numpy as np
import pytest

from floating_point import(
    machine_epsilon,
    next_float,
    absolute_error,
    relative_error,
    approximately_equal,
    cancellation_example,
    summation_forward,
    summation_reverse,
)

def test_machine_epsilon():
    eps = machine_epsilon()

    assert eps == pytest.approx(np.finfo(np.float64).eps)
    assert 1.0 + eps > 1.0

def test_next_float_is_greater():
    x = 1.0
    next_value = next_float(x)
    
    assert next_value > x

def test_next_float_matches_numpy():
    x = 10.0

    assert next_float(x) == pytest.approx(
        np.nextafter(x, np.inf)
    )

def test_absolute_error():
    approximation = 3.14
    exact = np.pi

    result = absolute_error(approximation, exact)

    assert result == pytest.approx(abs(3.14 - np.pi))

def test_relative_error():
    approximation = 9.9
    exact = 10.0

    assert relative_error(approximation, exact) == pytest.approx(0.01)

def test_relative_error_zero_exact_value():
    with pytest.raises(ValueError):
        relative_error(1.0, 0.0)

def test_approximately_equal():
    a = 0.1 + 0.2
    b = 0.3

    assert approximately_equal(a, b)

def test_approximately_equal_false():
    assert not approximately_equal(1.0, 1.1)

def test_negative_tolerance_rejected():
    with pytest.raises(ValueError):
        approximately_equal(
            1.0,
            1.0,
            relative_tolerance = -1e9,
        )

def test_cancellation_example_stable_form_is_more_accurate():
    x = 1e16

    direct, stable = cancellation_example(x)

    expected = 1.0 / (
        np.sqrt(x + 1.0) + np.sqrt(x)
    )

    direct_error = abs(direct - expected)
    stable_error = abs(stable - expected)

    assert stable_error <= direct_error

def test_cancellation_example_negative_input():
    with pytest.raises(ValueError):
        cancellation_example(-1.0)

def test_summation_order_can_matter():
    values = np.array([
        1e16,
        -1e16,
        1.0,
    ])

    forward = summation_forward(values)
    reverse = summation_reverse(values)

    assert forward != reverse

def test_forward_sum():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert summation_forward(values) == pytest.approx(10.0)

def test_reverse_sum():
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert summation_reverse(values) == pytest.approx(10.0)

def test_summation_matches_for_well_behaved_values():
    values = np.array([0.5, 1.5, 2.5, 3.5])

    assert summation_forward(values) == pytest.approx(
        summation_reverse(values)
    )