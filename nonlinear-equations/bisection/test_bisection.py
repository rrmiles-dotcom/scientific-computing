import math

import pytest

from bisection import bisection

def test_finds_square_root_of_two():
    root = bisection(
        lambda x: x**2 - 2.0,
        1.0,
        2.0,
    )

    assert root == pytest.approx(
        math.sqrt(2.0),
        abs = 1e-9,
    )

def test_finds_root_of_cubic():
    root = bisection(
        lambda x: x**3 - x - 2.0,
        1.0,
        2.0,
    )

    assert root == pytest.approx(
        1.5213797068,
        abs = 1e-9,
    )

def test_finds_negative_root():
    root = bisection(
        lambda x: x**2 - 4.0,
        -3.0,
        -1.0,
    )

    assert root == pytest.approx(
        -2.0,
        abs = 1e-9,
    )

def test_returns_left_endpoint_when_it_is_root():
    root = bisection(
        lambda x: x - 2.0,
        2.0,
        5.0,
    )

    assert root == pytest.approx(2.0)

def test_returns_right_endpoint_when_it_is_root():
    root = bisection(
        lambda x: x - 5.0,
        2.0,
        5.0,
    )

    assert root == pytest.approx(5.0)

def test_result_has_small_residual():
    function = lambda x: math.cos(x) - x

    root = bisection(
        function,
        0.0,
        1.0,
    )

    assert abs(function(root)) <= 1e-9

def test_tighter_tolerance_imporives_accuracy():
    function = lambda x: x**2 - 2.0

    coarse = bisection(
        function,
        1.0,
        2.0,
        tolerance = 1e-3,
    )

    fine = bisection(
        function,
        1.0,
        2.0,
        tolerance = 1e-10,
    )

    exact = math.sqrt(2.0)

    assert abs(fine - exact) < abs(coarse - exact)

def test_rejects_interval_without_sign_change():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x**2 + 1.0,
            -1.0,
            1.0,
        )

def test_rejects_reversed_interval():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x,
            2.0,
            1.0,
        )

def test_rejects_equal_endpoints():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x,
            1.0,
            1.0,
        )

def test_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x,
            -1.0,
            1.0,
            tolerance = 0.0,
        )

def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x,
            -1.0,
            1.0,
            tolerance = -1e-6,
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        bisection(
            lambda x: x,
            -1.0,
            1.0,
            max_iterations = 0,
        )

def test_raises_when_iteration_limit_is_too_small():
    with pytest.raises(RuntimeError):
        bisection(
            lambda x: x**2 - 2.0,
            1.0,
            2.0,
            tolerance = 1e-15,
            max_iterations = 1,
        )