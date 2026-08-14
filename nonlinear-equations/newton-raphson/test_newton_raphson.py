import math

import pytest

from newton_raphson import newton_raphson

def test_finds_square_root_of_two():
    root = newton_raphson(
        lambda x: x**2 - 2.0,
        lambda x: 2.0 * x,
        initial_guess = 1.5,
    )

    assert root == pytest.approx(
        math.sqrt(2.0),
        abs = 1e-9,
    )

def test_finds_cubic_root():
    root = newton_raphson(
        lambda x: x**3 - x - 2.0,
        lambda x: 3.0 * x**2 - 1.0,
        initial_guess = 1.5,
    )

    assert root == pytest.approx(
        1.5213797068,
        abs = 1e-9,
    )

def test_finds_negative_root():
    root = newton_raphson(
        lambda x: x**2 - 4.0,
        lambda x: 2.0 * x,
        initial_guess = -3.0,
    )

    assert root == pytest.approx(
        -2.0,
        abs = 1e-9,
    )

def test_initial_guess_already_root():
    root = newton_raphson(
        lambda x: x - 5.0,
        lambda x: 1.0,
        initial_guess = 5.0,
    )

    assert root == pytest.approx(5.0)

def test_result_has_small_residual():
    function = lambda x: math.cos(x) - x
    derivative = lambda x: -math.sin(x) - 1.0

    root = newton_raphson(
        function,
        derivative,
        initial_guess = 1.0,
    )

    assert abs(function(root)) <= 1e-9

def test_tighter_tolerance_improves_accuracy():
    function = lambda x: x**2 - 2.0
    derivative = lambda x: 2.0 * x

    coarse = newton_raphson(
        function,
        derivative,
        initial_guess = 1.5,
        tolerance = 1e-3,
    )

    fine = newton_raphson(
        function,
        derivative,
        initial_guess = 1.5,
        tolerance = 1e-12,
    )

    exact = math.sqrt(2.0)

    assert abs(fine - exact) <= abs(coarse - exact)

def test_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: x,
            lambda x: 1.0,
            initial_guess = 1.0,
            tolerance = 0.0,
        )

def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: x,
            lambda x: 1.0,
            initial_guess = 1.0,
            tolerance = -1e-6, 
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: x,
            lambda x: 1.0,
            initial_guess = 1.0,
            max_iterations = 0,
        )

def test_rejects_near_zero_derivative():
    with pytest.raises(ValueError):
        newton_raphson(
            lambda x: x**2 + 1.0,
            lambda x: 2.0 * x,
            initial_guess = 0.0,
        )

def test_raises_when_iteration_limit_is_too_small():
    with pytest.raises(RuntimeError):
        newton_raphson(
            lambda x: x**2 - 2.0,
            lambda x: 2.0 * x,
            initial_guess = 10.0,
            tolerance = 1e-15,
            max_iterations = 1,
        )

def test_linear_function_converges_in_one_step():
    root = newton_raphson(
        lambda x: 3.0 * x - 9.0,
        lambda x: 3.0,
        initial_guess = 0.0,
    )

    assert root == pytest.approx(3.0)
