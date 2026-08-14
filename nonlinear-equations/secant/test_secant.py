import math

import pytest

from secant import secant

def test_finds_square_root_of_two():
    root = secant(
        lambda x: x**2 - 2.0,
        first_guess = 1.0,
        second_guess = 2.0,  
    )

    assert root == pytest.approx(
        math.sqrt(2.0),
        abs = 1e-9,
    )

def test_finds_cubic_root():
    root = secant(
        lambda x: x**3 - x - 2.0,
        first_guess = 1.0,
        second_guess = 2.0, 
    )

    assert root == pytest.approx(
        1.5213797068,
        abs = 1e-9,
    )

def test_finds_negative_root():
    root = secant(
        lambda x: x**2 - 4.0,
        first_guess = -3.0,
        second_guess = -1.0,
    )

    assert root == pytest.approx(
        -2.0,
        abs = 1e-9,
    )

def test_first_guess_already_root():
    root = secant(
        lambda x: x - 3.0,
        first_guess = 3.0,
        second_guess = 5.0,
    )

    assert root == pytest.approx(3.0)

def test_second_guess_already_root():
    root = secant(
        lambda x: x - 3.0,
        first_guess = 1.0,
        second_guess = 3.0,
    )

    assert root == pytest.approx(3.0)

def test_result_has_small_residual():
    function = lambda x: math.cos(x) - x

    root = secant(
        function,
        first_guess = 0.0,
        second_guess = 1.0,
    )

    assert abs(function(root)) <= 1e-9

def test_linear_function_converges():
    root = secant(
        lambda x: 4.0 * x - 12.0,
        first_guess = 0.0,
        second_guess = 1.0,
    )

    assert root == pytest.approx(3.0)

def test_tighter_tolerance_improves_accuracy():
    function = lambda x: x**2 - 2.0

    coarse = secant(
        function,
        first_guess = 1.0,
        second_guess = 2.0,
        tolerance = 1e-3,
    )

    fine = secant(
        function,
        first_guess = 1.0,
        second_guess = 2.0,
        tolerance = 1e-12,
    )

    exact = math.sqrt(2.0)

    assert abs(fine - exact) <= abs(coarse - exact)

def test_rejects_equal_initial_guesses():
    with pytest.raises(ValueError):
        secant(
            lambda x: x**2 - 2.0,
            first_guess = 1.0,
            second_guess = 1.0,
        )

def test_rejects_zero_tolerance():
    with pytest.raises(ValueError):
        secant(
            lambda x: x,
            first_guess = -1.0,
            second_guess = 1.0,
            tolerance = 0.0,
        )

def test_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        secant(
            lambda x: x,
            first_guess = -1.0,
            second_guess = 1.0,
            tolerance = -1e-6,
        )

def test_rejects_nonpositive_max_iterations():
    with pytest.raises(ValueError):
        secant(
            lambda x: x,
            first_guess = -1.0,
            second_guess = 1.0,
            max_iterations = 0, 
        )

def test_rejects_zero_secant_slope():
    with pytest.raises(ValueError):
        secant(
            lambda x: x**2,
            first_guess = -1.0,
            second_guess = 1.0,
        )

def test_raises_when_iteration_limit_is_too_small():
    with pytest.raises(RuntimeError):
        secant(
            lambda x: x**2 - 2.0,
            first_guess = 1.0,
            second_guess = 2.0,
            tolerance = 1e-15,
            max_iterations = 1, 
        )