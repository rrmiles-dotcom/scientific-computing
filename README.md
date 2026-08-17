# Scientific Computing

A collection of numerical methods implemented in Python, with an emphasis on the mathematics behind the algorithms, numerical behaviour, and reliable testing.

## About

This repository grew out of a systematic study of scientific computing and numerical analysis.

Rather than treating numerical algorithms as black-box library calls, the methods here are implemented directly and tested against problems with known behaviour. The goal is to understand not only how an algorithm works, but also when it works well, how quickly it converges, where numerical error enters, and what can cause it to fail.

The repository covers topics ranging from floating-point arithmetic and numerical linear algebra to differential equations, optimization, and Monte Carlo methods.

## Contents

### Numerical Python
- NumPy and vectorization
- Broadcasting
- Matrix operations
- Floating-point arithmetic
- Conditioning and numerical error

### Numerical Linear Algebra
- Forward and backward substitution
- Gaussian elimination with pivoting
- LU decomposition
- QR decomposition
- Least-squares problems
- Eigenvalue algorithms
- Condition numbers

### Nonlinear Equations
- Bisection method
- Newton-Raphson method
- Secant method
- Convergence analysis

### Approximation
- Polynomial interpolation
- Lagrange interpolation
- Newton interpolation
- Splines
- Least-squares approximation

### Numerical Calculus
- Forward differences
- Backward differences
- Central differences
- Trapezoidal rule
- Simpson's rule
- Error and convergence analysis

### Ordinary Differential Equations
- Euler method
- Second-order Runge-Kutta (RK2)
- Fourth-order Runge-Kutta (RK4)
- Systems of ODEs
- Stability analysis
- Convergence analysis

### Optimization
- Gradient descent
- Momentum
- Newton optimization
- Backtracking line search
- Convergence analysis

### Stochastic Methods
- Random sampling
- Monte Carlo integration
- Monte Carlo error scaling
- Variance and confidence intervals
- Statistical simulation

## Approach

Most topics follow the same basic workflow:

1. Start from the mathematical formulation.
2. Implement the numerical method directly in Python.
3. Test the implementation on problems with known solutions or theoretical behaviour.
4. Measure error, convergence, stability, or uncertainty where relevant.
5. Include edge cases and failure conditions rather than testing only the ideal case.

The implementations intentionally favour clarity over unnecessary abstraction. NumPy and related libraries are used for numerical primitives, while the algorithms themselves remain explicit.

## Numerical Analysis

A recurring theme throughout the repository is that obtaining a numerical answer is only part of the problem.

The implementations and tests examine properties such as:

- floating-point error and finite precision
- absolute and relative approximation error
- conditioning of numerical problems
- convergence rates and observed convergence order
- stability of ODE solvers
- step-size behaviour
- convergence of optimization algorithms
- Monte Carlo error scaling with $begin:math:text$N\^\{\-1\/2\}$end:math:text$
- sampling variance and standard error
- confidence intervals
- estimator bias, variance, and mean squared error

This makes it possible to compare algorithms based on their numerical behaviour rather than only their final output.

## Testing

Automated tests are written with `pytest` and form an important part of the repository.

Depending on the method, tests verify:

- exact results where analytical solutions are available
- approximation accuracy within numerical tolerances
- theoretical convergence behaviour
- stability properties
- reproducibility of stochastic experiments
- dimensional and parameter validation
- singular, degenerate, and invalid cases

The complete test suite can be run from the repository root:

```bash
pytest -q
```

## Repository Structure

```text
scientific-computing/
├── numerical-python/
├── numerical-linear-algebra/
├── nonlinear-equations/
├── approximation/
├── numerical-calculus/
├── odes/
├── optimization/
└── stochastic-methods/
```

Each section is divided into individual methods or concepts, with implementation and test files kept together.

## Tools

- Python
- NumPy
- SciPy
- Matplotlib
- pytest

## Design Principles

A few principles guided the repository throughout its development:

- Understand the mathematics before abstracting the implementation.
- Prefer readable numerical code over unnecessary complexity.
- Quantify numerical error instead of assuming accuracy.
- Measure convergence rather than relying on visual inspection.
- Treat edge cases and numerical failure modes as part of the algorithm.
- Keep stochastic experiments reproducible.
- Test mathematical behaviour, not just whether code executes.

## Status

**Complete — all 43 planned topics have been implemented and tested.**

The repository now covers the full planned progression from numerical foundations and deterministic algorithms through optimization and stochastic simulation.
