# scientific-computing
Implementations and analysis of numerical methods and scientific computing techniques in Python.

## Overview

This repository documents the implementation and study of numerical methods used in scientific computing.

The focus extends beyond obtaining numerical results to understanding mathematical foundations, approximation error, convergence, numerical stability, and computational trade-offs.

---

## Scope

The repository follows the following structure:

### Numerical Python
- NumPy / vectorization
- Broadcasting
- Matrix operations
- Floating-point arithmetic
- Conditioning and numerical error

### Numerical Linear Algebra
- Forward / backward substitution
- Gaussian elimination with pivoting
- LU decomposition
- QR decomposition
- Least squares
- Eigenvalue algorithms
- Condition numbers

### Nonlinear Equations
- Bisection
- Newton-Raphson
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
- RK2
- RK4
- Systems of ODEs
- Stability
- Convergence

### Optimization
- Gradient descent
- Momentum
- Newton optimization
- Line search
- Convergence

### Stochastic Methods
- Random sampling
- Monte Carlo integration
- Monte Carlo convergence
- Variance and confidence intervals
- Simulation

---

## Objectives

- Implement numerical algorithms from first principles.
- Understand the mathematical principles underlying each method.
- Analyse numerical error, stability, conditioning, and convergence.
- Compare implementations against established numerical libraries.
- Develop well-tested and reproducible computational experiments.

---

## Methodology

Each major numerical method is studied through:

1. Mathematical formulation
2. Implementation from first principles
3. Numerical testing
4. Error and convergence analysis
5. Comparison with reference implementations where appropriate

---

## Tools

- Python
- NumPy
- SciPy
- Matplotlib
- pytest

---

## Principles

- Mathematical understanding before abstraction.
- Correctness before optimisation.
- Numerical accuracy must be quantified.
- Failure cases are part of the analysis.
- Tests should verify both correctness and numerical behaviour.
- Experiments should be reproducible.

---

## Status

This repository is under active development.

Current focus:

**Numerical Python → NumPy / vectorization**