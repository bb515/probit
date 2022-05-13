probit
======

Please do not distribute this private repository.

probit, by Ben Boys (2021), is a lightweight, open-source and high-performance python package for solving ordinal regression problems with Gaussian Processes. It is implemented in [Python](https://www.python.org/) and the performance critical parts will be implemented in [Cython](https://cython.org/). The plan is that the numpy code will be converted to JAX code.

probit allows users to write their code in pure Python. Simulations are then executed seamlessly using high performance and numerically stable code.

Features
--------
- Exact inference with MCMC (Gibbs sampler or Elliptical slice sampler)
- Approximate inference with Variational Bayes, Laplace approximation or Expectation Propagation
- Sparse approximate inference with the Nystr\"{o}m approximation applied to Variational Bayes and Laplace approximation.
- Sparse approximate inference with sparse Expectation Propagation.
- Gaussian, ARD, linear and polynomial kernel functions
- Fully Bayesian pseudo-marginal inference

Planned Features
--------
- TensorFlow backend


Get started (preferred)
-----------------------

### Building and Installation ###

- The package requires Python 3.7+

### Running examples ###

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).

Get started from the GitHub repository (for developers)
-------------------------------------------------------

### Building and Installation ###

### Running examples ###

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).
