# [probit](http://github.com/bb515/probit_jax)
[![CI]]()
[![Coverage Status]]()
[![Latest Docs]]()
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)]()

GP regression and classification in JAX in not many lines of code.

Contents:

- [Installation](#installation)
- [Usage](#usage)
- probit uses [MLKernels](https://github.com/wesselb/mlkernels) for the GP prior, see the available [means](https://github.com/wesselb/mlkernels#available-means) and [kernels](https://github.com/wesselb/mlkernels#available-kernels) with [compositional design](https://github.com/wesselb/mlkernels#compositional-design).
- [Doesn't haves](#doesnthaves)

TLDR:
```python
>>> from probit.approximators import LaplaceGP as GP
>>> from probit.utilities import log_gaussian_likelihood
>>> from mlkernels import EQ
>>>
>>> def prior(prior_parameters):
>>>     lengthscale, signal_variance = prior_parameters
>>>     # Here you can define the kernel that defines the Gaussian process
>>>     return signal_variance * EQ().stretch(lengthscale).periodic(0.5)
>>>
>>> gaussian_process = GP(data=(X, y), prior=prior, log_likelihood=log_gaussian_likelihood)
>>> likelihood_parameters = 1.0
>>> prior_parameters = (1.0, 1.0)
>>> params = (likelihood_parameters, prior_parameters)
>>> weight, precision = gaussian_process.approximate_posterior(params)
>>> predictive_mean, predictive_variance = gaussian_process.predict(
>>>     X_test,
>>>     params, weight, precision)
```


Get started
-----------

### Building and Installation ###
- The package requires Python 3.9+
- Clone the repository `git clone git@github.com:bb515/probit.git`
- Install using pip `pip install -e .` from the root directory of the repository (see the `setup.py` for the requirements that this command installs).

### Running examples ###

### Regression and hyperparameter optimization
```python
>>> from probit_jax.utilities import log_gaussian_likelihood
>>> from probit_jax.approximators import LaplaceGP as GP
>>> import jax.numpy as jnp
>>> import jax.random as random
>>> from mlkernels import EQ
>>> from varz import Vars, minimise_l_bfgs_b, parametrised, Positive
>>>
>>>
>>> def prior(prior_parameters):
>>>     lengthscale, signal_variance = prior_parameters
>>>     # Here you can define the kernel that defines the Gaussian process
>>>     return signal_variance * EQ().stretch(lengthscale).periodic(0.5)
>>>
>>> # Generate data
>>> noise_std = 0.2
>>> (X, y, X_show, f_show, N_show) = generate_data(noise_std)
>>>
>>> gaussian_process = GP(data=(X, y), prior=prior,
>>>     log_likelihood=log_gaussian_likelihood)
>>> evidence = gaussian_process.objective()
>>>
>>> vs = Vars(jnp.float32)
>>>
>>> def model(vs):
>>>     p = vs.struct
>>>     return (p.lengthscale.positive(), p.signal_variance.positive()), (p.noise_variance.positive(),)
>>>
>>> def objective(vs):
>>>     return evidence(model(vs))
>>>
>>> # Initiate parameters
>>> parameters = model(vs)
>>>
>>> # Approximate posterior
>>> weight, precision = gaussian_process.approximate_posterior(parameters)
>>> mean, variance = gaussian_process.predict(
>>>     X_show,
>>>     parameters,
>>>     weight, precision)
>>>     noise_variance = parameters[1][0]**2
>>> obs_variance = variance + noise_variance
>>> plot((X, y), (X_show, f_show), mean, obs_variance, fname="readme_simple_regression_before.png")
>>> print("Before optimization, evidence={},\nparams={}".format(objective(vs), parameters))
```
![Prediction](https://raw.githubusercontent.com/bb515/probit_jax/master/readme_regression.png)

```python
>>> # Optimize
>>> minimise_l_bfgs_b(objective, vs)
>>> parameters = model(vs)
>>> print("After optimization, evidence={},\nparams={}".format(objective(vs), model(vs)))
>>>
>>> # Approximate posterior
>>> weight, precision = gaussian_process.approximate_posterior(parameters)
>>> mean, variance = gaussian_process.predict(
>>>     X_show,
>>>     parameters,
>>>     weight, precision)
>>> noise_variance = parameters[1][0]**2
>>> obs_variance = variance + noise_variance
>>> variance = variance + noise_std**2
>>> plot((X, y), (X_show, f_show), mean, variance, fname="readme_simple_regression_after.png")
```

### Ordinal regression and hyperparameter optimization

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).

Doesn't haves
-------------
- [Variational Gaussian Process](https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/theory/vgp_notes.html) or [Sparse Variational Gaussian Process](https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/theory/SGPR_notes.html).

References
----------

Most of the algorithms in this package were ported from pre-existing code. In particular, the code was ported from the following papers and repositories

Laplace approximation http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html
@article{Chu2005,
author = {Chu, Wei and Ghahramani, Zoubin},
year = {2005},
month = {07},
pages = {1019-1041},
title = {Gaussian Processes for Ordinal Regression.},
volume = {6},
journal = {Journal of Machine Learning Research},
howpublished = {\url{http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html}},
}

Variational inference via factorizing assumption and free form minimization
@article{Girolami2005,
  author="M. Girolami and S. Rogers",
  journal="Neural Computation", 
  title="Variational Bayesian Multinomial Probit Regression with Gaussian Process Priors", 
  year="2006",
  volume="18",
  number="8",
  pages="1790-1817"
 }
 and
 @Misc{King2005,
  title = 	 {Variational Inference in <span>G</span>aussian Processes via Probabilistic Point Assimilation},
  author = 	 {King, Nathaniel J. and Lawrence, Neil D.},
  year = 	 {2005},
  number = {CS-05-06},
  url = 	 {http://inverseprobability.com/publications/king-ppa05.html},
  abstract = 	 {We introduce a novel variational approach for approximate inference in Gaussian process (GP) models. The key advantages of our approach are the ease with which different noise models can be incorporated and improved speed of convergence. We refer to the algorithm as probabilistic point assimilation (PPA). We introduce the algorithm firstly using the ‘weight space’ view and then through its Gaussian process formulation. We illustrate the approach on several benchmark data sets.}
}



