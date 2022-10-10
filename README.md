probit
======

Please do not distribute this private repository.

probit, is an open-source and high-performance python package for solving ordinal regression problems with Gaussian Processes. It is implemented in [Python](https://www.python.org/).

References
----------

Most of the algorithms in this package were ported from pre-existing code. In particular, the code was ported from the following papers and repositories

Laplace approximation, Expectation Propagation http://www.gatsby.ucl.ac.uk/~chuwei/ordinalregression.html
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

PowerEP https://github.com/thangbui/sparseGP_powerEP
@article{Bui2017,
    author = {Bui, T.D. and Yan, Josiah and Turner, Richard},
    year = {2017},
    month = {10},
    pages = {1-72},
    title = {A Unifying Framework for Gaussian Process Pseudo-Point Approximations using Power Expectation Propagation},
    volume = {18},
    journal = {Journal of Machine Learning Research}
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
  number =       {CS-05-06},
  url = 	 {http://inverseprobability.com/publications/king-ppa05.html},
  abstract = 	 {We introduce a novel variational approach for approximate inference in Gaussian process (GP) models. The key advantages of our approach are the ease with which different noise models can be incorporated and improved speed of convergence. We refer to the algorithm as probabilistic point assimilation (PPA). We introduce the algorithm firstly using the ‘weight space’ view and then through its Gaussian process formulation. We illustrate the approach on several benchmark data sets.}
}


Pseudo-marginal Bayesian inference https://www.eurecom.fr/~filippon/pages/publications.html
@article{Filippone2014,
author = {Filippone, Maurizio and Girolami, Mark},
year = {2014},
month = {11},
pages = {},
title = {Pseudo-Marginal Bayesian Inference for Gaussian Processes},
journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
doi = {10.1109/TPAMI.2014.2316530}
}


probit allows users to write their code in pure Python. Simulations are then executed seamlessly using high performance and numerically stable code.

Features
--------
- Exact inference with MCMC (Gibbs sampler or Elliptical slice sampler)
- Approximate inference with mean field Variational Bayes, Laplace approximation, Expectation Propagation or Power Expectation Propagation
- Sparse approximate inference applied to Power Expectation Propagation and mean field Variational Bayes
- Gaussian, ARD, linear and polynomial kernel functions
- Fully Bayesian inference with pseudo-marginal approach, auxiliary augmentation or sufficient augmentation.

Planned Features
--------
- Make numerically stable custom derivatives
- Make faster than manual differentiation package, `probit`
- Documentation
- Python Package index release

Get started (preferred)
-----------------------

### Building and Installation ###

- The package requires Python 3.7+
- TODO: include any autodiff and linear algebra backend dependencies

### Running examples ###

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).

Get started from the GitHub repository (for developers)
-------------------------------------------------------

### Building and Installation ###

### Running examples ###

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).
