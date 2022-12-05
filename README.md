probit
======

Please do not distribute this private repository. This is WIP until release - so some functionality may be temporarily broken and poorly documented.

probit, is an open-source python package for solving ordinal regression problems with Gaussian Processes. 

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


Features
--------
- Supports MLKernels priors
- Approximate inference with mean field Variational Bayes, Laplace approximation 
- Type II maximum likelihood via manual and numerical gradients. A first version of autodiff is available on branch feature/implicit_layer

Planned Features
--------
- Make autodiff capabilities numerically stable
- Complete documentation
- Make faster than manual differentiation package, `probit`
- Python Package index release

Get started
-----------

### Building and Installation ###
- The package requires Python 3.9+
- TODO: include any autodiff and linear algebra backend dependencies

### Running examples ###

### Running the tests ###

The tests for this project use [pytest](https://pytest.org/en/latest/).
