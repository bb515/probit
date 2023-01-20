from abc import ABC, abstractmethod
import pathlib
import jax
import lab.jax as B
import jax.numpy as jnp
from jax import value_and_grad, grad, jit, vmap
from probit_jax.implicit.solvers import (
    fwd_solver, newton_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit_jax.implicit.Laplace import f_LA, objective_LA
from probit_jax.implicit.VB import f_VB, objective_VB


class Approximator(ABC):
    """
    Base class for GP classification approximators.

    This class allows users to define a classification problem,
    get predictions using an approximate Bayesian inference. Here, N is the
    number of training datapoints, D is the data input dimensions, J is the
    number of ordinal classes, N_test is the number of testing datapoints.

    All approximators must define an init method, which may or may not
        inherit Sampler as a parent class using `super()`.
    All approximators that inherit Approximator define a number of methods that
        return the approximate posterior.
    All approximators must define a :meth:`approximate_posterior` that can be
        used to approximate the posterior and get ELBO gradients with respect
        to the hyperparameters.
    All approximators must define a :meth:`_approximate_initiate` that is used
        to initiate approximate.
    All approximators must define a :meth:`predict` can be used to make
        predictions given test data.
    """

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class
        from the string.

        This method should be implemented in every concrete Approximator.
        """

    @abstractmethod
    def __init__(
            self, data, prior, log_likelihood,
            grad_log_likelihood=None, hessian_log_likelihood=None,
            tolerance=1e-5):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

        :arg prior: method, when evaluated prior(*args, **kwargs) returns
            a valid `:class:MLKernels.Kernel` object
            that is the kernel of the Gaussian Process.
        :arg data: The data tuple. (X_train, y_train), where  
            X_train is the (N, D) The data vector and y_train (N, ) is the
            target vector. 
        :type data: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        :arg log_likelihood: method, when evaluated
            log_likelihood(*args, **kwargs) returns the log likelihood. Takes
            in arguments as log_likelihood(f, y, likelihood_parameters)
            where likelihood_parameters are the (trainable) parameters
            of the likelihood, f is a latent variable and y is a datum.
        :arg grad_log_likelihood: Optional argument supplying the
            (scalar) gradient of the log_likelihood wrt to its first argument,
            the latent variables, f.
        :arg hessian_log_likelihood: Optional argument supplying the
            (scalar) second derivative of the log_likelihood wrt to its first
            argument, the latent variables, f.

        :returns: A :class:`Approximator` object
        """
        self.tolerance = tolerance  # tolerance for the solvers
        self.prior = prior
        if grad_log_likelihood is None:  # Try JAX grad
            grad_log_likelihood = grad(log_likelihood)
        if hessian_log_likelihood is None:  # Try JAX grad
            hessian_log_likelihood = grad(
                lambda f, y, x: grad(log_likelihood)(f, y, x))
        self.log_likelihood = jit(vmap(
            log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
        self.grad_log_likelihood = jit(vmap(
            grad_log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
        self.hessian_log_likelihood = jit(vmap(
            hessian_log_likelihood, in_axes=(0, 0, None), out_axes=(0)))

        # Get data and calculate the prior
        X_train, _ = data
        (self.N, self.D) = jnp.shape(X_train)
        self.data = data
        # Set up a JAX-transformable function for a custom VJP rule definition
        fixed_point_layer.defvjp(
            fixed_point_layer_fwd, fixed_point_layer_bwd)

    @abstractmethod
    def construct(self):
        """
        The parameterized function, which takes in parameters, that is the
        function in fixed point iteration.

        This method should be implemented in every concrete Approximator.
        """

    @abstractmethod
    def objective(self):
        """
        The parameterized function, which takes in parameters, that is the
        objective function, such as a model evidence or lower bound thereof.

        This method should be implemented in every concrete Approximator.
        """

    def value_and_grad(self):
        """Value and grad of the objective at the fix point."""
        return jit(value_and_grad(self.objective()))

    @abstractmethod
    def weight(self):
        """
        The weight, that is part of the solution of GP regression.

        This method should be implemented in every concrete Approximator.
        """

    @abstractmethod
    def precision(self):
        """
        The precision, that is part of the solution of GP regression.

        This method should be implemented in every concrete Approximator.
        """

    def predict(
        self,
        X_test,
        parameters,
        weight, precision):
        """
        Make posterior prediction over ordinal classes of X_test.

        :arg X_test: The new data points, array like (N_test, D).
        :arg weight: The solution of GP regression.
        :type weight: Array like (N,)`
        :arg precision: Solution of GP regression.
            Precisions used in calculation of posterior
            predictions. Array like (N,).
        :type precision: :class:`numpy.ndarray`
        :return: Gaussian process predictive mean and variance array.
        :rtype tuple: ((N_test,), (N_test,))
        """
        kernel = self.prior(parameters[0])
        Kss = B.flatten(kernel.elwise(X_test, X_test))
        Kfs = kernel(self.data[0], X_test)
        Kff = kernel(self.data[0])
        K = Kff + B.diag(1. / precision)
        predictive_posterior_variance = Kss - B.einsum(
            'ij, ij -> j', B.dense(Kfs), B.solve(K, Kfs))
        predictive_posterior_mean = B.flatten(Kfs.T @ weight)
        return predictive_posterior_mean, predictive_posterior_variance

    def posterior_mean(self, weight):
        K = B.dense(self.prior(parameters[0])(self.data[0]))
        return K @ weight

    def approximate_posterior(self, parameters):
        w = self.weight(parameters)
        p, _ = self.precision(w, parameters)
        return w, p


class LaplaceGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Laplace
    approximation.

    Inherits the Approximator ABC.

    This class allows users to define a classification problem and get
    predictions using approximate Bayesian inference. It is for ordinal
    likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian
    Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class
        from the string.
        """
        return "LaplaceGP"

    def __init__(
            self, *args, **kwargs):
        """
        Create an :class:`LaplaceGP` Approximator object.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)

    def construct(self):
        """Fixed point iteration function"""
        return lambda parameters, weight: f_LA(
            prior_parameters=parameters[0],
            likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            weight=weight, data=self.data)

    def weight(self, parameters):
        """
        :returns: A JAX array.
        """
        f = self.construct()
        return newton_solver(lambda z: f(parameters, z),
            jnp.zeros(self.N), self.tolerance)

    def precision(self, weight, parameters):
        """
        :returns: A JAX array.
        """
        K = B.dense(self.prior(parameters[0])(self.data[0]))
        posterior_mean = K @ weight
        precision = -self.hessian_log_likelihood(
            posterior_mean, self.data[1], parameters[1])
        return precision, posterior_mean

    def objective(self):
        return lambda parameters: objective_LA(
                parameters[0], parameters[1],
                self.prior,
                self.log_likelihood,
                self.hessian_log_likelihood,
                fixed_point_layer(jnp.zeros(self.N), self.tolerance,
                    newton_solver, self.construct(), parameters),
                self.data)
    

class VBGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Variational Bayes (VB)
    approximation.
 
    Inherits the Approximator ABC. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference. It is for the ordinal likelihood. For this a
    :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class
        from the string.
        """
        return "VBGP"

    def __init__(
            self, *args, **kwargs):
        """
        Create an :class:`VBGP` Approximator object.

        :returns: A :class:`VBGP` object.
        """
        super().__init__(*args, **kwargs)

    def construct(self):
        """Fixed point iteration function"""
        return lambda parameters, weight: f_VB(
            prior_parameters=parameters[0],
            likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            weight=weight, data=self.data)

    def weight(self, parameters):
        """
        :returns: A JAX array.
        """
        f = self.construct()
        return fwd_solver(lambda z: f(parameters, z),
            jnp.zeros(self.N), self.tolerance)

    def precision(self, weight, parameters):
        """
        :returns: A JAX array.
        """
        K = B.dense(self.prior(parameters[0])(self.data[0]))
        posterior_mean = K @ weight
        return 1./ parameters[1][0]**2 * jnp.ones(weight.shape[0]), posterior_mean

    def objective(self):
        return lambda parameters: objective_VB(
            parameters[0], parameters[1],
            self.prior,
            self.log_likelihood,
            fixed_point_layer(jnp.zeros(self.N), self.tolerance,
                fwd_solver, self.construct(), parameters),
            self.data)

