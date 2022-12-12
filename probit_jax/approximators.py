from abc import ABC, abstractmethod
import pathlib
import jax
import lab.jax as B
import jax.numpy as jnp
from jax import grad, jit, vmap
from mlkernels import Delta
from probit_jax.utilities import (
    read_array)
from probit_jax.solvers import (
    fwd_solver, newton_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit_jax.implicit.Laplace import f_LA
from probit_jax.implicit.Laplace import (
    f_LA, objective_LA)
from probit_jax.implicit.VB import (
    f_VB, objective_VB)


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
    All approximators must define a :meth:`_approximate_initiate` that is used to
        initiate approximate.
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
            self, prior, log_likelihood,
            grad_log_likelihood=None, hessian_log_likelihood=None,
            data=None, read_path=None, single_precision=True):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

        :arg prior: method, when evaluated prior(*args, **kwargs)
            returns the MLKernel class that is the kernel of the Gaussian Process.
        :arg log_likelihood: method, when evaluated log_likelihood(*args, **kwargs)
            returns the log likelihood. Takes in arguments as
            log_likelihood(f, y, likelihood_parameters)
            where likelihood_parameters are the (trainable) parameters
            of the likelihood, f is a latent variable and y is a datum.
        :arg grad_log_likelihood: Optional argument supplying the
            (scalar) gradient of the log_likelihood wrt to its first argument,
            the latent variables, f.
        :arg hessian_log_likelihood: Optional argument supplying the
            (scalar) second derivative of the log_likelihood wrt to its first argument,
            the latent variables, f.
        :arg data: The data tuple. (X_train, y_train), where  
            X_train is the (N, D) The data vector and y_train (N, ) is the
            target vector. Default `None`, if `None`, then the data and prior
            are assumed cached in `read_path` and are attempted to be read.
        :type data: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        :arg str read_path: Read path for outputs. If no data is provided,
            then it assumed that this is the path to the data and cached
            prior covariance(s).

        :returns: A :class:`Approximator` object
        """
        self.tolerance = 1e-3  # tolerance for the solvers
        # Read/write
        if read_path is None:
            self.read_path = None
        else:
            self.read_path = pathlib.Path(read_path)

        # prior is a method that takes in prior_parameters and returns an `:class:MLKernels.Kernel` object
        self.prior = prior
        # Maybe just best to let precision of likelihood be preset by the user
        if grad_log_likelihood is None:  # Try JAX grad
            grad_log_likelihood = grad(log_likelihood)
        if hessian_log_likelihood is None:  # Try JAX grad
            hessian_log_likelihood = grad(lambda f, y, x: grad(log_likelihood)(f, y, x))
  
        self.log_likelihood = jit(vmap(log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
        self.grad_log_likelihood = jit(vmap(grad_log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
        self.hessian_log_likelihood = jit(vmap(hessian_log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
        # Get data and calculate the prior
        if data is not None:
            X_train, y_train = data
            if y_train.dtype not in [int, jnp.int32]:
                raise TypeError(
                    "t must contain only integer values (got {})".format(
                        y_train.dtype))
            else:
                y_train = y_train.astype(int)

            self.N = X_train.shape[0]
            # self._update_prior()  # TODO: here would amortize e.g. gram matrix if want to store it
        else:
            # Try read model from file
            try:
                X_train = read_array(self.read_path, "X_train")
                y_train = read_array(self.read_path, "y_train")
                self.N = self.X_train.shape[0]
                # self._load_cached_prior()  # TODO: here would amortize e.g. gram matrix if want to store it
            except KeyError:
                # The array does not exist in the model file
                raise
            except OSError:
                # Model file does not exist
                raise
        self.data = (X_train, y_train)
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
    def take_grad(self):
        """
        The parameterized function, which takes in parameters, that is the
        function in fixed point iteration.

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
        :arg cov: A covariance matrix used in calculation of posterior
            predictions. (\sigma^2I + K)^{-1} Array like (N, N).
        :type cov: :class:`numpy.ndarray`
        :arg weight: The approximate inverse-covariance-posterior-mean.
            .. math::
                \nu = (\mathbf{K} + \sigma^{2}\mathbf{I})^{-1} \mathbf{y}
                = \mathbf{K}^{-1} \mathbf{f}
            Array like (N,).
        :type weight: :class:`numpy.ndarray`
        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: The noise variance.
        :arg bool numerically_stable: Use matmul or triangular solve.
            Default `False`. 
        :return: Gaussian process predictive mean and std array.
        :rtype tuple: ((N_test,), (N_test,))
        """
        kernel = self.prior(parameters[0])
        Kss = B.dense(kernel*Delta()(X_test))
        Kfs = B.dense(kernel(self.data[0], X_test))
        Kff = B.dense(kernel(self.data[0]))
        posterior_variance = Kss - B.einsum(
            'ij, ij -> j', Kfs, B.solve(Kff + B.diag(1. / precision), Kfs))
        posterior_mean = Kfs.T @ weight
        return posterior_mean, posterior_variance


class LaplaceGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Laplace
    approximation.

    Inherits the Approximator ABC.

    Evidence maximization algorithm as written in Appendix A
    Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes for Ordinal
    Regression.. Journal of Machine Learning Research. 6. 1019-1041.

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
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            hessian_log_likelihood=self.hessian_log_likelihood,
            weight=weight, data=self.data)
    
    def approximate_posterior(self, theta):
        weight = fixed_point_layer(jnp.zeros(self.N), self.tolerance, newton_solver, self.construct(), theta),
        K = B.dense(self.prior(theta[0])(self.data[0]))
        posterior_mean = K @ weight
        precision = -self.hessian_log_likelihood(
            posterior_mean, self.data[1], theta[1])
        return weight, precision

    def take_grad(self):
        return jit(jax.value_and_grad(
            lambda theta: objective_LA(
                theta[0], theta[1],
                self.prior,
                self.log_likelihood,
                self.grad_log_likelihood,
                self.hessian_log_likelihood,
                fixed_point_layer(jnp.zeros(self.N), self.tolerance, newton_solver, self.construct(), theta),
                self.data)))
    

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
        Return a string representation of this class, used to import the class from
        the string.
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
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            weight=weight, data=self.data)

    def approximate_posterior(self, theta):
        weight = fixed_point_layer(jnp.zeros(self.N), self.tolerance, newton_solver, self.construct(), theta),
        precision = 1./ theta[1][0]**2
        return weight, precision

    def take_grad(self):
        """Value and grad of the objective at the fix point."""
        return jit(jax.value_and_grad(
            lambda theta: objective_VB(
                theta[0], theta[1],
                self.prior,
                self.log_likelihood,
                self.grad_log_likelihood,
                fixed_point_layer(jnp.zeros(self.N), self.tolerance, newton_solver, self.construct(), theta),
                self.data)))


class InvalidApproximator(Exception):
    """An invalid approximator has been passed to `PseudoMarginal`"""

    def __init__(self, approximator):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`PseudoMarginal` as the approximator
            argument.
        :rtype: :class:`InvalidApproximator`
        """
        message = (
            f"{approximator} is not an instance of "
            "probit.approximators.Approximator"
        )

        super().__init__(message)
