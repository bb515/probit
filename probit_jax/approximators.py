from abc import ABC, abstractmethod
from mlkernels import Delta
import pathlib
import warnings
import lab.jax as B
import jax.numpy as jnp
import jax
from jax import lax, grad, jit, vmap
from probit_jax.utilities import (
    check_cutpoints,
    read_array)
from probit_jax.lab.utilities import (
    predict_reparameterised)
from probit_jax.solvers import (
    fwd_solver, newton_solver, anderson_solver, jax_opt_solver,
    fixed_point_layer, fixed_point_layer_fwd, fixed_point_layer_bwd)
from probit_jax.implicit.Laplace import f_LA
# Change probit_jax.<linalg backend>.<Approximator>, as appropriate
from probit_jax.implicit.Laplace import (
    f_LA, jacobian_LA, objective_LA)
from probit_jax.implicit.VB import (
    f_VB, jacobian_VB, objective_VB)


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
        # Read/write
        if read_path is None:
            self.read_path = None
        else:
            self.read_path = pathlib.Path(read_path)

        self.max_steps = 100
        if single_precision is True:
            # Numerical stability when taking Cholesky decomposition
            # See GPML by Williams et al. for an explanation of jitter
            self.epsilon = 1e-8  # Strong regularisation
            # Decreasing tolerance will lead to more accurate solutions up to a
            # point but a longer convergence time. Acts as a machine tolerance.
            # Single precision linear algebra libraries won't converge smaller than
            # tolerance = 1e-3. Probably don't put much smaller than 1e-6.
            self.tolerance = 1e-6
            # self.tolerance = 1e-2  # Single precision
            self.single_precision = single_precision
        else:  # Double precision
            self.epsilon = 1e-12  # Default regularisation- If too small, 1e-10
            self.tolerance = 1e-6
            self.single_precision = False

        # prior is a method that takes in prior_parameters and returns an `:class:MLKernels.Kernel` object
        self.prior = prior
        # Maybe just best to let precision of likelihood be preset by the user
        if grad_log_likelihood is None:  # Try JAX grad
            grad_log_likelihood = grad(log_likelihood)
        if hessian_log_likelihood is None:  # Try JAX grad
            hessian_log_likelihood = grad(lambda f, y, x: grad(log_likelihood)(f, y, x))
            print("hessian_log_likelihood shape:", type(hessian_log_likelihood))
        self.log_likelihood= jit(vmap(log_likelihood, in_axes=(0, 0, None), out_axes=(0)))
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

            self.N = jnp.shape(X_train)[0]
            # self._update_prior()  # TODO: here would amortize e.g. gram matrix if want to store it
        else:
            # Try read model from file
            try:
                X_train = read_array(self.read_path, "X_train")
                y_train = read_array(self.read_path, "y_train")
                self.N = jnp.shape(self.X_train)[0]
                # self._load_cached_prior()  # TODO: here would amortize e.g. gram matrix if want to store it
            except KeyError:
                # The array does not exist in the model file
                raise
            except OSError:
                # Model file does not exist
                raise
        self.data = (X_train, y_train)
        self.D = jnp.shape(X_train)[1]
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

    def approximate_posterior(self, theta):
        return fixed_point_layer(jnp.zeros(self.N), self.tolerance, fwd_solver, self.construct(), theta),

    def predict(
            self, X_test, cov, f, reparameterised=True, whitened=False):
        """
        Return the posterior predictive distribution over classes.

        :arg X_test: The new data points, array like (N_test, D).
        :type X_test: :class:`numpy.ndarray`.
        :arg cov: The approximate
            covariance-posterior-inverse-covariance matrix. Array like (N, N).
        :type cov: :class:`numpy.ndarray`.
        :arg f: Array like (N,).
        :type f: :class:`numpy.ndarray`.
        :arg bool reparametrised: Boolean variable that is `True` if f is
            reparameterised, and `False` if not.
        :arg bool whitened: Boolean variable that is `True` if f is whitened,
            and `False` if not.
        :return: The ordinal class probabilities.
        """
        if whitened is True:
            raise NotImplementedError("Not implemented.")
        elif reparameterised is True:
            print(predict_reparameterised(
                self.kernel*Delta()(X_test),
                self.kernel(self.X_train, X_test),
                cov, weight=f,
                cutpoints=self.cutpoints,
                noise_variance=self.noise_variance,
                single_precision=self.single_precision))
            # TODO: make as a function of the likelihood
            return predict_reparameterised(
                self.kernel*Delta()(X_test),
                self.kernel(self.X_train, X_test),
                cov, weight=f,
                cutpoints=self.cutpoints,
                noise_variance=self.noise_variance,
                single_precision=self.single_precision)
        else:
            raise NotImplementedError("Not implemented.")


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
        return lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            hessian_log_likelihood=self.hessian_log_likelihood,
            posterior_mean=posterior_mean, data=self.data)
    
    def get_latents(self, params):
        return fixed_point_layer(z_init=jnp.zeros(self.N), tolerance = self.tolerance, 
        solver = jax_opt_solver, f = self.construct(), params = params)

    def take_grad(self):
        return jax.value_and_grad(
            lambda theta: objective_LA(
                theta[0], theta[1],
                self.prior,
                self.log_likelihood,
                self.grad_log_likelihood,
                self.hessian_log_likelihood,
                fixed_point_layer(z_init=jnp.zeros(self.N), tolerance = self.tolerance, 
                solver = jax_opt_solver, f = self.construct(), params = theta),
                self.data))
    

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
        return lambda parameters, posterior_mean: f_VB(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior, grad_log_likelihood=self.grad_log_likelihood,
            posterior_mean=posterior_mean, data=self.data)

    def get_latents(self, params):
        return fixed_point_layer(z_init=jnp.zeros(self.N), tolerance = self.tolerance, 
        solver = jax_opt_solver, f = self.construct(), params = params)
    
    def take_grad(self):
        """Value and grad of the objective at the fix point."""
        return jit(jax.value_and_grad(
            lambda theta: objective_VB(
                theta[0], theta[1],
                self.prior,
                self.log_likelihood,
                self.grad_log_likelihood,
                fixed_point_layer(z_init=jnp.zeros(self.N), tolerance = self.tolerance, 
                solver = jax_opt_solver, f = self.construct(), params = theta),
                self.data)), static_argnames=[''])


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

