from abc import ABC, abstractmethod
from mlkernels import Delta
import pathlib
import warnings
import lab.jax as B
import jax.numpy as jnp
import jax
from jax import lax
from functools import partial
from probit_jax.utilities import (
    check_cutpoints,
    read_array)
from probit_jax.lab.utilities import (
    predict_reparameterised)

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
            self, prior, parameters,
            data=None, read_path=None,
            theta_hyperparameters=None, cutpoints_hyperparameters=None,
            noise_std_hyperparameters=None, single_precision=True):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
        :arg int J: The number of (ordinal) classes.
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
        self.prior = prior
        # Initiate hyper-hyper-parameters in case of MCMC or Variational
        # inference over theta
        self.initiate_hyperhyperparameters(
            theta_hyperparameters=theta_hyperparameters,
            cutpoints_hyperparameters=cutpoints_hyperparameters,
            noise_std_hyperparameters=noise_std_hyperparameters)
 
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
            self.tolerance = 1e-3  # Single precision
            self.single_precision = single_precision
        else:  # Double precision
            self.epsilon = 1e-12  # Default regularisation- If too small, 1e-10
            self.tolerance = 1e-6
            self.single_precision = False

        # Get data and calculate the prior
        if data is not None:
            X_train, y_train = data
            self.X_train = X_train
            if y_train.dtype not in [int, jnp.int32]:
                raise TypeError(
                    "t must contain only integer values (got {})".format(
                        y_train.dtype))
            else:
                y_train = y_train.astype(int)
                self.y_train = y_train

            self.N = jnp.shape(self.X_train)[0]
            self._update_prior()
        else:
            # Try read model from file
            try:
                self.X_train = read_array(self.read_path, "X_train")
                self.y_train = read_array(self.read_path, "y_train")
                self.N = jnp.shape(self.X_train)[0]
                self._load_cached_prior()
            except KeyError:
                # The array does not exist in the model file
                raise
            except OSError:
                # Model file does not exist
                raise
        self.D = jnp.shape(self.X_train)[1]
        # Set up a JAX-transformable function for a custom VJP rule definition
        self.fixed_point_layer.defvjp(
            self.fixed_point_layer_fwd, self.fixed_point_layer_bwd)

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

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
    def fixed_point_layer(solver, z_init, f, params):
        """
        Following the tutorial, Chapter 2 of
        http://implicit-layers-tutorial.org/implicit_functions/

        A wrapper function for the parameterized fixed point sovler.

        :arg solver: Root finding numerical solver
        :arg params: Parameters for the non-linear set of equations that must
            be satisfied
            .. math::
                z = f(a, z)
            where :math:`a` are the parameters and :math:`z` are the latent
            variables, and :math:`f` is a non-linear function.
        """ 
        z_star = solver(
            lambda z: f(params, z), z_init=z_init)
        return z_star

    def fixed_point_layer_fwd(self, solver, params):
        z_star = self.fixed_point_layer(solver, params)
        return z_star, (params, z_star)

    def fixed_point_layer_bwd(self, solver, res, z_star_bar):
        params, z_star = res
        _, vjp_a = jax.vjp(lambda params: self.f(params, z_star), params)
        _, vjp_z = jax.vjp(lambda z: self.f(params, z), z_star)
        return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar,
                            z_init=jnp.zeros_like(z_star)))

    def fwd_solver(self, f, z_init):
        """
        Using fix point iteration, return the latent variables at the fix
        point.
        """
        def cond_fun(carry):
            z_prev, z = carry
            return (jnp.linalg.norm(z_prev - z) > self.tolerance)  # TODO: This is a very unsafe implementation, can lead to infinite while loops!

        def body_fun(carry):
            _, z = carry
            return z, f(z)

        init_carry = (z_init, f(z_init))
        _, z_star = lax.while_loop(cond_fun, body_fun, init_carry)
        return z_star

    def newton_solver(self, f, z_init):
        """
        Using Newton's method, return the latent variables at the fix point.
        """
        f_root = lambda z: f(z) - z
        g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
        return self.fwd_solver(g, z_init)

    def anderson_solver(
            f, z_init, m=5, lam=1e-4, max_iter=50, tol=1e-5, beta=1.0):
        """
        Using Anderson acceleration, return the latent  variables at the fix
        point.
        """
        x0 = z_init
        x1 = f(x0)
        x2 = f(x1)
        X = jnp.concatenate(
            [jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
        F = jnp.concatenate(
            [jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

        def step(n, k, X, F):
            G = F[:n] - X[:n]
            GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
            H = jnp.block([[jnp.zeros((1, 1)), jnp.ones((1, n))],
                        [ jnp.ones((n, 1)), GTG]]) + lam * jnp.eye(n + 1)
            alpha = jnp.linalg.solve(H, jnp.zeros(n+1).at[0].set(1))[1:]

            xk = beta * jnp.dot(alpha, F[:n])\
                + (1-beta) * jnp.dot(alpha, X[:n])
            X = X.at[k % m].set(xk)
            F = F.at[k % m].set(f(xk))
            return X, F

        # unroll the first m steps
        for k in range(2, m):
            X, F = step(k, k, X, F)
            res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
            if res < tol or k + 1 >= max_iter:
                return X[k], k

        # run the remaining steps in a lax.while_loop
        def body_fun(carry):
            k, X, F = carry
            X, F = step(m, k, X, F)
            return k + 1, X, F

        def cond_fun(carry):
            k, X, F = carry
            kmod = (k - 1) % m
            res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
            return (k < max_iter) & (res >= tol)

        k, X, F = lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
        return X[(k - 1) % m]

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

    def get_phi(self, trainables):
        """
        Get the parameters (phi) for unconstrained optimization.

        :arg trainables: Indicator array of the hyperparameters to optimize over.
            TODO: it is not clear, unless reading the code from this method,
            that trainables[0] means noise_variance, etc. so need to change the
            interface to expect a dictionary with keys the hyperparameter
            names and values a bool that they are fixed?
        :type trainables: :class:`numpy.ndarray`
        :returns: The unconstrained parameters to optimize over, phi.
        :rtype: :class:`numpy.array`
        """
        phi = []
        if trainables[0]:
            phi.append(0.5 * jnp.log(self.noise_variance))
        if trainables[1]:
            phi.append(self.cutpoints[1])
        for j in range(2, self.J):
            if trainables[j]:
                phi.append(jnp.log(self.cutpoints[j] - self.cutpoints[j - 1]))
        if trainables[self.J]:
            phi.append(0.5 * jnp.log(self.kernel.variance))
        if self.kernel._ARD:
            for d in range(self.D):
                if trainables[self.J + 1][d]:
                    phi.append(jnp.log(self.kernel.theta[d]))
        else:
            if trainables[self.J + 1]:
                phi.append(jnp.log(self.kernel.theta))
        return jnp.array(phi)

    def _hyperparameters_update(
        self, cutpoints=None, theta=None, variance=None, noise_variance=None):
        """
        Reset kernel hyperparameters, generating new prior covariances.
 
        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg theta: The kernel hyper-parameters.
        :type theta: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg noise variance:
        :type noise variance:
        """
        if cutpoints is not None:
            self.cutpoints_ts = self.cutpoints[self.y_train]
            self.cutpoints_tplus1s = self.cutpoints[self.y_train + 1]
        if theta is not None or variance is not None:
            self.kernel.update_hyperparameter(
                theta=theta, variance=variance)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
            warnings.warn("Done updating prior covariance")
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = jnp.sqrt(noise_variance)

    def initiate_hyperhyperparameters(self,
            variance_hyperparameters=None,
            theta_hyperparameters=None,
            cutpoints_hyperparameters=None, noise_std_hyperparameters=None):
        """TODO: For MCMC over these parameters. Could it be a part
        of sampler?"""
        if variance_hyperparameters is not None:
            self.variance_hyperparameters = variance_hyperparameters
        else:
            self.variance_hyperparameters = None
        if theta_hyperparameters is not None:
            self.theta_hyperparameters = theta_hyperparameters
        else:
            self.theta_hyperparameters = None
        if cutpoints_hyperparameters is not None:
            self.cutpoints_hyperparameters = cutpoints_hyperparameters
        else:
            self.cutpoints_hyperparameters = None
        if noise_std_hyperparameters is not None:
            self.noise_std_hyperparameters = noise_std_hyperparameters
        else:
            self.noise_std_hyperparameters = None

    def hyperparameters_update(
        self, cutpoints=None, theta=None, variance=None, noise_variance=None):
        """
        Wrapper function for :meth:`_hyperparameters_update`.
        """
        return self._hyperparameters_update(
            cutpoints=cutpoints, theta=theta, variance=variance,
            noise_variance=noise_variance)

    def _load_cached_prior(self):
        """
        Load cached prior covariances.
        """
        self.K = read_array(self.read_path, "K")
        self.partial_K_theta = read_array(
            self.read_path, "partial_K_theta")
        self.partial_K_variance = read_array(
            self.read_path, "partial_K_variance")

    def _update_prior(self):
        """Update prior covariances."""
        warnings.warn("Updating prior covariance.")
        # self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        # self.partial_K_theta = self.kernel.kernel_partial_derivative_theta(
        #     self.X_train, self.X_train)
        # self.partial_K_variance = self.kernel.kernel_partial_derivative_variance(
        #     self.X_train, self.X_train)
        warnings.warn("Done updating prior covariance.")
        # TODO: When it is not necessary to calculate the partial derivatives - when no gradient eval is required.
        # if phi is not None:
        #     # If the unconstrained optimization input (phi) is defined then
        #     # we need to calculate some derivatives of the Gram matrix
        #     # with respect to the hyperparameters.
        #     # This can be done via automatic differentiation, here
        #     # or by a manual function. I have chosen to evaluate manually.


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

    # def construct(self):
    #     """Fixed point iteration function"""
    #     return lambda params, posterior_mean: f_LA(
    #         posterior_mean, self.model(params), self.single_precision)

    def construct(self):
        """Fixed point iteration function"""
        return lambda parameters, posterior_mean: f_LA(
            prior_parameters=parameters[0], likelihood_parameters=parameters[1],
            prior=self.prior, grad_likelihood=self.grad_likelihood,
            posterior_mean=posterior_mean, data=self.data
        )

    def take_grad(self):
        f = self.construct()
        # AD for fixed point solvers for \theta
        self.fixed_point_layer.defvjp(self.fixed_point_layer_fwd, self.fixed_point_layer_bwd)
        z_star = self.fixed_point_layer(self.newton_solver, jnp.zeros(self.N), f,
            (self.prior_parameters, self.likelihood_parameters))
        print(z_star)
        assert 0
        return jax.value_and_grad(
            lambda theta: objective_LA_auto(
                self.prior_parameters, self.likelihood_parameters,
                self.prior, self.grad_likelihood,
                self.hessian_likelihood, z_star, self.data))
        # return jax.value_and_grad(
        #     lambda theta: objective_LA(
        #         z_star,
        #         # self.fixed_point_layer(self.newton_solver, jnp.zeros(self.N), f, theta),
        #         self.noise_std, self.cutpoints_ts, self.cutpoints_tplus1s,
        #         B.dense(self.kernel.stretch(theta)(self.X_train)), self.upper_bound))

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
        return lambda params, posterior_mean: f_VB(
            posterior_mean, self.model(params), (self.X_train, self.y_train),
            self.N, self.upper_bound)

    def take_grad(self):
        f = self.construct()
        # AD for fixed point solvers for \theta
        self.fixed_point_layer.defvjp(self.fixed_point_layer_fwd, self.fixed_point_layer_bwd)
        z_star = self.fixed_point_layer(self.fwd_solver, jnp.zeros(self.N), f, self.parameters)
        print(z_star)
        return jax.value_and_grad(lambda params:
            objective_VB(
                self.fixed_point_layer(self.fwd_solver, jnp.zeros(self.N), f, self.parameters),
                # z_star,  # This cannot work, since it cannot be a constant
                # B.dense(self.fixed_point_layer(self.fwd_solver, jnp.zeros(self.N), f, theta)),
                self.model(params),
                (self.X_train, self.y_train), self.N, self.upper_bound))

    def hyperparameters_update(
        self, cutpoints=None, theta=None, variance=None, noise_variance=None,
        theta_hyperparameters=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since theta can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg theta: The kernel hyper-parameters.
        :type theta: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg float noise_variance: The noise variance.
        :type noise_variance:
        :arg theta_hyperparameters:
        :type theta_hyperparameters:
        """
        self._hyperparameters_update(
            cutpoints=cutpoints, theta=theta,
            variance=variance, noise_variance=noise_variance)
        if theta_hyperparameters is not None:
            self.kernel.update_hyperparameter(
                theta_hyperparameters=theta_hyperparameters)
        # Update posterior covariance
        warnings.warn("Updating posterior covariance.")
        # (self.L_cov, self.cov, self.log_det_cov, self.trace_cov,
        # self.trace_posterior_cov_div_var) = update_posterior_covariance_VB(
        #     self.noise_variance, self.N, self.K)
        warnings.warn("Done updating posterior covariance.")


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


