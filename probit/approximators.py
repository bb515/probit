from abc import ABC, abstractmethod
import pathlib
from tqdm import trange
import warnings
import numpy as np
from probit.numpy.utilities import (
    check_cutpoints,
    read_array,
    posterior_covariance,
    norm_cdf,
    truncated_norm_normalising_constant)

# Change probit.<linalg backend>.<Approximator>, as appropriate
from probit.jax.Laplace import (update_posterior_LA,
    compute_weights_LA, objective_LA, objective_gradient_LA)
from probit.numpy.VB import (update_posterior_mean_VB,
    update_posterior_covariance_VB, update_hyperparameter_posterior_VB,
    objective_VB, objective_gradient_VB)
from probit.numpy.EP import (update_posterior_EP,
    objective_EP, objective_gradient_EP,
    compute_weights_EP, compute_integrals_vector_EP)
from probit.numpy.PEP import (update_posterior_parallel_PEP,
    update_posterior_sequential_PEP, objective_PEP, objective_gradient_PEP)


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
            self, kernel, J, data=None, read_path=None,
            theta_hyperparameters=None, cutpoints_hyperparameters=None,
            noise_std_hyperparameters=None):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

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
        # Initiate hyper-hyper-parameters in case of MCMC or Variational
        # inference over theta
        self.initiate_hyperhyperparameters(
            theta_hyperparameters=theta_hyperparameters,
            cutpoints_hyperparameters=cutpoints_hyperparameters,
            noise_std_hyperparameters=noise_std_hyperparameters)

        # if not (isinstance(kernel, Kernel)):
        #     raise InvalidKernel(kernel)
        # else:
        self.kernel = kernel
        self.J = J

        # Read/write
        if read_path is None:
            self.read_path = None
        else:
            self.read_path = pathlib.Path(read_path)

        # Numerical stability when taking Cholesky decomposition
        # See GPML by Williams et al. for an explanation of jitter
        self.epsilon = 1e-12  # Default regularisation TODO: may be too small, try 1e-10
        # self.epsilon = 1e-8  # Strong regularisation

        # Decreasing tolerance will lead to more accurate solutions up to a
        # point but a longer convergence time. Acts as a machine tolerance.
        # Single precision linear algebra libraries won't converge smaller than
        # tolerance = 1e-3. Probably don't put much smaller than 1e-6.
        self.tolerance = 1e-3  # Single precision
        # self.tolerance = 1e-6  # Double precision
        self.tolerance2 = self.tolerance**2

        # Threshold of single sided standard deviations that
        # normal cdf can be approximated to 0 or 1
        # More than this + redundancy leads to numerical instability
        # due to catestrophic cancellation
        # Less than this leads to a poor approximation due to series
        # expansion at infinity truncation
        # Good values found between 4 and 6
        # self.upper_bound = 4  # For single precision
        self.upper_bound = 6  # For double precision

        # More than this + redundancy leads to numerical
        # instability due to overflow
        # Less than this results in poor approximation due to
        # neglected probability mass in the tails
        # Good values found between 18 and 30
        # Try decreasing if experiencing infs or NaNs
        # self.upper_bound = 18  # For single precision
        self.upper_bound2 = 30  # For double precision

        # Get data and calculate the prior
        if data is not None:
            X_train, y_train = data
            self.X_train = X_train
            if y_train.dtype not in [int, np.int32]:
                raise TypeError(
                    "t must contain only integer values (got {})".format(
                        y_train.dtype))
            else:
                y_train = y_train.astype(int)
                self.y_train = y_train
            self.N = np.shape(self.X_train)[0]
            self._update_prior()
        else:
            # Try read model from file
            try:
                self.X_train = read_array(self.read_path, "X_train")
                self.y_train = read_array(self.read_path, "y_train")
                self.N = np.shape(self.X_train)[0]
                self._load_cached_prior()
            except KeyError:
                # The array does not exist in the model file
                raise
            except OSError:
                # Model file does not exist
                raise
        self.D = np.shape(self.X_train)[1]
        self.grid = np.ogrid[0:self.N]  # For indexing sets of self.y_train
        self.indices_where_0 = np.where(self.y_train == 0)
        self.indices_where_J_1 = np.where(self.y_train == self.J - 1)

    @abstractmethod
    def approximate_posterior(self):
        """
        Return the lower bound on the marginal likelihood and its gradients
        with respect to the hyperparameters and, optionally, the posterior mean
        and covariance (or some parameterisation thereof)

        This method should be implemented in every concrete Approximator.
        """

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
            return self._predict(
                X_test, cov, weight=f,
                cutpoints=self.cutpoints,
                noise_variance=self.noise_variance)
        else:
            raise NotImplementedError("Not implemented.")

    def _ordinal_predictive_distributions(
        self, posterior_pred_mean, posterior_pred_std, N_test, cutpoints
    ):
        """
        TODO: Replace with truncated_norm_normalizing_constant
        Return predictive distributions for the ordinal likelihood.
        """
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            z1 = np.divide(np.subtract(
                cutpoints[j + 1], posterior_pred_mean), posterior_pred_std)
            z2 = np.divide(
                np.subtract(cutpoints[j],
                posterior_pred_mean), posterior_pred_std)
            predictive_distributions[:, j] = norm_cdf(z1) - norm_cdf(z2)
        return predictive_distributions

    def _predict(
            self, X_test, cov, weight, cutpoints, noise_variance):
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
        :return: A Monte Carlo estimate of the class probabilities.
        :rtype tuple: ((N_test, J), (N_test,), (N_test,))
        """
        N_test = np.shape(X_test)[0]
        Kss = self.kernel.kernel_prior_diagonal(X_test)
        Kfs = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        temp = cov @ Kfs
        posterior_variance = Kss - np.einsum(
            'ij, ij -> j', Kfs, temp)
        posterior_std = np.sqrt(posterior_variance)
        posterior_pred_mean = Kfs.T @ weight
        posterior_pred_variance = posterior_variance + noise_variance
        posterior_pred_std = np.sqrt(posterior_pred_variance)
        return (
            self._ordinal_predictive_distributions(
            posterior_pred_mean, posterior_pred_std, N_test, cutpoints),
            posterior_pred_mean, posterior_std)

    def get_log_likelihood(self, m):
        """
        Likelihood of ordinal regression. This is product of scalar normal cdf.

        If np.ndim(m) == 2, vectorised so that it returns (num_samples,)
        vector from (num_samples, N) samples of the posterior mean.

        Note that numerical stability has been turned off in favour of
        exactness - but experiments should be run twice with numerical
        stability turned on to see if it makes a difference.
        """
        Z, *_ = truncated_norm_normalising_constant(
            self.cutpoints_ts, self.cutpoints_tplus1s,
            self.noise_std, m,
            upper_bound=self.upper_bound,
            # upper_bound2=self.upper_bound2,  # optional
            # tolerance=self.tolerance  # optional
            )
        if np.ndim(m) == 2:
            return np.sum(np.log(Z), axis=1)  # (num_samples,)
        elif np.ndim(m) == 1:
               return np.sum(np.log(Z))  # (1,)

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
            phi.append(0.5 * np.log(self.noise_variance))
        if trainables[1]:
            phi.append(self.cutpoints[1])
        for j in range(2, self.J):
            if trainables[j]:
                phi.append(np.log(self.cutpoints[j] - self.cutpoints[j - 1]))
        if trainables[self.J]:
            phi.append(0.5 * np.log(self.kernel.variance))
        if self.kernel._ARD:
            for d in range(self.D):
                if trainables[self.J + 1][d]:
                    phi.append(np.log(self.kernel.theta[d]))
        else:
            if trainables[self.J + 1]:
                phi.append(np.log(self.kernel.theta))
        return np.array(phi)

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
            self.cutpoints = check_cutpoints(cutpoints, self.J)
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
            self.noise_std = np.sqrt(noise_variance)

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

    def _hyperparameter_training_step_initialise(
            self, phi, trainables, verbose=False):
        """
        TODO: this doesn't look correct, for example if only training a subset
        Initialise the hyperparameter training step.

        :arg phi: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}}
                \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\theta}],

            where :math:`\sigma` is the noise standard deviation,
            :math:`\b_{1}` is the first cutpoint, :math:`\Delta_{l}` is the
            :math:`l`th cutpoint interval, :math:`\theta` is the single
            shared lengthscale parameter or vector of parameters in which
            there are in the most general case J * D parameters.
            If `None` then no hyperperameter update is performed.
        :type phi: :class:`numpy.ndarray`
        :return: (intervals, steps, error, iteration, gx_where, gx)
        :rtype: (6,) tuple
        """
        # Initiate at None since those that are None do not get updated        
        noise_variance = None
        cutpoints = None
        variance = None
        theta = None
        index = 0
        if trainables is not None:
            if trainables[0]:
                noise_std = np.exp(phi[index])
                noise_variance = noise_std**2
                if verbose:
                    if noise_variance < 1.0e-04:
                        warnings.warn(
                            "WARNING: noise variance is very low - numerical"
                            " stability issues may arise "
                            "(noise_variance={}).".format(noise_variance))
                    elif noise_variance > 1.0e3:
                        warnings.warn(
                            "WARNING: noise variance is very large - numerical"
                            " stability issues may arise "
                        "(noise_variance={}).".format(noise_variance))
                index += 1
            if np.any(trainables[1:self.J]):
                # Get cutpoints from classifier
                if cutpoints is None: cutpoints = self.cutpoints
                # Then need to get all values of phi
                phi_cutpoints = np.empty(self.J-1)
                phi_cutpoints[0] = cutpoints[1]
                for j in range(2, self.J):
                    phi_cutpoints[j - 1] = np.log(
                        cutpoints[j] - cutpoints[j - 1])
                # update phi_cutpoints
                if trainables[1]:
                    phi_cutpoints[0] = phi[index]
                    index += 1
                for j in range(2, self.J):
                    if trainables[j]:
                        phi_cutpoints[j - 1] = phi[index]
                        index += 1
                # update cutpoints
                cutpoints[1] = phi_cutpoints[0]
                for j in range(2, self.J):
                    cutpoints[j] = cutpoints[j - 1] + np.exp(
                        phi_cutpoints[j - 1])
            if trainables[self.J]:
                std_dev = np.exp(phi[index])
                variance = std_dev**2
                index += 1
            if self.kernel._ARD:
                if theta is None: theta = self.kernel.theta
                for d in range(self.D):
                    if trainables[self.J + 1][d]:
                        theta[d] = np.exp(phi[index])
                        index += 1
            else:
                if trainables[self.J + 1]:
                    theta = np.exp(phi[index])
                    index += 1
        # Update prior and posterior covariance
        # TODO: this should include an argument as to whether derivatives need to be calculated. Perhaps this is given by trainables.
        self.hyperparameters_update(
            cutpoints=cutpoints, theta=theta, variance=variance,
            noise_variance=noise_variance)
        if self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        intervals = self.cutpoints[2:self.J] - self.cutpoints[1:self.J - 1]
        error = np.inf
        iteration = 0
        from collections.abc import Iterable
        def flatten(x):
            if isinstance(x, Iterable):
                return [a for i in x for a in flatten(i)]
            else:
                return [x]
        gx_where = np.where(flatten(trainables))
        return (intervals, error, iteration, gx_where, gx)

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
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.partial_K_theta = self.kernel.kernel_partial_derivative_theta(
            self.X_train, self.X_train)
        self.partial_K_variance = self.kernel.kernel_partial_derivative_variance(
            self.X_train, self.X_train)
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
            self, cutpoints, noise_variance, *args, **kwargs):
        """
        Create an :class:`LaplaceGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)
        # Initiate hyperparameters
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _approximate_initiate(
            self, posterior_mean_0=None):
        """
        Initialise the Approximator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Approximator.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If
             `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg psi_0: Initialisation of hyperhyperparameters. If `None`
            then initialised to ones, default `None`.
        :type psi_0: :class:`numpy.ndarray` or float
        :return: Containers for the approximate posterior means of parameters and
            hyperparameters.
        :rtype: (12,) tuple.
        """
        if posterior_mean_0 is None:
            posterior_mean_0 = self.cutpoints_ts.copy()
            posterior_mean_0[self.indices_where_0] = self.cutpoints_tplus1s[
                self.indices_where_0]
        error = 0.0
        posterior_means = []
        containers = (posterior_means)
        return (posterior_mean_0, containers, error)

    def approximate(
            self, steps, posterior_mean_0=None, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Laplace approximation via Newton-Raphson iteration as
        written in
        Appendix A Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.

        Laplace imposes an inverse covariance for the approximating Gaussian
        equal to the negative Hessian of the log of the target density.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg bool write: Boolean variable to store and write arrays of
            interest. If set to "True", the method will output non-empty
            containers of evolution of the statistics over the steps.
            If set to "False", statistics will not be written and those
            containers will remain empty.
        :return: convergence error, the regression weights and
            approximate posterior mean. Containers for write values.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate
            posterior means, other statistics and tuple of lists of per-step
            evolution of those statistics.
        """
        (posterior_mean, containers, error) = self._approximate_initiate(
            posterior_mean_0)
        (posterior_means) = containers
        for _ in trange(1, 1 + steps,
                        desc="Laplace GP approximator progress",
                        unit="iterations", disable=True):
            (error, weight, precision, cov, log_det_cov,
                    posterior_mean) = update_posterior_LA(
                self.noise_std, self.noise_variance, posterior_mean,
                self.cutpoints_ts, self.cutpoints_tplus1s, self.K, self.N,
                # upper_bound=self.upper_bound,  # optional
                # upper_bound=self.upper_bound2,  # optional
                # tolerance=self.tolerance  # not recommended
                )
            if write is True:
                posterior_means.append(posterior_mean)
        containers = (posterior_means)
        return (error, weight, precision, cov, log_det_cov, posterior_mean,
            containers)

    def approximate_posterior(
            self, phi, trainables, steps, verbose=False,
            return_reparameterised=None, posterior_mean_0=None):
        """
        Newton-Raphson procedure for convex optimization to find MAP point and
        curvature.

        :arg phi: (log-)hyperparameters to be optimised.
        :type phi:
        :arg trainables:
        :type trainables:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, error, iteration, gx_where,
                gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, verbose=verbose)
        posterior_mean = posterior_mean_0
        while error / steps > self.tolerance:
            iteration += 1
            (error, weight, precision, cov, log_det_cov, posterior_mean,
                    containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean, write=False)
            if verbose:
                print("({}), error={}".format(iteration, error))
        (weight, precision,
        w1, w2, g1, g2, v1, v2, q1, q2,
        L_cov, cov, Z, log_det_cov) = compute_weights_LA(
            posterior_mean, self.cutpoints_ts, self.cutpoints_tplus1s,
            self.noise_std, self.noise_variance,
            self.N, self.K,
            # upper_bound=self.upper_bound,  # optional
            # upper_bound2=self.upper_bound2,  # optional
            # tolerance=self.tolerance2,  # not recommended
            )
        fx = objective_LA(weight, posterior_mean, precision, L_cov, Z)
        gx = objective_gradient_LA(
            gx, intervals, w1, w2, g1, g2, v1, v2, q1, q2, cov, weight,
            precision, self.y_train, trainables, self.K,
            self.partial_K_theta, self.partial_K_variance,
            self.noise_std, self.noise_variance,
            self.kernel.theta, self.kernel.variance,
            self.N, self.J, self.D, self.kernel._ARD)
        gx = gx[gx_where]
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        if return_reparameterised is False:
            posterior_covariance_ = posterior_covariance(
                self.K, cov, precision)
            return None, None, log_det_cov, weight, precision, posterior_mean, (
                posterior_covariance_, False)
        elif return_reparameterised is None:
            if verbose:
                print(
                "\ncutpoints={}, theta={}, noise_variance={}, variance={},"
                "\nfunction_eval={}, \nfunction_grad={}".format(
                    self.cutpoints, self.kernel.theta, self.noise_variance,
                    self.kernel.variance, fx, gx))
            return fx, gx


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
            self, cutpoints, noise_variance, *args, **kwargs):
        """
        Create an :class:`VBGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`VBGP` object.
        """
        super().__init__(*args, **kwargs)
        # Initiate hyperparameters
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

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
        (self.L_cov, self.cov, self.log_det_cov, self.trace_cov,
        self.trace_posterior_cov_div_var) = update_posterior_covariance_VB(
            self.noise_variance, self.N, self.K)
        warnings.warn("Done updating posterior covariance.")

    def _approximate_initiate(self, posterior_mean_0):
        """
        Initialise the approximator.

        :arg m_0: The initial state of the approximate posterior mean (N,).
            If `None` then initialised to zeros, default `None`. 
        :type m_0: :class:`numpy.ndarray`
        """
        if posterior_mean_0 is None:
            # posterior_mean_0 = np.random.rand(self.N)  # TODO: justification for this?
            posterior_mean_0 = np.zeros(self.N)
        posterior_means = []
        thetas = []
        psis = []
        fxs = []
        containers = (posterior_means, thetas, psis, fxs)
        return posterior_mean_0, containers

    def approximate(
            self, steps, posterior_mean_0=None,
            write=False):
        """
        Estimating the posterior means are a 3 step iteration over
        posterior_mean, theta and psi Eq.(8), (9), (10), respectively or,
        optionally, just an iteration over posterior_mean.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg bool write: Boolean variable to store and write arrays of
            interest. If set to "True", the method will output non-empty
            containers of evolution of the statistics over the steps. If
            set to "False", statistics will not be written and those
            containers will remain empty.
        :return: Approximate posterior means and covariances.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate
            posterior means, other statistics and tuple of lists of per-step
            evolution of those statistics.
        """
        posterior_mean, containers = self._approximate_initiate(
            posterior_mean_0)
        posterior_means, thetas, psis, fxs = containers
        for _ in trange(1, 1 + steps,
                        desc="VB GP approximator progress", unit="samples",
                        disable=True):
            posterior_mean, weight = update_posterior_mean_VB(
                self.noise_std, posterior_mean, self.cov,
                self.cutpoints_ts, self.cutpoints_tplus1s, self.K,
                self.upper_bound, self.upper_bound2)
            if self.kernel.theta_hyperhyperparameters is not None:
                # Variational Bayes update for kernel hyperparameters
                # Kernel hyperparameters are treated as latent variables here
                # TODO maybe this shouldn't be performed at every step.
                (theta,
                theta_hyperparameters) = update_hyperparameter_posterior_VB(
                    posterior_mean, theta, theta_hyperparameters)
                self.hyperparameters_update(
                    theta=theta,
                    theta_hyperparameters=theta_hyperparameters)
            if write:
                Z, *_ = truncated_norm_normalising_constant(
                    self.cutpoints_ts, self.cutpoints_tplus1s,
                    self.noise_std, posterior_mean)
                fx = objective_VB(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
                posterior_means.append(posterior_mean)
                if self.kernel.theta_hyperparameters is not None:
                    thetas.append(self.kernel.theta)
                    psis.append(self.kernel.theta_hyperparameters)
                fxs.append(fx)
        containers = (posterior_means, thetas, psis, fxs)
        return posterior_mean, weight, containers

    def approximate_posterior(
            self, phi, trainables, steps, verbose=False,
            return_reparameterised=None):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :arg trainables:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, error, iteration, gx_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, verbose=verbose)
        fx_old = np.inf
        posterior_mean = None
        while error / steps > self.tolerance:
            iteration += 1
            (posterior_mean, weight, *_) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                write=False)
            (Z, norm_pdf_z1s, norm_pdf_z2s,
                    *_ )= truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean)
            if self.kernel.theta_hyperhyperparameters is not None:
                fx = objective_VB(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
            else:
                # Only terms in posterior_mean change for fixed hyperparameters
                fx = -0.5 * posterior_mean.T @ weight - np.sum(Z)
            error = np.abs(fx_old - fx)
            fx_old = fx
            if verbose:
                print("({}), error={}".format(iteration, error))
        fx = objective_VB(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
        gx = objective_gradient_VB(
                gx.copy(), intervals, self.cutpoints_ts,
                self.cutpoints_tplus1s,
                self.kernel.theta, self.kernel.variance,
                self.noise_variance, self.noise_std,
                self.y_train, posterior_mean, weight, self.cov, self.trace_cov,
                self.partial_K_theta, self.partial_K_variance,
                self.N, self.J, self.D, self.kernel._ARD,
                self.upper_bound, self.upper_bound2, Z,
                norm_pdf_z1s, norm_pdf_z2s, trainables,
                numerical_stability=True, verbose=False)
        gx = gx[gx_where]
        if return_reparameterised is True:
            return fx, gx, weight, (self.cov, True)
        elif return_reparameterised is False:
            precision = np.ones(self.N) / self.noise_variance
            return fx, gx, self.log_det_cov, weight, precision, posterior_mean, (
                self.noise_variance * self.K @ self.cov, False)
        elif return_reparameterised is None:
            if verbose:
                print(
                "\ncutpoints={}, theta={}, noise_variance={}, variance={},"
                "\nfunction_eval={}, \nfunction_grad={}".format(
                    self.cutpoints, self.kernel.theta, self.noise_variance,
                    self.kernel.variance, fx, gx))
            return fx, gx


class EPGP(Approximator):
    """
    A GP classifier for ordinal likelihood using Expectation Propagation (EP).

    Inherits the Approximator ABC.

    Expectation propagation algorithm as written in Appendix B
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
        return "EPGP"

    def __init__(
            self, cutpoints, noise_variance, *args, **kwargs):
        """
        Create an :class:`EPGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to 1.0, default `None`.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)
        # Initiate hyperparameters
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _approximate_initiate(
            self, posterior_mean_0=None, posterior_cov_0=None,
            mean_EP_0=None, precision_EP_0=None, amplitude_EP_0=None):
        """
        Initialise the Approximator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Approximator.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If
             `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N,). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros,
            default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site)
            amplitudes (N,). If `None` then initialised to ones,
            default `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg psi_0: Initialisation of hyperhyperparameters. If `None`
            then initialised to ones, default `None`.
        :type psi_0: :class:`numpy.ndarray` or float
        :arg dlogZ_dcavity_mean_0: Initialisation of the EP weights,
            which are gradients of the approximate marginal
            likelihood wrt to the 'cavity distribution mean'. If `None`
            then initialised to zeros, default `None`.
        :type dlogZ_dcavity_mean_0: :class:`numpy.ndarray`
        :return: Containers for the approximate posterior means of parameters
            and hyperparameters.
        :rtype: (12,) tuple.
        """
        if posterior_cov_0 is None:
            # The first EP approximation before data-update is the GP prior cov
            # TODO: may run into errors when elements of K are zero (e.g., when lengthscale very small)
            posterior_cov_0 = self.K
        if mean_EP_0 is None:
            mean_EP_0 = np.zeros((self.N,))
        if precision_EP_0 is None:
            precision_EP_0 = np.zeros((self.N,))
        if amplitude_EP_0 is None:
            amplitude_EP_0 = np.ones((self.N,))
        if posterior_mean_0 is None:
            posterior_mean_0 = (
                posterior_cov_0 @ np.diag(precision_EP_0)) @ mean_EP_0
        error = 0.0
        dlogZ_dcavity_mean_0 = np.zeros(self.N)  # Initialisation
        posterior_means = []
        posterior_covs = []
        mean_EPs = []
        amplitude_EPs = []
        precision_EPs = []
        approximate_marginal_likelihoods = []
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_marginal_likelihoods)
        return (posterior_mean_0, posterior_cov_0, mean_EP_0,
                precision_EP_0, amplitude_EP_0, dlogZ_dcavity_mean_0,
                containers, error)

    def approximate(self, indices, steps, posterior_mean_0=None,
            posterior_cov_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Expectation propagation iteration as written in
        Appendix B Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning Research. 6.
        1019-1041.

        EP does not attempt to learn a posterior distribution over
        hyperparameters, but instead tries to approximate
        the joint posterior given some hyperparameters. The hyperparameters
        have to be optimized with model selection step.

        :arg indices: The set of indices of the data in this swipe.
            Could be e.g., a minibatch, the whole dataset.
        :type indices: :class:`numpy.ndarray`
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N, N). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros, default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site)
            amplitudes (N,). If `None` then initialised to ones, default
            `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg bool fix_hyperparameters: Must be `True`, since the hyperparameter
            approximate posteriors are of the hyperparameters are not
            calculated in this EP approximation.
        :arg bool write: Boolean variable to store and write arrays of
            interest. If set to "True", the method will output non-empty
            containers of evolution of the statistics over the steps.
            If set to "False", statistics will not be written and those
            containers will remain empty.
        :return: approximate posterior mean and covariances.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate
            posterior means, other statistics and tuple of lists of per-step
            evolution of those statistics.
        """
        for _ in range(steps):
            (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, dlogZ_dcavity_mean, containers,
                error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)
            (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, dlogZ_dcavity_mean, error) = update_posterior_EP(
                    indices, posterior_mean, posterior_cov,
                    mean_EP, precision_EP, amplitude_EP, dlogZ_dcavity_mean,
                    error,
                    self.y_train, self.cutpoints, self.noise_variance, self.J,
                    self.upper_bound, self.tolerance, self.tolerance2)
        return (error, dlogZ_dcavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP)

    def approximate_posterior(
            self, phi, trainables, steps, verbose=False,
            return_reparameterised=None, posterior_mean_0=None,
            posterior_cov_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :type phi:
        :arg trainables:
        :type trainables:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg return_reparameterised:
        :type return_reparameterised:
        :arg posterior_cov_0:
        :type posterior_cov_0:
        :arg mean_EP_0:
        :type mean_EP_0:
        :arg precision_EP_0:
        :type precision_EP_0:
        :arg amplitude_EP_0:
        :type amplitude_EP_0:
        :arg bool write:
        :arg bool verbose:
        :return: fx the objective and gx the objective gradient
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, error, iteration, gx_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, verbose=verbose)
        posterior_mean = posterior_mean_0
        posterior_cov = posterior_cov_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        # random permutation of data
        indices = np.arange(self.N)
        while error / (steps * self.N) > self.tolerance:
            iteration += 1
            (error, dlogZ_dcavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP) = self.approximate(
                indices, steps, posterior_mean_0=posterior_mean,
                posterior_cov_0=posterior_cov, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP)
            if verbose:
                print("error = {}".format(error))
        (weight, precision_EP, L_cov, cov) = compute_weights_EP(
            precision_EP, mean_EP, dlogZ_dcavity_mean, self.K, self.tolerance,
            self.tolerance2, self.N)
        # Compute gradients of the hyperparameters
        t1, t2, t3, t4, t5 = compute_integrals_vector_EP(
            np.diag(posterior_cov), posterior_mean, self.noise_variance,
            self.cutpoints_ts, self.cutpoints_tplus1s, self.indices_where_J_1,
            self.indices_where_0, self.N, self.tolerance, self.tolerance2)
        fx = objective_EP(precision_EP, posterior_mean, t1,
            L_cov, cov, weight, self.tolerance2)
        if self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = objective_gradient_EP(
            gx, intervals, self.kernel.theta, self.kernel.variance,
            self.noise_variance, self.cutpoints, t2, t3, t4, t5, self.y_train,
            cov, weight, trainables, self.partial_K_theta,
            self.partial_K_variance, self.J, self.D, self.kernel._ARD)
        gx = gx[gx_where]
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        elif return_reparameterised is False:
            log_det_cov = -2 * np.sum(np.log(np.diag(L_cov)))
            return fx, gx, log_det_cov, weight, precision_EP, posterior_mean, (
                posterior_cov, False)
        elif return_reparameterised is None:
            if verbose:
                print(
                "\ncutpoints={}, theta={}, noise_variance={}, variance={},"
                "\nfunction_eval={}, \nfunction_grad={}".format(
                    self.cutpoints, self.kernel.theta, self.noise_variance,
                    self.kernel.variance, fx, gx))
            return fx, gx


class PEPGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Expectation Propagation
    (EP) approximation.

    Inherits EP. TODO: possibly restructure so EP inherits PEP, since PEP is strictly more general.

    Expectation propagation algorithm as written in Appendix B
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
        return "PEPGP"

    def __init__(
            self, cutpoints, noise_variance, alpha, minibatch_size=None,
            gauss_hermite_points=20, *args, **kwargs):
        """
        Create an :class:`PEPGP` Approximator object.

        :arg float alpha: Power EP tuning parameter.
        :arg int minibatch_size: How many parallel updates

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to 1.0, default `None`.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)
        # Initiate hyperparameters
        self.hyperparameters_update(cutpoints=cutpoints, noise_variance=noise_variance)
        if minibatch_size is None:
            self.minibatch_size = self.N
        else:
            self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.gauss_hermite_points = gauss_hermite_points

    def _update_prior(self):
        """
        Update prior covariances with inducing points.
        """
        # Use full GP
        self.M = self.N
        warnings.warn(
            "Updating prior covariance")
        self.Kuu = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.K = self.Kuu
        self.Kfu = self.Kuu
        self.Kfdiag = self.kernel.kernel_prior_diagonal(self.X_train)
        # self.Kuuinv, self.L_K = matrix_inverse(self.Kuu + self.epsilon * np.eye(self.N))
        self.KuuinvKuf = np.eye(self.N)
        self.partial_K_theta = self.kernel.kernel_partial_derivative_theta(
            self.X_train, self.X_train)
        self.partial_K_variance = self.kernel.kernel_partial_derivative_variance(
            self.X_train, self.X_train)
        warnings.warn(
            "Done updating prior covariance")

    def approximate(
            self, sequential, indices, steps, beta_0, gamma_0, mean_EP_0,
            variance_EP_0, write):
        if sequential:
            for _ in range(steps):
                (beta, gamma, mean_EP, variance_EP, log_lik, grads_logZtilted,
                containers, error) = self._approximate_initiate(
                    beta_0, gamma_0, mean_EP_0, variance_EP_0)
                (error, beta, gamma, mean_EP, variance_EP,log_lik,
                    grads_logZtilted) = update_posterior_sequential_PEP(
                        indices, beta_0=beta, gamma_0=gamma, mean_EP_0=mean_EP,
                        variance_EP_0=variance_EP, write=write)
        else:
            for _ in range(steps):
                (beta, gamma, mean_EP, variance_EP, log_lik, grads,
                    containers, error) = self._approximate_parallel_initiate(
                    beta_0, gamma_0, mean_EP_0, variance_EP_0)
                (error, beta, gamma, mean_EP, variance_EP, log_lik,
                    grads_logZtilted) = update_posterior_parallel_PEP(
                        indices, beta_0=beta, gamma_0=gamma, mean_EP_0=mean_EP,
                        variance_EP_0=variance_EP, write=write)
        return (error, beta, gamma, mean_EP, variance_EP,
            log_lik, grads_logZtilted)

    def _approximate_initiate(
            self, beta_0=None, gamma_0=None, mean_EP_0=None,
            variance_EP_0=None):
        """
        Initialise the Approximator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Approximator.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If
             `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N,). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros,
            default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :return: Containers for the approximate posterior means of parameters
            and hyperparameters.
        :rtype: (12,) tuple.
        """
        if beta_0 is None:
            # The first EP approximation before data-update is the GP prior cov
            beta_0 = np.zeros((self.M, self.M))
        if gamma_0 is None:
            gamma_0 = np.zeros((self.M,))
        if mean_EP_0 is None:
            mean_EP_0 = np.zeros((self.N,))
        if variance_EP_0 is None:
            variance_EP_0 = 1e20 * np.ones((self.N,))
        error = 0.0
        log_like_0 = 0.0
        grads_logZtilted_0 = 0.0
        betas = []
        gammas = []
        mean_EPs = []
        variance_EPs = []
        containers = (betas, gammas, mean_EPs, variance_EPs)
        return (beta_0, gamma_0, mean_EP_0, variance_EP_0,
            log_like_0, grads_logZtilted_0, containers, error)

    def _approximate_parallel_initiate(
            self, beta_0=None, gamma_0=None, mean_EP_0=None,
            variance_EP_0=None):
        """
        Initialise the Approximator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Approximator.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If
             `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N,). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros,
            default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :return: Containers for the approximate posterior means of parameters
            and hyperparameters.
        :rtype: (12,) tuple.
        """
        if beta_0 is None:
            # The first EP approximation before data-update is the GP prior cov
            beta_0 = np.zeros((self.M, self.M))
        if gamma_0 is None:
            gamma_0 = np.zeros(self.M)
        if mean_EP_0 is None:
            mean_EP_0 = np.zeros((self.N, 1))
        if variance_EP_0 is None:
            variance_EP_0 = 5.7 * np.ones((self.N, 1))
        error = 0.0
        log_like_0 = 0.0
        grads_0 = 0.0
        betas = []
        gammas = []
        mean_EPs = []
        variance_EPs = []
        containers = (betas, gammas, mean_EPs, variance_EPs)
        return (beta_0, gamma_0, mean_EP_0, variance_EP_0,
            log_like_0, grads_0, containers, error)

    def approximate_posterior(
            self, phi, trainables, steps, verbose=False,
            return_reparameterised=None,
            beta_0=None, gamma_0=None,
            mean_EP_0=None, variance_EP_0=None, sequential=False):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :type phi:
        :arg trainables:
        :type trainables:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg return_reparameterised:
        :type return_reparameterised:
        :arg posterior_cov_0:
        :type posterior_cov_0:
        :arg mean_EP_0:
        :type mean_EP_0:
        :arg precision_EP_0:
        :type precision_EP_0:
        :arg amplitude_EP_0:
        :type amplitude_EP_0:
        :arg bool write:
        :arg bool verbose:
        :return: fx the objective and gx the objective gradient
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, error, iteration, gx_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, verbose=verbose)
        beta = beta_0
        gamma = gamma_0
        mean_EP = mean_EP_0
        variance_EP = variance_EP_0
        # random permutation of data
        indices = np.arange(self.N)
        while error / (steps * self.N) > self.tolerance2:
            iteration += 1
            (error, beta, gamma, mean_EP, variance_EP,
                log_lik, grads) = self.approximate(
            sequential, indices, steps, beta_0=beta, gamma_0=gamma,
            mean_EP_0=mean_EP, variance_EP_0=variance_EP, write=False)
            print("{}/iterations, error={}".format(iteration, error))
        # Compute posterior TODO: does it need to be done twice?
        (posterior_mean, posterior_cov) = self._compute_posterior(
            self.Kuu, gamma, beta)
        fx = objective_PEP(self.N, self.alpha, self.minibatch_size,
            posterior_mean, posterior_cov, log_lik)
        if self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = objective_gradient_PEP(
            gx, trainables)
        gx = gx[gx_where]
        if verbose:
            print(
            "\ncutpoints={}, theta={}, noise_variance={}, variance={},"
            "\nfunction_eval={}, \nfunction_grad={}".format(
                self.cutpoints, self.kernel.theta, self.noise_variance,
                self.kernel.variance, fx, gx))
        if return_reparameterised is True:
            return fx, gx, gamma, (beta, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (posterior_cov, False)
        elif return_reparameterised is None:
            return fx, gx


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
            f"{approximator} is not an instance of"
            "probit.approximators.Approximator"
        )

        super().__init__(message)
