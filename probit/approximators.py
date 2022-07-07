from abc import ABC, abstractmethod
import enum
from re import A
from probit.kernels import Kernel, InvalidKernel
import pathlib
import random
from tqdm import trange
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
from probit.utilities import (
    check_cutpoints,
    read_array,
    norm_z_pdf, norm_cdf,
    truncated_norm_normalising_constant,
    p, dp,
    sample_varphis,
    fromb_t1_vector, fromb_t2_vector, fromb_t3_vector, fromb_t4_vector,
    fromb_t5_vector)
# NOTE Usually the numba implementation is not faster
# from .numba.utilities import (
#     fromb_t1_vector, fromb_t2_vector,
#     fromb_t3_vector, fromb_t4_vector, fromb_t5_vector)
from scipy.linalg import cho_solve, cho_factor, solve_triangular


class Approximator(ABC):
    """
    Base class for variational Bayes approximators.

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
            self, kernel, J, data=None, read_path=None):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
        :arg int J: The number of (ordinal) classes.
        :arg data: The data tuple. (X_train, t_train), where  
            X_train is the (N, D) The data vector and t_train (N, ) is the
            target vector. Default `None`, if `None`, then the data and prior
            are assumed cached in `read_path` and are attempted to be read.
        :type data: (:class:`numpy.ndarray`, :class:`numpy.ndarray`)
        :arg str read_path: Read path for outputs. If no data is provided,
            then it assumed that this is the path to the data and cached
            prior covariance(s).

        :returns: A :class:`Approximator` object
        """
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

        # Numerical stability
        # See GPML by Williams et al. for a good explanation of jitter
        self.jitter = 1e-10
    
        # Threshold of single sided standard deviations that
        # normal cdf can be approximated to 0 or 1
        # More than this + redundancy leads to numerical instability
        # due to catestrophic cancellation
        # Less than this leads to a poor approximation due to series
        # expansion at infinity truncation
        # Good values found between 4 and 6
        self.upper_bound = 6

        # More than this + redundancy leads to numerical
        # instability due to overflow
        # Less than this results in poor approximation due to
        # neglected probability mass in the tails
        # Good values found between 18 and 30
        self.upper_bound2 = 30  # Try decreasing if experiencing infs or NaNs

        # Get data and calculate the prior
        if data is not None:
            X_train, t_train = data
            self.X_train = X_train
            if t_train.dtype not in [int, np.int32]:
                raise TypeError(
                    "t must contain only integer values (got {})".format(
                        t_train.dtype))
            else:
                t_train = t_train.astype(int)
                self.t_train = t_train
            self._update_prior()
        else:
            # Try read model from file
            try:
                self.X_train = read_array(self.read_path, "X_train")
                self.t_train = read_array(self.read_path, "t_train")
                self._load_cached_prior()
            except KeyError:
                # The array does not exist in the model file
                raise
            except OSError:
                # Model file does not exist
                raise
        # TODO: deprecate self.N and self.D
        self.N = np.shape(self.X_train)[0]
        self.D = np.shape(self.X_train)[1]
        self.grid = np.ogrid[0:self.N]  # For indexing sets of self.t_train
        self.indices_where_0 = np.where(self.t_train == 0)
        self.indices_where_J_1 = np.where(self.t_train == self.J - 1)

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
        Return predictive distributions for the ordinal likelihood.
        """
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(
                cutpoints[j + 1], posterior_pred_mean), posterior_pred_std)
            Z2 = np.divide(
                np.subtract(cutpoints[j],
                posterior_pred_mean), posterior_pred_std)
            predictive_distributions[:, j] = norm_cdf(Z1) - norm_cdf(Z2)
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
            self.noise_std, m, self.EPS,
            upper_bound=self.upper_bound,
            # upper_bound2=self.upper_bound2,  # optional
            # numerically_stable=True  # optional
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
            phi.append(np.log(np.sqrt(self.noise_variance)))
        if trainables[1]:
            phi.append(self.cutpoints[1])
        for j in range(2, self.J):
            if trainables[j]:
                phi.append(np.log(self.cutpoints[j] - self.cutpoints[j - 1]))
        if trainables[self.J]:
            phi.append(np.log(np.sqrt(self.kernel.variance)))
        # TODO: replace this with kernel number of hyperparameters.
        if trainables[self.J + 1]:
            phi.append(np.log(self.kernel.varphi))
        return np.array(phi)

    def _hyperparameters_update(
        self, cutpoints=None, varphi=None, variance=None, noise_variance=None):
        """
        TODO: Is the below still relevant?
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The
        strange thing is that hyperparameters can be absorbed into the set of
        variables and so the definition of hyperparameters and variables
        becomes muddled. Since varphi can be a variable or a parameter, then
        optionally initiate it as a parameter, and then intitate it as a
        variable within :meth:`approximate`. Problem is, if it changes at
        approximate time, then a hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        """
        if cutpoints is not None:
            self.cutpoints = check_cutpoints(cutpoints, self.J)
            self.cutpoints_ts = self.cutpoints[self.t_train]
            self.cutpoints_tplus1s = self.cutpoints[self.t_train + 1]
        if varphi is not None or variance is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, variance=variance)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
            warnings.warn("Done updating prior covariance")
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)

    def hyperparameters_update(
        self, cutpoints=None, varphi=None, variance=None, noise_variance=None):
        """
        Wrapper function for :meth:`_hyperparameters_update`.
        """
        return self._hyperparameters_update(
            cutpoints=cutpoints, varphi=varphi, variance=variance,
            noise_variance=noise_variance)

    def _hyperparameter_training_step_initialise(
            self, phi, trainables, steps):
        """
        TODO: this doesn't look correct, for example if only training a subset
        Initialise the hyperparameter training step.

        :arg phi: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}}
                \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\varphi}],

            where :math:`\sigma` is the noise standard deviation,
            :math:`\b_{1}` is the first cutpoint, :math:`\Delta_{l}` is the
            :math:`l`th cutpoint interval, :math:`\varphi` is the single
            shared lengthscale parameter or vector of parameters in which
            there are in the most general case J * D parameters.
            If `None` then no hyperperameter update is performed.
        :type phi: :class:`numpy.ndarray`
        :return: (intervals, steps, error, iteration, trainables_where, gx)
        :rtype: (6,) tuple
        """
        # Initiate at None since those that are None do not get updated        
        noise_variance = None
        cutpoints = None
        variance = None
        varphi = None
        index = 0
        if trainables is not None:
            if trainables[0]:
                noise_std = np.exp(phi[index])
                noise_variance = noise_std**2
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
            if cutpoints is None and np.any(trainables[1:self.J]):
                # Get cutpoints from classifier
                cutpoints = self.cutpoints
            if trainables[1]:
                cutpoints[1] = phi[index]
                index += 1
            for j in range(2, self.J):
                if trainables[j]:
                    print(np.exp(phi[index]))
                    cutpoints[j] = cutpoints[j - 1] + np.exp(phi[index])
                    index += 1
            if trainables[self.J]:
                std_dev = np.exp(phi[index])
                variance = std_dev**2
                index += 1
            if trainables[self.J + 1]:
                varphi = np.exp(phi[index])
                index += 1
        # Update prior and posterior covariance
        # TODO: this should include an argument as to whether derivatives need to be calculated. Perhaps this is given by trainables.
        self.hyperparameters_update(
            cutpoints=cutpoints, varphi=varphi, variance=variance,
            noise_variance=noise_variance)
        gx = np.zeros(1 + self.J - 1 + 1 + 1)
        intervals = self.cutpoints[2:self.J] - self.cutpoints[1:self.J - 1]
        error = np.inf
        iteration = 0
        trainables_where = np.where(trainables!=0)
        return (intervals, steps, error, iteration, trainables_where, gx)

    def hyperparameter_training_step(
            self, phi, trainables, verbose, steps,
            first_step=0, write=False):
        """
        Optimisation routine for hyperparameters.
        :return: fx, gx
        """
        return self.approximate_posterior(
            phi, trainables, first_step=first_step, steps=steps,
            posterior_mean_0=None, return_reparameterised=None,
            verbose=verbose)  # returns (fx, gx)

    def _grid_over_hyperparameters_initiate(
            self, res, domain, trainables, cutpoints):
        """
        Initiate metadata and hyperparameters for plotting the objective
        function surface over hyperparameters.

        :arg int res:
        :arg domain: ((2,)tuple, (2.)tuple) of the start/stop in the domain to
            grid over, for each axis, like: ((start, stop), (start, stop)).
        :type domain:
        :arg trainables: Indicator array of the hyperparameters to sample over.
        :type trainables: :class:`numpy.ndarray`
        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        """
        index = 0
        label = []
        axis_scale = []
        space = []
        if trainables[0]:
            # Grid over noise_std
            label.append(r"$\sigma$")
            axis_scale.append("log")
            space.append(
                np.logspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        if trainables[1]:
            # Grid over b_1, the first cutpoint
            label.append(r"$b_{}$".format(1))
            axis_scale.append("linear")
            space.append(
                np.linspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        for j in range(2, self.J):
            if trainables[j]:
                # Grid over b_j - b_{j-1}, the differences between cutpoints
                label.append(r"$b_{} - b_{}$".format(j, j-1))
                axis_scale.append("log")
                space.append(
                    np.logspace(
                        domain[index][0], domain[index][1], res[index]))
                index += 1
        if trainables[self.J]:
            # Grid over variance
            label.append("$variance$")
            axis_scale.append("log")
            space.append(
                np.logspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        # TODO: 
        # gx_0 = np.empty(1 + self.J - 1 + 1 + self.J * self.D)
        # # In this case, then there is a scale parameter,
        # #  the first cutpoint, the interval parameters,
        # # and lengthvariances parameter for each dimension and class
        # for j in range(self.J * self.D):
        #     if trainables[self.J + 1 + j]:
        #         # grid over this particular hyperparameter
        #         raise ValueError("TODO")
        #         index += 1
        gx_0 = np.empty(1 + self.J - 1 + 1 + 1)
        if trainables[self.J + 1]:
            # Grid over only kernel hyperparameter, varphi
            label.append(r"$\varphi$")
            axis_scale.append("log")
            space.append(
                np.logspace(
                    domain[index][0], domain[index][1], res[index]))
            index +=1
        if index == 2:
            meshgrid = np.meshgrid(space[0], space[1])
            thetas = np.dstack(meshgrid)
            thetas = thetas.reshape((len(space[0]) * len(space[1]), 2))
            fxs = np.empty(len(thetas))
            gxs = np.empty((len(thetas), 2))
        elif index == 1:
            meshgrid = (space[0], None)
            space.append(None)
            axis_scale.append(None)
            label.append(None)
            thetas = space[0]
            fxs = np.empty(len(thetas))
            gxs = np.empty(len(thetas))
        else:
            raise ValueError(
                "Too many independent variables to plot objective over!"
                " (got {}, expected {})".format(
                index, "1, or 2"))
        assert len(axis_scale) == 2
        assert len(meshgrid) == 2
        assert len(space) ==  2
        assert len(label) == 2
        intervals = cutpoints[2:self.J] - cutpoints[1:self.J - 1]
        trainables_where = np.where(trainables != 0)
        return (
            space[0], space[1],
            label[0], label[1],
            axis_scale[0], axis_scale[1],
            meshgrid[0], meshgrid[1],
            thetas, fxs, gxs, gx_0, intervals, trainables_where)

    def _grid_over_hyperparameters_update(
        self, theta, trainables, cutpoints):
        """
        Update the hyperparameters, theta.

        :arg theta: The updated values of the hyperparameters.
        :type theta:
        :arg trainables:
        :type trainables:
        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        """
        index = 0
        if trainables[0]:
            if np.isscalar(theta):
                noise_std = theta
            else:
                noise_std = theta[index]
            noise_variance = noise_std**2
            noise_variance_update = noise_variance
            # Update kernel parameters, update prior and posterior covariance
            index += 1
        else:
            noise_variance_update = None
        if trainables[1]:
            cutpoints = np.empty((self.J + 1,))
            cutpoints[0] = np.NINF
            cutpoints[-1] = np.inf
            if np.isscalar(theta):
                cutpoints[1] = theta
            else:
                cutpoints[1] = theta[index]
            index += 1
        for j in range(2, self.J):
            if trainables[j]:
                if np.isscalar(theta):
                    cutpoints[j] = cutpoints[j-1] + theta
                else:
                    cutpoints[j] = cutpoints[j-1] + theta[index]
                index += 1
        if trainables[self.J]:
            std_dev = theta[index]
            variance = std_dev**2
            index += 1
            variance_update = variance
        else:
            variance_update = None
        if trainables[self.J + 1]:  # TODO: replace this with kernel number of hyperparameters.
            if np.isscalar(theta):
                varphi = theta
            else:
                varphi = theta[index]
            varphi_update = varphi
            index += 1
        else:
            varphi_update = None
        # Update kernel parameters, update prior and posterior covariance
        self.hyperparameters_update(
                cutpoints=cutpoints, 
                noise_variance=noise_variance_update,
                variance=variance_update,
                varphi=varphi_update)

    def _load_cached_prior(self):
        """
        Load cached prior covariances.
        """
        self.K = read_array(self.read_path, "K")
        self.partial_K_varphi = read_array(
            self.read_path, "partial_K_varphi")
        self.partial_K_variance = read_array(
            self.read_path, "partial_K_variance")

    def _update_prior(self):
        """Update prior covariances."""
        warnings.warn("Updating prior covariance.")
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
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
            self, cutpoints, noise_variance=1.0, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`VBGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`VBGP` object.
        """
        super().__init__(*args, **kwargs)
        # if cutpoints_hyperparameters is not None:
        #     warnings.warn("cutpoints_hyperparameters set as {}".format(cutpoints_hyperparameters))
        #     self.cutpoints_hyperparameters = cutpoints_hyperparameters
        # else:
        #     self.cutpoints_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        #self.EPS = 0.000001  # Acts as a machine tolerance, controls error
        #self.EPS = 0.0000001  # Probably wouldn't go much smaller than this
        self.EPS = 1e-4  # perhaps not low enough.
        # self.EPS = 1e-8
        #self.EPS_2 = 1e-7
        self.EPS_2 = self.EPS**2
        #self.EPS = 0.001  # probably too large, will affect convergence 
        # Tends to work well in practice - should it be made smaller?
        # Just keep it consistent
        #self.jitter = 1e-6
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _update_posterior(self):
        """Update posterior covariances."""
        # Note that this scipy implementation returns an upper triangular matrix
        # whereas numpy, tf, scipy.cholesky return a lower triangular,
        # then the position of the matrix transpose in the code would change.
        (self.L_cov, self.lower) = cho_factor(
            self.noise_variance * np.eye(self.N) + self.K)

        # cho_solve defaults as upper triangular, whereas
        # linalg.cholesky is lower triangular by default.
        # Therefore I don't need any transpose in my algorithm.
        # But if I revert to a lower triangular cholesky decomp,
        # then I will exactly follow e.g., gpflow.

        # Using scipy.linalg.cholesky() and scipy.linalg.cho_factor()
        # probably uses the same code.
        # If the same cholesky is computed twice,
        # then the second time the result is cached (?) and will take no time to
        # evaluate (scipy will just point to the memory object that it has
        # already calculated)

        # TODO: If jax @jit works really well with the GPU for cho_solve,
        # it is worth not storing this matrix - due to storage cost, and it
        # will be faster. See alternative implementation on feature/cho_solve
        # For the CPU, storing self.cov saves solving for the gradient and the
        # fx. Maybe have it as part of a seperate method.
        # TODO: should be using  cho_solve and not solve_triangular, unless I used it because that is what is used
        # in tensorflow for whatever reason (maybe tensorflow has no cho_solve)
        # Note that Tensorflow uses tf.linalg.triangular_solve
        L_covT_inv = solve_triangular(
            self.L_cov.T, np.eye(self.N), lower=True)
        # TODO Is it necessary to do this triangular solve, or can it be done each step instead of matmul. A: This is 3-4 times slower on CPU, what about with jit compiled CPU or GPU?
        self.cov = solve_triangular(self.L_cov, L_covT_inv, lower=False)

        # log det (\sigma^{2}I + K)^{-1} = -2 * trace(log(L)),
        # where LL^{T} = (\sigma^{2}I + K)
        self.log_det_cov = -2 * np.sum(np.log(np.diag(self.L_cov)))
        self.trace_cov = np.sum(np.diag(self.cov))
        self.trace_posterior_cov_div_var = np.einsum(
            'ij, ij -> ', self.K, self.cov)

    def hyperparameters_update(
        self, cutpoints=None, varphi=None, variance=None, noise_variance=None,
        varphi_hyperparameters=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg variance:
        :type variance:
        :arg float noise_variance: The noise variance.
        :type noise_variance:
        :arg varphi_hyperparameters:
        :type varphi_hyperparameters:
        """
        self._hyperparameters_update(
            cutpoints=cutpoints, varphi=varphi,
            variance=variance, noise_variance=noise_variance)
        if varphi_hyperparameters is not None:
            self.kernel.update_hyperparameter(
                varphi_hyperparameters=varphi_hyperparameters)
        # Update posterior covariance
        warnings.warn("Updating posterior covariance.")
        self._update_posterior()
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
        gs = []
        posterior_means = []
        varphis = []
        psis = []
        fxs = []
        containers = (posterior_means, gs, varphis, psis, fxs)
        return posterior_mean_0, containers

    def approximate(
            self, steps, posterior_mean_0=None, first_step=0,
            write=False):
        """
        Estimating the posterior means are a 3 step iteration over
        posterior_mean, varphi and psi Eq.(8), (9), (10), respectively or,
        optionally, just an iteration over posterior_mean.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg int first_step: The first step. Useful for burn in algorithms.
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
        posterior_means, gs, varphis, psis, fxs = containers
        for _ in trange(first_step, first_step + steps,
                        desc="VB GP approximator progress", unit="samples",
                        disable=True):
            p_ = p(
                posterior_mean, self.cutpoints_ts, self.cutpoints_tplus1s,
                self.noise_std, self.EPS, self.upper_bound, self.upper_bound2)
            g = self._g(
                p_, posterior_mean, self.noise_std)
            posterior_mean, weight = self._posterior_mean(
                g, self.cov, self.K)
            if self.kernel.varphi_hyperhyperparameters is not None:
                # Posterior mean update for kernel hyperparameters
                # Kernel hyperparameters are variables here
                # TODO maybe this shouldn't be performed at every step.
                varphi = self._varphi(
                    posterior_mean, self.kernel.varphi_hyperparameters,
                    n_samples=10)
                varphi_hyperparameters = self._varphi_hyperparameters(
                    self.kernel.varphi)
                self.hyperparameters_update(
                    varphi=varphi,
                    varphi_hyperparameters=varphi_hyperparameters)
            if write:
                Z, *_ = truncated_norm_normalising_constant(
                    self.cutpoints_ts, self.cutpoints_tplus1s,
                    self.noise_std, posterior_mean, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance,
                    self.log_det_cov)
                posterior_means.append(posterior_mean)
                gs.append(g)
                if self.kernel.varphi_hyperparameters is not None:
                    varphis.append(self.kernel.varphi)
                    psis.append(self.kernel.varphi_hyperparameters)
                fxs.append(fx)
        containers = (posterior_means, gs, varphis, psis, fxs)
        return posterior_mean, weight, g, p, containers

    def _varphi_hyperparameters(self, varphi):
        """
        Return the approximate posterior mean of the kernel
        varphi hyperparameters.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
        Probit Regression with Gaussian Process Priors," in Neural Computation,
        vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
        doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(10).

        :arg varphi: Posterior mean approximate of varphi.
        :type varphi: :class:`numpy.ndarray`
        :return: The approximate posterior mean of the hyperhyperparameters psi
            Girolami and Rogers Page 9 Eq.(10).
        """
        return np.divide(
            np.add(1, self.kernel.varphi_hyperhyperparameters[0]),
            np.add(self.kernel.varphi_hyperhyperparameters[1], varphi))

    def _varphi(
            self, posterior_mean, varphi_hyperparameters, n_samples=10,
            vectorised=False):
        """
        Return the weights of the importance sampler for varphi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
        Probit Regression with Gaussian Process Priors," in Neural Computation,
        vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
        doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(9).

        :arg posterior_mean: approximate posterior mean.
        :arg varphi_hyperparameters: approximate posterior mean of the kernel
            hyperparameters.
        :arg int n_samples: The number of samples for the importance sampling
            estimate, 500 is used in 2005 Page 13.
        """
        # (n_samples, N, N) for ISO case. Depends on the shape of psi.
        varphis = sample_varphis(
            varphi_hyperparameters, n_samples)  # (n_samples, )
        print(varphis)
        log_varphis = np.log(varphis)
        Ks_samples = self.kernel.kernel_matrices(
            self.X_train, self.X_train, varphis)  # (n_samples, N, N)
        Ks_samples = np.add(Ks_samples, self.jitter * np.eye(self.N))
        if vectorised:
            raise ValueError("TODO")
        else:
            log_ws = np.empty((n_samples,))
            # Scalar version
            for i in range(n_samples):
                (prior_L, lower) = cho_factor(
                    Ks_samples[i])
                half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L)))
                # weight = cho_solve(
                #     (prior_L, lower), posterior_mean)
                # print(posterior_mean.T @ weight)
                # TODO something not quite right - converges to zero
                prior_LT_inv = solve_triangular(
                    prior_L.T, np.eye(self.N), lower=True)
                K_inv = solve_triangular(
                    prior_L, prior_LT_inv, lower=False)
                log_ws[i] = -0.5 * np.log(2 * np.pi) - half_log_det_prior_cov\
                    - 0.5 * posterior_mean.T @ K_inv @ posterior_mean
        # Normalise the w vectors
        max_log_ws = np.max(log_ws)
        log_normalising_constant = max_log_ws + np.log(
            np.sum(np.exp(log_ws - max_log_ws), axis=0))
        log_ws = np.subtract(log_ws, log_normalising_constant)
        print(np.sum(np.exp(log_ws)))
        element_prod = np.add(log_varphis, log_ws)
        element_prod = np.exp(element_prod)
        return np.sum(element_prod, axis=0)

    def _posterior_mean(self, g, cov, K):
        """
        Return the approximate posterior mean of m.

        2021 Page Eq.()

        :arg y: (N,) array
        :type y: :class:`numpy.ndarray`
        :arg cov:
        :type cov:
        :arg K:
        :type K:
        """
        weight = cov @ g
        ## TODO: This is 3-4 times slower on CPU, what about with jit compiled CPU or GPU?
        # weight = cho_solve((self.L_cov, self.lower), y)
        return K @ weight, weight  # (N,), (N,)

    def _g(self, p, posterior_mean, noise_std):
        """
        Calculate y elements 2021 Page Eq.().

        :arg p:
        :type p:
        :arg posterior_mean:
        :type posterior_mean:
        :arg float noise_std: The square root of the noise variance.
        """
        return np.add(posterior_mean, noise_std * p)

    def objective(
        self, N, posterior_mean, weight, trace_cov,
        trace_posterior_cov_div_var, Z, noise_variance,
        log_det_cov, verbose=False):
        """
        # TODO: tidy
        Calculate fx, the variational lower bound of the log marginal
        likelihood.

        .. math::
                \mathcal{F(	heta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
                marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg int N: The number of datapoints.
        :arg m: The posterior mean.
        :type m: :class:`numpy.ndarray`
        :arg y: The posterior mean.
        :type y: :class:`numpy.ndarray`
        :arg K: The prior covariance.
        :type K: :class:`numpy.ndarray`
        :arg float noise_variance: The noise variance.
        :arg float log_det_cov: The log determinant of (a factor in) the
            posterior covariance.
        :arg Z: The array of normalising constants.
        :type Z: :class:`numpy.ndarray`
        :return: fx
        :rtype: float

        """
        trace_K_inv_posterior_cov = noise_variance * trace_cov
        one = - trace_posterior_cov_div_var / 2
        three = - trace_K_inv_posterior_cov / 2
        four = - posterior_mean.T @ weight / 2
        five = (N * np.log(noise_variance) + log_det_cov) / 2
        six = N / 2
        seven = np.sum(Z)
        fx = one + three + four + five + six  + seven
        if verbose:
            print("one ", one)
            print("three ", three)
            print("four ", four)  # Sometimes largest contribution
            print("five ", five)
            print("six ", six)
            print("seven ", seven)
            print('fx = {}'.format(fx))
        return -fx

    def objectiveSS(
            self, N, posterior_mean, weight, trace_cov, trace_posterior_cov_div_var, Z,
            noise_variance,
            log_det_K, log_det_cov, verbose=False):
        """
        # TODO log_det_K cancels out of this calculation!!!
        Calculate fx, the variational lower bound of the log marginal
        likelihood.

        .. math::
                \mathcal{F(\theta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
                marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg int N: The number of datapoints.
        :arg m: The posterior mean.
        :type m: :class:`numpy.ndarray`
        :arg y: The posterior mean.
        :type y: :class:`numpy.ndarray`
        :arg K: The prior covariance.
        :type K: :class:`numpy.ndarray`
        :arg float noise_variance: The noise variance.
        :arg float log_det_K: The log determinant of the prior covariance.
        :arg float log_det_cov: The log determinant of (a factor in) the
            posterior covariance.
        :arg Z: The array of normalising constants.
        :type Z: :class:`numpy.ndarray`
        :return: fx
        :rtype: float
        """
        trace_K_inv_posterior_cov = noise_variance * trace_cov
        log_det_posterior_cov = log_det_K + N * np.log(noise_variance)\
            + log_det_cov  # or should this be negative?
        one = - trace_posterior_cov_div_var / 2
        two = - log_det_K / 2
        three = - trace_K_inv_posterior_cov / 2
        four = - posterior_mean.T @ weight / 2
        five = log_det_posterior_cov / 2
        six = N / 2
        seven = np.sum(Z)
        fx = one + two + three + four + five + six  + seven
        if verbose:
            print("one ", one)
            print("two ", two)
            print("three ", three)
            print("four ", four)  # Sometimes largest contribution
            print("five ", five)
            print("six ", six)
            print("seven ", seven)
            print('fx = {}'.format(fx))
        return -fx

    def objective_gradient(
            self, gx, intervals, cutpoints_ts, cutpoints_tplus1s, varphi,
            noise_variance, noise_std,
            m, weight, cov, trace_cov, partial_K_varphi, N,
            Z, norm_pdf_z1s, norm_pdf_z2s, trainables,
            numerical_stability=True, verbose=False):
        """
        Calculate gx, the jacobian of the variational lower bound of the log
        marginal likelihood at the VB equilibrium,

        .. math::
                \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

            where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`\theta` is the set of hyperparameters, :math:`h`,
            :math:`\Pi`, :math:`K`. #TODO

        :arg intervals: The vector of the first cutpoint and the intervals
            between cutpoints for unconstrained optimisation of the cutpoint
            parameters.
        :type intervals: :class:`numpy.ndarray`
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg float noise_variance: The noise variance.
        :arg float noise_std:
        :arg m: The posterior mean.
        :type m: :class:`numpy.ndarray`
        :arg cov: An intermediate matrix in calculating the posterior
            covariance, posterior_cov.
        :type cov: :class:`numpy.ndarray`
        :arg posterior_cov: The posterior covariance.
        :type posterior_cov: :class:`numpy.ndarray`
        :arg K_inv: The inverse of the prior covariance.
        :type K_inv: :class:`numpy.ndarray`
        :arg Z: The array of normalising constants.
        :type Z: :class:`numpy.ndarray`
        :arg bool numerical_stability: If the function is evaluated in a
            numerically stable way, default `True`.
        :return: fx
        :rtype: float
        """
        if trainables is not None:
            # For gx[0] -- ln\sigma  # TODO: currently seems analytically incorrect
            if trainables[0]:
                one = N - noise_variance * trace_cov
                sigma_dp = dp(m, cutpoints_ts, cutpoints_tplus1s, noise_std,
                    self.upper_bound, self.upper_bound2)
                two = - (1. / noise_std) * np.sum(sigma_dp)
                if verbose:
                    print("one ", one)
                    print("two ", two)
                    print("gx_sigma = ", one + two)
                gx[0] = one + two
            # For gx[1] -- \b_1
            if np.any(trainables[1:self.J]):  # TODO: analytic and numeric gradients do not match
                # TODO: treat these with numerical stability, or fix them
                temp_1s = np.divide(norm_pdf_z1s, Z)
                temp_2s = np.divide(norm_pdf_z2s, Z)
                idx = np.where(self.t_train == 0)  # TODO factor out
                gx[1] += np.sum(temp_1s[idx])
                for j in range(2, self.J):
                    idx = np.where(self.t_train == j - 1)  # TODO: factor it out seems inefficient. Is there a better way?
                    gx[j - 1] -= np.sum(temp_2s[idx])
                    gx[j] += np.sum(temp_1s[idx])
                # gx[self.J] -= 0  # Since J is number of classes
                gx[1:self.J] /= noise_std
                # For gx[2:self.J] -- ln\Delta^r
                gx[2:self.J] *= intervals
                if verbose:
                    print(gx[2:self.J])
            # For gx[self.J] -- s
            if trainables[self.J]:
                raise ValueError("TODO")
            # For kernel parameters
            if trainables[self.J + 1]:
                if numerical_stability is True:
                    # Update gx[-1], the partial derivative of the lower bound
                    # wrt the lengthscale. Using matrix inversion Lemma
                    one = (varphi / 2) * weight.T @ partial_K_varphi @ weight
                    # TODO: slower but what about @jit compile CPU or GPU?
                    # D = solve_triangular(
                    #     L_cov.T, partial_K_varphi, lower=True)
                    # D_inv = solve_triangular(L_cov, D, lower=False)
                    # two = - (varphi / 2) * np.trace(D_inv)
                    two = - (varphi / 2) * np.einsum(
                        'ij, ji ->', partial_K_varphi, cov)
                    gx[self.J + 1] = one + two
                    if verbose:
                        print("one", one)
                        print("two", two)
                        print("gx = {}".format(gx[self.J + 1]))
        return -gx

    def grid_over_hyperparameters(
            self, domain, res, trainables=None, posterior_mean_0=None,
            verbose=False, steps=100):
        """
        TODO: Can this be moved to a plot.py
        Return meshgrid values of fx and directions of gx over hyperparameter
        space.

        The particular hyperparameter space is inferred from the user inputs
        - the rule is that if any of the
        variables are None, then those are the variables to grid over. We can
        only visualise these surfaces for
        maximum of 2 variables, so the number of combinations is Mc2 + Mc1
        where M is the total no. of hyperparameters.

        Special cases are frequent: log and non log variables. 2 axis vs 1
        axis objective function, calculate
        new Gram matrix or not. So the simplest way is to combinate manually.
        """
        (
        x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        thetas,
        fxs, gxs, gx_0,
        intervals, trainables_where) = self._grid_over_hyperparameters_initiate(
            res, domain, trainables, self.cutpoints)
        error = np.inf
        fx_old = np.inf
        for i, theta in enumerate(thetas):
            self._grid_over_hyperparameters_update(
                theta, trainables, self.cutpoints)
            # Reset error and posterior mean
            iteration = 0
            error = np.inf
            fx_old = np.inf
            # TODO: reset m_0 is None?
            # Convergence is sometimes very fast so this may not be necessary
            while error / steps > self.EPS:
                iteration += 1
                (posterior_mean_0, weight, y, p, *_) = self.approximate(
                    steps, posterior_mean_0=posterior_mean_0,
                    first_step=0, write=False)
                (Z,
                norm_pdf_z1s,
                norm_pdf_z2s,
                z1s,
                z2s,
                *_) = truncated_norm_normalising_constant(
                    self.cutpoints_ts, self.cutpoints_tplus1s,
                    self.noise_std, posterior_mean_0, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean_0, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
                error = np.abs(fx_old - fx)
                fx_old = fx
                print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(thetas)))
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.cutpoints_ts,
                self.cutpoints_tplus1s, self.kernel.varphi,
                self.noise_variance, self.noise_std, posterior_mean_0,
                weight, self.cov, self.trace_cov,
                self.partial_K_varphi, self.N, Z,
                norm_pdf_z1s, norm_pdf_z2s, trainables,
                numerical_stability=True, verbose=False)
            fxs[i] = fx
            gxs[i] = gx[trainables_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print(
                    "cutpoints={}, varphi={}, noise_variance={}, variance={}, "
                    "fx={}, gx={}".format(
                    self.cutpoints, self.kernel.varphi, self.noise_variance,
                    self.kernel.variance, fx, gxs[i]))
        if x2s is not None:
            return (
                fxs.reshape((len(x1s), len(x2s))), gxs,
                xx, yy, xlabel, ylabel, xscale, yscale)
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def approximate_posterior(
            self, phi, trainables, steps=None, first_step=0, max_iter=2,
            return_reparameterised=False, verbose=False):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :arg trainables:
        :arg first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, steps, error, iteration, trainables_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, steps)
        fx_old = np.inf
        posterior_mean = None
        # Convergence is sometimes very fast so this may not be necessary
        while error / steps > self.EPS and iteration < max_iter:
            iteration += 1
            (posterior_mean, weight, *_) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=False)
            (Z, norm_pdf_z1s, norm_pdf_z2s,
                    *_ )= truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS)
            if self.kernel.varphi_hyperhyperparameters is not None:
                fx = self.objective(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
            else:
                # Only terms in f matter
                fx = -0.5 * posterior_mean.T @ weight - np.sum(Z)
            error = np.abs(fx_old - fx)
            fx_old = fx
            if verbose:
                print("({}), error={}".format(iteration, error))
        fx = self.objective(
                    self.N, posterior_mean, weight, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_cov)
        gx = self.objective_gradient(
                gx.copy(), intervals, self.cutpoints_ts,
                self.cutpoints_tplus1s,
                self.kernel.varphi, self.noise_variance, self.noise_std,
                posterior_mean, weight, self.cov, self.trace_cov,
                self.partial_K_varphi, self.N, Z,
                norm_pdf_z1s, norm_pdf_z2s, trainables,
                numerical_stability=True, verbose=False)
        gx = gx[trainables_where]
        if verbose:
            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (self.cov, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (
                self.noise_variance * self.K @ self.cov, False)
        elif return_reparameterised is None:
            return fx, gx


class EPGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Expectation Propagation
    (EP) approximation.

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
        self, cutpoints, noise_variance=1.0, *args, **kwargs):
        # cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`EPGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to 1.0, default `None`.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)
        # if cutpoints_hyperparameters is not None:
        #     warnings.warn("cutpoints_hyperparameters set as {}".format(cutpoints_hyperparameters))
        #     self.cutpoints_hyperparameters = cutpoints_hyperparameters
        # else:
        #     self.cutpoints_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        self.EPS = 1e-4  # perhaps too large
        # self.EPS = 1e-6  # Decreasing EPS will lead to more accurate solutions but a longer convergence time.
        self.EPS_2 = self.EPS**2
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.hyperparameters_update(cutpoints=cutpoints, noise_variance=noise_variance)

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
        :arg grad_Z_wrt_cavity_mean_0: Initialisation of the EP weights,
            which are gradients of the approximate marginal
            likelihood wrt to the 'cavity distribution mean'. If `None`
            then initialised to zeros, default `None`.
        :type grad_Z_wrt_cavity_mean_0: :class:`numpy.ndarray`
        :return: Containers for the approximate posterior means of parameters
            and hyperparameters.
        :rtype: (12,) tuple.
        """
        if posterior_cov_0 is None:
            # The first EP approximation before data-update is the GP prior cov
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
        grad_Z_wrt_cavity_mean_0 = np.zeros(self.N)  # Initialisation
        posterior_means = []
        posterior_covs = []
        mean_EPs = []
        amplitude_EPs = []
        precision_EPs = []
        approximate_marginal_likelihoods = []
        # random ints
        permutation = np.random.choice(self.N, self.N, replace=False)
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_marginal_likelihoods)
        return (posterior_mean_0, posterior_cov_0, mean_EP_0,
                precision_EP_0, amplitude_EP_0, grad_Z_wrt_cavity_mean_0,
                permutation, containers, error)

    def approximate(
            self, trainables, posterior_mean_0=None, posterior_cov_0=None,
            mean_EP_0=None, precision_EP_0=None, amplitude_EP_0=None,
            write=False):
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

        :arg int steps: The number of iterations the Approximator takes.
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
        :arg int first_step: The first step. Useful for burn in algorithms.
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
        (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, grad_Z_wrt_cavity_mean,
                permutation, containers, error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)
        (posterior_means, posterior_covs, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_log_marginal_likelihoods) = containers
        for index in trainables:
            target = self.t_train[index]
            (cavity_mean_n, cavity_variance_n,
            posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old
                    )= self._remove(
                posterior_cov[index, index], posterior_mean[index],
                mean_EP[index], precision_EP[index], amplitude_EP[index])
            # Tilt/ moment match
            (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n, posterior_covariance_n_new,
            z1, z2, nu_n) = self._include(
                target, cavity_mean_n, cavity_variance_n,
                self.cutpoints[target], self.cutpoints[target + 1],
                self.noise_variance)
            # Update EP weight (alpha)
            grad_Z_wrt_cavity_mean[index] = grad_Z_wrt_cavity_mean_n
            diff = precision_EP_n - precision_EP_n_old
            if (np.abs(diff) > self.EPS
                    and Z_n > self.EPS
                    and precision_EP_n > 0.0
                    and posterior_covariance_n_new > 0.0):
                posterior_mean, posterior_cov = self._update(
                    index, posterior_mean, posterior_cov,
                    posterior_mean_n, posterior_variance_n,
                    mean_EP_n_old, precision_EP_n_old,
                    grad_Z_wrt_cavity_mean_n, diff)
                # Update EP parameters
                error += (diff**2
                          + (mean_EP_n - mean_EP_n_old)**2
                          + (amplitude_EP_n - amplitude_EP_n_old)**2)
                precision_EP[index] = precision_EP_n
                mean_EP[index] = mean_EP_n
                amplitude_EP[index] = amplitude_EP_n
                if write:
                    # approximate_log_marginal_likelihood = \
                    # self._approximate_log_marginal_likelihood(
                    # posterior_cov, precision_EP, mean_EP)
                    # posterior_means.append(posterior_mean)
                    # posterior_covs.append(posterior_cov)
                    # mean_EPs.append(mean_EP)
                    # precision_EPs.append(precision_EP)
                    # amplitude_EPs.append(amplitude_EP)
                    # approximate_log_marginal_likelihood.append(
                    #   approximate_marginal_log_likelihood)
                    pass
            else:
                if precision_EP_n < 0.0 or posterior_covariance_n_new < 0.0:
                    print(
                        "Skip {} update z1={}, z2={}, nu={} p_new={},"
                        " p_old={}.\n".format(
                        index, z1, z2, nu_n,
                        precision_EP_n, precision_EP_n_old))
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_log_marginal_likelihoods)
        return (
            error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP, containers)

    def _remove(
            self, posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old):
        """
        Calculate the product of approximate posterior factors with the current
        index removed.

        This is called the cavity distribution,
        "a bit like leaving a hole in the dataset".

        :arg float posterior_variance_n: Variance of latent function at index.
        :arg float posterior_mean_n: The state of the approximate posterior
            mean.
        :arg float mean_EP_n: The state of the individual (site) mean.
        :arg precision_EP_n: The state of the individual (site) variance.
        :arg amplitude_EP_n: The state of the individual (site) amplitudes.
        :returns: A (8,) tuple containing cavity mean and variance, and old
            site states.
        """
        if posterior_variance_n > 0:
            cavity_variance_n = posterior_variance_n / (
                1 - posterior_variance_n * precision_EP_n_old)
            if cavity_variance_n > 0:
                cavity_mean_n = (posterior_mean_n
                    + cavity_variance_n * precision_EP_n_old * (
                        posterior_mean_n - mean_EP_n_old))
            else:
                raise ValueError(
                    "cavity_variance_n must be non-negative (got {})".format(
                        cavity_variance_n))
        else:
            raise ValueError(
                "posterior_cov_nn must be non-negative (got {})".format(
                    posterior_variance_n))
        return (cavity_mean_n, cavity_variance_n,
            posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old)

    def _assert_valid_values(self, nu_n, variance, cavity_mean_n,
            cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1, norm_pdf_z2,
            grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n):
        if math.isnan(grad_Z_wrt_cavity_mean_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_mean is nan (got {})".format(
                grad_Z_wrt_cavity_mean_n))
        if math.isnan(grad_Z_wrt_cavity_variance_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_variance is nan (got {})".format(
                    grad_Z_wrt_cavity_variance_n))
        if nu_n <= 0:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError("nu_n must be positive (got {})".format(nu_n))
        if nu_n > 1.0 / variance + self.EPS:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "nu_n must be less than 1.0 / (cavity_variance_n + "
                "noise_variance) = {}, got {}".format(
                    1.0 / variance, nu_n))

    def _include(
            self, target, cavity_mean_n, cavity_variance_n,
            cutpoints_t, cutpoints_tplus1, noise_variance,
            numerically_stable=False):
        """
        Update the approximate posterior by incorporating the message
        p(t_i|m_i) into Q^{\i}(\bm{f}).
        Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)
        This includes one true-observation likelihood, and 'tilts' the
        approximation towards the true posterior. It updates the approximation
        to the true posterior by minimising a moment-matching KL divergence
        between the tilted distribution and the posterior distribution. This
        gives us an approximate posterior in the approximating family. The
        update to posterior_cov is a rank-1 update (see the outer product of
        two 1d vectors), and so it essentially constructs a piecewise low rank
        approximation to the GP posterior covariance matrix, until convergence
        (by which point it will no longer be low rank).
        :arg int target: The ordinal class index of the current site
            (the class of the datapoint that is "left out").
        :arg float cavity_mean_n: The cavity mean of the current site.
        :arg float cavity_variance_n: The cavity variance of the current site.
        :arg float cutpoints_t: The upper cutpoint parameters.
        :arg float cutpoints_tplus1: The lower cutpoint parameter.
        :arg float noise_variance: Initialisation of noise variance. If
            `None` then initialised to one, default `None`.
        :arg bool numerically_stable: Boolean variable for assert valid
            numerical values. Default `False'.
        :returns: A (10,) tuple containing cavity mean and variance, and old
            site states.
        """
        variance = cavity_variance_n + noise_variance
        std_dev = np.sqrt(variance)
        # Compute Z
        norm_cdf_z2 = 0.0
        norm_cdf_z1 = 1.0
        norm_pdf_z1 = 0.0
        norm_pdf_z2 = 0.0
        z1 = 0.0
        z2 = 0.0
        if target == 0:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z1_abs = np.abs(z1)
            if z1_abs > self.upper_bound:
                z1 = np.sign(z1) * self.upper_bound
            Z_n = norm_cdf(z1) - norm_cdf_z2
            norm_pdf_z1 = norm_z_pdf(z1)
        elif target == self.J - 1:
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            z2_abs = np.abs(z2)
            if z2_abs > self.upper_bound:
                z2 = np.sign(z2) * self.upper_bound
            Z_n = norm_cdf_z1 - norm_cdf(z2)
            norm_pdf_z2 = norm_z_pdf(z2)
        else:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            Z_n = norm_cdf(z1) - norm_cdf(z2)
            norm_pdf_z1 = norm_z_pdf(z1)
            norm_pdf_z2 = norm_z_pdf(z2)
        if Z_n < self.EPS:
            if np.abs(np.exp(-0.5*z1**2 + 0.5*z2**2) - 1.0) > self.EPS**2:
                grad_Z_wrt_cavity_mean_n = (z1 * np.exp(
                        -0.5*z1**2 + 0.5*z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5 * z1 ** 2) + 0.5 * z2 ** 2) - 1.0)
                        * variance
                )
                grad_Z_wrt_cavity_variance_n = (
                    -1.0 + (z1**2 + 0.5 * z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5*z1**2 + 0.5 * z2**2) - 1.0)
                        * 2.0 * variance)
                )
                grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
                nu_n = (
                    grad_Z_wrt_cavity_mean_n_2
                    - 2.0 * grad_Z_wrt_cavity_variance_n)
            else:
                grad_Z_wrt_cavity_mean_n = 0.0
                grad_Z_wrt_cavity_mean_n_2 = 0.0
                grad_Z_wrt_cavity_variance_n = -(
                    1.0 - self.EPS)/(2.0 * variance)
                nu_n = (1.0 - self.EPS) / variance
                warnings.warn(
                    "Z_n must be greater than tolerance={} (got {}): "
                    "SETTING to Z_n to approximate value\n"
                    "z1={}, z2={}".format(
                        self.EPS, Z_n, z1, z2))
            if nu_n >= 1.0 / variance:
                nu_n = (1.0 - self.EPS) / variance
            if nu_n <= 0.0:
                nu_n = self.EPS * variance
        else:
            grad_Z_wrt_cavity_variance_n = (
                - z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (
                    2.0 * variance * Z_n)  # beta
            grad_Z_wrt_cavity_mean_n = (
                - norm_pdf_z1 + norm_pdf_z2) / (
                    std_dev * Z_n)  # alpha/gamma
            grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
            nu_n = (grad_Z_wrt_cavity_mean_n_2
                - 2.0 * grad_Z_wrt_cavity_variance_n)
        # Update alphas
        if numerically_stable:
            self._assert_valid_values(
                nu_n, variance, cavity_mean_n, cavity_variance_n, target,
                z1, z2, Z_n, norm_pdf_z1,
                norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                grad_Z_wrt_cavity_mean_n)
        # posterior_mean_n_new = (  # Not used for anything
        #     cavity_mean_n + cavity_variance_n * grad_Z_wrt_cavity_mean_n)
        posterior_covariance_n_new = (
            cavity_variance_n - cavity_variance_n**2 * nu_n)
        precision_EP_n = nu_n / (1.0 - cavity_variance_n * nu_n)
        mean_EP_n = cavity_mean_n + grad_Z_wrt_cavity_mean_n / nu_n
        amplitude_EP_n = Z_n * np.sqrt(
            cavity_variance_n * precision_EP_n + 1.0) * np.exp(
                0.5 * grad_Z_wrt_cavity_mean_n_2 / nu_n)
        return (
            mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n,
            posterior_covariance_n_new, z1, z2, nu_n)

    def _update(
            self, index, posterior_mean, posterior_cov,
            posterior_mean_n, posterior_variance_n,
            mean_EP_n_old, precision_EP_n_old,
            grad_Z_wrt_cavity_mean_n, diff):
        """
        Update the posterior mean and covariance.

        Projects the tilted distribution on to an approximating family.
        The update for the t_n is a rank-1 update. Constructs a low rank
        approximation to the GP posterior covariance matrix.

        :arg int index: The index of the current likelihood (the index of the
            datapoint that is "left out").
        :arg float mean_EP_n_old: The state of the individual (site) mean (N,).
        :arg posterior_cov: The current approximate posterior covariance
            (N, N).
        :type posterior_cov: :class:`numpy.ndarray`
        :arg float posterior_variance_n: The current approximate posterior
            site variance.
        :arg float posterior_mean_n: The current site approximate posterior
            mean.
        :arg float precision_EP_n_old: The state of the individual (site)
            variance (N,).
        :arg float grad_Z_wrt_cavity_mean_n: The gradient of the log
            normalising constant with respect to the site cavity mean
            (The EP "weight").
        :arg float posterior_mean_n_new: The state of the site approximate
            posterior mean.
        :arg float posterior_covariance_n_new: The state of the site
            approximate posterior variance.
        :arg float diff: The differance between precision_EP_n and
            precision_EP_n_old.
        :returns: The updated approximate posterior mean and covariance.
        :rtype: tuple (`numpy.ndarray`, `numpy.ndarray`)
        """
        rho = diff / (1 + diff * posterior_variance_n)
        eta = (
            grad_Z_wrt_cavity_mean_n
            + precision_EP_n_old * (posterior_mean_n - mean_EP_n_old)) / (
                1.0 - posterior_variance_n * precision_EP_n_old)
        # Update posterior mean and rank-1 covariance
        a_n = posterior_cov[:, index]
        posterior_mean += eta * a_n
        posterior_cov = posterior_cov - rho * np.outer(
            a_n, a_n) 
        return posterior_mean, posterior_cov

    def _approximate_log_marginal_likelihood(
            self, posterior_cov, precision_EP, amplitude_EP, mean_EP,
            numerical_stability):
        """
        Calculate the approximate log marginal likelihood.
        TODO: need to finish this. Probably not useful if using EP.

        :arg posterior_cov: The approximate posterior covariance.
        :type posterior_cov:
        :arg precision_EP: The state of the individual (site) variance (N,).
        :type precision_EP:
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :type amplitude EP:
        :arg mean_EP: The state of the individual (site) mean (N,).
        :type mean_EP:
        :arg bool numerical_stability: If the calculation is made in a
            numerically stable manner.
        """
        precision_matrix = np.diag(precision_EP)
        inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
        log_amplitude_EP = np.log(amplitude_EP)
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp\
                - temp.T @ mean_EP
        if numerical_stability is True:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.trace(
                    np.log(inverse_precision_matrix)))
            approximate_marginal_likelihood = np.add(
                    approximate_marginal_likelihood, B/2)
            approximate_marginal_likelihood = np.subtract(
                approximate_marginal_likelihood, 0.5 * np.trace(
                    np.log(self.K + inverse_precision_matrix)))
            return np.sum(approximate_marginal_likelihood)
        else:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.log(np.linalg.det(
                    inverse_precision_matrix)))  # TODO: use log det C trick
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, B/2
            )
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, 0.5 * np.log(
                    np.linalg.det(self.K + inverse_precision_matrix))
            )  # TODO: use log det C trick
            return np.sum(approximate_marginal_likelihood)

    def grid_over_hyperparameters(
            self, domain, res,
            trainables=None,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None, amplitude_EP_0=None,
            first_step=0, write=False, verbose=False):
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        trainables.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        thetas, fxs,
        gxs, gx_0, intervals,
        trainables_where) = self._grid_over_hyperparameters_initiate(
            res, domain, trainables, self.cutpoints)
        for i, phi in enumerate(thetas):
            self._grid_over_hyperparameters_update(
                phi, trainables, self.cutpoints)
            if verbose:
                print(
                    "cutpoints_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "variance_0 = {}".format(
                        self.cutpoints, self.kernel.varphi, self.noise_variance,
                        self.kernel.variance))
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            posterior_cov = posterior_cov_0
            mean_EP = mean_EP_0
            precision_EP = precision_EP_0
            amplitude_EP = amplitude_EP_0
            while error / steps > self.EPS**2:
                iteration += 1
                (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
                 precision_EP, amplitude_EP, containers) = self.approximate(
                    steps, posterior_mean_0=posterior_mean, posterior_cov_0=posterior_cov,
                    mean_EP_0=mean_EP, precision_EP_0=precision_EP,
                    amplitude_EP_0=amplitude_EP,
                    first_step=first_step, write=False)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(thetas)))
            weight, precision_EP, L_cov, cov = self.compute_weights(
                precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
            t1, t2, t3, t4, t5 = self.compute_integrals_vector(
                np.diag(posterior_cov), posterior_mean, self.noise_variance)
            fx = self.objective(
                precision_EP, posterior_mean,
                t1, L_cov, cov, weight)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi,
                self.noise_variance,
                t2, t3, t4, t5, cov, weight, trainables)
            gxs[i] = gx[trainables_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(
                    self.kernel.varphi, self.noise_variance, fx))
        if x2s is not None:
            return (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
        else:
            return (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    def approximate_posterior(
            self, phi, trainables, steps=None,
            posterior_mean_0=None, return_reparameterised=False,
            posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=0, verbose=True):
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
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx the objective and gx the objective gradient
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, steps, error, iteration, trainables_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, steps)
        posterior_mean = posterior_mean_0
        posterior_cov = posterior_cov_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP,
            containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                posterior_cov_0=posterior_cov, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP,
                amplitude_EP_0=amplitude_EP,
                first_step=first_step, write=False)
        # TODO: this part requires an inverse, could it be sparsified
        # by putting q(f) = p(f_m) R(f_n). This is probably how FITC works
        (weight, precision_EP, L_cov, cov) = self.compute_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        # Try optimisation routine
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            np.diag(posterior_cov), posterior_mean, self.noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1,
            L_cov, cov, weight)
        gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, self.noise_variance,
            t2, t3, t4, t5, cov, weight, trainables)
        gx = gx[np.where(trainables != 0)]
        if verbose:
            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (posterior_cov, False)
        elif return_reparameterised is None:
            return fx, gx

    def compute_integrals_vector(
            self, posterior_variance, posterior_mean, noise_variance):
        """
        Compute the integrals required for the gradient evaluation.
        """
        noise_std = np.sqrt(noise_variance)
        mean_ts = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_ts) / (
                noise_variance + posterior_variance)
        mean_tplus1s = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_tplus1s) / (
                noise_variance + posterior_variance)
        sigma = np.sqrt(
            (noise_variance * posterior_variance) / (
            noise_variance + posterior_variance))
        a_ts = mean_ts - 5.0 * sigma
        b_ts = mean_ts + 5.0 * sigma
        h_ts = b_ts - a_ts
        a_tplus1s = mean_tplus1s - 5.0 * sigma
        b_tplus1s = mean_tplus1s + 5.0 * sigma
        h_tplus1s = b_tplus1s - a_tplus1s
        y_0 = np.zeros((20, self.N))
        t1 = fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_variance,
                self.cutpoints_ts, self.cutpoints_tplus1s,
                noise_std, self.EPS, self.EPS_2, self.N)
        t2 = fromb_t2_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t2[self.indices_where_0] = 0.0
        t3 = fromb_t3_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t3[self.indices_where_J_1] = 0.0
        t4 = fromb_t4_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t4[self.indices_where_J_1] = 0.0
        t5 = fromb_t5_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t5[self.indices_where_0] = 0.0
        return t1, t2, t3, t4, t5

    def objective(
            self, precision_EP, posterior_mean, t1, L_cov, cov,
            weights):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\theta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg precision_EP:
        :type precision_EP:
        :arg posterior_mean:
        :type posterior_mean:
        :arg t1:
        :type t1:
        :arg L_cov:
        :type L_cov:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :returns: fx
        :rtype: float
        """
        # Fill possible zeros in with machine precision
        precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        fx = -np.sum(np.log(np.diag(L_cov)))  # log det cov
        fx -= 0.5 * posterior_mean.T @ weights
        fx -= 0.5 * np.sum(np.log(precision_EP))
        # cov = L^{-1} L^{-T}  # requires a backsolve with the identity
        # TODO: check if there is a simpler calculation that can be done
        fx -= 0.5 * np.sum(np.divide(np.diag(cov), precision_EP))
        fx += np.sum(t1)
        # Regularisation - penalise large varphi (overfitting)
        # fx -= 0.1 * self.kernel.varphi
        return -fx

    def objective_gradient(
            self, gx, intervals, varphi, noise_variance,
            t2, t3, t4, t5, cov, weights, trainables):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

            where :math:`F(\theta)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\theta` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg intervals:
        :type intervals:
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg t2:
        :type t2:
        :arg t3:
        :type t3:
        :arg t4:
        :type t4:
        :arg t5:
        :type t5:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :return: gx
        :rtype: float
        """
        if trainables is not None:
            # Update gx
            if trainables[0]:
                # For gx[0] -- ln\sigma
                gx[0] = np.sum(t5 - t4)
                # gx[0] *= -0.5 * noise_variance  # This is a typo in the Chu code
                gx[0] *= np.sqrt(noise_variance)
            # For gx[1] -- \b_1
            if trainables[1]:
                gx[1] = np.sum(t3 - t2)
            # For gx[2] -- ln\Delta^r
            for j in range(2, self.J):
                if trainables[j]:
                    targets = self.t_train[self.grid]
                    gx[j] = np.sum(t3[targets == j - 1])
                    gx[j] -= np.sum(t2[targets == self.J - 1])
                    # TODO: check this, since it may be an `else` condition!!!!
                    gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                    gx[j] *= intervals[j - 2]
            # For gx[self.J] -- variance
            if trainables[self.J]:
                # For gx[self.J] -- s
                # TODO: Need to check this is correct: is it directly analogous to
                # gradient wrt log varphi?
                partial_K_s = self.kernel.kernel_partial_derivative_s(
                    self.X_train, self.X_train)
                # VC * VC * a' * partial_K_varphi * a / 2
                gx[self.J] = varphi * 0.5 * weights.T @ partial_K_s @ weights  # That's wrong. not the same calculation.
                # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                gx[self.J] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_s))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[self.J] -= 0.1 * varphi
                gx[self.J] *= 2.0  # since varphi = kappa / 2
            # For gx[self.J + 1] -- varphi
            if trainables[self.J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    self.X_train, self.X_train)
                # elif 1:
                #     gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights
                # TODO: This needs fixing/ checking vs original code
                if 0:
                    for l in range(self.kernel.L):
                        K = self.kernel.num_hyperparameters[l]
                        KK = 0
                        for k in range(K):
                            gx[self.J + KK + k] = varphi[l] * 0.5 * weights.T @ partial_K_varphi[l][k] @ weights
                        KK += K
                else:
                    # VC * VC * a' * partial_K_varphi * a / 2
                    gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights  # That's wrong. not the same calculation.
                    # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                    gx[self.J + 1] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_varphi))
                    # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                    # gx[self.J] -= 0.1 * varphi
        return -gx

    def approximate_evidence(self, mean_EP, precision_EP, amplitude_EP, posterior_cov):
        """
        TODO: check and return line could be at risk of overflow
        Compute the approximate evidence at the EP solution.

        :return:
        """
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp - np.multiply(
            temp, mean_EP)
        Pi_inv = np.diag(1. / precision_EP)
        return (
            np.prod(
                amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
                / np.sqrt(np.linalg.det(np.add(Pi_inv, self.K))))

    def compute_weights(
        self, precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L_cov=None, cov=None, numerically_stable=False):
        """
        TODO: There may be an issue, where grad_Z_wrt_cavity_mean is updated
        when it shouldn't be, on line 2045.

        Compute the regression weights required for the gradient evaluation,
        and check that they are in equilibrium with
        the gradients of Z wrt cavity means.

        A matrix inverse is always required to evaluate fx.

        :arg precision_EP:
        :arg mean_EP:
        :arg grad_Z_wrt_cavity_mean:
        :arg L_cov: . Default `None`.
        :arg cov: . Default `None`.
        """
        if np.any(precision_EP == 0.0):
            # TODO: Only check for equilibrium if it has been updated in this swipe
            warnings.warn("Some sample(s) have not been updated.\n")
            precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        Pi_inv = np.diag(1. / precision_EP)
        if L_cov is None or cov is None:
            (L_cov, lower) = cho_factor(
                Pi_inv + self.K)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.N), lower=True)
            # TODO It is necessary to do this triangular solve to get
            # diag(cov) for the lower bound on the marginal likelihood
            # calculation. Note no tf implementation for diag(A^{-1}) yet.
            cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        if numerically_stable:
            # This is 3-4 times slower on CPU,
            # what about with jit compiled CPU or GPU?
            # Is this ever more stable than a matmul by the inverse?
            g = cho_solve((L_cov, False), mean_EP)
            weight = cho_solve((L_cov.T, True), g)
        else:
            weight = cov @ mean_EP
        if np.any(
            np.abs(weight - grad_Z_wrt_cavity_mean) > np.sqrt(self.EPS)):
            warnings.warn("Fatal error: the weights are not in equilibrium wit"
                "h the gradients".format(
                    weight, grad_Z_wrt_cavity_mean))
        return weight, precision_EP, L_cov, cov


class PEPGP(EPGP):
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
        self, alpha=0.8, minibatch_size=None, *args, **kwargs):
        # cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
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
        self.minibatch_size = minibatch_size  # TODO: could put as approximate argument
        self.alpha = alpha  # TODO: could put as approximate argument
        super().__init__(*args, **kwargs)

    def update_pep_variables(self, Kuuinv, posterior_mean, posterior_cov):
        """TODO: collapse"""
        return Kuuinv @ posterior_mean, Kuuinv @ (Kuuinv - posterior_cov)

    def compute_posterior(self, Kuu, gamma, beta):
        """TODO: collapse"""
        return Kuu @ gamma, Kuu - Kuu @ (beta @ Kuu)

    def _delete(
            self, posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old):
        pass 

    def _remove(self, target, index):
        pass

    def _project(self):
        pass

    def _include(self):
        pass

    def _update(self):
        pass

    def approximate(self, steps, posterior_mean_0, posterior_cov_0=None,
            first_step=0, write=False):


    def approximate(
            self, steps, posterior_mean_0=None, posterior_cov_0=None,
            mean_EP_0=None, precision_EP_0=None, amplitude_EP_0=None,
            first_step=0, write=False):
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

        :arg int steps: The number of iterations the Approximator takes.
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
        :arg int first_step: The first step. Useful for burn in algorithms.
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
        (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, grad_Z_wrt_cavity_mean,
                permutation, containers, error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)
        (posterior_means, posterior_covs, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_log_marginal_likelihoods) = containers
        for step in trange(first_step, first_step + steps,
                        desc="PEP GP approximator progress",
                        unit="iterations", disable=True):
            index = permutation[step]
            target = self.t_train[index]
            # Find the mean and variance of the leave-one-out
            # posterior distribution Q^{\backslash i}(\bm{f})
            # Also, this factors out some fancy indexing
            (posterior_mean_n, posterior_variance_n, cavity_mean_n,
            cavity_variance_n, mean_EP_n_old,
            precision_EP_n_old, amplitude_EP_n_old) = self._remove(
                posterior_cov[index, index], posterior_mean[index],
                mean_EP[index], precision_EP[index], amplitude_EP[index])
            # Tilt/ moment match
            (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n, posterior_mean_n_new,
            posterior_covariance_n_new, z1, z2, nu_n) = self._include(
                target, cavity_mean_n, cavity_variance_n,
                self.cutpoints[target], self.cutpoints[target + 1],
                self.noise_variance)
            # Update EP weight (alpha)
            grad_Z_wrt_cavity_mean[index] = grad_Z_wrt_cavity_mean_n
            #print(grad_Z_wrt_cavity_mean)
            diff = precision_EP_n - precision_EP_n_old
            if (np.abs(diff) > self.EPS
                    and Z_n > self.EPS
                    and precision_EP_n > 0.0
                    and posterior_covariance_n_new > 0.0):
                # Update posterior mean and rank-1 covariance
                posterior_cov, posterior_mean = self._update(
                    index, mean_EP_n_old, posterior_cov,
                    posterior_mean_n, posterior_variance_n,
                    precision_EP_n_old, grad_Z_wrt_cavity_mean_n,
                    posterior_mean_n_new, posterior_mean,
                    posterior_covariance_n_new, diff)
                # Update EP parameters
                precision_EP[index] = precision_EP_n
                mean_EP[index] = mean_EP_n
                amplitude_EP[index] = amplitude_EP_n
                error += (diff**2
                          + (mean_EP_n - mean_EP_n_old)**2
                          + (amplitude_EP_n - amplitude_EP_n_old)**2)
                if write:
                    # approximate_log_marginal_likelihood = \
                    # self._approximate_log_marginal_likelihood(
                    # posterior_cov, precision_EP, mean_EP)
                    posterior_means.append(posterior_mean)
                    posterior_covs.append(posterior_cov)
                    mean_EPs.append(mean_EP)
                    precision_EPs.append(precision_EP)
                    amplitude_EPs.append(amplitude_EP)
                    # approximate_log_marginal_likelihood.append(
                    #   approximate_marginal_log_likelihood)
            else:
                if precision_EP_n < 0.0 or posterior_covariance_n_new < 0.0:
                    print(
                        "Skip {} update z1={}, z2={}, nu={} p_new={},"
                        " p_old={}.\n".format(
                        index, z1, z2, nu_n,
                        precision_EP_n, precision_EP_n_old))
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_log_marginal_likelihoods)
        return (
            error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP, containers)
        # TODO: are there some other inputs missing here?
        # error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
        #  precision_EP, amplitude_EP, *_

    def _remove(
            self, posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old):
        """
        Calculate the product of approximate posterior factors with the current
        index removed.

        This is called the cavity distribution,
        "a bit like leaving a hole in the dataset".

        :arg float posterior_variance_n: Variance of latent function at index.
        :arg float posterior_mean_n: The state of the approximate posterior
            mean.
        :arg float mean_EP_n: The state of the individual (site) mean.
        :arg precision_EP_n: The state of the individual (site) variance.
        :arg amplitude_EP_n: The state of the individual (site) amplitudes.
        :returns: A (8,) tuple containing cavity mean and variance, and old
            site states.
        """
        if posterior_variance_n > 0:
            cavity_variance_n = posterior_variance_n / (
                1 - posterior_variance_n * precision_EP_n_old)
            if cavity_variance_n > 0:
                cavity_mean_n = (posterior_mean_n
                    + cavity_variance_n * precision_EP_n_old * (
                        posterior_mean_n - mean_EP_n_old))
            else:
                raise ValueError(
                    "cavity_variance_n must be non-negative (got {})".format(
                        cavity_variance_n))
        else:
            raise ValueError(
                "posterior_cov_nn must be non-negative (got {})".format(
                    posterior_variance_n))
        return (
            posterior_mean_n, posterior_variance_n,
            cavity_mean_n, cavity_variance_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old)

    def _assert_valid_values(self, nu_n, variance, cavity_mean_n,
            cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1, norm_pdf_z2,
            grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n):
        if math.isnan(grad_Z_wrt_cavity_mean_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_mean is nan (got {})".format(
                grad_Z_wrt_cavity_mean_n))
        if math.isnan(grad_Z_wrt_cavity_variance_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_variance is nan (got {})".format(
                    grad_Z_wrt_cavity_variance_n))
        if nu_n <= 0:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError("nu_n must be positive (got {})".format(nu_n))
        if nu_n > 1.0 / variance + self.EPS:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "nu_n must be less than 1.0 / (cavity_variance_n + "
                "noise_variance) = {}, got {}".format(
                    1.0 / variance, nu_n))
        return 0

    def _include(
            self, target, cavity_mean_n, cavity_variance_n,
            cutpoints_t, cutpoints_tplus1, noise_variance,
            numerically_stable=False):
        """
        Update the approximate posterior by incorporating the message
        p(t_i|m_i) into Q^{\i}(\bm{f}).
        Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)
        This includes one true-observation likelihood, and 'tilts' the
        approximation towards the true posterior. It updates the approximation
        to the true posterior by minimising a moment-matching KL divergence
        between the tilted distribution and the posterior distribution. This
        gives us an approximate posterior in the approximating family. The
        update to posterior_cov is a rank-1 update (see the outer product of
        two 1d vectors), and so it essentially constructs a piecewise low rank
        approximation to the GP posterior covariance matrix, until convergence
        (by which point it will no longer be low rank).
        :arg int target: The ordinal class index of the current site
            (the class of the datapoint that is "left out").
        :arg float cavity_mean_n: The cavity mean of the current site.
        :arg float cavity_variance_n: The cavity variance of the current site.
        :arg float cutpoints_t: The upper cutpoint parameters.
        :arg float cutpoints_tplus1: The lower cutpoint parameter.
        :arg float noise_variance: Initialisation of noise variance. If
            `None` then initialised to one, default `None`.
        :arg bool numerically_stable: Boolean variable for assert valid
            numerical values. Default `False'.
        :returns: A (10,) tuple containing cavity mean and variance, and old
            site states.
        """
        variance = cavity_variance_n + noise_variance
        std_dev = np.sqrt(variance)
        # Compute Z
        norm_cdf_z2 = 0.0
        norm_cdf_z1 = 1.0
        norm_pdf_z1 = 0.0
        norm_pdf_z2 = 0.0
        z1 = 0.0
        z2 = 0.0
        if target == 0:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z1_abs = np.abs(z1)
            if z1_abs > self.upper_bound:
                z1 = np.sign(z1) * self.upper_bound
            Z_n = norm_cdf(z1) - norm_cdf_z2
            norm_pdf_z1 = norm_z_pdf(z1)
        elif target == self.J - 1:
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            z2_abs = np.abs(z2)
            if z2_abs > self.upper_bound:
                z2 = np.sign(z2) * self.upper_bound
            Z_n = norm_cdf_z1 - norm_cdf(z2)
            norm_pdf_z2 = norm_z_pdf(z2)
        else:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            Z_n = norm_cdf(z1) - norm_cdf(z2)
            norm_pdf_z1 = norm_z_pdf(z1)
            norm_pdf_z2 = norm_z_pdf(z2)
        if Z_n < self.EPS:
            if np.abs(np.exp(-0.5*z1**2 + 0.5*z2**2) - 1.0) > self.EPS**2:
                grad_Z_wrt_cavity_mean_n = (z1 * np.exp(
                        -0.5*z1**2 + 0.5*z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5 * z1 ** 2) + 0.5 * z2 ** 2) - 1.0)
                        * variance
                )
                grad_Z_wrt_cavity_variance_n = (
                    -1.0 + (z1**2 + 0.5 * z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5*z1**2 + 0.5 * z2**2) - 1.0)
                        * 2.0 * variance)
                )
                grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
                nu_n = (
                    grad_Z_wrt_cavity_mean_n_2
                    - 2.0 * grad_Z_wrt_cavity_variance_n)
            else:
                grad_Z_wrt_cavity_mean_n = 0.0
                grad_Z_wrt_cavity_mean_n_2 = 0.0
                grad_Z_wrt_cavity_variance_n = -(
                    1.0 - self.EPS)/(2.0 * variance)
                nu_n = (1.0 - self.EPS) / variance
                warnings.warn(
                    "Z_n must be greater than tolerance={} (got {}): "
                    "SETTING to Z_n to approximate value\n"
                    "z1={}, z2={}".format(
                        self.EPS, Z_n, z1, z2))
            if nu_n >= 1.0 / variance:
                nu_n = (1.0 - self.EPS) / variance
            if nu_n <= 0.0:
                nu_n = self.EPS * variance
        else:
            grad_Z_wrt_cavity_variance_n = (
                - z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (
                    2.0 * variance * Z_n)  # beta
            grad_Z_wrt_cavity_mean_n = (
                - norm_pdf_z1 + norm_pdf_z2) / (
                    std_dev * Z_n)  # alpha/gamma
            grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
            nu_n = (grad_Z_wrt_cavity_mean_n_2
                - 2.0 * grad_Z_wrt_cavity_variance_n)
        # Update alphas
        if numerically_stable:
            self._assert_valid_values(
                nu_n, variance, cavity_mean_n, cavity_variance_n, target,
                z1, z2, Z_n, norm_pdf_z1,
                norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                grad_Z_wrt_cavity_mean_n)
        # hnew = loomean + loovar * alpha;
        posterior_mean_n_new = (
            cavity_mean_n + cavity_variance_n * grad_Z_wrt_cavity_mean_n)
        # cnew = loovar - loovar * nu * loovar;
        posterior_covariance_n_new = (
            cavity_variance_n - cavity_variance_n**2 * nu_n)
        # pnew = nu / (1.0 - loovar * nu);
        precision_EP_n = nu_n / (1.0 - cavity_variance_n * nu_n)
        # print("posterior_mean_n_new", posterior_mean_n_new)
        # print("nu_n", nu_n)
        # print("precision_EP_n", precision_EP_n)
        # mnew = loomean + alpha / nu;
        mean_EP_n = cavity_mean_n + grad_Z_wrt_cavity_mean_n / nu_n
        # snew = Zi * sqrt(loovar * pnew + 1.0)*exp(0.5 * alpha * alpha / nu);
        amplitude_EP_n = Z_n * np.sqrt(
            cavity_variance_n * precision_EP_n + 1.0) * np.exp(
                0.5 * grad_Z_wrt_cavity_mean_n_2 / nu_n)
        return (
            mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n,
            posterior_mean_n_new, posterior_covariance_n_new, z1, z2, nu_n)

    def _update(
        self, index, mean_EP_n_old, posterior_cov,
        posterior_mean_n, posterior_variance_n,
        precision_EP_n_old,
        grad_Z_wrt_cavity_mean_n, posterior_mean_n_new, posterior_mean,
        posterior_covariance_n_new, diff, numerically_stable=False):
        """
        Update the posterior mean and covariance.

        Projects the tilted distribution on to an approximating family.
        The update for the t_n is a rank-1 update. Constructs a low rank
        approximation to the GP posterior covariance matrix.

        :arg int index: The index of the current likelihood (the index of the
            datapoint that is "left out").
        :arg float mean_EP_n_old: The state of the individual (site) mean (N,).
        :arg posterior_cov: The current approximate posterior covariance
            (N, N).
        :type posterior_cov: :class:`numpy.ndarray`
        :arg float posterior_variance_n: The current approximate posterior
            site variance.
        :arg float posterior_mean_n: The current site approximate posterior
            mean.
        :arg float precision_EP_n_old: The state of the individual (site)
            variance (N,).
        :arg float grad_Z_wrt_cavity_mean_n: The gradient of the log
            normalising constant with respect to the site cavity mean
            (The EP "weight").
        :arg float posterior_mean_n_new: The state of the site approximate
            posterior mean.
        :arg float posterior_covariance_n_new: The state of the site
            approximate posterior variance.
        :arg float diff: The differance between precision_EP_n and
            precision_EP_n_old.
        :returns: The updated approximate posterior mean and covariance.
        :rtype: tuple (`numpy.ndarray`, `numpy.ndarray`)
        """
        rho = diff / (1 + diff * posterior_variance_n)
        eta = (
            grad_Z_wrt_cavity_mean_n
            + precision_EP_n_old * (posterior_mean_n - mean_EP_n_old)) / (
                1.0 - posterior_variance_n * precision_EP_n_old)
        a_n = posterior_cov[:, index]  # The index'th column of posterior_cov
        posterior_cov = posterior_cov - rho * np.outer(a_n, a_n)
        posterior_mean += eta * a_n
        if numerically_stable is True:
            # TODO is this inequality meant to be the other way around?
            # TODO is hnew meant to be the EP weights, grad_Z_wrt_cavity_mean_n
            # assert(fabs((settings->alpha+index)->postcov[index]-alpha->cnew)<EPS)
            if np.abs(
                    posterior_covariance_n_new
                    - posterior_cov[index, index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_covariance_n_new - posterior_cov[index, "
                    "index]) must be less than some tolerance. Got (posterior_"
                    "covariance_n_new={}, posterior_cov_index_index={}, diff="
                    "{})".format(
                    posterior_covariance_n_new, posterior_cov[index, index],
                    posterior_covariance_n_new - posterior_cov[index, index]))
            # assert(fabs((settings->alpha+index)->pair->postmean-alpha->hnew)<EPS)
            if np.abs(posterior_mean_n_new - posterior_mean[index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_mean_n_new - posterior_mean[index]) must"
                    " be less than some tolerance. Got (posterior_mean_n_new="
                    "{}, posterior_mean_index={}, diff={})".format(
                        posterior_mean_n_new, posterior_mean[index],
                        posterior_mean_n_new - posterior_mean[index]))
        return posterior_cov, posterior_mean

    def _approximate_log_marginal_likelihood(
            self, posterior_cov, precision_EP, amplitude_EP, mean_EP,
            numerical_stability):
        """
        Calculate the approximate log marginal likelihood.
        TODO: need to finish this. Probably not useful if using EP.

        :arg posterior_cov: The approximate posterior covariance.
        :type posterior_cov:
        :arg precision_EP: The state of the individual (site) variance (N,).
        :type precision_EP:
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :type amplitude EP:
        :arg mean_EP: The state of the individual (site) mean (N,).
        :type mean_EP:
        :arg bool numerical_stability: If the calculation is made in a
            numerically stable manner.
        """
        precision_matrix = np.diag(precision_EP)
        inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
        log_amplitude_EP = np.log(amplitude_EP)
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp\
                - temp.T @ mean_EP
        if numerical_stability is True:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.trace(
                    np.log(inverse_precision_matrix)))
            approximate_marginal_likelihood = np.add(
                    approximate_marginal_likelihood, B/2)
            approximate_marginal_likelihood = np.subtract(
                approximate_marginal_likelihood, 0.5 * np.trace(
                    np.log(self.K + inverse_precision_matrix)))
            return np.sum(approximate_marginal_likelihood)
        else:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.log(np.linalg.det(
                    inverse_precision_matrix)))  # TODO: use log det C trick
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, B/2
            )
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, 0.5 * np.log(
                    np.linalg.det(self.K + inverse_precision_matrix))
            )  # TODO: use log det C trick
            return np.sum(approximate_marginal_likelihood)

    def grid_over_hyperparameters(
            self, domain, res,
            trainables=None,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None, amplitude_EP_0=None,
            first_step=0, write=False, verbose=False):
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        trainables.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        thetas, fxs,
        gxs, gx_0, intervals,
        trainables_where) = self._grid_over_hyperparameters_initiate(
            res, domain, trainables, self.cutpoints)
        for i, phi in enumerate(thetas):
            self._grid_over_hyperparameters_update(
                phi, trainables, self.cutpoints)
            if verbose:
                print(
                    "cutpoints_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "variance_0 = {}".format(
                        self.cutpoints, self.kernel.varphi, self.noise_variance,
                        self.kernel.variance))
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            posterior_cov = posterior_cov_0
            mean_EP = mean_EP_0
            precision_EP = precision_EP_0
            amplitude_EP = amplitude_EP_0
            while error / steps > self.EPS**2:
                iteration += 1
                (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
                 precision_EP, amplitude_EP, containers) = self.approximate(
                    steps, posterior_mean_0=posterior_mean, posterior_cov_0=posterior_cov,
                    mean_EP_0=mean_EP, precision_EP_0=precision_EP,
                    amplitude_EP_0=amplitude_EP,
                    first_step=first_step, write=False)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(thetas)))
            weight, precision_EP, L_cov, cov = self.compute_weights(
                precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
            t1, t2, t3, t4, t5 = self.compute_integrals_vector(
                np.diag(posterior_cov), posterior_mean, self.noise_variance)
            fx = self.objective(
                precision_EP, posterior_mean,
                t1, L_cov, cov, weight)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi,
                self.noise_variance,
                t2, t3, t4, t5, cov, weight, trainables)
            gxs[i] = gx[trainables_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(
                    self.kernel.varphi, self.noise_variance, fx))
        if x2s is not None:
            return (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
        else:
            return (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    def approximate_posterior(
            self, phi, trainables, steps=None,
            posterior_mean_0=None, return_reparameterised=False,
            posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=0, verbose=True):
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
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx the objective and gx the objective gradient
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, steps, error, iteration, trainables_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, steps)
        posterior_mean = posterior_mean_0
        posterior_cov = posterior_cov_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP,
            containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                posterior_cov_0=posterior_cov, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP,
                amplitude_EP_0=amplitude_EP,
                first_step=first_step, write=False)
        # TODO: this part requires an inverse, could it be sparsified
        # by putting q(f) = p(f_m) R(f_n). This is probably how FITC works
        (weight, precision_EP, L_cov, cov) = self.compute_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        # Try optimisation routine
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            np.diag(posterior_cov), posterior_mean, self.noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1,
            L_cov, cov, weight)
        gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, self.noise_variance,
            t2, t3, t4, t5, cov, weight, trainables)
        gx = gx[np.where(trainables != 0)]
        if verbose:
            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (posterior_cov, False)
        elif return_reparameterised is None:
            return fx, gx

    def compute_integrals_vector(
            self, posterior_variance, posterior_mean, noise_variance):
        """
        Compute the integrals required for the gradient evaluation.
        """
        noise_std = np.sqrt(noise_variance)
        mean_ts = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_ts) / (
                noise_variance + posterior_variance)
        mean_tplus1s = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_tplus1s) / (
                noise_variance + posterior_variance)
        sigma = np.sqrt(
            (noise_variance * posterior_variance) / (
            noise_variance + posterior_variance))
        a_ts = mean_ts - 5.0 * sigma
        b_ts = mean_ts + 5.0 * sigma
        h_ts = b_ts - a_ts
        a_tplus1s = mean_tplus1s - 5.0 * sigma
        b_tplus1s = mean_tplus1s + 5.0 * sigma
        h_tplus1s = b_tplus1s - a_tplus1s
        y_0 = np.zeros((20, self.N))
        t1 = fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_variance,
                self.cutpoints_ts, self.cutpoints_tplus1s,
                noise_std, self.EPS, self.EPS_2, self.N)
        t2 = fromb_t2_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t2[self.indices_where_0] = 0.0
        t3 = fromb_t3_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t3[self.indices_where_J_1] = 0.0
        t4 = fromb_t4_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t4[self.indices_where_J_1] = 0.0
        t5 = fromb_t5_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t5[self.indices_where_0] = 0.0
        return t1, t2, t3, t4, t5

    def objective(
            self, precision_EP, posterior_mean, t1, L_cov, cov,
            weights):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\theta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg precision_EP:
        :type precision_EP:
        :arg posterior_mean:
        :type posterior_mean:
        :arg t1:
        :type t1:
        :arg L_cov:
        :type L_cov:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :returns: fx
        :rtype: float
        """
        # Fill possible zeros in with machine precision
        precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        fx = -np.sum(np.log(np.diag(L_cov)))  # log det cov
        fx -= 0.5 * posterior_mean.T @ weights
        fx -= 0.5 * np.sum(np.log(precision_EP))
        # cov = L^{-1} L^{-T}  # requires a backsolve with the identity
        # TODO: check if there is a simpler calculation that can be done
        fx -= 0.5 * np.sum(np.divide(np.diag(cov), precision_EP))
        fx += np.sum(t1)
        # Regularisation - penalise large varphi (overfitting)
        # fx -= 0.1 * self.kernel.varphi
        return -fx

    def objective_gradient(
            self, gx, intervals, varphi, noise_variance,
            t2, t3, t4, t5, cov, weights, indices):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

            where :math:`F(\theta)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\theta` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg intervals:
        :type intervals:
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg t2:
        :type t2:
        :arg t3:
        :type t3:
        :arg t4:
        :type t4:
        :arg t5:
        :type t5:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :return: gx
        :rtype: float
        """
        if trainables is not None:
            # Update gx
            if trainables[0]:
                # For gx[0] -- ln\sigma
                gx[0] = np.sum(t5 - t4)
                # gx[0] *= -0.5 * noise_variance  # This is a typo in the Chu code
                gx[0] *= np.sqrt(noise_variance)
            # For gx[1] -- \b_1
            if trainables[1]:
                gx[1] = np.sum(t3 - t2)
            # For gx[2] -- ln\Delta^r
            for j in range(2, self.J):
                if trainables[j]:
                    targets = self.t_train[self.grid]
                    gx[j] = np.sum(t3[targets == j - 1])
                    gx[j] -= np.sum(t2[targets == self.J - 1])
                    # TODO: check this, since it may be an `else` condition!!!!
                    gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                    gx[j] *= intervals[j - 2]
            # For gx[self.J] -- variance
            if trainables[self.J]:
                # For gx[self.J] -- s
                # TODO: Need to check this is correct: is it directly analogous to
                # gradient wrt log varphi?
                partial_K_s = self.kernel.kernel_partial_derivative_s(
                    self.X_train, self.X_train)
                # VC * VC * a' * partial_K_varphi * a / 2
                gx[self.J] = varphi * 0.5 * weights.T @ partial_K_s @ weights  # That's wrong. not the same calculation.
                # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                gx[self.J] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_s))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[self.J] -= 0.1 * varphi
                gx[self.J] *= 2.0  # since varphi = kappa / 2
            # For gx[self.J + 1] -- varphi
            if trainables[self.J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    self.X_train, self.X_train)
                # elif 1:
                #     gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights
                # TODO: This needs fixing/ checking vs original code
                if 0:
                    for l in range(self.kernel.L):
                        K = self.kernel.num_hyperparameters[l]
                        KK = 0
                        for k in range(K):
                            gx[self.J + KK + k] = varphi[l] * 0.5 * weights.T @ partial_K_varphi[l][k] @ weights
                        KK += K
                else:
                    # VC * VC * a' * partial_K_varphi * a / 2
                    gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights  # That's wrong. not the same calculation.
                    # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                    gx[self.J + 1] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_varphi))
                    # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                    # gx[self.J] -= 0.1 * varphi
        return -gx

    def approximate_evidence(self, mean_EP, precision_EP, amplitude_EP, posterior_cov):
        """
        TODO: check and return line could be at risk of overflow
        Compute the approximate evidence at the EP solution.

        :return:
        """
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp - np.multiply(
            temp, mean_EP)
        Pi_inv = np.diag(1. / precision_EP)
        return (
            np.prod(
                amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
                / np.sqrt(np.linalg.det(np.add(Pi_inv, self.K))))

    def compute_weights(
        self, precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L_cov=None, cov=None, numerically_stable=False):
        """
        TODO: There may be an issue, where grad_Z_wrt_cavity_mean is updated
        when it shouldn't be, on line 2045.

        Compute the regression weights required for the gradient evaluation,
        and check that they are in equilibrium with
        the gradients of Z wrt cavity means.

        A matrix inverse is always required to evaluate fx.

        :arg precision_EP:
        :arg mean_EP:
        :arg grad_Z_wrt_cavity_mean:
        :arg L_cov: . Default `None`.
        :arg cov: . Default `None`.
        """
        if np.any(precision_EP == 0.0):
            # TODO: Only check for equilibrium if it has been updated in this swipe
            warnings.warn("Some sample(s) have not been updated.\n")
            precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        Pi_inv = np.diag(1. / precision_EP)
        if L_cov is None or cov is None:
            (L_cov, lower) = cho_factor(
                Pi_inv + self.K)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.N), lower=True)
            # TODO It is necessary to do this triangular solve to get
            # diag(cov) for the lower bound on the marginal likelihood
            # calculation. Note no tf implementation for diag(A^{-1}) yet.
            cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        if numerically_stable:
            # This is 3-4 times slower on CPU,
            # what about with jit compiled CPU or GPU?
            # Is this ever more stable than a matmul by the inverse?
            g = cho_solve((L_cov, False), mean_EP)
            weight = cho_solve((L_cov.T, True), g)
        else:
            weight = cov @ mean_EP
        if np.any(
            np.abs(weight - grad_Z_wrt_cavity_mean) > np.sqrt(self.EPS)):
            warnings.warn("Fatal error: the weights are not in equilibrium wit"
                "h the gradients".format(
                    weight, grad_Z_wrt_cavity_mean))
        return weight, precision_EP, L_cov, cov


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
        self, cutpoints, noise_variance=1.0, *args, **kwargs):
        # cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`LaplaceGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: An :class:`EPGP` object.
        """
        super().__init__(*args, **kwargs)
        # if cutpoints_hyperparameters is not None:
        #     warnings.warn("cutpoints_hyperparameters set as {}".format(cutpoints_hyperparameters))
        #     self.cutpoints_hyperparameters = cutpoints_hyperparameters
        # else:
        #     self.cutpoints_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        # self.EPS = 0.001  # Acts as a machine tolerance
        # self.EPS = 1e-4
        self.EPS = 1e-2
        # self.EPS = 1e-6
        self.EPS_2 = self.EPS**2
        # self.jitter = 1e-4  # Try increasing the noise variance if jitter has to be this large
        # self.jitter = 1e-6  # 1e-10 was too small when the noise variance is very low, resulting in infs or nans in chol
        self.jitter = 1e-10  # 1e-8, 1e-10 was too small for covariance parameterisation
        # Initiate hyperparameters
        self.hyperparameters_update(
            cutpoints=cutpoints, noise_variance=noise_variance)

    def _approximate_log_marginal_likelihood(
            self, posterior_cov, precision_EP,
            amplitude_EP, mean_EP, numerical_stability):
        """
        Calculate the approximate log marginal likelihood. TODO: need to finish this.

        :arg posterior_cov: The approximate posterior covariance.
        :arg mean_EP: The state of the individual (site) mean (N,).
        :arg precision_EP: The state of the individual (site) variance (N,).
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :arg bool numerical_stability: If the calculation is made in a
            numerically stable manner.
        """
        precision_matrix = np.diag(precision_EP)
        inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
        log_amplitude_EP = np.log(amplitude_EP)
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp - temp.T @ mean_EP
        if numerical_stability is True:
            approximate_marginal_likelihood = np.add(log_amplitude_EP, 0.5 * np.trace(np.log(inverse_precision_matrix)))
            approximate_marginal_likelihood = np.add(approximate_marginal_likelihood, B/2)
            approximate_marginal_likelihood = np.subtract(
                approximate_marginal_likelihood, 0.5 * np.trace(np.log(self.K + inverse_precision_matrix)))
            return np.sum(approximate_marginal_likelihood)
        else:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.log(np.linalg.det(inverse_precision_matrix)))  # TODO: use log det C trick
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, B/2
            )
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, 0.5 * np.log(np.linalg.det(self.K + inverse_precision_matrix))
            )  # TODO: use log det C trick
            return np.sum(approximate_marginal_likelihood)

    def grid_over_hyperparameters(
            self, domain, res,
            trainables=None,
            posterior_mean_0=None,
            first_step=0, verbose=True):  # TODO False
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        trainables.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        thetas, fxs,
        gxs, gx_0, intervals,
        trainables_where) = self._grid_over_hyperparameters_initiate(
            res, domain, trainables, self.cutpoints)
        for i, phi in enumerate(thetas):
            self._grid_over_hyperparameters_update(
                phi, trainables, self.cutpoints)
            if verbose:
                print(
                    "cutpoints_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "variance_0 = {}".format(
                        self.cutpoints, self.kernel.varphi, self.noise_variance,
                        self.kernel.variance))
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            while error / steps > self.EPS_2:
                iteration += 1
                (error, weight, posterior_mean, containers) = self.approximate(
                    steps, posterior_mean_0=posterior_mean,
                    first_step=first_step, write=False)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(thetas)))
            (weight, precision,
            w1, w2, g1, g2, v1, v2, q1, q2,
            L_cov, cov, Z) = self.compute_weights(
                posterior_mean)
            fx = self.objective(weight, precision, L_cov, Z)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), (self.X_train, self.t_train), self.grid, self.J,
                intervals, self.kernel.varphi, self.noise_variance,
                self.noise_std,
                w1, w2, g1, g2, v1, v2, q1, q2, cov, weight,
                self.N, self.K, precision, trainables)
            gxs[i] = gx[trainables_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(
                    self.kernel.varphi, self.noise_variance, fx))
        if x2s is not None:
            return (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
        else:
            return (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    def compute_weights(
        self, posterior_mean):
        """
        Compute the regression weights required for the objective evaluation
        and its gradients.

        A matrix inverse is always required to evaluate the objective.

        :arg posterior_mean:
        """
        # Numerically stable calculation of ordinal likelihood!
        (Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s, *_) = truncated_norm_normalising_constant(
            self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
            posterior_mean, self.EPS,
            upper_bound=self.upper_bound,
            upper_bound2=self.upper_bound2)
        w1 = norm_pdf_z1s / Z
        w2 = norm_pdf_z2s / Z
        z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
        z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
        g1 = z1s * w1
        g2 = z2s * w2
        v1 = z1s * g1
        v2 = z2s * g2
        q1 = z1s * v1
        q2 = z2s * v2
        weight = (w1 - w2) / self.noise_std
        precision = weight**2 + (g2 - g1) / self.noise_variance
        (L_cov, lower) = cho_factor(
            np.diag(1./ precision) + self.K)
        L_covT_inv = solve_triangular(
            L_cov.T, np.eye(self.N), lower=True)
        # TODO: Necessary to calculate? probably only for marginals. or maybe not
        cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        return weight, precision, w1, w2, g1, g2, v1, v2, q1, q2, L_cov, cov, Z

    def objective(self, weight, posterior_mean, precision, L_cov, Z):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\theta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg weight: 
        :arg precision:
        :arg L_cov:
        :arg Z:

        :returns: fx
        :rtype: float
        """
        fx = -np.sum(np.log(Z))
        fx += 0.5 * posterior_mean.T @ weight
        fx += np.sum(np.log(np.diag(L_cov)))
        fx += 0.5 * np.sum(np.log(precision))
        # fx = -fx
        return fx

    def objective_gradient(
            self, gx, data, grid, J,
            intervals, varphi, noise_variance, noise_std,
            w1, w2, g1, g2, v1, v2, q1, q2,
            cov, weight, N, K, precision, trainables):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

            where :math:`F(\theta)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\theta` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg gx: 
        :type gx:
        :arg data:
        :type data:
        :arg grid:
        :type grid:
        :arg J:
        :type J:
        :arg intervals:
        :type intervals:
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg float noise_variance: The noise variance.
        :arg float noise_std:
        :arg w1:
        :type w1:
        :arg w2:
        :type w2:
        :arg g1:
        :type g1:
        :arg g2:
        :type g2:
        :arg v1:
        :type v1:
        :arg v2:
        :type v2:
        :arg q1:
        :type q1:
        :arg q2:
        :type q2:
        :arg cov:
        :type cov:
        :arg weight:
        :type weight:
        :arg N:
        :type N:
        :arg K:
        :type K:
        :arg precision:
        :type precision:
        :arg trainables:
        :type trainables:
        :return: gx
        :rtype: float
        """
        X_train, t_train = data
        if trainables is not None:
            # compute a diagonal
            dsigma = cov @ K
            diag = np.diag(dsigma) / precision
            # partial lambda / partial phi_b = - partial lambda / partial f (* SIGMA)
            t1 = ((w2 - w1) - 3.0 * (w2 - w1) * (g2 - g1) - 2.0 * (w2 - w1)**3 - (v2 - v1)) / noise_variance
            # Update gx
            if trainables[0]:
                # For gx[0] -- ln\sigma
                cache = (w2 - w1) * ((g2 - g1) - (w2 - w1) + (v2 - v1)) / noise_variance
                # prepare D f / D delta_l
                t2 = - dsigma @ cache / precision
                tmp = (
                    - 2.0 * precision
                    + 2.0 * (w2 - w1) * (v2 - v1)
                    + 2.0 * (w2 - w1)**2 * (g2 - g1)
                    - (g2 - g1)
                    + (g2 - g1)**2
                    + (q2 - q1) / noise_variance)
                gx[0] = np.sum(g2 - g1 + 0.5 * (tmp - t2 * t1) * diag)
                gx[0] = - gx[0] / 2.0 * noise_variance
            # For gx[1] -- \b_1
            if trainables[1]:
                # For gx[1], \phi_b^1
                t2 = dsigma @ precision
                t2 = t2 / precision
                gx[1] -= np.sum(w2 - w1)
                gx[1] += 0.5 * np.sum(t1 * (1 - t2) * diag)
                gx[1] = gx[1] / noise_std
            # For gx[2] -- ln\Delta^r
            for j in range(2, J):
                targets = t_train[grid]
                print(targets)
                print(t_train)
                # Prepare D f / D delta_l
                cache0 = -(g2 + (w2 - w1) * w2) / noise_variance
                cache1 = - (g2 - g1 + (w2 - w1)**2) / noise_variance
                if trainables[j]:
                    idxj = np.where(targets == j - 1)
                    idxg = np.where(targets > j - 1)
                    idxl = np.where(targets < j - 1)
                    cache = np.zeros(N)
                    cache[idxj] = cache0[idxj]
                    cache[idxg] = cache1[idxg]
                    t2 = dsigma @ cache
                    t2 = t2 / precision
                    gx[j] -= np.sum(w2[idxj])
                    temp = (
                        w2[idxj]
                        - 2.0 * (w2[idxj] - w1[idxj]) * g2[idxj]
                        - 2.0 * (w2[idxj] - w1[idxj])**2 * w2[idxj]
                        - v2[idxj]
                        - (g2[idxj] - g1[idxj]) * w2[idxj]) / noise_variance
                    gx[j] += 0.5 * np.sum((temp - t2[idxj] * t1[idxj]) * diag[idxj])
                    gx[j] -= np.sum(w2[idxg] - w1[idxg])
                    gx[j] += 0.5 * np.sum(t1[idxg] * (1.0 - t2[idxg]) * diag[idxg])
                    gx[j] += 0.5 * np.sum(-t2[idxl] * t1[idxl] * diag[idxl])
                    gx[j] = gx[j] * intervals[j - 2] / noise_std
            # For gx[J] -- variance
            if trainables[J]:
                raise ValueError("TODO")
            # For gx[J + 1] -- varphi
            if trainables[J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    X_train, X_train)
                dmat = partial_K_varphi @ cov
                t2 = (dmat @ weight) / precision
                gx[J + 1] -= varphi * 0.5 * weight.T @ partial_K_varphi @ weight
                gx[J + 1] += varphi * 0.5 * np.sum((-diag * t1 * t2) / (noise_std))
                gx[J + 1] += varphi * 0.5 * np.sum(np.multiply(cov, partial_K_varphi))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[J] -= 0.1 * varphi
        return gx

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
        posterior_precisions = []
        containers = (posterior_means, posterior_precisions)
        return (posterior_mean_0, containers, error)

    def _update_posterior(self, Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
            noise_std, noise_variance, posterior_mean):
        """Update Laplace approximation posterior covariance in Newton step."""
        weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
        z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
        z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
        precision  = weight**2 + (
            z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
            ) / Z / noise_variance
        m = - self.K @ weight + posterior_mean
        # TODO: temp
        L_cov, _ = cho_factor(self.K + np.diag(1. / precision))
        L_covT_inv = solve_triangular(
            L_cov.T, np.eye(self.N), lower=True)
        cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        t1 = - (cov @ m) / precision
        posterior_mean += t1
        error = np.abs(max(t1.min(), t1.max(), key=abs))
        return error, weight, posterior_mean

    def approximate(
            self, steps, posterior_mean_0=None, first_step=0, write=False):
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
        :arg int first_step: The first step. Useful for burn in algorithms.
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
        (posterior_means, posterior_precisions) = containers
        for _ in trange(first_step, first_step + steps,
                        desc="Laplace GP approximator progress",
                        unit="iterations", disable=True):
            (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
                    _, _) = truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS, upper_bound=self.upper_bound,
                )
                #upper_bound2=self.upper_bound2)  # TODO Turn this off!
            error, weight, posterior_mean = self._update_posterior(
                Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
                self.noise_std, self.noise_variance, posterior_mean)
            if write is True:
                posterior_means.append(posterior_mean)
                posterior_precisions.append(posterior_precisions)
        containers = (posterior_means, posterior_precisions)
        return error, weight, posterior_mean, containers

    def approximate_posterior(
            self, phi, trainables, steps=None,
            posterior_mean_0=None,
            return_reparameterised=False, first_step=0, verbose=False):
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
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, steps, error, iteration, trainables_where,
                gx) = self._hyperparameter_training_step_initialise(
            phi, trainables, steps)
        posterior_mean = posterior_mean_0
        while error / steps > self.EPS_2 and iteration < 10:  # TODO is this overkill?
            iteration += 1
            (error, weight, posterior_mean, containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=False)
            if verbose:
                print("({}), error={}".format(iteration, error))
        # Calculates weights and matrix inverses one more time.
        (weight, precision,
        w1, w2, g1, g2, v1, v2, q1, q2,
        L_cov, cov, Z) = self.compute_weights(
            posterior_mean)
        fx = self.objective(weight, posterior_mean, precision, L_cov, Z)
        gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, (self.X_train, self.t_train), self.grid,
            self.J, intervals, self.kernel.varphi, self.noise_variance,
            self.noise_std,
            w1, w2, g1, g2, v1, v2, q1, q2, cov, weight,
            self.N, self.K, precision, trainables)
        gx = gx[np.nonzero(trainables)]
        if verbose:
            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        if return_reparameterised is False:
            return fx, gx, posterior_mean, (
                self.noise_variance * self.K @ cov, False)
        elif return_reparameterised is None:
            return fx, gx


class CutpointValueError(Exception):
    """
    An invalid cutpoint argument was used to construct the classifier model.
    """

    def __init__(self, cutpoint):
        """
        Construct the exception.

        :arg cutpoint: The cutpoint parameters array.
        :type cutpoint: :class:`numpy.array` or list

        :rtype: :class:`CutpointValueError`
        """
        message = (
                "The cutpoint list or array "
                "must be in ascending order, "
                f" {cutpoint} was given."
                )

        super().__init__(message)


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
