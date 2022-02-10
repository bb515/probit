from abc import ABC, abstractmethod

from scipy.stats.stats import weightedtau
from sklearn.metrics import precision_recall_curve
from .kernels import Kernel, InvalidKernel
import pathlib
import random
from tqdm import trange
import warnings
import math
import matplotlib.pyplot as plt
import numpy as np
from .utilities import (
    norm_z_pdf, norm_cdf,
    norm_logpdf, norm_logcdf,
    truncated_norm_normalising_constant,
    p, dp)
# Usually the numba implementation is not faster
# from .numba.utilities import (
#     fromb_t1_vector, fromb_t2_vector,
#     fromb_t3_vector, fromb_t4_vector, fromb_t5_vector)
from scipy.stats import norm
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from .utilities import (
    sample_varphis, unnormalised_log_multivariate_normal_pdf,
    vectorised_unnormalised_log_multivariate_normal_pdf,
    fromb_t1_vector, fromb_t2_vector, fromb_t3_vector, fromb_t4_vector,
    fromb_t5_vector)


class Approximator(ABC):
    """
    TODO: separate the approximation from the likelihood, e.g. probit.models.EP(likelihood = ordinal),
        probit.models.Laplace(likelihood = ordinal), probit.models.VGP(likelihood = probit).
    Base class for variational Bayes approximators.

    This class allows users to define a classification problem,
    get predictions using an approximate Bayesian inference.

    All approximators must define an init method, which may or may not
        inherit Sampler as a parent class using `super()`.
    All approximators that inherit Approximator define a number of methods that
        return the approximate posterior.
    All approximators must define a :meth:`approximate` that can be used to approximate
        the posterior.
    All approximators must define a :meth:`_approximate_initiate` that is used to
        initiate approximate.
    All approximators must define a :meth:`predict` can be used to make
        predictions given test data.
    """

    @abstractmethod
    def __init__(self, kernel, X_train, t_train, J, write_path=None):
        """
        Create an :class:`Approximator` object.

        This method should be implemented in every concrete Approximator.

        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.    
        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :arg t_train: (N, ) The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg J: The number of (ordinal) classes.
        :arg str write_path: Write path for outputs.

        :returns: A :class:`Approximator` object
        """
        if not (isinstance(kernel, Kernel)):
            raise InvalidKernel(kernel)
        else:
            self.kernel = kernel
        if write_path is None:
            self.write_path = pathlib.Path()
        else:
            self.write_path = pathlib.Path(write_path)
        self.N = np.shape(X_train)[0]
        self.D = np.shape(X_train)[1]
        self.X_train = X_train
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise TypeError(
                "t must contain only integer values (got {})".format(t_train))
        if t_train.dtype != int:
            raise TypeError(
                "t must contain only integer values (got {})".format(
                    t_train))
        else:
            self.t_train = t_train
        self.J = J
        self.grid = np.ogrid[0:self.N]  # For indexing sets of self.t_train
        if self.kernel._ARD:
            sigma = np.reshape(self.kernel.sigma, (self.J, 1))
            tau = np.reshape(self.kernel.tau, (self.J, 1))
            self.sigma = np.tile(sigma, (1, self.D))  # (J, D)
            self.tau = np.tile(tau, (1, self.D))  # (J, D)
        else:
            self.sigma = self.kernel.sigma
            self.tau = self.kernel.tau
        # See GPML by Williams et al. for a good explanation of jitter
        # self.jitter = 1e-8 
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
        self.upper_bound2 = 30  # May need decreasing if experience infs or NaNs
        warnings.warn("Updating prior covariance.")
        self._update_prior()

    @abstractmethod
    def _approximate_initiate(self):
        """
        Initialise the Approximator.

        This method should be implemented in every concrete Approximator.
        """

    @abstractmethod
    def approximate(self):
        """
        Return the samples

        This method should be implemented in every concrete Approximator.
        """

    @abstractmethod
    def predict(self):
        """
        Return the samples

        This method should be implemented in every concrete Approximator.
        """

    def get_log_likelihood(self, m):
        """
        TODO: once self.gamma is factored out,
            this needs to go somewhere else. Like a KL divergence/likelihoods folder.

        Likelihood of ordinal regression. This is product of scalar normal cdf.

        If np.ndim(m) == 2, vectorised so that it returns (num_samples,)
        vector from (num_samples, N) samples of the posterior mean.

        Note that numerical stability has been turned off in favour of exactness - but experiments should=
        be run twice with numerical stability turned on to see if it makes a difference.
        """
        Z, *_ = truncated_norm_normalising_constant(
            self.gamma_ts, self.gamma_tplus1s,
            self.noise_std, m, self.EPS, upper_bound=self.upper_bound)
            #  upper_bound=self.upper_bound, upper_bound2=self.upper_bound2)  #  , numerically_stable=True)
        if np.ndim(m) == 2:
            return np.sum(np.log(Z), axis=1)  # (num_samples,)
        elif np.ndim(m) == 1:
               return np.sum(np.log(Z))  # (1,)

    def _log_multivariate_normal_pdf(self, x, cov_inv, half_log_det_cov, mean=None):
        """Get the pdf of the multivariate normal distribution."""
        if mean is not None:
            x = x - mean
        return -0.5 * np.log(2 * np.pi) - half_log_det_cov - 0.5 * x.T @ cov_inv @ x  # log likelihood

    def _log_multivariate_normal_pdf_vectorised(self, xs, cov_inv, half_log_det_cov, mean=None):
        """Get the pdf of the multivariate normal distribution."""
        if mean is not None:
            xs = xs - mean
        return -0.5 * np.log(2 * np.pi) - half_log_det_cov - 0.5 * np.einsum(
            'kj, kj -> k', np.einsum('ij, ki -> kj', cov_inv, xs), xs)

    def _psi(self, varphi):
        """
        Return the approximate posterior mean of the hyperhyperparameters psi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
        Probit Regression with Gaussian Process Priors," in Neural Computation,
        vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
        doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(10).

        This is the same for all categorical approximators, and so can live in the
        Abstract Base Class.

        :arg varphi: Posterior mean approximate of varphi.
        :type varphi: :class:`numpy.ndarray`
        :return: The approximate posterior mean of the hyperhyperparameters psi
            Girolami and Rogers Page 9 Eq.(10).
        """
        return np.divide(np.add(1, self.sigma), np.add(self.tau, varphi))  # TODO self.sigma does/should not exist probs

    def get_theta(self, indices):
        """
        Get the parameters (theta) for unconstrained optimization.

        :arg indices: Indicator array of the hyperparameters to optimize over.
        :type indices: :class:`numpy.ndarray`
        :returns: The unconstrained parameters to optimize over, theta.
        :rtype: :class:`numpy.array`
        """
        theta = []
        if indices[0]:
            theta.append(np.log(np.sqrt(self.noise_variance)))
        if indices[1]:
            theta.append(self.gamma[1])
        for j in range(2, self.J):
            if indices[j]:
                theta.append(np.log(self.gamma[j] - self.gamma[j - 1]))
        if indices[self.J]:
            theta.append(np.log(np.sqrt(self.kernel.scale)))
        # TODO: replace this with kernel number of hyperparameters.
        if indices[self.J + 1]:
            theta.append(np.log(self.kernel.varphi))
        return np.array(theta)

    def _hyperparameter_training_step_initialise(
            self, theta, indices, steps):
        """
        TODO: this doesn't look correct, for example if only training a subset
        Initialise the hyperparameter training step.

        :arg theta: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}}
                \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\varphi}],

            where :math:`\sigma` is the noise standard deviation,
            :math:`\b_{1}` is the first cutpoint, :math:`\Delta_{l}` is the
            :math:`l`th cutpoint interval, :math:`\varphi` is the single
            shared lengthscale parameter or vector of parameters in which
            there are in the most general case J * D parameters.
            If `None` then no hyperperameter update is performed.
        :type theta: :class:`numpy.ndarray`
        :return: (intervals, steps, error, iteration, indices_where, gx)
        :rtype: (6,) tuple
        """
        # Initiate at None since those that are None do not get updated        
        noise_variance = None
        gamma = None
        scale = None
        varphi = None
        index = 0
        if indices is not None:
            if indices[0]:
                noise_std = np.exp(theta[index])
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
            for j in range(1, self.J):
                if gamma is None and indices[j]:
                    # Get gamma from classifier
                    gamma = self.gamma
            if indices[1]:
                gamma[1] = theta[index]
                index += 1
            for j in range(2, self.J):
                if indices[j]:
                    gamma[j] = gamma[j - 1] + np.exp(theta[index])
                    index += 1
            if indices[self.J]:
                scale_std = np.exp(theta[index])
                scale = scale_std**2
                index += 1
            if indices[self.J + 1]:
                if self.kernel._general and self.kernel._ARD:
                    # In this case, then there is a scale parameter, the first
                    # cutpoint, the interval parameters,
                    # and lengthscales parameter for each dimension and class
                    varphi = np.exp(
                        np.reshape(
                            theta[self.J:self.J + self.J * self.D],
                            (self.J, self.D)))
                    index += self.J * self.D
                else:
                    # In this case, then there is a scale parameter, the first
                    # cutpoint, the interval parameters,
                    # and a single, shared lengthscale parameter
                    varphi = np.exp(theta[index])
                    index += 1
        # Update prior and posterior covariance
        self.hyperparameters_update(
            gamma=gamma, varphi=varphi, scale=scale,
            noise_variance=noise_variance)
        if self.kernel._general and self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.J * self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        intervals = self.gamma[2:self.J] - self.gamma[1:self.J - 1]
        if steps is None:
            steps = np.max([200, self.N // 10 ])  # TODO: justify, should it be different for each Laplace, EP, VB
        error = np.inf
        iteration = 0
        indices_where = np.where(indices!=0)
        return (intervals, steps, error, iteration, indices_where, gx)

    def _grid_over_hyperparameters_initiate(
            self, res, domain, indices, gamma):
        """
        Initiate metadata and hyperparameters for plotting the objective
        function surface over hyperparameters.

        :arg int res:
        :arg domain: ((2,)tuple, (2.)tuple) of the start/stop in the domain to
            grid over, for each axis, like: ((start, stop), (start, stop)).
        :type domain:
        :arg indices: Indicator array of the hyperparameters to sample over.
        :type indices: :class:`numpy.ndarray`
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        """
        index = 0
        label = []
        axis_scale = []
        space = []
        if indices[0]:
            # Grid over noise_std
            label.append(r"$\sigma$")
            axis_scale.append("log")
            space.append(
                np.logspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        if indices[1]:
            # Grid over b_1
            label.append(r"$\gamma_{1}$")
            axis_scale.append("linear")
            space.append(
                np.linspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                # Grid over b_j
                label.append(r"$\gamma_{} - \gamma{}$".format(j, j-1))
                axis_scale.append("log")
                space.append(
                    np.logspace(
                        domain[index][0], domain[index][1], res[index]))
                index += 1
        if indices[self.J]:
            # Grid over scale
            label.append("$scale$")
            axis_scale.append("log")
            space.append(
                np.logspace(domain[index][0], domain[index][1], res[index]))
            index += 1
        if self.kernel._general and self.kernel._ARD:
            gx_0 = np.empty(1 + self.J - 1 + 1 + self.J * self.D)
            # In this case, then there is a scale parameter,
            #  the first cutpoint, the interval parameters,
            # and lengthscales parameter for each dimension and class
            for j in range(self.J * self.D):
                if indices[self.J + 1 + j]:
                    # grid over this particular hyperparameter
                    raise ValueError("TODO")
                    index += 1
        else:
            gx_0 = np.empty(1 + self.J - 1 + 1 + 1)
            if indices[self.J + 1]:
                # Grid over only kernel hyperparameter, varphi
                label.append(r"$\varphi$")
                axis_scale.append("log")
                space.append(
                    np.logspace(
                        domain[index][0], domain[index][1], res[index]))
                index +=1
        if index == 2:
            meshgrid = np.meshgrid(space[0], space[1])
            Phi_new = np.dstack(meshgrid)
            Phi_new = Phi_new.reshape((len(space[0]) * len(space[1]), 2))
            fxs = np.empty(len(Phi_new))
            gxs = np.empty((len(Phi_new), 2))
        elif index == 1:
            meshgrid = (space[0], None)
            space.append(None)
            axis_scale.append(None)
            label.append(None)
            Phi_new = space[0]
            fxs = np.empty(len(Phi_new))
            gxs = np.empty(len(Phi_new))
        else:
            raise ValueError(
                "Too many independent variables to plot objective over!"
                " (got {}, expected {})".format(
                index, "1, or 2"))
        assert len(axis_scale) == 2
        assert len(meshgrid) == 2
        assert len(space) ==  2
        assert len(label) == 2
        intervals = gamma[2:self.J] - gamma[1:self.J - 1]
        indices_where = np.where(indices != 0)
        return (
            space[0], space[1],
            label[0], label[1],
            axis_scale[0], axis_scale[1],
            meshgrid[0], meshgrid[1],
            Phi_new, fxs, gxs, gx_0, intervals, indices_where)

    def _grid_over_hyperparameters_update(
        self, phi, indices, gamma):
        """
        Update the hyperparameters, phi.

        :arg phi: The updated values of the hyperparameters.
        :type phi:
        :arg indices:
        :type indices:
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        """
        index = 0
        if indices[0]:
            if np.isscalar(phi):
                noise_std = phi
            else:
                noise_std = phi[index]
            noise_variance = noise_std**2
            noise_variance_update = noise_variance
            # Update kernel parameters, update prior and posterior covariance
            index += 1
        else:
            noise_variance_update = None
        if indices[1]:
            gamma = np.empty((self.J + 1,))
            gamma[0] = np.NINF
            gamma[-1] = np.inf
            gamma[1] = phi[index]
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                if np.isscalar(phi):
                    gamma[j] = gamma[j-1] + phi
                else:
                    gamma[j] = gamma[j-1] + phi[index]
                index += 1
        if indices[self.J]:
            scale_std = phi[index]
            scale = scale_std**2
            index += 1
            scale_update = scale
        else:
            scale_update = None
        if indices[self.J + 1]:  # TODO: replace this with kernel number of hyperparameters.
            if np.isscalar(phi):
                varphi = phi
            else:
                varphi = phi[index]
            varphi_update = varphi
            index += 1
        else:
            varphi_update = None
        # Update kernel parameters, update prior and posterior covariance
        self.hyperparameters_update(
                gamma=gamma, 
                noise_variance=noise_variance_update,
                scale=scale_update,
                varphi=varphi_update)
        return 0

    def _update_prior(self):
        """Update prior covariances."""
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
            self.X_train, self.X_train)
        self.partial_K_scale = self.kernel.kernel_partial_derivative_scale(
            self.X_train, self.X_train)


class VBOrdinalGP(Approximator):
    """
    A GP classifier for ordinal likelihood using the Variational Bayes (VB)
    approximation.
 
    Inherits the Approximator ABC. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference. It is for the ordinal likelihood. For this a
    :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """

    def __init__(
            self, gamma, noise_variance=1.0, *args, **kwargs):
            #gamma_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`VBOrderedGP` Approximator object.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: A :class:`VBOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        # if gamma_hyperparameters is not None:
        #     warnings.warn("gamma_hyperparameters set as {}".format(gamma_hyperparameters))
        #     self.gamma_hyperparameters = gamma_hyperparameters
        # else:
        #     self.gamma_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        if self.kernel._ARD:
            raise ValueError(
                "The kernel must not be ARD type (kernel._ARD=1),"
                " but ISO type (kernel._ARD=0). (got {}, expected)".format(
                    self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError(
                "The kernel must not be general "
                "type (kernel._general=1), but simple type "
                "(kernel._general=0). (got {}, expected)".format(
                    self.kernel._general, 0))
        #self.EPS = 0.000001  # Acts as a machine tolerance, controls error
        #self.EPS = 0.0000001  # Probably wouldn't go much smaller than this
        self.EPS = 1e-4  # perhaps not low enough.
        # self.EPS = 1e-8
        self.EPS_2 = self.EPS**2 
        #self.EPS = 0.001  # probably too large, will affect convergence 
        # Tends to work well in practice - should it be made smaller?
        # Just keep it consistent
        # self.jitter = 1e-6
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.hyperparameters_update(gamma=gamma, noise_variance=noise_variance)

    def _update_posterior(self):
        """Update posterior covariances."""
        # TODO: Is this really the best cholesky to take. What are the eigenvalues?
        # are they bounded?
        # Note that this scipy implementation returns an upper triangular matrix
        # whereas numpy, tf, scipy.cholesky return a lower triangular,
        # then the position of the matrix transpose in the code would change.
        (self.L_cov, self.lower) = cho_factor(
            self.noise_variance * np.eye(self.N) + self.K)
        # Unfortunately, it is necessary to take this cho_factor,
        # only for log_det_K
        (L_K, lower) = cho_factor(self.K + self.jitter * np.eye(self.N))

        # tmp
        L_KT_inv = solve_triangular(
            L_K.T, np.eye(self.N), lower=True)
        self.K_inv = solve_triangular(L_K, L_KT_inv, lower=False)

        self.log_det_K = 2 * np.sum(np.log(np.diag(L_K)))
        self.log_det_cov = -2 * np.sum(np.log(np.diag(self.L_cov)))
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
        self.cov = solve_triangular(self.L_cov, L_covT_inv, lower=False)
        self.trace_cov = np.sum(np.diag(self.cov))
        self.trace_posterior_cov_div_var = np.einsum('ij, ij -> ', self.K, self.cov)

    def hyperparameters_update(
        self, gamma=None, varphi=None, scale=None, noise_variance=None):
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

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi:
        :type varphi:
        :arg scale:
        :type scale:
        :arg noise_variance:
        :type noise_variance:
        """
        if gamma is not None:
            # Convert gamma to numpy array
            gamma = np.array(gamma)
            # Not including -\infty or \infty
            if np.shape(gamma)[0] == self.J - 1:
                gamma = np.append(gamma, np.inf)  # Append \infty
                gamma = np.insert(gamma, np.NINF)  # Insert -\infty at index 0
                pass  # Correct format
            # Not including one cutpoints
            elif np.shape(gamma)[0] == self.J: 
                if gamma[-1] != np.inf:
                    if gamma[0] != np.NINF:
                        raise ValueError(
                            "The last cutpoint parameter must be numpy.inf, or"
                            " the first cutpoint parameter must be numpy.NINF "
                            "(got {}, expected {})".format(
                            [gamma[0], gamma[-1]], [np.inf, np.NINF]))
                    else:  # gamma[0] is -\infty
                        gamma.append(np.inf)
                        pass  # correct format
                else:
                    gamma = np.insert(gamma, np.NINF)
                    pass  # correct format
            # Including all the cutpoints
            elif np.shape(gamma)[0] == self.J + 1:
                if gamma[0] != np.NINF:
                    raise ValueError(
                        "The cutpoint parameter \gamma must be numpy.NINF "
                        "(got {}, expected {})".format(gamma[0], np.NINF))
                if gamma[-1] != np.inf:
                    raise ValueError(
                        "The cutpoint parameter \gamma_J must be "
                        "numpy.inf (got {}, expected {})".format(
                            gamma[-1], np.inf))
                pass  # correct format
            else:
                raise ValueError(
                    "Could not recognise gamma shape. "
                    "(np.shape(gamma) was {})".format(np.shape(gamma)))
            assert gamma[0] == np.NINF
            assert gamma[-1] == np.inf
            assert np.shape(gamma)[0] == self.J + 1
            if not all(
                    gamma[i] <= gamma[i + 1]
                    for i in range(self.J)):
                raise CutpointValueError(gamma)
            self.gamma = gamma
            self.gamma_ts = gamma[self.t_train]
            self.gamma_tplus1s = gamma[self.t_train + 1]
        if varphi is not None or scale is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, scale=scale)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
            warnings.warn("Done posterior covariance.")
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
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
            posterior_mean_0 = np.random.rand(self.N)  # TODO: justification for this?
            # posterior_mean_0 = np.zeros(self.N)
        ys = []
        posterior_means = []
        varphis = []
        psis = []
        fxs = []
        containers = (posterior_means, ys, varphis, psis, fxs)
        return posterior_mean_0, containers

    def approximate(
            self, steps, posterior_mean_0=None, first_step=1, write=False, plot=False):
        """
        Estimating the posterior means are a 3 step iteration over posterior_mean,
        varphi and psi Eq.(8), (9), (10), respectively or,
        optionally, just an iteration over posterior_mean.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior mean
            (N,). If `None` then initialised to zeros, default `None`.
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
        posterior_mean, containers = self._approximate_initiate(posterior_mean_0)
        ms, ys, varphis, psis, fxs = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples",
                        disable=True):
            p_ = p(
                posterior_mean, self.gamma_ts, self.gamma_tplus1s, self.noise_std,
                self.EPS, self.upper_bound, self.upper_bound2)
            y = self._y(
                p_, posterior_mean, self.noise_std)
            posterior_mean, nu = self._posterior_mean(
                y, self.cov, self.K)
            if plot:
                plt.scatter(self.X_train, y, label="y")
                plt.scatter(self.X_train, posterior_mean, label="posterior_mean")
                plt.legend()
                plt.savefig("VB_approximate_posterior_mean_and_latents.png")
                plt.close()
            if 0:
            #if self.kernel.varphi_hyperparameters is not None:
                # Posterior mean update for kernel hyperparameters
                # Kernel hyperparameters are variables here
                varphi = self._varphi(
                    posterior_mean, self.kernel.varphi_hyperparameters, n_samples=1000)
                varphi_hyperparameters = self._varphi_hyperparameters(self.kernel.varphi)
                self.hyperparameters_update(varphi=varphi)
            if write:
                Z, *_ = truncated_norm_normalising_constant(
                    self.gamma_ts, self.gamma_tplus1s,
                    self.noise_std, posterior_mean, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean, nu, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance, self.log_det_K,
                    self.log_det_cov)
                ms.append(posterior_mean)
                ys.append(y)
                if self.kernel.varphi_hyperparameters is not None:
                    varphis.append(self.kernel.varphi)
                    psis.append(self.kernel.varphi_hyperparameters)
                fxs.append(fx)
        containers = (ms, ys, varphis, psis, fxs)
        return posterior_mean, nu, y, p, containers

    def _predict_vector(
            self, gamma, cov, y, noise_variance, X_test):
        """
        TODO: probably a way to do this with just the posterior mean?
        Make variational Bayes prediction over classes of X_test given the
        posterior samples.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg cov:
        :arg y:
        :arg varphi:
        :arg noise_variance:
        :arg X_test:

        :return: The class probabilities.
        :rtype tuple: ((N_test, J), (N_test,), (N_test,))
        """
        N_test = np.shape(X_test)[0]
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        c_news = self.kernel.kernel_prior_diagonal(X_test)  # (N_test,)
        intermediate_vectors = cov @ C_news  # (N, N_test)
        intermediate_scalars = np.sum(
            np.multiply(C_news, intermediate_vectors), axis=0)  # (N_test,)
        # Calculate posterior_mean_new # TODO: test this.
        # TODO: Could This just be a cov @ y @ C_news then a sum?
        posterior_predictive_m = np.einsum(
            'ij, i -> j', intermediate_vectors, y)  # (N_test,)
        # plt.scatter(self.X_train, y)
        # plt.plot(X_test, posterior_predictive_m)
        # plt.hlines(gamma[[1, 2]], -0.5, 1.5)
        # plt.show()
        posterior_var = c_news - intermediate_scalars
        posterior_std = np.sqrt(posterior_var)
        posterior_predictive_var = posterior_var + noise_variance  # (N_test,)
        posterior_predictive_std = np.sqrt(posterior_predictive_var)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(
                gamma[j + 1], posterior_predictive_m), posterior_predictive_std)
            Z2 = np.divide(np.subtract(
                gamma[j], posterior_predictive_m), posterior_predictive_std)
            predictive_distributions[:, j] = norm.cdf(Z1) - norm.cdf(Z2)
        return predictive_distributions, posterior_predictive_m, posterior_std

    def predict(
        self, gamma, cov, y, noise_variance,
        X_test, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg cov:
        :arg y: The approximate posterior mean of the latent variables y.
        :arg varphi: The approximate posterior mean of the
            hyperparameters varphi.
        :arg noise_variance:
        :arg X_test: The new data points, array like (N_test, D).
        :arg bool vectorised:

        
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError(
                "For the ordinal likelihood approximator, the kernel must not be"
                " ARD type (kernel._ARD=1), but ISO type (kernel._ARD=0). "
                "(got {}, expected)".format(self.kernel._ARD, 0))
        else:
            if vectorised:
                return self._predict_vector(
                    gamma, cov, y, noise_variance, X_test)
            else:
                return ValueError(
                    "The scalar implementation has been superseded. Please use"
                    " the vector implementation.")

    def _varphi(self, posterior_mean, varphi_hyperparameters, n_samples=10, vectorised=True):
        """
        Return the w values of the sample

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
        Probit Regression with Gaussian Process Priors," in Neural Computation,
        vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
        doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(9).

        :arg posterior_mean: approximate posterior mean.
        :arg varphi_hyperparameters: approximate posterior mean of the kernel hyperparameters.
        :arg int n_samples: The number of samples for the importance sampling
            estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from
        # (n_samples, J, D) in general and _ARD, (n_samples, ) for single
        # shared kernel and ISO case. Depends on the
        # shape of psi.
        varphis = sample_varphis(varphi_hyperparameters, n_samples)  # (n_samples, )
        log_varphis = np.log(varphis)
        # (n_samples, J, N, N) in general and _ARD, (n_samples, N, N) for
        # single shared kernel and ISO case. Depends on
        # the shape of psi.
        Cs_samples = self.kernel.kernel_matrices(
            self.X_train, self.X_train, varphis)  # (n_samples, N, N)
        Cs_samples = np.add(Cs_samples, 1e-5 * np.eye(self.N))
        if vectorised:
            log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(
                posterior_mean, mean=None, covs=Cs_samples)
        else:
            log_ws = np.empty((n_samples,))
            # Scalar version
            for i in range(n_samples):
                log_ws[i] = unnormalised_log_multivariate_normal_pdf(
                    posterior_mean, mean=None, cov=Cs_samples[i])
        # Normalise the w vectors
        max_log_ws = np.max(log_ws)
        log_normalising_constant = max_log_ws + np.log(
            np.sum(np.exp(log_ws - max_log_ws), axis=0))
        log_ws = np.subtract(log_ws, log_normalising_constant)
        element_prod = np.add(log_varphis, log_ws)
        element_prod = np.exp(element_prod)
        magic_number = 2.0
        print("varphi", magic_number * np.sum(element_prod, axis=0))
        return magic_number * np.sum(element_prod, axis=0)

    def _posterior_mean(self, y, cov, K):
        """
        Return the approximate posterior mean of m.

        2021 Page Eq.()

        :arg y: (N,) array
        :type y: :class:`np.ndarray`
        :arg cov:
        :type cov:
        :arg K:
        :type K:
        """
        nu = cov @ y
        ## TODO: This is 3-4 times slower on CPU, what about with jit compiled CPU or GPU?
        # nu = cho_solve((self.L_cov, self.lower), y)
        return K @ nu, nu  # (N, J)

    def _y(self, p, posterior_mean, noise_std):
        """
        Calculate Y elements 2021 Page Eq.().

        :arg p:
        :type p:
        :arg posterior_mean:
        :type posterior_mean:
        :arg noise_std:
        :type noise_std:
        """
        return np.add(posterior_mean, noise_std * p)

    def objective(
            self, N, m, nu, trace_cov, trace_posterior_cov_div_var, Z,
            noise_variance,
            log_det_K, log_det_cov, verbose=False):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood.

        .. math::
                \mathcal{F(\Phi)} =,

            where :math:`F(\Phi)` is the variational lower bound of the log
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
        :arg bool numerical_stability: If the function is evaluated in a
            numerically stable way, default `True`. `False`
            is NOT recommended as often np.linalg.det(C) returns a value 0.0.
        :return: fx
        :rtype: float
        """
        trace_K_inv_posterior_cov = noise_variance * trace_cov
        log_det_posterior_cov = log_det_K + N * np.log(noise_variance) + log_det_cov 
        one = - trace_posterior_cov_div_var / 2
        two = - log_det_K / 2
        three = - trace_K_inv_posterior_cov / 2
        four = - m.T @ nu / 2
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
            self, gx, intervals, gamma_ts, gamma_tplus1s, varphi,
            noise_variance, noise_std,
            m, nu, cov, trace_cov, partial_K_varphi, N,
            Z, norm_pdf_z1s, norm_pdf_z2s, indices,
            numerical_stability=True, verbose=False):
        """
        Calculate gx, the jacobian of the variational lower bound of the log
        marginal likelihood at the VB equilibrium,

        .. math::
                \mathcal{\frac{\partial F(\Phi)}{\partial \Phi}}

            where :math:`F(\Phi)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`\Phi` is the set of hyperparameters, :math:`h`,
            :math:`\Pi`, :math:`K`. #TODO

        :arg intervals: The vector of the first cutpoint and the intervals
            between cutpoints for unconstrained optimisation of the cutpoint
            parameters.
        :type intervals: :class:`numpy.ndarray`
        :arg varphi: The lengthscale parameters.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance:
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
        if indices is not None:
            # For gx[0] -- ln\sigma  # TODO: currently seems analytically incorrect
            if indices[0]:
                one = N - noise_variance * trace_cov
                sigma_dp = dp(m, gamma_ts, gamma_tplus1s, noise_std,
                    self.upper_bound, self.upper_bound2)
                two = - (1. / noise_std) * np.sum(sigma_dp)
                if verbose:
                    print("one ", one)
                    print("two ", two)
                    print("gx_sigma = ", one + two)
                gx[0] = one + two
            # For gx[1] -- \b_1
            if indices[1]:
                # TODO: treat these with numerical stability, or fix them
                intermediate_vector_1s = np.divide(norm_pdf_z1s, Z)
                intermediate_vector_2s = np.divide(norm_pdf_z2s, Z)
                indices = np.where(self.t_train == 0)
                gx[1] += np.sum(intermediate_vector_1s[indices])
                for j in range(2, self.J):
                    indices = np.where(self.t_train == j - 1)
                    gx[j - 1] -= np.sum(intermediate_vector_2s[indices])
                    gx[j] += np.sum(intermediate_vector_1s[indices])
                # gx[self.J] -= 0  # Since J is number of classes
                gx[1:self.J] /= noise_std
                # For gx[2:self.J] -- ln\Delta^r
                gx[2:self.J] *= intervals
                if verbose:
                    print(gx[2:self.J])
            # For gx[self.J] -- s
            if indices[self.J]:
                raise ValueError("TODO")
            # For kernel parameters
            if indices[self.J + 1]:
                if self.kernel._general and self.kernel._ARD:
                    raise ValueError("TODO")
                else:
                    if numerical_stability is True:
                        # Update gx[-1], the partial derivative of the lower bound
                        # wrt the lengthscale. Using matrix inversion Lemma
                        one = (varphi / 2) * nu.T @ partial_K_varphi @ nu
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
            self, domain, res, indices=None, posterior_mean_0=None, write=False,
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
        Phi_new,
        fxs, gxs, gx_0,
        intervals, indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.gamma)
        error = np.inf
        fx_old = np.inf
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, indices, self.gamma)
            # Reset error and posterior mean
            iteration = 0
            error = np.inf
            fx_old = np.inf
            # TODO: reset m_0 is None?
            # Convergence is sometimes very fast so this may not be necessary
            while error / steps > self.EPS:
                iteration += 1
                (posterior_mean_0, nu, y, p, *_) = self.approximate(
                    steps, posterior_mean_0=posterior_mean_0,
                    first_step=1, write=False)
                (Z,
                norm_pdf_z1s,
                norm_pdf_z2s,
                z1s,
                z2s,
                *_ )= truncated_norm_normalising_constant(
                    self.gamma_ts, self.gamma_tplus1s,
                    self.noise_std, posterior_mean_0, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean_0, nu, self.trace_cov, self.trace_posterior_cov_div_var,
                    Z,
                    self.noise_variance, self.log_det_K, self.log_det_cov)
                error = np.abs(fx_old - fx)
                fx_old = fx
                print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(Phi_new)))
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.gamma_ts, self.gamma_tplus1s,
                self.kernel.varphi, self.noise_variance, self.noise_std, posterior_mean_0,
                nu, self.cov, self.trace_cov,
                self.partial_K_varphi, self.N, Z,
                norm_pdf_z1s, norm_pdf_z2s, indices,
                numerical_stability=True, verbose=False)
            fxs[i] = fx
            gxs[i] = gx[indices_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print(
                    "gamma={}, varphi={}, noise_variance={}, scale={}, "
                    "fx={}, gx={}".format(
                    self.gamma, self.kernel.varphi, self.noise_variance,
                    self.kernel.scale, fx, gxs[i]))
        if x2s is not None:
            return (
                fxs.reshape((len(x1s), len(x2s))), gxs,
                xx, yy, xlabel, ylabel, xscale, yscale)
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def approximate_posterior(
            self, theta, indices, steps=None, first_step=1,
            calculate_posterior_cov=True, write=False, verbose=False):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :arg indices:
        :arg first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx, gx
        :rtype: float, `:class:numpy.ndarray`
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        fx_old = np.inf
        posterior_mean_0 = None
        # Convergence is sometimes very fast so this may not be necessary
        while error / steps > self.EPS:
            iteration += 1
            (posterior_mean_0, nu, *_) = self.approximate(
                steps, posterior_mean_0=posterior_mean_0,
                first_step=first_step, write=write)
            (Z,
            norm_pdf_z1s,
            norm_pdf_z2s,
            *_ )= truncated_norm_normalising_constant(
                self.gamma_ts, self.gamma_tplus1s, self.noise_std, posterior_mean_0,
                self.EPS)
            fx = self.objective(
                self.N, posterior_mean_0, nu, self.trace_cov, self.trace_posterior_cov_div_var,
                Z,
                self.noise_variance, self.log_det_K, self.log_det_cov)
            error = np.abs(fx_old - fx)
            fx_old = fx
            if verbose:
                print("({}), error={}".format(iteration, error))
        gx = self.objective_gradient(
            gx.copy(), intervals, self.gamma_ts, self.gamma_tplus1s,
            self.kernel.varphi, self.noise_variance, self.noise_std,
            posterior_mean_0, nu, self.cov, self.trace_cov,
            self.partial_K_varphi, self.N, Z,
            norm_pdf_z1s, norm_pdf_z2s, indices,
            numerical_stability=True, verbose=False)
        gx = gx[indices_where]
        if verbose:
            # print("gamma=", repr(self.gamma), ", ")
            # print("varphi=", self.kernel.varphi)
            # print("noise_variance=", self.noise_variance, ", ")
            # print("scale=", self.kernel.scale, ", ")
            # print("\nfunction_eval={}\n jacobian_eval={}".format(
            #     fx, gx))
            print(
                "gamma={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.gamma,
                    self.noise_variance,
                    self.kernel.varphi, fx))
        if calculate_posterior_cov:
            #return fx, gx, posterior_mean_0, (self.noise_variance * np.eye(self.N) + self.K_inv, True)
            return fx, gx, posterior_mean_0, (self.noise_variance * self.K @ self.cov, False)
        else:
            return fx, gx

    def hyperparameter_training_step(
            self, theta, indices, first_step=1, write=False, verbose=True):
        """
        Optimisation routine for hyperparameters.
        :return: fx, gx
        """
        return self.approximate_posterior(
            theta, indices, first_step=first_step, calculate_posterior_cov=False, write=write, verbose=verbose)


class EPOrdinalGP(Approximator):
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
    def __init__(
        self, gamma, noise_variance=1.0, *args, **kwargs):
        # gamma_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`EPOrderedGP` Approximator object.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: An :class:`EPOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        # if gamma_hyperparameters is not None:
        #     warnings.warn("gamma_hyperparameters set as {}".format(gamma_hyperparameters))
        #     self.gamma_hyperparameters = gamma_hyperparameters
        # else:
        #     self.gamma_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        if self.kernel._ARD:
            raise ValueError(
                "The kernel must not be _ARD type (kernel._ARD=1),"
                " but ISO type (kernel._ARD=0). (got {}, expected)".format(
                self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError(
                "The kernel must not be general type (kernel._general=1),"
                " but simple type (kernel._general=0). "
                "(got {}, expected)".format(self.kernel._general, 0))
        self.EPS = 1e-4  # perhaps too large
        # self.EPS = 1e-6  # Decreasing EPS will lead to more accurate solutions but a longer convergence time.
        self.EPS_2 = self.EPS**2
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.hyperparameters_update(gamma=gamma, noise_variance=noise_variance)

    def hyperparameters_update(
        self, gamma=None, varphi=None, scale=None, noise_variance=None):
        """
        TODO: can probably collapse this code into other hyperparameter update
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi:
        :type varphi:
        :arg scale:
        :type scale:
        :arg noise_variance:
        :type noise_variance:
        """
        # TODO: can't this be done in an _update_prior()
        #self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        if gamma is not None:
            # Convert gamma to numpy array
            gamma = np.array(gamma)
            # Not including -\infty or \infty
            if np.shape(gamma)[0] == self.J - 1:
                gamma = np.append(gamma, np.inf)  # Append \infty
                gamma = np.insert(gamma, np.NINF)  # Insert -\infty at index 0
                pass  # correct format
            # Not including one cutpoints
            elif np.shape(gamma)[0] == self.J:
                if gamma[-1] != np.inf:
                    if gamma[0] != np.NINF:
                        raise ValueError(
                            "The last cutpoint parameter must be numpy.inf, or"
                            " the first cutpoint parameter must be numpy.NINF "
                            "(got {}, expected {})".format(
                            [gamma[0], gamma[-1]], [np.inf, np.NINF]))
                    else:  #gamma[0] is -\infty
                        gamma.append(np.inf)
                        pass  # correct format
                else:
                    gamma = np.insert(gamma, np.NINF)
                    pass  # correct format
            # Including all the cutpoints
            elif np.shape(gamma)[0] == self.J + 1:
                if gamma[0] != np.NINF:
                    raise ValueError(
                        "The cutpoint parameter \gamma must be numpy.NINF "
                        "(got {}, expected {})".format(gamma[0], np.NINF))
                if gamma[-1] != np.inf:
                    raise ValueError(
                        "The cutpoint parameter \gamma_J must be "
                        "numpy.inf (got {}, expected {})".format(
                            gamma[-1], np.inf))
                pass  # correct format
            else:
                raise ValueError(
                    "Could not recognise gamma shape. "
                    "(np.shape(gamma) was {})".format(np.shape(gamma)))
            assert gamma[0] == np.NINF
            assert gamma[-1] == np.inf
            assert np.shape(gamma)[0] == self.J + 1
            if not all(
                    gamma[i] <= gamma[i + 1]
                    for i in range(self.J)):
                raise CutpointValueError(gamma)
            self.gamma = gamma
            self.gamma_ts = gamma[self.t_train]
            self.gamma_tplus1s = gamma[self.t_train + 1]
        if varphi is not None or scale is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, scale=scale)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
        # Posterior covariance is calculated iteratively in EP,
        # so no update here.

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
        :arg posterior_cov_0: The initial state of the posterior covariance (N,). If 
            `None` then initialised to prior covariance, default `None`.
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
        :return: Containers for the approximate posterior means of parameters and
            hyperparameters.
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
            posterior_mean_0 = (posterior_cov_0 @ np.diag(precision_EP_0)) @ mean_EP_0
        error = 0.0
        grad_Z_wrt_cavity_mean_0 = np.zeros(self.N)  # Initialisation
        posterior_means = []
        posterior_covs = []
        mean_EPs = []
        amplitude_EPs = []
        precision_EPs = []
        approximate_marginal_likelihoods = []
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_marginal_likelihoods)
        return (posterior_mean_0, posterior_cov_0, mean_EP_0,
                precision_EP_0, amplitude_EP_0, grad_Z_wrt_cavity_mean_0,
                containers, error)

    def approximate(
            self, steps, posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None, amplitude_EP_0=None,
            first_step=1, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Expectation propagation iteration as written in
        Appendix B Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.

        EP does not attempt to learn a posterior distribution over
        hyperparameters, but instead tries to approximate
        the joint posterior given some hyperparameters. The hyperpamaeters
        have to be optimized with model selection step.

        :arg int steps: The number of iterations the Approximator takes.
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi: Initialisation of hyperparameter approximate posterior means.
            If `None` then initialised to ones, default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance (N, N).
            If `None` then initialised to prior covariance, default `None`.
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
        amplitude_EP, grad_Z_wrt_cavity_mean, containers,
        error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)
        (posterior_means, posterior_covs, mean_EPs, precision_EPs, amplitude_EPs,
         approximate_log_marginal_likelihoods) = containers
        for step in trange(first_step, first_step + steps,
                        desc="EP GP priors Approximator Progress",
                        unit="iterations", disable=True):
            index = self.new_point(step, random_selection=False)
            target = self.t_train[index]
            # Find the mean and variance of the leave-one-out
            # posterior distribution Q^{\backslash i}(\bm{f})
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
                self.gamma[target], self.gamma[target + 1],
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

    def new_point(self, step, random_selection=True):
        """
        Return a new point based on some policy.

        :arg int step: The current iteration step.
        :arg bool random_selection: If random_selection is true, then returns
            a random point from the ordering.
            Otherwise, it returns a sequential point. Default `True`.
        :return: index
        """
        if random_selection:
            return random.randint(0, self.N-1)
        else:
            # There is an issue here. If steps < N, then some points never
            # get updated at all.
            return step % self.N

    def _remove(
            self, posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old):
        """
        Calculate the product of approximate posterior factors with the current
        index removed.

        This is called the cavity distribution,
        "a bit like leaving a hole in the dataset".

        :arg float posterior_variance_n: Variance of latent function at index.
        :arg float posterior_mean_n: The state of the approximate posterior mean.
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
                "posterior_cov_nn must be non-negative (got {})".format(posterior_variance_n))
        return (
            posterior_mean_n, posterior_variance_n,
            cavity_mean_n, cavity_variance_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old)

    def _assert_valid_values(self, nu_n, variance, cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1,
            norm_pdf_z2, grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n):
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
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1,
                    norm_pdf_z2, grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n))
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
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1,
                    norm_pdf_z2, grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n))
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
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1,
                    norm_pdf_z2, grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n))
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
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1,
                    norm_pdf_z2, grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "nu_n must be less than 1.0 / (cavity_variance_n + "
                "noise_variance) = {}, got {}".format(
                    1.0 / variance, nu_n))
        return 0

    def _include(
            self, target, cavity_mean_n, cavity_variance_n,
            gamma_t, gamma_tplus1, noise_variance, numerically_stable=False):
        """
        Update the approximate posterior by incorporating the message
        p(t_i|m_i) into Q^{\i}(\bm{f}).
        Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)
        This includes one true-observation likelihood, and 'tilts' the
        approximation towards the true posterior. It updates the approximation
        to the true posterior by minimising a moment-matching KL divergence
        between the tilted distribution and the posterior distribution. This
        gives us an approximate posterior in the approximating family. The
        update to posterior_cov is a rank-1 update (see the outer product of two 1d
        vectors), and so it essentially constructs a piecewise low rank
        approximation to the GP posterior covariance matrix, until convergence
        (by which point it will no longer be low rank).
        :arg int target: The ordinal class index of the current site
            (the class of the datapoint that is "left out").
        :arg float cavity_mean_n: The cavity mean of the current site.
        :arg float cavity_variance_n: The cavity variance of the current site.
        :arg float gamma_t: The upper cutpoint parameters.
        :arg float gamma_tplus1: The lower cutpoint parameter.
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
            z1 = (gamma_tplus1 - cavity_mean_n) / std_dev
            z1_abs = np.abs(z1)
            if z1_abs > self.upper_bound:
                z1 = np.sign(z1) * self.upper_bound
            Z_n = norm_cdf(z1) - norm_cdf_z2
            norm_pdf_z1 = norm_z_pdf(z1)
        elif target == self.J - 1:
            z2 = (gamma_t - cavity_mean_n) / std_dev
            z2_abs = np.abs(z2)
            if z2_abs > self.upper_bound:
                z2 = np.sign(z2) * self.upper_bound
            Z_n = norm_cdf_z1 - norm_cdf(z2)
            norm_pdf_z2 = norm_z_pdf(z2)
        else:
            z1 = (gamma_tplus1 - cavity_mean_n) / std_dev
            z2 = (gamma_t - cavity_mean_n) / std_dev
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
        :arg posterior_cov: The current approximate posterior covariance (N, N).
        :type posterior_cov: :class:`numpy.ndarray`
        :arg float posterior_variance_n: The current approximate posterior site variance.
        :arg float posterior_mean_n: The current site approximate posterior mean.
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
        # rho = diff/(1+diff*Aii);
        rho = diff / (1 + diff * posterior_variance_n)
		# eta = (alpha+epinvvar*(postmean-epmean))/(1.0-Aii*epinvvar) ;
        eta = (
            grad_Z_wrt_cavity_mean_n
            + precision_EP_n_old * (posterior_mean_n - mean_EP_n_old)) / (
                1.0 - posterior_variance_n * precision_EP_n_old)
        # ai[i] = Retrieve_Posterior_Covariance (i, index, settings) ;
        a_n = posterior_cov[:, index]  # The index'th column of posterior_cov
        ##a_n = posterior_cov[index, :]]
        #
        # postcov[j]-=rho*ai[i]*ai[j] ;
        #posterior_cov -= (rho * np.outer(a_n, a_n))
        # Delta = - rho * np.outer(a_n, a_n)
        posterior_cov = posterior_cov - rho * np.outer(a_n, a_n)
        # posterior_cov = posterior_cov - rho * np.outer(a_n, a_n)
        # postmean+=eta*ai[i];
        posterior_mean += eta * a_n
        if numerically_stable is True:
            # TODO is hnew meant to be the EP weights, grad_Z_wrt_cavity_mean_n
            # assert(fabs((settings->alpha+index)->pair->postmean-alpha->hnew)<EPS)
            if np.abs(posterior_covariance_n_new - posterior_cov[index, index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_covariance_n_new - posterior_cov[index, index]) must"
                    " be less than some tolerance. Got (posterior_covariance_n_"
                    "new={}, posterior_cov_index_index={}, diff={})".format(
                    posterior_covariance_n_new, posterior_cov[index, index],
                    posterior_covariance_n_new - posterior_cov[index, index]))
            # assert(fabs((settings->alpha+index)->postcov[index]-alpha->cnew)<EPS)
            if np.abs(posterior_mean_n_new - posterior_mean[index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_mean_n_new - posterior_mean[index]) must be "
                    "less than some tolerance. Got (posterior_mean_n_new={}, "
                    "posterior_mean_index={}, diff={})".format(
                        posterior_mean_n_new, posterior_mean[index],
                        posterior_mean_n_new - posterior_mean[index]))
        return posterior_cov, posterior_mean

    def _approximate_log_marginal_likelihood(
        self, posterior_cov, precision_EP, amplitude_EP, mean_EP, numerical_stability):
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
        intermediate_vector = np.multiply(mean_EP, precision_EP)
        B = intermediate_vector.T @ posterior_cov @ intermediate_vector - intermediate_vector.T @ mean_EP
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

    def _predict_vector(
        self, gamma, posterior_cov, mean_EP, precision_EP, varphi, noise_variance,
        X_test, Lambda):
        """
        Make EP prediction over classes of X_test given the posterior samples.
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg posterior_cov: TODO: don't need posterior_cov here
        :arg posterior_mean:
        :arg varphi:
        :arg X_test: The new data points, array like (N_test, D).
        :arg Lambda: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        :rtype tuple: ((N_test, J), (N_test,), (N_test,))
        """
        # error = 0.0
        # absolute_error = 0.0
        Pi_inv = np.diag(1. / precision_EP)
        # Lambda = np.linalg.inv(np.add(Pi_inv, self.K))  # (N, N)
        Lambda_chol = np.linalg.cholesky(np.add(Pi_inv, self.K))
        Lambda_chol_inv = np.linalg.inv(Lambda_chol)
        Lambda = Lambda_chol_inv.T @ Lambda_chol_inv
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
        self.kernel.varphi = varphi
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        c_news = self.kernel.kernel_prior_diagonal(X_test)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Lambda @ C_news  # (N, N_test)
        intermediate_scalars = np.einsum(
            'ij, ij -> j', C_news, intermediate_vectors)
        posterior_var = c_news - intermediate_scalars
        posterior_pred_var = posterior_var + noise_variance
        posterior_std = np.sqrt(posterior_var)
        posterior_pred_std = np.sqrt(posterior_pred_var)
        posterior_pred_mean = np.einsum(
            'ij, i -> j', intermediate_vectors, mean_EP)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(
                gamma[j + 1], posterior_pred_mean), posterior_pred_std)
            Z2 = np.divide(
                np.subtract(gamma[j], posterior_pred_mean), posterior_pred_std)
            predictive_distributions[:, j] = norm.cdf(Z1) - norm.cdf(Z2)
        return predictive_distributions, posterior_pred_mean, posterior_std

    def predict(
        self, gamma, posterior_cov,
        mean_EP, precision_EP,
        varphi, noise_variance, X_test, Lambda, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg posterior_cov: The approximate posterior covariance.
        :arg y: The approximate posterior mean of the latent variable Y.
        :arg varphi: The approximate posterior mean of the
            hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError(
                "For the ordinal likelihood approximator,the kernel "
                "must not be _ARD type (kernel._ARD=1), but"
                " ISO type (kernel._ARD=0). (got {}, expected)".format(
                    self.kernel._ARD, 0))
        else:
            if vectorised:
                return self._predict_vector(
                    gamma, posterior_cov,
                    mean_EP, precision_EP,
                    varphi, noise_variance, X_test, Lambda)
            else:
                return ValueError(
                    "The scalar implementation has been "
                    "superseded. Please use "
                    "the vector implementation.")

    def grid_over_hyperparameters(
            self, domain, res,
            indices=None,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None, amplitude_EP_0=None,
            first_step=1, write=False, verbose=False):
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        indices.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        Phi_new, fxs,
        gxs, gx_0, intervals,
        indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.gamma)
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, indices, self.gamma)
            if verbose:
                print(
                    "gamma_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "scale_0 = {}".format(
                        self.gamma, self.kernel.varphi, self.noise_variance,
                        self.kernel.scale))
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
                    first_step=first_step, write=write)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(Phi_new)))
            (weights, precision_EP,
            Lambda_cholesky, Lambda) = self.compute_weights(
                precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
            t1, t2, t3, t4, t5 = self.compute_integrals_vector(
                np.diag(posterior_cov), posterior_mean, self.noise_variance)
            fx = self.objective(
                precision_EP, posterior_mean,
                t1, Lambda_cholesky, Lambda, weights)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi,
                self.noise_variance,
                t2, t3, t4, t5, Lambda, weights, indices)
            gxs[i] = gx[indices_where]
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
            self, theta, indices, steps=None,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=True):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :type theta:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
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
        :return:
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        posterior_mean = posterior_mean_0
        posterior_cov = posterior_cov_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
             precision_EP, amplitude_EP, containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                posterior_cov_0=posterior_cov, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP,
                amplitude_EP_0=amplitude_EP,
                first_step=first_step, write=write)
            if verbose:
                print("({}), error={}".format(iteration, error))
        (weights,
        precision_EP,
        Lambda_cholesky,
        Lambda) = self.compute_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        if write:
            (posterior_means, posterior_covs, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_marginal_likelihoods) = containers
        # Try optimisation routine
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            np.diag(posterior_cov), posterior_mean, self.noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1,
            Lambda_cholesky, Lambda, weights)
        if self.kernel._general and self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.J * self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, self.noise_variance,
            t2, t3, t4, t5, Lambda, weights, indices)
        gx = gx[np.where(indices != 0)]
        if verbose:
            # print("gamma=", repr(self.gamma), ", ")
            # print("varphi=", self.kernel.varphi, ", ")
            # # print("varphi=", self.kernel.constant_variance, ", ")
            # print("noise_variance=", self.noise_variance, ", ")
            # print("scale=", self.kernel.scale, ", ")
            # print("\nfunction_eval={}\n jacobian_eval={}".format(
            #     fx, gx))
            print(
                "\ngamma={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.gamma, self.noise_variance, self.kernel.varphi, fx))
        # # Inverse of K
        # L_K, lower = cho_factor(self.K + self.jitter * np.eye(self.N))
        # L_KT_inv = solve_triangular(
        # L_K.T, np.eye(self.N), lower=True)
        # K_inv = solve_triangular(L_K, L_KT_inv, lower=False)
        # posterior_inv_cov = K_inv + np.diag(precision_EP)
        # #return fx, gx, posterior_mean, (posterior_inv_cov, True)
        return fx, gx, posterior_mean, (posterior_cov, False)

    def hyperparameter_training_step(
            self, theta, indices,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=True):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :type theta:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
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
        :return:
        """
        fx, gx, *_ = self.approximate_posterior(
            theta, indices, posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0, amplitude_EP_0, first_step,
            write, verbose)
        return fx, gx

    def compute_integrals_vector(
            self, posterior_variance, posterior_mean, noise_variance):
        """
        Compute the integrals required for the gradient evaluation.
        """
        noise_std = np.sqrt(noise_variance)
        mean = (posterior_mean * noise_variance
            + posterior_variance * self.gamma_ts) / (
                noise_variance + posterior_variance)
        sigma = np.sqrt(
            (noise_variance * posterior_variance) / (
            noise_variance + posterior_variance))
        a = mean - 5.0 * sigma
        b = mean + 5.0 * sigma
        h = b - a
        y_0 = np.zeros((20, self.N))
        t2 = np.zeros((self.N,))
        t3 = np.zeros((self.N,))
        t4 = np.zeros((self.N,))
        t5 = np.zeros((self.N,))
        t2 = fromb_t2_vector(
                y_0.copy(), mean, sigma,
                a, b, h,
                posterior_mean,
                posterior_variance,
                self.gamma_ts,
                self.gamma_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t3 = fromb_t3_vector(
                y_0.copy(), mean, sigma,
                a, b,
                h, posterior_mean,
                posterior_variance,
                self.gamma_ts,
                self.gamma_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t4 = fromb_t4_vector(
                y_0.copy(), mean, sigma,
                a, b,
                h, posterior_mean,
                posterior_variance,
                self.gamma_ts,
                self.gamma_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N),
        t5 = fromb_t5_vector(
                y_0.copy(), mean, sigma,
                a, b, h,
                posterior_mean,
                posterior_variance,
                self.gamma_ts,
                self.gamma_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        return (
            fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_variance,
                self.gamma_ts, self.gamma_tplus1s,
                noise_std, self.EPS, self.EPS_2, self.N),
            t2,
            t3,
            t4,
            t5
        )

    def objective(
        self, precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda,
        weights):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\Phi)} =,

            where :math:`F(\Phi)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg precision_EP:
        :type precision_EP:
        :arg posterior_mean:
        :type posterior_mean:
        :arg t1:
        :type t1:
        :arg Lambda_cholesky:
        :type Lambda_cholesky:
        :arg Lambda:
        :type Lambda:
        :arg weights:
        :type weights:
        :returns: fx
        :rtype: float
        """
        # Fill possible zeros in with machine precision
        precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        fx = -np.sum(np.log(np.diag(Lambda_cholesky)))  # log det Lambda
        fx -= 0.5 * posterior_mean.T @ weights
        fx -= 0.5 * np.sum(np.log(precision_EP))
        fx -= 0.5 * np.sum(np.divide(np.diag(Lambda), precision_EP))
        fx += np.sum(t1)
        # Regularisation - penalise large varphi (overfitting)
        # fx -= 0.1 * self.kernel.varphi
        return -fx

    def objective_gradient(
            self, gx, intervals, varphi, noise_variance,
            t2, t3, t4, t5, Lambda, weights, indices):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\Phi)}{\partial \Phi}}

            where :math:`F(\Phi)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\Phi` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg intervals:
        :type intervals:
        :arg varphi:
        :type varphi:
        :arg noise_variance:
        :type noise_variance:
        :arg t2:
        :type t2:
        :arg t3:
        :type t3:
        :arg t4:
        :type t4:
        :arg t5:
        :type t5:
        :arg Lambda:
        :type Lambda:
        :arg weights:
        :type weights:
        :return: gx
        :rtype: float
        """
        if indices is not None:
            # Update gx
            if indices[0]:
                # For gx[0] -- ln\sigma
                gx[0] = np.sum(t5 - t4)
                # gx[0] *= -0.5 * noise_variance  # This is a typo in the Chu code
                gx[0] *= np.sqrt(noise_variance)
            # For gx[1] -- \b_1
            if indices[1]:
                gx[1] = np.sum(t3 - t2)
            # For gx[2] -- ln\Delta^r
            for j in range(2, self.J):
                if indices[j]:
                    targets = self.t_train[self.grid]
                    gx[j] = np.sum(t3[targets == j - 1])
                    gx[j] -= np.sum(t2[targets == self.J - 1])
                    # TODO: check this, since it may be an `else` condition!!!!
                    gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                    gx[j] *= intervals[j - 2]
            # For gx[self.J] -- scale
            if indices[self.J]:
                # For gx[self.J] -- s
                # TODO: Need to check this is correct: is it directly analogous to
                # gradient wrt log varphi?
                partial_K_s = self.kernel.kernel_partial_derivative_s(
                    self.X_train, self.X_train)
                # VC * VC * a' * partial_K_varphi * a / 2
                gx[self.J] = varphi * 0.5 * weights.T @ partial_K_s @ weights  # That's wrong. not the same calculation.
                # equivalent to -= varphi * 0.5 * np.trace(Lambda @ partial_K_varphi)
                gx[self.J] -= varphi * 0.5 * np.sum(np.multiply(Lambda, partial_K_s))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[self.J] -= 0.1 * varphi
                gx[self.J] *= 2.0  # since varphi = kappa / 2
            # For gx[self.J + 1] -- varphi
            if indices[self.J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    self.X_train, self.X_train)
                if self.kernel._general and self.kernel._ARD:
                    raise ValueError("TODO")
                # elif 1:
                #     gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights
                # TODO: This needs fixing/ checking vs original code
                elif 0:
                    for l in range(self.kernel.L):
                        K = self.kernel.num_hyperparameters[l]
                        KK = 0
                        for k in range(K):
                            gx[self.J + KK + k] = varphi[l] * 0.5 * weights.T @ partial_K_varphi[l][k] @ weights
                        KK += K
                else:
                    # VC * VC * a' * partial_K_varphi * a / 2
                    gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights  # That's wrong. not the same calculation.
                    # equivalent to -= varphi * 0.5 * np.trace(Lambda @ partial_K_varphi)
                    gx[self.J + 1] -= varphi * 0.5 * np.sum(np.multiply(Lambda, partial_K_varphi))
                    # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                    # gx[self.J] -= 0.1 * varphi
        return -gx

    def approximate_evidence(self, mean_EP, precision_EP, amplitude_EP, posterior_cov):
        """
        TODO: check and return line could be at risk of overflow
        Compute the approximate evidence at the EP solution.

        :return:
        """
        intermediate_vector = np.multiply(mean_EP, precision_EP)
        B = intermediate_vector.T @ posterior_cov @ intermediate_vector - np.multiply(
            intermediate_vector, mean_EP)
        Pi_inv = np.diag(1. / precision_EP)
        return (
            np.prod(
                amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
                / np.sqrt(np.linalg.det(np.add(Pi_inv, self.K))))

    def compute_weights(
        self, precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L=None, Lambda=None):
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
        :arg L: . Default `None`.
        :arg Lambda: . Default `None`.
        """
        if np.any(precision_EP == 0.0):
            # TODO: Only check for equilibrium if it has been updated in this swipe
            warnings.warn("Some sample(s) have not been updated.\n")
            precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        Pi_inv = np.diag(1. / precision_EP)
        if L is None or Lambda is None:
            # SS
            # Cholesky factorisation  -- O(N^3)
            L = np.linalg.cholesky(np.add(Pi_inv, self.K))
            # Inverse at each hyperparameter update - TODO: DOES NOT USE CHOLESKY
            L_inv = np.linalg.inv(L)
            Lambda = L_inv.T @ L_inv  # (N, N)
            # (L, lower) = cho_factor(np.add(Pi_inv, self.K))
            # # Back substitution
            # L_T_inv = solve_triangular(L.T, np.eye(self.N), lower=True)
            # # Forward substitution
            # Lambda = solve_triangular(L, L_T_inv, lower=False)
        # TODO: Need initialisation for the cholesky factors
        weights = Lambda @ mean_EP
        if np.any(
            np.abs(weights - grad_Z_wrt_cavity_mean) > np.sqrt(self.EPS)):
            warnings.warn("Fatal error: the weights are not in equilibrium wit"
                "h the gradients".format(
                    weights, grad_Z_wrt_cavity_mean))
        return weights, precision_EP, L, Lambda


class LaplaceOrdinalGP(Approximator):
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
    def __init__(
        self, gamma, noise_variance=1.0, *args, **kwargs):
        # gamma_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`LaplaceOrdinalGP` Approximator object.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.

        :returns: An :class:`EPOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        # if gamma_hyperparameters is not None:
        #     warnings.warn("gamma_hyperparameters set as {}".format(gamma_hyperparameters))
        #     self.gamma_hyperparameters = gamma_hyperparameters
        # else:
        #     self.gamma_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        if self.kernel._ARD:
            raise ValueError(
                "The kernel must not be _ARD type (kernel._ARD=1),"
                " but ISO type (kernel._ARD=0). (got {}, expected)".format(
                self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError(
                "The kernel must not be general type (kernel._general=1),"
                " but simple type (kernel._general=0). "
                "(got {}, expected)".format(self.kernel._general, 0))
        # self.EPS = 0.001  # Acts as a machine tolerance
        self.EPS = 1e-4
        # self.EPS = 1e-6
        self.EPS_2 = self.EPS**2
        # self.jitter = 1e-6
        self.jitter = 1e-10  # 1e-8, 1e-10 was too small for covariance parameterisation
        # Initiate hyperparameters
        self.hyperparameters_update(gamma=gamma, noise_variance=noise_variance)

    def hyperparameters_update(
        self, gamma=None, varphi=None, scale=None, noise_variance=None):
        """
        TODO: can probably collapse this code into other hyperparameter update
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        approximator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        :meth:`approximate`. Problem is, if it changes at approximate time, then a
        hyperparameter update needs to be called.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi:
        :type varphi:
        :arg scale:
        :type scale:
        :arg noise_variance:
        :type noise_variance:
        """
        # TODO: can't this be done in an _update_prior()
        #self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        if gamma is not None:
            # Convert gamma to numpy array
            gamma = np.array(gamma)
            # Not including -\infty or \infty
            if np.shape(gamma)[0] == self.J - 1:
                gamma = np.append(gamma, np.inf)  # Append \infty
                gamma = np.insert(gamma, np.NINF)  # Insert -\infty at index 0
                pass  # correct format
            # Not including one cutpoints
            elif np.shape(gamma)[0] == self.J:
                if gamma[-1] != np.inf:
                    if gamma[0] != np.NINF:
                        raise ValueError(
                            "The last cutpoint parameter must be numpy.inf, or"
                            " the first cutpoint parameter must be numpy.NINF "
                            "(got {}, expected {})".format(
                            [gamma[0], gamma[-1]], [np.inf, np.NINF]))
                    else:  #gamma[0] is -\infty
                        gamma.append(np.inf)
                        pass  # correct format
                else:
                    gamma = np.insert(gamma, np.NINF)
                    pass  # correct format
            # Including all the cutpoints
            elif np.shape(gamma)[0] == self.J + 1:
                if gamma[0] != np.NINF:
                    raise ValueError(
                        "The cutpoint parameter \gamma must be numpy.NINF "
                        "(got {}, expected {})".format(gamma[0], np.NINF))
                if gamma[-1] != np.inf:
                    raise ValueError(
                        "The cutpoint parameter \gamma_J must be "
                        "numpy.inf (got {}, expected {})".format(
                            gamma[-1], np.inf))
                pass  # correct format
            else:
                raise ValueError(
                    "Could not recognise gamma shape. "
                    "(np.shape(gamma) was {})".format(np.shape(gamma)))
            assert gamma[0] == np.NINF
            assert gamma[-1] == np.inf
            assert np.shape(gamma)[0] == self.J + 1
            if not all(
                    gamma[i] <= gamma[i + 1]
                    for i in range(self.J)):
                raise CutpointValueError(gamma)
            self.gamma = gamma
            self.gamma_ts = gamma[self.t_train]
            self.gamma_tplus1s = gamma[self.t_train + 1]
        if varphi is not None or scale is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, scale=scale)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior()
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
        # Posterior covariance is calculated iteratively in EP,
        # so no update here.

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
        intermediate_vector = np.multiply(mean_EP, precision_EP)
        B = intermediate_vector.T @ posterior_cov @ intermediate_vector - intermediate_vector.T @ mean_EP
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

    def _predict_vector(
            self, gamma, invcov,
            posterior_mean, varphi, noise_variance, X_test):
        """
        Make EP prediction over classes of X_test given the posterior samples.
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg posterior_mean:
        :arg varphi:
        :arg X_test: The new data points, array like (N_test, D).
        :arg Lambda: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        :rtype tuple: ((N_test, J), (N_test,), (N_test,))
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
        self.kernel.varphi = varphi
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        c_news = self.kernel.kernel_prior_diagonal(X_test)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = invcov @ C_news  # (N, N_test)
        intermediate_scalars = np.einsum(
            'ij, ij -> j', C_news, intermediate_vectors)
        posterior_var = c_news - intermediate_scalars
        posterior_pred_var = posterior_var + noise_variance
        posterior_std = np.sqrt(posterior_var)
        posterior_pred_std = np.sqrt(posterior_pred_var)
        posterior_pred_mean = np.einsum(
            'ij, i -> j', intermediate_vectors, posterior_mean)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(
                gamma[j + 1], posterior_pred_mean), posterior_pred_std)
            Z2 = np.divide(
                np.subtract(gamma[j], posterior_pred_mean), posterior_pred_std)
            predictive_distributions[:, j] = norm.cdf(Z1) - norm.cdf(Z2)
        return predictive_distributions, posterior_pred_mean, posterior_std

    def predict(
        self, gamma, invcov, posterior_mean,
        varphi, noise_variance, X_test, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg y: The approximate posterior mean of the latent variable Y.
        :arg varphi: The approximate posterior mean of the
            hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError(
                "For the ordinal likelihood approximator,the kernel "
                "must not be _ARD type (kernel._ARD=1), but"
                " ISO type (kernel._ARD=0). (got {}, expected)".format(
                    self.kernel._ARD, 0))
        else:
            if vectorised:
                return self._predict_vector(
                    gamma, invcov, posterior_mean, varphi, noise_variance, X_test)
            else:
                return ValueError(
                    "The scalar implementation has been "
                    "superseded. Please use "
                    "the vector implementation.")

    def grid_over_hyperparameters(
            self, domain, res,
            indices=None,
            posterior_mean_0=None,
            first_step=1, write=False, verbose=True):  # TODO False
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        indices.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        Phi_new, fxs,
        gxs, gx_0, intervals,
        indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.gamma)
        for i, phi in enumerate(Phi_new):
            self._grid_over_hyperparameters_update(
                phi, indices, self.gamma)
            if verbose:
                print(
                    "gamma_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "scale_0 = {}".format(
                        self.gamma, self.kernel.varphi, self.noise_variance,
                        self.kernel.scale))
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            while error / steps > self.EPS_2:
                iteration += 1
                (error, weight, posterior_mean, containers) = self.approximate(
                    steps, posterior_mean_0=posterior_mean,
                    first_step=first_step, write=write)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(Phi_new)))
            (weight, precision,
            w1, w2, g1, g2, v1, v2, q1, q2,
            L_cov, cov, Z) = self.compute_weights(
                posterior_mean)
            if write:
                (posterior_means, approximate_marginal_likelihoods) = containers
            fx = self.objective(weight, precision, L_cov, Z)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi, self.noise_variance, self.noise_std,
                w1, w2, g1, g2, v1, v2, q1, q2, cov, weight, self.N, self.K, precision, indices)
            gxs[i] = gx[indices_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(
                    self.kernel.varphi, self.noise_variance, fx))
        if x2s is not None:
            return (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
        else:
            return (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    def hyperparameter_training_step(
            self, theta, indices,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=True):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :type theta:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
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
        :return:
        """
        fx, gx, *_ = self.approximate_posterior(
            theta, indices, posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0, amplitude_EP_0, first_step,
            write, verbose)
        return fx, gx

    def compute_weights(
        self, posterior_mean):
        """
        Compute the regression weights required for the objective evaluation and its gradients.

        A matrix inverse is always required to evaluate the objective.

        :arg posterior_mean:
        """
        (Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s, *_) = truncated_norm_normalising_constant(
            self.gamma_ts, self.gamma_tplus1s, self.noise_std, posterior_mean, self.EPS)
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
        inv_cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        return weight, precision, w1, w2, g1, g2, v1, v2, q1, q2, L_cov, inv_cov, Z

    def objective(self, weight, precision, L_cov, Z):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\Phi)} =,

            where :math:`F(\Phi)` is the variational lower bound of the log
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
        fx += 0.5 * weight.T @ self.K @ weight
        fx += np.sum(np.log(np.diag(L_cov)))
        fx += 0.5 * np.sum(np.log(precision))
        # fx = -fx
        return fx

    def objective_gradient(
            self, gx, intervals, varphi, noise_variance, noise_std,
            w1, w2, g1, g2, v1, v2, q1, q2, cov, weight, N, K, precision, indices):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\Phi)}{\partial \Phi}}

            where :math:`F(\Phi)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\Phi` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg gx: 
        :type gx:
        :arg intervals:
        :type intervals:
        :arg varphi:
        :type varphi:
        :arg float noise_variance:
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
        :arg indices:
        :type indices:
        :return: gx
        :rtype: float
        """
        if indices is not None:
            # compute a diagonal
            dsigma = cov @ K
            diag = np.diag(dsigma) / precision
            # partial lambda / partial theta_b = - partial lambda / partial f (* SIGMA)
            t1 = ((w2 - w1) - 3.0 * (w2 - w1) * (g2 - g1) - 2.0 * (w2 - w1)**3 - (v2 - v1)) / noise_variance
            # Update gx
            if indices[0]:
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
            if indices[1]:
                # For gx[1], \theta_b^1
                t2 = dsigma @ precision
                t2 = t2 / precision
                gx[1] -= np.sum(w2 - w1)
                gx[1] += 0.5 * np.sum(t1 * (1 - t2) * diag)
                gx[1] = gx[1] / noise_std
            # For gx[2] -- ln\Delta^r
            for j in range(2, self.J):
                targets = self.t_train[self.grid]
                # Prepare D f / D delta_l
                cache0 = -(g2 + (w2 - w1) * w2) / noise_variance
                cache1 = - (g2 - g1 + (w2 - w1)**2) / noise_variance
                if indices[j]:
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
                    gx[j] += 0.5 * np.sum((temp[idxj] - t2[idxj] * t1[idxj]) * diag[idxj])
                    gx[j] -= np.sum(w2[idxg] - w1[idxg])
                    gx[j] += 0.5 * np.sum(t1[idxg] * (1.0 - t2[idxg]) * diag[idxg])
                    gx[j] += 0.5 * np.sum(-t2[idxl] * t1[idxl] * diag[idxl])
                    gx[j] = gx[j] * intervals[j - 1] / noise_std
            # For gx[self.J] -- scale
            if indices[self.J]:
                raise ValueError("TODO")
            # For gx[self.J + 1] -- varphi
            if indices[self.J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    self.X_train, self.X_train)
                if self.kernel._general and self.kernel._ARD:
                    raise ValueError("TODO")
                else:
                    dmat = partial_K_varphi @ cov
                    t2 = (dmat @ weight) / precision
                    gx[self.J + 1] -= varphi * 0.5 * weight.T @ partial_K_varphi @ weight
                    gx[self.J + 1] += varphi * 0.5 * np.sum((-diag * t1 * t2) / (noise_std))
                    gx[self.J + 1] += varphi * 0.5 * np.sum(np.multiply(cov, partial_K_varphi))
                    # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                    # gx[self.J] -= 0.1 * varphi
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
            posterior_mean_0 = self.gamma_ts.copy()
            idx = np.where(self.t_train == 0)
            posterior_mean_0[idx] = self.gamma_tplus1s[idx]
        error = 0.0
        weight_0 = np.empty(self.N)
        inverse_variance_0 = np.empty(self.N)
        posterior_means = []
        posterior_precisions = []
        containers = (posterior_means, posterior_precisions)
        return (weight_0, inverse_variance_0, posterior_mean_0, containers, error)

    def approximate(
            self, steps, posterior_mean_0=None, first_step=1, write=False):
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
        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi: Initialisation of hyperparameter approximate posterior means.
            If `None` then initialised to ones, default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to one, default `None`.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
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
        (weight_0,
        inverse_variance_0,
        posterior_mean, containers, error) = self._approximate_initiate(
            posterior_mean_0)
        (posterior_means, posterior_precisions) = containers
        for step in trange(first_step, first_step + steps,
                        desc="Laplace GP priors Approximator Progress",
                        unit="iterations", disable=True):
            (Z, norm_pdf_z1s, norm_pdf_z2s,
                z1s, z2s,
                norm_cdf_z1s, norm_cdf_z2s
                ) = truncated_norm_normalising_constant(
                self.gamma_ts, self.gamma_tplus1s, self.noise_std,
                posterior_mean, self.EPS, upper_bound=self.upper_bound)
            weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / self.noise_std
            z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
            z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
            precision  = weight**2 + (z2s * norm_pdf_z2s - z1s * norm_pdf_z1s) / Z / self.noise_variance
            L_cov = self.K + np.diag(1. / precision)
            m = - self.K @ weight + posterior_mean
            L_cov, _ = cho_factor(L_cov)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.N), lower=True)
            invcov = solve_triangular(L_cov, L_covT_inv, lower=False)
            t1 = - (invcov @ m) / precision
            posterior_mean += t1
            error = np.abs(max(t1.min(), t1.max(), key=abs))
            if write is True:
                posterior_means.append(posterior_mean)
                posterior_precisions.append(posterior_precisions)
        containers = (posterior_means, posterior_precisions)
        return error, weight, posterior_mean, containers

    def approximate_posterior(
            self, theta, indices, steps=None,
            posterior_mean_0=None,
            calculate_posterior_cov=True, first_step=1,
            write=False, verbose=False):
        """
        Newton-Raphson.

        :arg theta: (log-)hyperparameters to be optimised.
        :type theta:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        posterior_mean = posterior_mean_0
        while error / steps > self.EPS_2 and iteration < 10:  # TODO is this overkill?
            iteration += 1
            (error, weight, posterior_mean, containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=write)
            if verbose:
                print("({}), error={}".format(iteration, error))
        (weight, precision,
        w1, w2, g1, g2, v1, v2, q1, q2,
        L_cov, cov, Z) = self.compute_weights(
            posterior_mean)
        if write:
            (posterior_means, approximate_marginal_likelihoods) = containers
        fx = self.objective(weight, precision, L_cov, Z)
        if self.kernel._general and self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.J * self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, self.noise_variance, self.noise_std,
            w1, w2, g1, g2, v1, v2, q1, q2, cov, weight, self.N, self.K, precision, indices)
        gx = gx[np.where(indices != 0)]
        if verbose:
            # print("gamma=", repr(self.gamma), ", ")
            # print("varphi=", self.kernel.varphi, ", ")
            # # print("varphi=", self.kernel.constant_variance, ", ")
            # print("noise_variance=", self.noise_variance, ", ")
            # print("scale=", self.kernel.scale, ", ")
            # print("\nfunction_eval={}\n jacobian_eval={}".format(
            #     fx, gx))
            print(
                "\ngamma={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.gamma, self.noise_variance, self.kernel.varphi, fx))

        if calculate_posterior_cov:
            # Inverse of K
            L_K, lower = cho_factor(self.K + self.jitter * np.eye(self.N))
            L_KT_inv = solve_triangular(
                L_K.T, np.eye(self.N), lower=True)
            K_inv = solve_triangular(L_K, L_KT_inv, lower=False)
            posterior_inv_cov = K_inv + np.diag(precision)
            return fx, gx, posterior_mean, (posterior_inv_cov, True)
        elif calculate_posterior_cov:
            return fx, gx, posterior_mean, (self.noise_variance * self.K @ cov, False)
        elif calculate_posterior_cov is None:
            return fx, gx, posterior_mean, (cov, True)


        

  
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
