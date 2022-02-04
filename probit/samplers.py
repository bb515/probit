from abc import ABC, abstractmethod

from numpy import False_
from probit.approximators import Approximator, InvalidApproximator, EPOrdinalGP
from probit.priors import prior, prior_reparameterised
from probit.proposals import proposal, proposal_reparameterised, proposal_initiate
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
from .utilities import (
    norm_z_pdf, norm_cdf, sample_y,
    truncated_norm_normalising_constant)
import numba_scipy  # Numba overloads for scipy and scipy.special
from scipy.stats import norm, uniform, expon
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from tqdm import trange
import warnings
import matplotlib.pyplot as plt


class Sampler(ABC):
    """
    # TODO: The Base class for samplers is very similar to the base class for
    # estimators. Consider merging them.
    Base class for samplers. This class allows users to define a classification
    problem, get predictions using a exact Bayesian inference.

    All samplers must define an init method, which may or may not inherit
        Sampler as a parent class using `super()`.
    All samplers that inherit Sampler define a number of methods that return
        the posterior samples.
    All samplers must define a _sample_initiate method that is used to initate
        the sampler.
    All samplers must define an predict method can be  used to make predictions
        given test data.
    """

    @abstractmethod
    def __init__(self, kernel, X_train, t_train, J, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete Sampler.

        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.    
        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :arg t_train: (N, ) The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg J: The number of (ordinal) classes.
        :arg str write_path: Write path for outputs.

        :returns: A :class:`Sampler` object
        """
        if not (isinstance(kernel, Kernel)):
            raise InvalidKernel(kernel)
        else:
            self.kernel = kernel
        if write_path is None:
            self.write_path = None
        else:
            self.write_path = pathlib.Path(write_path)
        self.N = np.shape(X_train)[0]
        self.D = np.shape(X_train)[1]
        self.X_train = X_train
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise ValueError(
                "t must contain only integer values (got {})".format(t_train))
        if t_train.dtype != int:
            raise TypeError(
                "t must contain only integer values (got {})".format(
                    t_train))
        else:
            self.t_train = t_train
        self.J = J
        if self.kernel._ARD:
            sigma = np.reshape(self.kernel.sigma, (self.J, 1))
            tau = np.reshape(self.kernel.tau, (self.J, 1))
            self.sigma = np.tile(sigma, (1, self.D))  # (J, D)
            self.tau = np.tile(tau, (1, self.D))  # (J, D)
        else:
            self.sigma = self.kernel.sigma
            self.tau = self.kernel.tau
        # See GPML by Williams et al. for a good explanation of jitter
        self.jitter = 1e-8 
        self.upper_bound = 6.0 # TODO: needed?
        self.upper_bound2 = 18.0
        self.EPS = 0.0001 
        warnings.warn("Updating prior covariance.")
        self._update_prior()

    @abstractmethod
    def _sample_initiate(self):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """

    @abstractmethod
    def sample(self):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
        """

    @abstractmethod
    def predict(self):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
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
            self.noise_std, m, self.EPS)
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
        # return -0.5 * np.log(2 * np.pi) - half_log_det_cov - 0.5 * np.einsum('ik, ik -> k', (cov_inv @ xs.T), xs.T)  # log likelihoods

    # def _log_multivariate_normal_pdf(self, x, cov_inv, half_log_det_cov, mean=None):
    #     """Get the pdf of the multivariate normal distribution."""
    #     if mean is not None:
    #         x -= mean
    #     # Backwards substitution
    #     intermediate_vector = solve_triangular(L_cov.T, x, lower=True)
    #     # Forwards substitution
    #     intermediate_vector = solve_triangular(L_cov, intermediate_vector, lower=False)
    #     return -0.5 * np.log(2 * np.pi) - half_log_det_cov - 0.5 * np.dot(x, intermediate_vector)  # log likelihood

    def _vector_probit_likelihood(self, m_ns, gamma):
        """
        Get the probit likelihood given GP posterior mean samples and gamma sample.

        :return distribution_over_classes: the (N_samples, K) array of values.
        """
        N_samples = np.shape(m_ns)[0]
        distribution_over_classess = np.empty((N_samples, self.K))
        # Special case for gamma[-1] == np.NINF, gamma[0] == 0.0
        distribution_over_classess[:, 0] = norm.cdf(np.subtract(gamma[0], m_ns))
        for k in range(1, self.K + 1):
            gamma_k = gamma[k]
            gamma_k_1 = gamma[k - 1]
            distribution_over_classess[:, k - 1] = norm.cdf(
                np.subtract(gamma_k, m_ns)) - norm.cdf(np.subtract(gamma_k_1, m_ns))
        return distribution_over_classess

    def _probit_likelihood(self, m_n, gamma):
        """
                TODO: 01/03 this was refactored without testing. Test it.
        Get the probit likelihood given GP posterior mean sample and gamma sample.

        :return distribution_over_classes: the (K, ) array of.
        """
        distribution_over_classes = np.empty(self.K)
        # Special case for gamma[-1] == np.NINF, gamma[0] == 0.0
        distribution_over_classes[0] = norm.cdf(gamma[0] - m_n)
        for k in range(1, self.K + 1):
            gamma_k = gamma[k]  # remember these are the upper bounds of the classes
            gamma_k_1 = gamma[k - 1]
            distribution_over_classes[k - 1] = norm.cdf(gamma_k - m_n) - norm.cdf(gamma_k_1 - m_n)
        return distribution_over_classes

    def _predict_scalar(self, y_samples, gamma_samples, x_test):
        """
            TODO: This code was refactored on 01/03/2021 and 21/06/2021 without testing. Test it.
        Superseded by _predict_vector.

        Make gibbs prediction over classes of X_test[0] given the posterior samples.

        :arg Y_samples: The Gibbs samples of the latent variable Y.
        :arg x_test: The new data point, array like (1, D).
        :return: A Monte Carlo estimate of the class probabilities.
        """
        cs_new = np.diag(self.kernel.kernel(x_test[0], x_test[0]))  # (1, )
        Cs_new = self.kernel.kernel_vector(x_test, self.X_train)
        intermediate_vector = self.Sigma @ Cs_new  # (N, N_test)
        intermediate_scalar = Cs_new.T @ intermediate_vector
        n_posterior_samples = np.shape(y_samples)[0]
        # Sample pmf over classes
        distribution_over_classes_samples = []
        for i in range(n_posterior_samples):
            y = y_samples[i]
            gamma = gamma_samples[i]
            # Take a sample of m from a GP regression
            mean = y.T @ intermediate_vector  # (1, )
            var = cs_new - intermediate_scalar  # (1, )
            m = norm.rvs(loc=mean, scale=var)
            # Take a sample of gamma from the posterior gamma|y, t
            # This is proportional to the likelihood y|gamma, t since we have a flat prior
            uppers = np.empty(self.K - 2)
            locs = np.empty(self.K - 2)
            for k in range(1, self.K - 1):
                indeces = np.where(self.t_train == k)
                indeces2 = np.where(self.t_train == k + 1)
                if indeces2:
                    uppers[k - 1] = np.min(np.append(y[indeces2], gamma[k + 2]))
                else:
                    uppers[k - 1] = gamma[k + 2]
                if indeces:
                    locs[k - 1] = np.max(np.append(y[indeces], gamma[k]))
                else:
                    locs[k - 1] = gamma[k]
            gamma[0] = np.NINF
            gamma[1:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            gamma[-1] = np.inf
            # Calculate the measurable function and append the resulting MC sample
            distribution_over_classes_samples.append(self._probit_likelihood(m, gamma))

        monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
        return monte_carlo_estimate
   
    def _predict_vector(self, y_samples, gamma_samples, X_test):
        """
        Make gibbs prediction over classes of X_test given the posterior samples.

        :arg y_samples: The Gibbs samples of the latent variable Y.
        :arg X_test: The new data points, array like (N_test, D).
        :return: A Monte Carlo estimate of the class probabilities.
        """
        # N_test = np.shape(X_test)[0]
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = self.Sigma @ Cs_news  # (N, N_test)
        intermediate_vectors_T = intermediate_vectors.T
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        n_posterior_samples = np.shape(y_samples)[0]
        # Sample pmf over classes
        distribution_over_classes_sampless = []
        for i in range(n_posterior_samples):
            y = y_samples[i]
            gamma = gamma_samples[i]
            mean = intermediate_vectors_T @ y  # (N_test, )
            var = cs_news - intermediate_scalars  # (N_test, )
            m_ns = norm.rvs(loc=mean, scale=var)
            # Take a sample of gamma from the posterior gamma|y, t
            # This is proportional to the likelihood y|gamma, t since we have a flat prior
            # uppers = np.empty(self.K - 2)
            # locs = np.empty(self.K - 2)
            # for k in range(1, self.K - 1):
            #     indeces = np.where(self.t_train == k)
            #     indeces2 = np.where(self.t_train == k + 1)
            #     if indeces2:
            #         uppers[k - 1] = np.min(np.append(y[indeces2], gamma[k + 1]))
            #     else:
            #         uppers[k - 1] = gamma[k + 1]
            #     if indeces:
            #         locs[k - 1] = np.max(np.append(y[indeces], gamma[k - 1]))
            #     else:
            #         locs[k - 1] = gamma[k - 1]
            # gamma[1:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            # gamma[0] = 0.0
            # gamma[-1] = np.inf
            # Calculate the measurable function and append the resulting MC sample
            distribution_over_classes_sampless.append(self._vector_probit_likelihood(m_ns, gamma))
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            # TODO: How do we know that 1000 samples is enough to converge?
            #  Goes with root n_samples but depends on the estimator variance
        # TODO: Could also get a variance from the MC estimate.
        return (1. / n_posterior_samples) * np.sum(distribution_over_classes_sampless, axis=0)

    def predict(self, y_samples, gamma_samples, X_test, vectorised=True):
        if self.kernel.general_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            return ValueError("ARD kernel may not be used in the ordered likelihood estimator.")
        else:
            if vectorised:
                return self._predict_vector(y_samples, gamma_samples, X_test)
            else:
                return self._predict_scalar(y_samples, gamma_samples, X_test)

    def _hyperparameter_initialise(self, theta, indices):
        """
        Initialise the hyperparameters.

        :arg theta: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}}
                \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\varphi}],

            where :math:`\sigma` is the noise standard deviation,
            :math:`\b_{1}` is the first cutpoint, :math:`\Delta_{l}` is the
            :math:`l`th cutpoint interval, :math:`\varphi` is the single
            shared lengthscale parameter or vector of parameters in which
            there are in the most general case J * D parameters.
        :type theta: :class:`numpy.ndarray`
        :return: (gamma, noise_variance) the updated cutpoints and noise variance.
        :rtype: (2,) tuple
        """
        # Initiate at None since those that are None do not get updated        
        noise_variance = None
        gamma = None
        scale = None
        varphi = None
        index = 0
        if indices[0]:
            noise_std = np.exp(theta[0])
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
            gamma[1] = theta[1]
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                gamma[j] = gamma[j - 1] + np.exp(theta[j])
                index += 1
        if indices[self.J]:
            scale_std = np.exp(theta[self.J])
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
                varphi = np.exp(theta[self.J])
                index += 1
        # Update prior and posterior covariance
        self.hyperparameters_update(
            gamma=gamma, varphi=varphi, scale=scale,
            noise_variance=noise_variance)
        # TODO not sure if needed
        intervals = self.gamma[2:self.J] - self.gamma[1:self.J - 1]
        return intervals

    def _update_prior(self, K=None):
        """
        Update prior covariances.

        :arg K: Optionally supply prior covariance if it has already been calculated.
        :type K:
        """
        if K is None:
            K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.K = K

    def _update_posterior(self, L_K=None, log_det_K=None):
        """
        Update posterior covariances.

        :arg L_K: Optionally supply prior covariance cholesky factor if it has already been calculated.
        :type L_K:        
        """
        # TODO: I should be consitant with which cholesky using.
        if L_K is None:
            self.L_K, lower = cho_factor(self.jitter * np.eye(self.N) + self.K)
            # self.L_K = np.linalg.cholesky(self.jitter * np.eye(self.N) + self.K)
        if log_det_K is None:
            self.log_det_K = 2 * np.sum(np.log(np.diag(self.L_K)))
        (self.L_cov, self.lower) = cho_factor(
            self.noise_variance * np.eye(self.N) + self.K)
        self.log_det_cov = -2 * np.sum(np.log(np.diag(self.L_cov)))
        L_covT_inv = solve_triangular(
            self.L_cov.T, np.eye(self.N), lower=True)
        self.cov = solve_triangular(self.L_cov, L_covT_inv, lower=False)
        #self.trace_cov = np.sum(np.diag(self.cov))
        #self.trace_Sigma_div_var = np.einsum('ij, ij -> ', self.K, self.cov)
        Sigma = self.noise_variance * self.K @ self.cov
        # TODO: Is there a better way? Also, decide on a convention for which
        # cholesky algorithm and stick to it
        self.L_Sigma = np.linalg.cholesky(Sigma + self.jitter * np.eye(self.N))

    def hyperparameters_update(
        self, gamma=None, varphi=None, scale=None, noise_variance=None,
        K=None, L_K=None, log_det_K=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior
        covariances. Note that hyperparameters are fixed parameters of the
        estimator, not variables that change during the estimation. The strange
        thing is that hyperparameters can be absorbed into the set of variables
        and so the definition of hyperparameters and variables becomes
        muddled. Since varphi can be a variable or a parameter, then optionally
        initiate it as a parameter, and then intitate it as a variable within
        estimate. Problem is, if it changes at estimate time, then a
        hyperparameter update needs to be called.

        :arg gamma: (J + 1, ) array of the cutpoints.
        :type gamma: :class:`np.ndarray`.
        :arg varphi:
        :type varphi:
        :arg scale:
        :type scale:
        :arg noise_variance:
        :type noise_variance:
        :arg K: Optional argument supplied if K is already known.
        :arg L_K: Optional argument supplied if L_K is already known.
        :arg log_det_K: Optional argument supplied if log_det_K is already known.
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
            self.gamma_tplus1s = gamma[self.t_trainplus1]
        if varphi is not None or scale is not None:
            self.kernel.update_hyperparameter(
                varphi=varphi, scale=scale)
            # Update prior covariance
            warnings.warn("Updating prior covariance.")
            self._update_prior(K=K)
            warnings.warn("Done posterior covariance.")
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
        # Update posterior covariance
        print("posterior update")
        warnings.warn("Updating posterior covariance.")
        self._update_posterior(L_K=L_K, log_det_K=log_det_K)
        warnings.warn("Done updating posterior covariance.")

    def get_theta(self, indices):
        """
        Get the parameters (theta) for unconstrained sampling.

        :arg indices: Indicator array of the hyperparameters to sample over.
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

    def _grid_over_hyperparameters_initiate(
            self, res, domain, indices, gamma):
        """
        Initiate metadata and hyperparameters for plotting the objective
        function surface over hyperparameters.

        :arg axis_scale:
        :type axis_scale:
        :arg int res:
        :arg range_x1:
        :type range_x1:
        :arg range_x2:
        :type range_x2:
        :arg int J:
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

        :arg kernel:
        :type kernel:
        :arg phi: The updated values of the hyperparameters.
        :type phi:
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


class GibbsOrdinalGP(Sampler):
    """
    Gibbs sampler for ordinal GP regression. Inherits the sampler ABC.
    """

    def __init__(self, gamma, noise_variance=1.0, *args, **kwargs):
        """
        Create an :class:`GibbsOrdinalGP` sampler object.

        :returns: An :class:`GibbsOrdinalGP` object.
        """
        super().__init__(*args, **kwargs)
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
        self.EPS = 0.0001
        self.EPS_2 = self.EPS**2 
        self.upper_bound = 6
        self.upper_bound2 = 30
        self.jitter = 1e-6
        self.t_trainplus1 = self.t_train + 1
        # Initiate hyperparameters
        self.hyperparameters_update(gamma=gamma, noise_variance=noise_variance)

    def _m_tilde(self, y, cov, K):
        """
        TODO: consider moving to a utilities file.
        Return the posterior mean of m given y.

        2021 Page Eq.()

        :arg y: (N,) array
        :type y: :class:`np.ndarray`
        :arg cov:
        :type cov:
        :arg K:
        :type K:
        """
        nu = cov @ y
        return K @ nu, nu  # (N, J), (N, )

    def sample_gibbs(self, m_0, steps, first_step=1):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: m (GP regression posterior means) and
            then over y (auxilliaries). In this sampler, gamma (cutpoint parameters) are fixed.

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.
        """
        # Initiate containers for samples
        m = m_0
        y_container = np.empty(self.N)
        m_samples = []
        y_samples = []
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Sample y from the usual full conditional
            y = sample_y(y_container.copy(), m, self.t_train, self.gamma, self.noise_std, self.N)
            # Calculate statistics, then sample other conditional
            m_tilde, _ = self._m_tilde(y, self.cov, self.K)
            m = m_tilde + self.L_Sigma @ norm.rvs(size=self.N)
            m_samples.append(m)
            y_samples.append(y)
        return np.array(m_samples), np.array(y_samples)

    def _sample_metropolis_within_gibbs_initiate(self, m_0, gamma_0):
        """
        Initialise variables for the sample method.
        TODO: 03/03/2021 The first Gibbs step is not robust to a poor choice of m_0 (it will never sample a y_1 within
            range of the cutpoints). Idea: just initialise y_0 and m_0 close to another.
            Start with an initial guess for m_0 based on a linear regression and then initialise y_0 with random N(0,1)
            samples around that. Need to test convergence for random init of y_0 and m_0.
        TODO: 26/12/2021 I think that I've fixed this issue.
        """
        m_samples = []
        y_samples = []
        gamma_samples = []
        gamma_0_prev_jplus1 = gamma_0[self.t_trainplus1]
        gamma_0_prev_j = gamma_0[self.t_train]
        return m_0, gamma_0, gamma_0_prev_jplus1, gamma_0_prev_j, m_samples, y_samples, gamma_samples

    def sample_metropolis_within_gibbs(self, indices, m_0, gamma_0, sigma_gamma, steps, first_step=1):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: m (GP regression posterior means) and
            then jointly (using a Metropolis step) over y (auxilliaries) and gamma (cutpoint parameters).
            The purpose of the Metroplis step is that it is allows quicker convergence of the iterates
            since the full conditional over gamma is really thin if the bins are full. We get around sampling
            from the full conditional by sampling from the joint full conditional y, \gamma using a
            Metropolis step.

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg y_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type y_0: :class:`np.ndarray`.
        :arg gamma_0: (K + 1, ) numpy.ndarray of the initial location of the sampler.
        :type gamma_0: :class:`np.ndarray`.
        :arg float sigma_gamma: The
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.
        """
        (m,
        gamma_prev, gamma_prev_jplus1, gamma_prev_j,
        m_samples, y_samples, gamma_samples,
        y_container) = self._sample_initiate(
            m_0, gamma_0)
        precision_gamma = 1. / sigma_gamma
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty gamma (J + 1, ) array to collect the upper cut-points for each class
            gamma = np.empty(self.J + 1)
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_J = +\infty
            gamma[0] = np.NINF
            gamma[-1] = np.inf
            for j in range(1, self.J):
                gamma_proposal = -np.inf
                if indices[j]:
                    while gamma_proposal <= gamma[j - 1] or gamma_proposal > gamma_prev[j + 1]:
                        gamma_proposal = norm.rvs(loc=gamma_prev[j], scale=sigma_gamma)
                else:
                    gamma_proposal = gamma_0[j]
                gamma[j] = gamma_proposal
            # Calculate acceptance probability
            num_2 = np.sum(np.log(
                    norm_cdf(precision_gamma * (gamma_prev[2:] - gamma_prev[1:-1]))
                    - norm_cdf(precision_gamma * (gamma[0:-2] - gamma_prev[1:-1]))
            ))
            den_2 = np.sum(np.log(
                    norm_cdf(precision_gamma * (gamma[2:] - gamma[1:-1]))
                    - norm_cdf(precision_gamma * (gamma_prev[0:-2] - gamma[1:-1]))
            ))
            gamma_jplus1 = gamma[self.t_trainplus1]
            gamma_prev_jplus1 = gamma_prev[self.t_trainplus1]
            gamma_j = gamma[self.t_train]
            gamma_prev_j = gamma_prev[self.t_train]
            num_1 = np.sum(np.log(norm_cdf(gamma_jplus1 - m) - norm_cdf(gamma_j - m)))
            den_1 = np.sum(np.log(norm_cdf(gamma_prev_jplus1 - m) - norm_cdf(gamma_prev_j - m)))
            log_A = num_1 + num_2 - den_1 - den_2
            threshold = np.random.uniform(low=0.0, high=1.0)
            if log_A > np.log(threshold):
                # Accept
                gamma_prev = gamma
                gamma_prev_jplus1 = gamma_jplus1
                gamma_prev_j = gamma_j
                # Sample y from the full conditional
                y = sample_y(y_container.copy(), self.t_train, gamma, self.noise_std, self.N)
            else:
                # Reject, and use previous \gamma, y sample
                gamma = gamma_prev
            # Calculate statistics, then sample other conditional
            m_tilde, nu = self._m_tilde(y, self.cov, self.K)  # TODO: Numba?
            m = m_tilde.flatten() + self.L_Sigma @ norm.rvs(size=self.N)
            # plt.scatter(self.X_train, m)
            # plt.show()
            # print(gamma)
            m_samples.append(m.flatten())
            y_samples.append(y.flatten())
            gamma_samples.append(gamma.flatten())
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)

    def _sample_initiate(self, m_0, gamma_0):
        self.indices = []
        for j in range(0, self.J -1):
            self.indices.append(np.where(self.t_train == j))
        m_samples = []
        y_samples = []
        gamma_samples = []
        return m_0, gamma_0, m_samples, y_samples, gamma_samples

    def sample(self, m_0, gamma_0, steps, first_step=1):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: y (auxilliaries), m (GP regression posterior means) and
        gamma (cutpoint parameters).

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg gamma_0: (K + 1, ) numpy.ndarray of the initial location of the sampler.
        :type gamma_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Gibbs samples. The acceptance rate for the Gibbs algorithm is 1.
        """
        m, gamma, gamma_prev, m_samples, y_samples, gamma_samples = self._sample_initiate(m_0, gamma_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            y = sample_y(y, m, self.t_train, gamma, self.noise_std, self.N)
            # Calculate statistics, then sample other conditional
            m_tilde, _ = self._m_tilde(y, self.cov, self.K)
            m = m_tilde.flatten() + self.L_Sigma @ norm.rvs(size=self.N)
            # Empty gamma (J + 1, ) array to collect the upper cut-points for each class
            gamma = -1. * np.ones(self.J + 1)
            uppers = -1. * np.ones(self.J - 2)
            locs = -1. * np.ones(self.J - 2)
            for j in range(self.J - 2):  # TODO change the index to the class.
                if self.indices[j+1]:
                    uppers[j] = np.min(np.append(y[self.indices[j + 1]], gamma_prev[j + 2]))
                else:
                    uppers[j] = gamma_prev[j + 2]
                if self.indeces[j]:
                    locs[j] = np.max(np.append(y[self.indeces[j]], gamma_prev[j]))
                else:
                    locs[j] = gamma_prev[j]
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_K = +\infty
            gamma[0] = np.NINF
            gamma[1:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            gamma[-1] = np.inf
            # update gamma prev
            gamma_prev = gamma
            m_samples.append(m)
            y_samples.append(y)
            gamma_samples.append(gamma)
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)


class EllipticalSliceOrdinalGP(Sampler):
    """
    Elliptical Slice sampling of the latent variables.
    """

    def __init__(self, gamma, noise_variance=1.0,
            gamma_hyperparameters=None, noise_std_hyperparameters=None,
            *args, **kwargs):
        """
        Create an :class:`EllipticalSliceOrdinalGP` sampler object.

        :returns: An :class:`EllipticalSliceOrdinalGP` object.
        """
        super().__init__(*args, **kwargs)
        if gamma_hyperparameters is not None:
            warnings.warn("gamma_hyperparameters set as {}".format(gamma_hyperparameters))
            self.gamma_hyperparameters = gamma_hyperparameters
        else:
            self.gamma_hyperparameters = None
        if noise_std_hyperparameters is not None:
            warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
            self.noise_std_hyperparameters = noise_std_hyperparameters
        else:
            self.noise_std_hyperparameters = None
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
        self.EPS = 0.0001
        self.EPS_2 = self.EPS**2 
        self.upper_bound = 6
        self.upper_bound2 = 30
        self.jitter = 1e-6
        self.t_trainplus1 = self.t_train + 1
        # Initiate hyperparameters
        self.hyperparameters_update(gamma=gamma, noise_variance=noise_variance)

    def _sample_initiate(self, m_0):
        """Initialise variables for the sample method."""
        m_samples = []
        log_likelihood_samples = []
        log_likelihood = self.get_log_likelihood(m_0)
        return m_0, log_likelihood, m_samples, log_likelihood_samples

    def sample(self, m_0, steps, first_step=1):
        """
        Sample from the latent variables posterior.

        Elliptical slice sampling tansition operator, for f only.
        Can then use a Gibbs sampler to sample from y.

        Sampling occurs in Gibbs blocks over the parameters: m (GP regression
            posterior means) and then over y (auxilliaries). In this sampler,
            gamma (cutpoint parameters) are fixed.

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.
        """
        (m, log_likelihood, m_samples, log_likelihood_samples) = self._sample_initiate(m_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Sample m
            m, log_likelihood = self.transition_operator(
                self.L_K, self.N, m, log_likelihood)
            m_samples.append(m.flatten())
            log_likelihood_samples.append(log_likelihood)
        return np.array(m_samples), log_likelihood_samples

    def transition_operator(self, L_K, N, m, log_likelihood, pi2=2 * np.pi):
        """
        Elliptical slice sampling transition operator.

        Draw samples from p(m|y, \theta).
        """
        auxiliary_nu = L_K @ norm.rvs(size=N)
        auxiliary_theta = np.random.uniform(low=0, high=2 * np.pi)
        auxiliary_theta_min = auxiliary_theta - pi2
        auxiliary_theta_max = auxiliary_theta
        log_likelihood_plus_uniform = log_likelihood + np.log(np.random.uniform())
        while True:
            m_proposed = m * np.cos(auxiliary_theta) + auxiliary_nu * np.sin(auxiliary_theta)
            log_likelihood_proposed = self.get_log_likelihood(m_proposed)
            if log_likelihood_proposed > log_likelihood_plus_uniform:  # Accept reject
                log_likelihood = log_likelihood_proposed
                break
            if auxiliary_theta < 0:
                auxiliary_theta_min = auxiliary_theta
            elif auxiliary_theta >= 0:
                auxiliary_theta_max = auxiliary_theta
            auxiliary_theta = np.random.uniform(low=auxiliary_theta_min, high=auxiliary_theta_max)
        return m_proposed, log_likelihood

    def ELLSS_transition_operatorSS(self, L_K, N, y, f, log_likelihood, classifier):
        """
        TODO: might supercede this with an existing implementation in C
        Draw samples from p(f|y, \theta). Similarly to probabilistic crank nicolson (implicit evaluation of the likelihood).

        Requires to evaluate the likelihood p(y|f, \theta).
        In GP metric regression this is a multivariate normal making use of the L_K.
        In GP ordinal regression this is product of independent probit functions parametrised by the cutpoints.

        If the y is ordinal data then the likelihood takes the form of the probit. But why not just use a factorisation
        of the posterior distribution.
        """
        x = y - f
        # Normal distribution
        z = np.random.normal(loc=0.0, scale=1.0, size=N)
        # Draw a sample from the prior TODO this looks incorrect
        z = L_K @  solve_triangular(L_K.T, x, lower=True)
        u = np.random.exponential(scale=1.0)
        alpha = np.random.uniform(low=0.0, high=2 * np.pi)
        alpha_bracket = [alpha - 2 * np.pi, alpha]
        while True:
            f_dash = f * np.cos(alpha) + z * np.sin(alpha)
            log_likelihood_dash = self.get_log_likelihood(f_dash, classifier)
            if log_likelihood_dash > log_likelihood - u:
                return f_dash, log_likelihood_dash
            else:
                if alpha < 0:
                    alpha_bracket[0] = 0.0
                else:
                    alpha_bracket[1] = 0.0
                print(alpha)
                alpha = np.random.uniform(low=alpha_bracket[0], high=alpha_bracket[1])


# class HyperparameterSampler(object):
#     """
#     """


class SufficientAugmentation(object):
    """
    A sufficient augmentation (SA) hyperparameter sampler.

    This class allows users to define a sampler of the SA posterior
        :math:`\tilde{p}(\theta| f)`, where
        :math:`\theta` and the hyperparameters of a Gaussian Process (GP) model.
    The sampler is defined by a particular GP model,
    for this an :class:`probit.samplers.Sampler` is required.
    For learning how to use Probit, see
    :ref:`complete documentation <probit_docs_mainpage>`, and for getting
    started, please see :ref:`quickstart <probit_docs_user_quickstart>`.

    ref: Filippone, Maurizio & Girolami, Mark. (2014). Pseudo-Marginal Bayesian Inference for Gaussian Processes.
        IEEE Transactions on Pattern Analysis and Machine Intelligence. 10.1109/TPAMI.2014.2316530. 
    """

    def __init__(self, sampler, write_path=None):
        """
        Create an :class:`SufficientAugmentation` object.

        :arg sampler: The approximator to use, see :mod:`probit.samplers` for options.    
        :arg str write_path: Write path for outputs.
        :returns: An :class:`SufficientAugmentation` object.
        """
        if not (isinstance(sampler, Sampler)):
            raise InvalidSampler(sampler)
        else:
            self.sampler = sampler

    def tmp_compute_marginal(self, m, theta, indices, proposal_L_cov, reparameterised=True):
        """Temporary function to compute the marginal given theta"""
        if reparameterised:
            gamma, varphi, scale, noise_variance, log_p_theta = prior_reparameterised(
                theta, indices, self.sampler.J, self.sampler.kernel.varphi_hyperparameters,
                self.sampler.noise_std_hyperparameters, self.sampler.gamma_hyperparameters,
                self.sampler.kernel.scale_hyperparameters, self.sampler.gamma)
        else:
            gamma, varphi, scale, noise_variance, log_p_theta = prior(
                theta, indices, self.sampler.J, self.sampler.kernel.varphi_hyperparameters,
                self.sampler.noise_std_hyperparameters, self.sampler.gamma_hyperparameters,
                self.sampler.kernel.scale_hyperparameters, self.sampler.gamma)
        nu = solve_triangular(self.sampler.L_K.T, m, lower=True)  # TODO just make sure this is the correct solve.
        log_p_m_giv_theta = - 0.5 * self.sampler.log_det_K - 0.5 * nu.T @ nu
        log_p_theta_giv_m = log_p_theta[0] + log_p_m_giv_theta
        return log_p_theta_giv_m

    def transition_operator(
            self, u, log_prior_u, log_jacobian_u,
            indices, proposal_L_cov, nu, log_p_y_given_m, reparameterised):
        """
        Transition operator for the metropolis step.

        Samples from p(theta|f) \propto p(f|theta)p(theta).
        """
        # SS the below is AA
        # Different nu requires recalculation of this
        log_p_m_given_u = - 0.5 * self.sampler.log_det_K - self.sampler.N / 2 - 0.5 * nu.T @ nu
        log_p_u_given_m = log_p_m_given_u + log_prior_u
        # Make copies of previous hyperparameters in case of reject
        # So we don't need to recalculate
        # TODO do I need copy?
        gamma = self.sampler.gamma.copy()
        varphi = self.sampler.kernel.varphi.copy()
        scale = self.sampler.scale.copy()
        noise_variance = self.sampler.noise_variance.copy()
        L_K = self.sampler.L_K.copy()
        log_det_K = self.sampler.log_det_K.copy()
        K = self.sampler.K.copy()
        cov = self.sampler.cov.copy()
        L_Sigma = self.sampler.L_Sigma.copy()
        # Evaluate priors and proposal conditionals
        if reparameterised:
            v, log_jacobian_v = proposal_reparameterised(u, indices, proposal_L_cov)
            log_prior_v = prior_reparameterised(v, indices)
        else:
            v, log_jacobian_v = proposal(u, indices, proposal_L_cov)
            log_prior_v = prior(v, indices)
        # Initialise proposed hyperparameters, and update prior and posterior covariances
        self.sampler._hyperparameter_initialise(v, indices)
        log_p_m_given_v = - 0.5 * self.sampler.log_det_K - self.sampler.N / 2 - 0.5 * nu.T @ nu
        log_p_v_given_m = log_p_m_given_v + log_prior_v
        print(log_p_v_given_m)
        # Log ratio
        log_a = (
            log_p_v_given_m + np.sum(log_prior_v) + np.sum(log_jacobian_v)
            - log_p_u_given_m - np.sum(log_prior_u) - np.sum(log_jacobian_u))
        log_A = np.minimum(0, log_a)
        threshold = np.random.uniform(low=0.0, high=1.0)
        if log_A > np.log(threshold):
            print(log_A, ">", np.log(threshold), " ACCEPT")
            #joint_log_likelihood = log p(theta,f,y) = log p(y|f) + log p (f|theta) + log p(theta)
            joint_log_likelihood = log_p_y_given_m + log_p_m_given_v + log_prior_v
            # Prior and posterior covariances have already been updated
            return (v, log_prior_v, log_jacobian_v, joing_log_likelihood, 1)
        else:
            print(log_A, "<", np.log(threshold), " REJECT")
            # Revert the hyperparameters and
            # prior and posterior covariances to previous values
            self.sampler.hyperparameters_update(
                gamma=gamma, varphi=varphi, scale=scale,
                noise_variance=noise_variance,
                K=K, L_K=L_K, log_det_K=log_det_K, cov=cov, L_Sigma=L_Sigma)
            joint_log_likelihood = log_p_y_given_m + log_p_m_given_u + log_prior_u
            return (u, log_prior_u, log_jacobian_u, joint_log_likelihood, 0)

    def _sample_initiate(self, theta_0, indices, proposal_cov, reparameterised):
        """Initiate method for `:meth:sample_sufficient_augmentation'.""" 
        # Get starting point for the Markov Chain
        if theta_0 is None:
            theta_0 = self.sampler.get_theta(indices)
        # Get type of proposal density
        proposal_L_cov = proposal_initiate(theta_0, indices, proposal_cov)
        # Evaluate priors and proposal conditionals
        if reparameterised:
            theta_0, log_jacobian_theta = proposal_reparameterised(theta_0, indices, proposal_L_cov)
            log_prior_theta = prior_reparameterised(theta_0, indices)
        else:
            theta_0, log_jacobian_theta = proposal(theta_0, indices, proposal_L_cov)
            log_prior_theta = prior(theta_0, indices)
        # Initialise hyperparameters, and update prior and posterior covariances
        self.sampler._hyperparameter_initialise(theta_0, indices)
        # Initiate containers for samples
        theta_samples = []
        m_samples = []
        joint_log_likelihood_samples = []
        return (
            theta_0, theta_samples, log_prior_theta,
            log_jacobian_theta, proposal_L_cov,
            m_samples, joint_log_likelihood_samples, 0)

    def sample(self, m_0, indices, proposal_cov, steps, first_step=1,
            theta_0=None, reparameterised=True):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: m (GP regression posterior means) and
            then over y (auxilliaries). In this sampler, gamma (cutpoint parameters) are fixed.

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.
        """
        (theta, theta_samples, log_prior_theta,
        log_jacobian_theta, log_marginal_likelihood_theta,
        proposal_L_cov, joint_log_likelihood_samples, naccepted) = self._sample_initiate(
            theta_0, indices, proposal_cov, reparameterised)
        (m, log_likelihood, m_samples, log_likelihood_samples) = self.sampler._sample_initiate(m_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Need p(v | \nu, y) and p(y | \ny, y), but need to condition on the same \nu, y
            # and since these change, then we need to change the variables.
            m, log_likelihood = self.sampler.transition_operator(
                self.L_K, self.N, m, log_likelihood)
            # In code_pseudo they have f | nu, L_chol, not f | L_cho, y. Weird.
            # Solve for nu
            # TODO: Is this in the correct order?
            # nu | f, theta, so should be different depending on which theta is used.
            # So required two solves per step?
            # No if m is not the same for each
            # Yes if m is the same for each
            nu = self.sampler._nu(m, self.sampler.K)
            (theta, log_prior_theta,
            log_jacobian_theta,
            joint_log_likelihood,
            bool) = self.transition_operator(
                theta, log_prior_theta, log_jacobian_theta,
                indices, proposal_L_cov, nu, reparameterised)
            naccepted += bool
            # Update hyperparameters from theta.
            # plt.scatter(self.sampler.X_train, m)
            # plt.show()
            # print(gamma)
            m_samples.append(m)
            theta_samples.append(theta)
            joint_log_likelihood_samples.append(joint_log_likelihood)
        return np.array(m_samples), naccepted/steps


class AncilliaryAugmentation(Sampler):
    """
    An ancilliary augmentation (AA) hyperparameter sampler.

    This class allows users to define a sampler of the AA posterior
        :math:`\tilde{p}(\theta| \nu, \y)`, where
        :math:`\theta` and the hyperparameters of a Gaussian Process (GP) model,
        :math:`\nu` are the variables auxiliary to theta and :math:`\y` are the data.
    The sampler is defined by a particular GP model,
    for this an :class:`probit.samplers.Sampler` is required.
    For learning how to use Probit, see
    :ref:`complete documentation <probit_docs_mainpage>`, and for getting
    started, please see :ref:`quickstart <probit_docs_user_quickstart>`.

    ref: Filippone, Maurizio & Girolami, Mark. (2014). Pseudo-Marginal Bayesian Inference for Gaussian Processes.
        IEEE Transactions on Pattern Analysis and Machine Intelligence. 10.1109/TPAMI.2014.2316530. 
    """

    def __init__(self, sampler, write_path=None):
        """
        Create an :class:`AuxilliaryAugmentation` object.

        :arg sampler: The approximator to use, see :mod:`probit.samplers` for options.    
        :arg str write_path: Write path for outputs.
        :returns: An :class:`SufficientAugmentation` object.
        """
        if not (isinstance(sampler, Sampler)):
            raise InvalidSampler(sampler)
        else:
            self.sampler = sampler

    def _nu(self, m, K):
        """
        Calculate the ancilliary augmentation "whitened" variables.

        :arg m: (N,) array
        :type m: :class:`np.ndarray`
        :arg cov:
        :type cov:
        :arg K:
        :type K:
        """
        # m = L \nu
        # m ~ N(0, K)
        # p(m | theta)
        # m.T @ K^{-1} @ m
        # \nu.T @ L.T @ K^{-1} @ L @ \nu
        # \nu.T @ L.T @ L.{-1}.T @ L^{-1} @ L @ \nu
        # \nu.T @ \nu
        # so m = L\nu
        # so \nu = solve_triangular(m, L)
        return solve_triangular(m, K)  # Why have I put (m, K)?

    def tmp_compute_marginal(self, m, theta, indices, proposal_L_cov, reparameterised=True):
        """Temporary function to compute the marginal given theta"""
        if reparameterised:
            gamma, varphi, scale, noise_variance, log_p_theta = prior_reparameterised(
                theta, indices, self.sampler.J, self.sampler.kernel.varphi_hyperparameters,
                self.sampler.noise_std_hyperparameters, self.sampler.gamma_hyperparameters,
                self.sampler.kernel.scale_hyperparameters, self.sampler.gamma)
        else:
            gamma, varphi, scale, noise_variance, log_p_theta = prior(
                theta, indices, self.sampler.J, self.sampler.kernel.varphi_hyperparameters,
                self.sampler.noise_std_hyperparameters, self.sampler.gamma_hyperparameters,
                self.sampler.kernel.scale_hyperparameters, self.sampler.gamma)
        log_p_y_giv_nu_theta = self.sampler.get_log_likelihood(m)
        log_p_theta_giv_y_nu = log_p_theta[0] + log_p_y_giv_nu_theta
        return log_p_theta_giv_y_nu

    def transition_operator(
            self, u, log_prior_u, log_jacobian_u,
            indices, proposal_L_cov, nu, log_p_y_given_m, reparameterised):
        """
        Transition operator for the metropolis step.

        Samples from p(theta|f) \propto p(f|theta)p(theta).
        """
        # Different nu requires recalculation of this
        log_p_m_given_u = - 0.5 * self.sampler.log_det_K - self.sampler.N / 2 - 0.5 * nu.T @ nu
        log_p_u_given_m = log_p_m_given_u + log_prior_u
        # Make copies of previous hyperparameters in case of reject
        # So we don't need to recalculate
        # TODO do I need copy?
        gamma = self.sampler.gamma.copy()
        varphi = self.sampler.kernel.varphi.copy()
        scale = self.sampler.scale.copy()
        noise_variance = self.sampler.noise_variance.copy()
        L_K = self.sampler.L_K.copy()
        log_det_K = self.sampler.log_det_K.copy()
        K = self.sampler.K.copy()
        cov = self.sampler.cov.copy()
        L_Sigma = self.sampler.L_Sigma.copy()
        # Evaluate priors and proposal conditionals
        if reparameterised:
            v, log_jacobian_v = proposal_reparameterised(u, indices, proposal_L_cov)
            log_prior_v = prior_reparameterised(v, indices)
        else:
            v, log_jacobian_v = proposal(u, indices, proposal_L_cov)
            log_prior_v = prior(v, indices)
        # Initialise proposed hyperparameters, and update prior and posterior covariances
        self.sampler._hyperparameter_initialise(v, indices)
        log_p_m_given_v = - 0.5 * self.sampler.log_det_K - 0.5 * nu.T @ nu
        log_p_v_given_m = log_p_m_given_v + log_prior_v
        print(log_p_v_given_m)
        # Log ratio
        log_a = (
            log_p_v_given_m + np.sum(log_prior_v) + np.sum(log_jacobian_v)
            - log_p_u_given_m - np.sum(log_prior_u) - np.sum(log_jacobian_u))
        log_A = np.minimum(0, log_a)
        threshold = np.random.uniform(low=0.0, high=1.0)
        if log_A > np.log(threshold):
            print(log_A, ">", np.log(threshold), " ACCEPT")
            #joint_log_likelihood = log p(theta,f,y) = log p(y|f) + log p (f|theta) + log p(theta)
            joint_log_likelihood = log_p_y_given_m + log_p_m_given_v + log_prior_v
            # Prior and posterior covariances have already been updated
            return (v, log_prior_v, log_jacobian_v, 1)
        else:
            print(log_A, "<", np.log(threshold), " REJECT")
            # Revert the hyperparameters and
            # prior and posterior covariances to previous values
            self.sampler.hyperparameters_update(
                gamma=gamma, varphi=varphi, scale=scale,
                noise_variance=noise_variance,
                K=K, L_K=L_K, log_det_K=log_det_K, cov=cov, L_Sigma=L_Sigma)
            joint_log_likelihood = log_p_y_given_m + log_p_m_given_u + log_prior_u
            return (u, log_prior_u, log_jacobian_u, 0)

    def _sample_initiate(self, theta_0, indices, proposal_cov, reparameterised):
        """Initiate method for `:meth:sample_sufficient_augmentation'.""" 
        # Get starting point for the Markov Chain
        if theta_0 is None:
            theta_0 = self.sampler.get_theta(indices)
        # Get type of proposal density
        proposal_L_cov = proposal_initiate(theta_0, indices, proposal_cov)
        # Evaluate priors and proposal conditionals
        if reparameterised:
            theta_0, log_jacobian_theta = proposal_reparameterised(theta_0, indices, proposal_L_cov)
            log_prior_theta = prior_reparameterised(theta_0, indices)
        else:
            theta_0, log_jacobian_theta = proposal(theta_0, indices, proposal_L_cov)
            log_prior_theta = prior(theta_0, indices)
        # Initialise hyperparameters, and update prior and posterior covariances
        self.sampler._hyperparameter_initialise(theta_0, indices)
        # Initiate containers for samples
        theta_samples = []
        m_samples = []
        joint_log_likelihood_samples = []
        return (
            theta_0, theta_samples, log_prior_theta,
            log_jacobian_theta, proposal_L_cov,
            m_samples, joint_log_likelihood_samples, 0)

    def sample(self, m_0, indices, proposal_cov, steps, first_step=1,
            theta_0=None, reparameterised=True):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: m (GP regression posterior means) and
            then over y (auxilliaries). In this sampler, gamma (cutpoint parameters) are fixed.

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.
        """
        (theta, theta_samples, log_prior_theta,
        log_jacobian_theta, log_marginal_likelihood_theta,
        proposal_L_cov, joint_log_likelihood_samples, naccepted) = self._sample_initiate(
            theta_0, indices, proposal_cov, reparameterised)
        (m, log_likelihood, m_samples, log_likelihood_samples) = self.sampler._sample_initiate(m_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Need p(v | \nu, y) and p(y | \ny, y), but need to condition on the same \nu, y
            # and since these change, then we need to change the variables.
            m, log_likelihood = self.sampler.transition_operator(
                self.L_K, self.N, m, log_likelihood)
            # In code_pseudo they have f | nu, L_chol, not f | L_cho, y. Weird.
            # Solve for nu
            # TODO: Is this in the correct order?
            # nu | f, theta, so should be different depending on which theta is used.
            # So required two solves per step?
            # No if m is not the same for each
            # Yes if m is the same for each
            nu = self._nu(m, self.sampler.K)
            (theta, log_prior_theta,
            log_jacobian_theta,
            bool) = self.transition_operator(
                theta, log_prior_theta, log_jacobian_theta,
                indices, proposal_L_cov, nu, reparameterised)
            naccepted += bool
            # Update hyperparameters from theta.
            # plt.scatter(self.sampler.X_train, m)
            # plt.show()
            # print(gamma)
            m_samples.append(m)
            theta_samples.append(theta)
            joint_log_likelihood_samples.append(joint_log_likelihood)
        return np.array(m_samples), naccepted/steps


class PseudoMarginal(object):
    """
    A pseudo-marginal (PM) hyperparameter sampler.

    This class allows users to define a sampler of the PM posterior
        :math:`\tilde{p}(\theta| y)`, where
        :math:`\theta` and the hyperparameters of a Gaussian Process (GP) model.
    The sampler is defined by a particular GP model,
    for this an :class:`probit.approximators.Approximator` is required.
    For learning how to use Probit, see
    :ref:`complete documentation <probit_docs_mainpage>`, and for getting
    started, please see :ref:`quickstart <probit_docs_user_quickstart>`.

    ref: Filippone, Maurizio & Girolami, Mark. (2014). Pseudo-Marginal Bayesian Inference for Gaussian Processes.
        IEEE Transactions on Pattern Analysis and Machine Intelligence. 10.1109/TPAMI.2014.2316530. 
    """

    def __init__(self, approximator, write_path=None):
        """
        Create an :class:`PseudoMarginal` object.

        :arg approximator: The approximator to use, see :mod:`probit.approximators` for options.    
        :arg str write_path: Write path for outputs.
        :returns: An :class:`PseudoMarginal` object.
        """
        if not (isinstance(approximator, Approximator)):
            raise InvalidKernel(approximator)
        else:
            self.approximator = approximator

    def _prior(self, theta, indices):
        """
        A priors defined over their usual domains, and so a transformation of random variables is used
        for sampling from proposal distrubutions defined over continuous domains.
        Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
        """
        gamma, varphi, scale, noise_variance, log_prior_theta = prior(theta, indices)
        # Update prior covariance
        self.approximator.hyperparameters_update(
            gamma=gamma, varphi=varphi, scale=scale,
            noise_variance=noise_variance)
        # intervals = self.approximator.gamma[2:self.approximator.J] - self.approximator.gamma[1:self.approximator.J - 1]
        return log_prior_theta

    # def _weight(
    #     self, f_samp, prior_L_cov, half_log_det_prior_cov, posterior_L_cov, half_log_det_posterior_cov, posterior_mean):
    #     return (
    #         self.approximator._get_log_likelihood(f_samp)
    #         + self.approximator._log_multivariate_normal_pdf(
    #             f_samp, prior_L_cov, half_log_det_prior_cov)
    #         - self.approximator._log_multivariate_normal_pdf(
    #             f_samp, posterior_L_cov, half_log_det_posterior_cov, mean=posterior_mean)
    #     )

    def _weight(
            self, f_samp, prior_cov_inv, half_log_det_prior_cov, posterior_cov_inv,
            half_log_det_posterior_cov, posterior_mean):
        return (
            self.approximator.get_log_likelihood(f_samp)
            + self.approximator._log_multivariate_normal_pdf(
                f_samp, prior_cov_inv, half_log_det_prior_cov)
            - self.approximator._log_multivariate_normal_pdf(
                f_samp, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean)
        )

    def _weight_vectorised(
            self, f_samps, prior_cov_inv, half_log_det_prior_cov, posterior_cov_inv,
            half_log_det_posterior_cov, posterior_mean):
        log_ws = (self.approximator.get_log_likelihood(f_samps)
            + self.approximator._log_multivariate_normal_pdf_vectorised(
                f_samps, prior_cov_inv, half_log_det_prior_cov)
            - self.approximator._log_multivariate_normal_pdf_vectorised(
                f_samps, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean))
        return log_ws

    def _importance_sampler_vectorised(
        self, num_importance_samples, prior_cov_inv, half_log_det_prior_cov,
        posterior_mean, posterior_cov_inv, half_log_det_posterior_cov, posterior_cholesky):
        """
        Sampling from an unbiased estimate of the marginal likelihood p(y|\theta) given the likelihood of the parameters
        p(y | f) and samples
        from an (unbiased) approximating distribution q(f|y, \theta).
        """
        zs = np.random.normal(0, 1, (num_importance_samples, self.approximator.N))
        posterior_L, is_inv  = posterior_cholesky
        if is_inv:
            zs = solve_triangular(posterior_L, zs.T, lower=False)
            zs = zs.T
        else:
            zs = zs @ np.triu(posterior_L)
        f_samps = zs + posterior_mean
        # plt.scatter(self.approximator.X_train, posterior_mean, color='k')
        # for i in range(3):
        #     plt.scatter(self.approximator.X_train, f_samps[i])
        # plt.show()
        log_ws = self._weight_vectorised(
            f_samps, prior_cov_inv, half_log_det_prior_cov,
            posterior_cov_inv, half_log_det_posterior_cov, posterior_mean)
        max_log_ws = np.max(log_ws)
        log_sum_exp = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws)))
        return log_sum_exp - np.log(num_importance_samples)

    def _importance_sampler(
            self, num_importance_samples, prior_L_cov, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, half_log_det_posterior_cov):
        """
        Sampling from an unbiased estimate of the marginal likelihood p(y|\theta) given the likelihood of the parameters
        p(y | f) and samples
        from an (unbiased) approximating distribution q(f|y, \theta).
        """
        log_ws = np.empty(num_importance_samples)
        # TODO: vectorise this function
        # This function is embarassingly paralellisable, however, there may be no easy way to do this with numba or jax.
        for i in range(num_importance_samples):
            # Draw sample from GP posterior
            z = np.random.normal(0, 1, self.approximator.N)
            f_samp = posterior_mean + np.dot(posterior_L_cov, z)
            #plt.scatter(self.approximator.X_train, f_samp)
            log_ws[i] = self._weight(
                f_samp, prior_L_cov, half_log_det_prior_cov, posterior_L_cov,
                half_log_det_posterior_cov, posterior_mean)
        # print(log_ws)
        #plt.show()
        # Normalise the w vectors using the log-sum-exp operator
        max_log_ws = np.max(log_ws)
        log_sum_exp = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws)))
        return log_sum_exp - np.log(num_importance_samples)

    def tmp_compute_marginal(
            self, theta, indices, steps=None, num_importance_samples=64, reparameterised=False):
        """Temporary function to compute the marginal given theta"""
        if reparameterised:
            gamma, varphi, scale, noise_variance, log_p_theta = prior_reparameterised(
                theta, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)
        else:
            gamma, varphi, scale, noise_variance, log_p_theta = prior(
                theta, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)
        fx, gx, posterior_mean, (posterior_matrix, is_inv) = self.approximator.approximate_posterior(
            theta, indices, steps=steps, first_step=1, write=False, verbose=False)
        prior_L_cov = np.linalg.cholesky(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))
        prior_cov_inv = np.linalg.inv(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        if is_inv:
            # Laplace
            posterior_cov_inv = posterior_matrix
            posterior_L_inv_cov, lower = cho_factor(posterior_cov_inv)
            half_log_det_posterior_cov = - np.sum(np.log(np.diag(posterior_L_inv_cov)))
            posterior_cholesky = (posterior_L_inv_cov, True)
        else:
            # EP - perhaps a simpler way to do this via the precisions
            posterior_cov = posterior_matrix
            (posterior_L_cov, lower) = cho_factor(
                posterior_cov + self.approximator.jitter * np.eye(self.approximator.N))  # Is this necessary?
            half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
            posterior_L_covT_inv = solve_triangular(
                posterior_L_cov.T, np.eye(self.approximator.N), lower=True)
            posterior_cov_inv = solve_triangular(posterior_L_cov, posterior_L_covT_inv, lower=False)
            posterior_cholesky = (posterior_L_cov, False)
        log_p_pseudo_marginals = np.empty(50)
        # TODO This could be parallelized using MPI
        for i in range(50):
            # log_p_pseudo_marginal = self._importance_sampler_vectorised(
            #     num_importance_samples, prior_cov_inv, half_log_det_prior_cov,
            #     posterior_mean, posterior_cov_inv, half_log_det_posterior_cov, posterior_cholesky)
            # print(log_p_pseudo_marginal)
            log_p_pseudo_marginals[i] = self._importance_sampler_vectorised(
                num_importance_samples, prior_cov_inv, half_log_det_prior_cov,
                posterior_mean, posterior_cov_inv, half_log_det_posterior_cov, posterior_cholesky)
        return np.array(log_p_pseudo_marginals) + log_p_theta[0], log_p_theta[0]

    def _transition_operator(
            self, u, log_p_u, log_jacobian_u, log_p_pseudo_marginal_u,
            indices, proposal_L_cov, num_importance_samples, reparameterised, verbose=False):
        "Transition operator for the metropolis step."
        # Evaluate priors and proposal conditionals
        if reparameterised:
            v, log_jacobian_v = proposal_reparameterised(u, indices, proposal_L_cov)
            gamma, varphi, scale, noise_variance, log_p_v = prior_reparameterised(
                v, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)
        else:
            u, log_jacobian_u = proposal(u, indices, proposal_L_cov, self.approximator.J)
            gamma, varphi, scale, noise_variance, log_p_u = prior(
                u, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)

        fx, gx, posterior_mean, (posterior_matrix, is_inv) = self.approximator.approximate_posterior(
            v, indices, first_step=1, write=False, verbose=False)
        prior_L_cov = np.linalg.cholesky(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))
        prior_cov_inv = np.linalg.inv(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        if is_inv:
            posterior_cov_inv = posterior_matrix
            posterior_L_inv_cov, lower = cho_factor(posterior_cov_inv)
            half_log_det_posterior_cov = - np.sum(np.log(np.diag(posterior_L_inv_cov)))
            posterior_cholesky = (posterior_L_inv_cov, True)
        else:
            posterior_cov = posterior_matrix
            (posterior_L_cov, lower) = cho_factor(
                posterior_cov + self.approximator.jitter * np.eye(self.approximator.N))  # Is this necessary?
            half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
            posterior_L_covT_inv = solve_triangular(
                posterior_L_cov.T, np.eye(self.approximator.N), lower=True)
            posterior_cov_inv = solve_triangular(posterior_L_cov, posterior_L_covT_inv, lower=False)
            posterior_cholesky = (posterior_L_cov, False)
        log_p_pseudo_marginal_v = self._importance_sampler_vectorised(
                num_importance_samples, prior_cov_inv, half_log_det_prior_cov,
                posterior_mean, posterior_cov_inv, half_log_det_posterior_cov, posterior_cholesky)
        # Log ratio
        log_a = (
            log_p_pseudo_marginal_v + np.sum(log_p_v) + np.sum(log_jacobian_v)
            - log_p_pseudo_marginal_u - np.sum(log_p_u) - np.sum(log_jacobian_u))
        log_A = np.minimum(0, log_a)
        threshold = np.random.uniform(low=0.0, high=1.0)
        if log_A > np.log(threshold):
            if verbose:
                print(log_A, ">", np.log(threshold), " ACCEPT, log_p_marginal={}".format(log_p_pseudo_marginal_u))
            return (v, log_p_v, log_jacobian_v, log_p_pseudo_marginal_v, 1.0)
        else:
            if verbose:
                print(log_A, "<", np.log(threshold), " REJECT, log_p_marginal={}".format(log_p_pseudo_marginal_v))
            return (u, log_p_u, log_jacobian_u, log_p_pseudo_marginal_u, 0.0)

    def _sample_initiate(
            self, theta_0, indices, proposal_cov, num_importance_samples, reparameterised):
        # Get starting point for the Markov Chain
        if theta_0 is None:
            theta_0 = self.approximator.get_theta(indices)
        fx, gx, posterior_mean, (posterior_matrix, is_inv) = self.approximator.approximate_posterior(
            theta_0, indices, first_step=1, write=False, verbose=False)
        prior_L_cov = np.linalg.cholesky(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))
        prior_cov_inv = np.linalg.inv(self.approximator.K + self.approximator.jitter * np.eye(self.approximator.N))
        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        if is_inv:
            posterior_cov_inv = posterior_matrix
            posterior_L_inv_cov, lower = cho_factor(posterior_cov_inv)
            half_log_det_posterior_cov = - np.sum(np.log(np.diag(posterior_L_inv_cov)))
            posterior_cholesky = (posterior_L_inv_cov, True)
        else:
            posterior_cov = posterior_matrix
            (posterior_L_cov, lower) = cho_factor(
                posterior_cov + self.approximator.jitter * np.eye(self.approximator.N))  # Is this necessary?
            half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
            posterior_L_covT_inv = solve_triangular(
                posterior_L_cov.T, np.eye(self.approximator.N), lower=True)
            posterior_cov_inv = solve_triangular(posterior_L_cov, posterior_L_covT_inv, lower=False)
            posterior_cholesky = (posterior_L_cov, False)

        log_p_pseudo_marginal = self._importance_sampler_vectorised(
                num_importance_samples, prior_cov_inv, half_log_det_prior_cov,
                posterior_mean, posterior_cov_inv, half_log_det_posterior_cov, posterior_cholesky)

        # Evaluate priors and proposal conditionals
        proposal_L_cov = proposal_initiate(theta_0, indices, proposal_cov)
        if reparameterised:
            theta_0, log_jacobian_theta = proposal_reparameterised(theta_0, indices, proposal_L_cov)
            gamma, varphi, scale, noise_variance, log_p_theta = prior_reparameterised(
                theta_0, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)
        else:
            theta_0, log_jacobian_theta = proposal(theta_0, indices, proposal_L_cov, self.approximator.J)
            gamma, varphi, scale, noise_variance, log_p_theta = prior(
                theta_0, indices, self.approximator.J, self.approximator.kernel.varphi_hyperparameters,
                None, None,
                self.approximator.kernel.scale_hyperparameters, self.approximator.gamma)
        theta_samples = []
        acceptance_rate = 0.0
        return (
            theta_0, theta_samples, log_p_theta,
            log_jacobian_theta, log_p_pseudo_marginal, proposal_L_cov, acceptance_rate)

    def sample(
            self, indices, proposal_cov, steps, first_step=1, num_importance_samples=80, theta_0=None,
            reparameterised=True, verbose=False):
        (theta, theta_samples, log_prior_theta,
        log_jacobian_theta, log_marginal_likelihood_theta, proposal_L_cov,
        acceptance_rate) = self._sample_initiate(
            theta_0, indices, proposal_cov, num_importance_samples, reparameterised)
        for _ in trange(
            first_step, first_step + steps, desc="Pseudo-marginal Sampler Progress", unit="samples"):
            (theta, log_prior_theta, log_jacobian_theta,
            log_marginal_likelihood_theta, accept) = self._transition_operator(
                theta, log_prior_theta, log_jacobian_theta, log_marginal_likelihood_theta,
                indices, proposal_L_cov,
                num_importance_samples, reparameterised, verbose=verbose)
            acceptance_rate += accept
            theta_samples.append(theta)
        acceptance_rate /= steps
        return np.array(theta_samples), acceptance_rate


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


class InvalidSampler(Exception):
    """An invalid sampler has been passed to `Sampler`"""

    def __init__(self, sampler):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Sampler` as the kernel
            argument.
        :rtype: :class:`InvalidSampler`
        """
        message = (
            f"{sampler} is not an instance of"
            "probit.samplers.Sampler"
        )

        super().__init__(message)
