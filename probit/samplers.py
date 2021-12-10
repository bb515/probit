from abc import ABC, abstractmethod
from operator import pos

from numpy.core.fromnumeric import prod
from probit.estimators import EPOrdinalGP
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
from scipy.stats import norm, multivariate_normal, uniform, expon
from scipy.stats import gamma as gamma_
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from tqdm import trange
from .utilities import sample_Us, matrix_of_differences, matrix_of_differencess
import warnings

import matplotlib.pyplot as plt

class Sampler(ABC):
    """
    Base class for samplers. This class allows users to define a classification problem, get predictions
    using a exact Bayesian inference.

    All samplers must define an init method, which may or may not inherit Sampler as a parent class using
        `super()`.
    All samplers that inherit Sampler define a number of methods that return the samples.
    All samplers must define a _sample_initiate method that is used to initate the sampler.
    All samplers must define an predict method can be  used to make predictions given test data.
    """

    @abstractmethod
    def __init__(self, X_train, t_train, kernel, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg X_train: The data vector.
        :type X_train: :class:`numpy.ndarray`
        :arg t_train: The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
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

        self.D = np.shape(X_train)[1]
        self.N = np.shape(X_train)[0]
        self.X_train = X_train  # (N, D)
        self.X_train_T = X_train.T
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        if np.all(t_train >= 0):
            self.K = int(np.max(t_train) + 1)  # the number of classes
        else:
            raise ValueError("t must contain only positive integer values (got {})").format(t_train)
        self.t_train = t_train

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


class GibbsMultinomialGP(Sampler):
    """
    TODO: was refactored to factor out the cholesky decomposition on 13/07 without testing
    Multinomial Probit regression using Gibbs sampling with GP priors. Inherits the sampler ABC
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.I = np.eye(self.K)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.eye(self.N) + self.C)  # Takes a different notation than in paper
        self.cov = self.C @ self.Sigma
        self.cov_cholesky = np.empty(np.size(self.cov))
        if self.kernel.general_kernel:
            for k in range(self.K):
                self.cov_cholesky[k] = np.linalg.cholesky(self.cov[k])
        else:
            self.cov_cholesky = np.linalg.cholesky(self.cov)

    def _sample_initiate(self, M_0):
        """Initialise variables for the sample method."""
        K = np.shape(M_0)[1]
        if K != self.K:
            raise ValueError("Shape of axis 0 of M_0 must equal K (the number of classes)"
                             " (expected {}, got {})".format(
                self.K, K))
        M_samples = []
        Y_samples = []
        return M_0, M_samples, Y_samples

    def sample(self, M_0, steps, first_step=1):
        """
        Sampling occurs in blocks over the parameters: Y (auxilliaries) and M.

        :arg M_0: (N, K) numpy.ndarray of the initial location of the sampler.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Gibbs samples. The acceptance rate for Gibbs is 1.
        """
        M, M_samples, Y_samples = self._sample_initiate(M_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty Y (N, K) matrix to collect y_n samples over
            Y = -1. * np.ones((self.N, self.K))
            for n, m in enumerate(M):  # i in range N
                # Class index, k, is the target class
                k_true = self.t_train[n]
                # Initiate yi at 0
                y_n = np.zeros(self.K)
                y_n[k_true] = -1.0  # this is a trick for the next line
                # Sample from the cone of the truncated multivariate Gaussian
                # TODO: find out a way to compare yi[k_true] to
                #  all the other values apart from itself, as that is the theory
                while y_n[k_true] < np.max(y_n):
                    # sample Y jointly
                    y_n = m + norm.rvs(size=self.N)
                    # y_n = multivariate_normal.rvs(mean=m, cov=self.I)
                # Add sample to the Y vector
                Y[n, :] = y_n[:]
            # Calculate statistics, then sample other conditional
            # Empty M_T (K, N) matrix to collect m_k samples over
            M_T = -1. * np.ones((self.K, self.N))
            # TODO: Explore if this needs a regularisation trick (Covariance matrices are poorly conditioned)
            # TODO: Explore if this can be vectorised.
            if self.kernel.general_kernel:
                for k in range(self.K):
                    mean = self.cov[k] @ Y.T[k]
                    m_k = mean + self.cov_cholesky[k] @ norm.rvs(size=self.N)
                    # m_k = multivariate_normal.rvs(mean=mean, cov=self.cov[k])
                    # Add sample to the M vector
                    M_T[k, :] = m_k
            else:
                for k in range(self.K):
                    mean = self.cov @ Y.T[k]
                    m_k = mean + self.cov_cholesky @ norm.rvs(size=self.N)
                    # m_k = multivariate_normal.rvs(mean=mean, cov=self.cov)
                    # Add sample to the M vector
                    M_T[k, :] = m_k
            M = M_T.T
            M_samples.append(M)
            Y_samples.append(Y)
        return np.array(M_samples), np.array(Y_samples)

    def _expectation_wrt_u(self, m, n_samples):
        """
        Calculate distribution over classes.
        :arg m: is an (K, ) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :type m: :class:`numpy.ndarray`
        """
        # Find antisymmetric matrix of differences
        difference = matrix_of_differences(m, self.K)  # (K, K)
        # Take samples
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        random_variables = np.add(difference, Us)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        log_cum_dists[:, range(self.K), range(self.K)] = 0
        # axis 0 is the n_samples samples, axis 2 is then the row index, which is the product of cdfs of interest
        log_samples = np.sum(log_cum_dists, axis=2)
        samples = np.exp(log_samples)
        # axis 0 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=0)

    def _vector_expectation_wrt_u(self, ms, n_samples):
        """
        Calculate distribution over classes for multiple m at the same time.

        :arg ms: An (N_test, K) array filled with m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object.
        :type ms: :class:`numpy.ndarray`
        :arg int n_samples: Number of samples to take in the monte carlo estimate.

        :returns: Distribution over classes
        """
        N_test = np.shape(ms)[0]
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(ms, self.K, N_test)  # (N_test, K, K) we will product across axis 2 (rows)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N_test, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N_test, n_samples, K, K)
        # Assume its okay to use the same random variables over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # Fill diagonals with 0
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # axis 0 is the N_test objects,
        # axis 1 is the n_samples samples, axis 3 is then the row index, which is the product of cdfs of interest
        log_samples = np.sum(log_cum_dists, axis=3)
        samples = np.exp(log_samples)
        # axis 1 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=1)

    def _predict_scalar(self, Y_samples, x_test, n_samples=1000):
        """
        Superseded by _predict_vector.

        TODO: This code was refactored on 01/03/2021 and 21/06/2021 without testing. Test it.
        Make gibbs prediction over classes of X_test[0] given the posterior samples.

        :arg Y_samples: The Gibbs samples of the latent variable Y.
        :arg x_test: The new data point, array like (1, D).
        :arg n_samples: The number of samples in the monte carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        cs_new = np.diag(self.kernel.kernel(x_test[0], x_test[0]))  # (1, )
        Cs_new = self.kernel.kernel_vector(x_test, self.X_train)
        intermediate_vector = self.Sigma @ Cs_new  # (N, N_test)
        intermediate_scalar = Cs_new.T @ intermediate_vector
        n_posterior_samples = np.shape(Y_samples)[0]
        # Sample pmf over classes
        distribution_over_classes_samples = []
        for Y in Y_samples:
            m = -1. * np.ones(self.K)  # Initiate m with null values
            for k, y_k in enumerate(Y.T):
                mean_k = y_k.T @ intermediate_vector  # (1, )
                var_k = cs_new - intermediate_scalar  # (1, )
                m[k] = norm.rvs(loc=mean_k, scale=var_k)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            distribution_over_classes_samples.append(self._expectation_wrt_u(m, n_samples))
        monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
        return monte_carlo_estimate

    def _predict_vector(self, Y_samples, X_test, n_samples=1000):
        """
        Make gibbs prediction over classes of X_test given the posterior samples.

        :arg Y_samples: The Gibbs samples of the latent variable Y.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the monte carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = self.Sigma @ Cs_news  # (N, N_test)
        intermediate_vectors_T = intermediate_vectors.T
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        n_posterior_samples = np.shape(Y_samples)[0]
        # Sample pmf over classes
        distribution_over_classes_sampless = []
        for Y in Y_samples:
            # Initiate m with null values
            ms = -1. * np.ones((N_test, self.K))
            for k, y_k in enumerate(Y.T):
                mean_k = intermediate_vectors_T @ y_k  # (N_test, )
                var_k = cs_news - intermediate_scalars  # (N_test, )
                ms[:, k] = norm.rvs(loc=mean_k, scale=var_k)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            # TODO: How do we know that 1000 samples is enough to converge?
            #  Goes with root n_samples but depends on the estimator variance
            distribution_over_classes_sampless.append(self._vector_expectation_wrt_u(ms, n_samples))
        # TODO: Could also get a variance from the MC estimate.
        return (1. / n_posterior_samples) * np.sum(distribution_over_classes_sampless, axis=0)

    def _predict_vector_generalised(self, Y_samples, X_test, n_samples=1000):
        """
        Make gibbs prediction over classes of X_test given the posterior samples.

        This is the general case where there are hyperparameters varphi (K, D)
            for all dimensions and classes.

        :arg Y_samples: The Gibbs samples of the latent variable Y.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the monte carlo estimate.
        :return: A monte carlo estimate of the class probabilities.
        """
        # X_new = np.append(X_test, self.X_train, axis=0)
        N_test = np.shape(X_test)[0]
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (K, N, N_test)
        # TODO: this is a bottleneck
        cs_news = [np.diag(self.kernel.kernel_matrix(X_test, X_test)[k]) for k in range(self.K)]  # (K, N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = self.Sigma @ Cs_news  # (K, N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (K, N_test, )
        n_posterior_samples = np.shape(Y_samples)[0]
        # TODO: Is there a way to do this without transposing and reshaping?
        Y_samples_T = np.transpose(Y_samples, (0, 2, 1)) #  (n_posterior_samples, K, N)
        ## Y_samples_T = np.reshape(Y_samples_T, (n_posterior_samples, self.K, self.N, 1))
        distribution_over_classes_sampless = []
        for Y_T in Y_samples_T:
            # Initiate m with null values
            M_new = np.empty((N_test, self.K))
            for k, y_k in enumerate(Y_T):
                M_new_tilde_k = intermediate_vectors_T[k] @ y_k  # (N_test, )
                var_new_tilde_k = cs_news[k] - intermediate_scalars[k]  # (N_test, )
                M_new[:, k] = norm.rvs(loc=M_new_tilde_k, scale=var_new_tilde_k)
            # # Vectorised version which is less readable, and likely slower TODO: test this.
            # M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_T)
            # M_new_tilde_T = np.reshape(M_new_tilde_T, (self.K, N_test))
            # var_new_tilde = np.subtract(cs_news, intermediate_scalars)
            # M_new = norm.rvs(loc=M_new_tilde_T.T, scale=var_new_tilde.T)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            # TODO: How do we know that 1000 samples is enough to converge?
            #  Goes with root n_samples but depends on the estimator variance
            distribution_over_classes_sampless.append(self._vector_expectation_wrt_u(M_new, n_samples))
        # TODO: Could also get a variance from the MC estimate.
        return (1. / n_posterior_samples) * np.sum(distribution_over_classes_sampless, axis=0)

    def predict(self, Y_samples, X_test, n_samples=1000, vectorised=True):
        if self.kernel.general_kernel:
            if vectorised:
                return self._predict_vector_generalised(Y_samples, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")
        else:
            if vectorised:
                return self._predict_vector(Y_samples, X_test, n_samples)
            else:
                return self._predict_scalar


class GibbsBinomial(Sampler):
    """
        A Gibbs sampler for linear regression. Inherits the sampler ABC
    """

    def __init__(self, X_train, t_train):
        """
        Create an :class:`GibbsBinomial` sampler object.

        :returns: An :class:`GibbsBinomial` object.
        """
        self.D = np.shape(X_train)[1]
        self.N = np.shape(X_train)[0]
        self.X_train = X_train  # (N, D)
        self.X_train_T = X_train.T
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        if t_train.all() not in [0, 1]:
            raise ValueError("In the binomial case, t must contain only 1s and/or 0s (got {})".format(t_train))
        self.K = int(np.max(t_train) + 1)  # the number of classes
        self.t_train = t_train
        self.cov = np.linalg.inv(self.X_train_T @ self.X_train)  # From Mark's lecture notes
        self.cov_cholesky = np.linalg.cholesky(self.cov)

    def _sample_initiate(self, beta_0):
        """Initialise variables for the sample method."""
        K_plus_1 = np.shape(beta_0)[0]
        if K_plus_1 != self.K + 1:
            raise ValueError("Shape of axis 0 of beta_0 must equal K + 1 (the number of classes plus one)"
                             " (expected {}, got {})".format(
                self.K + 1, K_plus_1))
        I = np.eye(self.K)  # (K, K)
        return I, beta_0

    def sample(self, beta_0, steps, first_step=1):
        """
        Take n Gibbs samples.

        :arg beta_0: The initial location of the sampler in parameter space (K + 1, ) ndarray.
        :type beta_0: :class:`numpy.ndarray`
        :arg int n: Number of samples.
        :return: Array of n samples.
        :rtype: :class:`numpy.ndarray`
        """
        I, beta = self._sample_initiate(beta_0)
        beta_samples = []
        Y_samples = []
        for _ in trange(first_step, first_step + steps,
                        desc="Regression Sampler Progress", unit="samples"):
            # Empty Y vector to collect Y_i samples over
            Y = []
            for i, x in enumerate(self.X_train):
                # Sample from truncated Gaussian depending on t
                if self.t_train[i] == 1:
                    yi = 0
                    while yi <= 0:
                        yi = norm.rvs(loc=np.dot(beta, x), scale=1)
                else:
                    yi = 0
                    while yi >= 0:
                        yi = norm.rvs(loc=np.dot(beta, x), scale=1)
                # Add sample to the Y vector
                Y.append(yi)
            # Calculate statistics, then sample other conditional
            mean = self.cov @ self.X_train_T @ np.array(Y)
            beta = mean + self.cov_cholesky @ norm.rvs(size=self.N)
            # beta = multivariate_normal.rvs(mean=mean, cov=self.cov)
            beta_samples.append(beta)
            Y_samples.append(Y)
        beta_samples = np.array(beta_samples, dtype=np.float64)
        Y_samples = np.array(Y_samples, dtype=np.float64)
        return beta_samples, Y_samples

    def predict(self, beta_samples, x_test):
        """Make gibbs prediction class of x_new given the beta posterior samples."""
        f = [norm.cdf(np.dot(beta, x_test)) for beta in beta_samples]
        return sum(f) / len(beta_samples)


class GibbsMultinomialOrderedGPTemp(Sampler):
    """
        A Gibbs sampler for Multinomial regression of ordered data. Inherits the sampler ABC
    """

    def __init__(self, K, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        if self.kernel.general_kernel:
            raise ValueError('The kernel must not be ARD type (kernel.general_kernel=1),'
                             ' but ISO type (kernel.general_kernel=0). (got {}, expected)'.format(
                self.kernel.general_kernel, 0))
        self.K = K
        self.I = np.eye(self.K)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.eye(self.N) + self.C)
        self.cov = self.C @ self.Sigma

    def _sample_initiate(self, m_0, y_0, gamma_0):
        """
        Initialise variables for the sample method.
        TODO: 13/07/2021 Got around to removing the zero cutpoint, and consider implimenting Elliptical slice sampling
        # Try to get code for Elliptical slice sampling
        TODO: 04/06/2021 The zero cutpoint is not necessary, so in this 'Temp' version, I remove it.
        TODO: 03/03/2021 The first Gibbs step is not robust to a poor choice of m_0 (it will never sample a y_1 within
            range of the cutpoints). Idea: just initialise y_0 and m_0 close to another.
            Start with an initial guess for m_0 based on a linear regression and then initialise y_0 with random N(0,1)
            samples around that. Need to test convergence for random init of y_0 and m_0.
        TODO: 01/03/2021 converted gamma to be a [np.NINF, 0.0, ..., np.inf] array. This may cause problems
            but those objects are both IEEE so should be okay.
        """
        # Treat user parsing of cutpoint parameters with just the upper cutpoints for each class
        # The first class t=0 is for y<=0
        if np.shape(gamma_0)[0] == self.K - 1:  # not including any of the fixed cutpoints: -\infty, 0, \infty
            gamma_0 = np.append(gamma_0, np.inf)  # append the infinity cutpoint
            gamma_0 = np.insert(gamma_0, np.NINF)  # insert the negative infinity cutpoint at index 0
            pass  # correct format
        elif np.shape(gamma_0)[0] == self.K:  # not including one of the infinity cutpoints
            if gamma_0[-1] != np.inf:
                if gamma_0[0] != np.NINF:
                    raise ValueError('The last cutpoint parameter must be numpy.inf, or the first cutpoint parameter'
                                     ' must be numpy.NINF (got {}, expected {})'.format(
                        gamma_0[-1], [np.inf, np.NINF]))

            else:
                gamma_0 = np.insert(gamma_0, np.NINF)
                pass  # correct format
        elif np.shape(gamma_0)[0] == self.K + 1:  # including all of the cutpoints
            if gamma_0[0] != np.NINF:
                raise ValueError('The cutpoint parameter \gamma_0 must be numpy.NINF (got {}, expected {})'.format(
                    gamma_0[0], np.NINF))
            if gamma_0[-1] != np.inf:
                raise ValueError('The cutpoint parameter \gamma_K must be numpy.inf (got {}, expected {})'.format(
                    gamma_0[-1], np.inf))
            pass  # correct format
        else:
            raise ValueError('Could not recognise gamma_0 shape. (np.shape(gamma_0) was {})'.format(np.shape(gamma_0)))
        assert gamma_0[0] == np.NINF
        assert gamma_0[-1] == np.inf
        assert np.shape(gamma_0)[0] == self.K + 1
        m_samples = []
        y_samples = []
        gamma_samples = []
        return m_0, y_0, gamma_0, m_samples, y_samples, gamma_samples

    def sample_metropolis_within_gibbs(self, m_0, y_0, gamma_0, sigma_gamma, steps, first_step=1):
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
        m, y, gamma_prev, m_samples, y_samples, gamma_samples = self._sample_initiate(m_0, y_0, gamma_0)
        precision_gamma = 1. / sigma_gamma
        i_gamma_k = np.add(self.t_train, 1)
        i_gamma_k_minus = self.t_train
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty gamma (K + 1, ) array to collect the upper cut-points for each class
            gamma = -1. * np.ones(self.K + 1)
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_K = +\infty
            gamma[0] = np.NINF
            gamma[-1] = np.inf

            # Vector
            for k in range(2, self.K):  # TODO: Can be made into a C binding as in Andrius' code
                gamma_proposal = -np.inf
                while gamma_proposal <= gamma[k - 1] or gamma_proposal > gamma_prev[k + 1]:
                    gamma_proposal = norm.rvs(loc=gamma_prev[k], scale=sigma_gamma)
                gamma[k] = gamma_proposal
            #print('gamma_proposal', gamma)
            # Calculate acceptance probability
            num_2 = np.sum(np.log(
                    norm.cdf(np.multiply(precision_gamma, gamma_prev[3:self.K + 1] - gamma_prev[2:self.K]))
                    - norm.cdf(np.multiply(precision_gamma, gamma[1:self.K - 1] - gamma_prev[2:self.K]))
            ))
            den_2 = np.sum(np.log(
                    norm.cdf(np.multiply(precision_gamma, gamma[3:self.K + 1] - gamma[2:self.K]))
                    - norm.cdf(np.multiply(precision_gamma, gamma_prev[1:self.K - 1] - gamma[2:self.K]))
            ))
            # Needs a product over m
            gamma_k = gamma[i_gamma_k]
            gamma_prev_k = gamma_prev[i_gamma_k]
            gamma_k_minus = gamma[i_gamma_k_minus]
            gamma_prev_k_minus = gamma_prev[i_gamma_k_minus]
            num_1 = np.sum(np.log(norm.cdf(gamma_k - m) - norm.cdf(gamma_k_minus - m)))
            den_1 = np.sum(np.log(norm.cdf(gamma_prev_k - m) - norm.cdf(gamma_prev_k_minus - m)))
            #print('num_1', num_1)
            #print('den_1', den_1)
            alpha = np.exp(num_1 + num_2 - den_1 - den_2)
            #print('alpha = ', alpha)

            # # Scalar version
            # alpha = 0
            # for k in range(2, self.K):
            #     # Propose cutpoint parameter
            #     gamma_proposal = -np.inf
            #     while gamma_proposal <= gamma[k - 1] or gamma_proposal > gamma_prev[k + 1]:
            #         gamma_proposal = norm.rvs(loc=gamma_prev[k], scale=sigma_gamma)
            #         print('gamma_proposal', gamma_proposal)
            #     gamma[k] = gamma_proposal
            #     alpha += np.log(
            #             norm.cdf(precision_gamma * (gamma_prev[k + 1] - gamma_prev[k]))
            #             - norm.cdf(precision_gamma * (gamma[k - 1] - gamma_prev[k]))
            #     ) - np.log(
            #         norm.cdf(precision_gamma * (gamma[k + 1] - gamma[k]))
            #         - norm.cdf(precision_gamma * (gamma_prev[k - 1] - gamma[k]))
            #     )
            #
            # # Needs a product over m
            # gamma_k = gamma[i_gamma_k]
            # gamma_prev_k = gamma_prev[i_gamma_k]
            # gamma_k_minus = gamma[i_gamma_k_minus]
            # gamma_prev_k_minus = gamma_prev[i_gamma_k_minus]
            # print('gamma_k', np.shape(gamma_k), gamma_k[1])
            # print('gamma_k_minus', np.shape(gamma_k_minus), gamma_k_minus[1])
            # print('gamma_prev_k', np.shape(gamma_prev_k), gamma_prev_k[1])
            # print('m', m[1])
            # num_1 = np.sum(np.log(norm.cdf(gamma_k - m) - norm.cdf(gamma_k_minus - m)))
            # den_1 = np.sum(np.log(norm.cdf(gamma_prev_k - m) - norm.cdf(gamma_prev_k_minus - m)))
            # alpha = np.exp(alpha + num_1 - den_1)
            # print('alpha = ', alpha)

            if uniform.rvs(0, 1) < alpha:
                # Accept
                gamma_prev = gamma
                # Sample y from the usual full conditional
                # Empty y (N, ) matrix to collect y sample over
                y = -1. * np.ones(self.N)
                for n, m_n in enumerate(m):  # i in range N
                    # Class index, k, is the target class
                    k_true = self.t_train[n]
                    # Initiate yi at 0
                    y_n = np.NINF  # this is a trick for the next line
                    # Sample from the truncated Gaussian
                    while y_n > gamma[k_true + 1] or y_n <= gamma[k_true]:
                        # sample y
                        y_n = norm.rvs(loc=m_n, scale=1)
                    # Add sample to the Y vector
                    y[n] = y_n
            else:
                # Reject, and use previous \gamma, y sample
                gamma = gamma_prev
            # Calculate statistics, then sample other conditional
            # TODO: Explore if this needs a regularisation trick (Covariance matrices are poorly conditioned)
            # TODO: Factorize cholesky
            mean = self.cov @ y
            m = multivariate_normal.rvs(mean=mean, cov=self.cov)
            m_samples.append(m)
            y_samples.append(y)
            gamma_samples.append(gamma)
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)

    def sample(self, m_0, y_0, gamma_0, steps, first_step=1):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: y (auxilliaries), m (GP regression posterior means) and
        gamma (cutpoint parameters).

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg y_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type y_0: :class:`np.ndarray`.
        :arg gamma_0: (K + 1, ) numpy.ndarray of the initial location of the sampler.
        :type gamma_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Gibbs samples. The acceptance rate for the Gibbs algorithm is 1.
        """
        m, y, gamma_prev, m_samples, y_samples, gamma_samples = self._sample_initiate(m_0, y_0, gamma_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty gamma (K + 1, ) array to collect the upper cut-points for each class
            gamma = -1. * np.ones(self.K + 1)
            uppers = -1. * np.ones(self.K - 2)
            locs = -1. * np.ones(self.K - 2)
            for k in range(0, self.K - 1):  # TODO change the index to the class.
                indeces = np.where(self.t_train == k)
                indeces2 = np.where(self.t_train == k + 1)
                if indeces2:
                    uppers[k] = np.min(np.append(y[indeces2], gamma_prev[k + 2]))
                else:
                    uppers[k] = gamma_prev[k + 2]
                if indeces:
                    locs[k] = np.max(np.append(y[indeces], gamma_prev[k]))
                else:
                    locs[k] = gamma_prev[k]
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_K = +\infty
            gamma[0] = np.NINF
            gamma[1:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            gamma[-1] = np.inf
            # update gamma prev
            gamma_prev = gamma
            # Empty y (N, ) matrix to collect y sample over
            y = -1. * np.ones(self.N)
            for n, m_n in enumerate(m):  # i in range N
                # Class index, k, is the target class
                k_true = self.t_train[n]
                # Initiate yi at 0
                y_n = np.NINF  # this is a trick for the next line
                # Sample from the truncated Gaussian
                # print(gamma[k_true + 1])
                # print(gamma[k_true])
                # print(m_n)
                while y_n > gamma[k_true + 1] or y_n <= gamma[k_true]:
                    # sample y
                    y_n = norm.rvs(loc=m_n, scale=1)
                # Add sample to the Y vector
                y[n] = y_n
            # Calculate statistics, then sample other conditional
            # TODO: Explore if this needs a regularisation trick (Covariance matrices are poorly conditioned)
            # TODO: Factorize cholesky
            mean = self.cov @ y
            m = multivariate_normal.rvs(mean=mean, cov=self.cov)
            # print(m, 'm')
            # print(y, 'y')
            # print(gamma, 'gamma')
            m_samples.append(m)
            y_samples.append(y)
            gamma_samples.append(gamma)
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)

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


class GibbsMultinomialOrderedGP(Sampler):
    """
        A Gibbs sampler for Multinomial regression of ordered data. Inherits the sampler ABC
    """

    def __init__(self, K, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        if self.kernel.general_kernel:
            raise ValueError('The kernel must not be ARD type (kernel.general_kernel=1),'
                             ' but ISO type (kernel.general_kernel=0). (got {}, expected)'.format(
                self.kernel.general_kernel, 0))
        self.K = K
        self.I = np.eye(self.K)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.eye(self.N) + self.C)
        self.cov = self.C @ self.Sigma

    def _sample_initiate(self, m_0, y_0, gamma_0):
        """
        Initialise variables for the sample method.
        TODO: 03/03/2021 The first Gibbs step is not robust to a poor choice of m_0 (it will never sample a y_1 within
            range of the cutpoints). Idea: just initialise y_0 and m_0 close to another.
            Start with an initial guess for m_0 based on a linear regression and then initialise y_0 with random N(0,1)
            samples around that. Need to test convergence for random init of y_0 and m_0.
        TODO: 01/03/2021 converted gamma to be a [np.NINF, 0.0, ..., np.inf] array. This may cause problems
            but those objects are both IEEE so should be okay.
        """
        # Treat user parsing of cutpoint parameters with just the upper cutpoints for each class
        # The first class t=0 is for y<=0
        if np.shape(gamma_0)[0] == self.K - 2:  # not including any of the fixed cutpoints: -\infty, 0, \infty
            gamma_0 = np.append(gamma_0, np.inf)  # append the infinity cutpoint
            gamma_0 = np.insert(gamma_0, 0.0)  # insert the zero cutpoint at index 0
            gamma_0 = np.insert(gamma_0, np.NINF)  # insert the negative infinity cutpoint at index 0
            pass  # correct format
        elif np.shape(gamma_0)[0] == self.K:  # not including one of the infinity cutpoints
            if gamma_0[-1] != np.inf:
                if gamma_0[0] != np.NINF:
                    raise ValueError('The last cutpoint parameter must be numpy.inf, or the first cutpoint parameter'
                                     ' must be numpy.NINF (got {}, expected {})'.format(
                        gamma_0[-1], [np.inf, np.NINF]))
                else:  # gamma_0[0] is negative infinity
                    if gamma_0[1] == 0.0:
                        gamma_0.append(np.inf)
                        pass  # correct format
                    else:
                        raise ValueError('The cutpoint parameter \gamma_1 must be 0.0 (got {}, expected {})'.format(
                            gamma_0[1], 0.0))
            else:
                if gamma_0[0] != 0.0:
                    raise ValueError('The cutpoint parameter \gamma_1 must be 0.0 (got {}, expected {})'.format(
                        gamma_0[0], 0.0))
                gamma_0 = np.insert(gamma_0, np.NINF)
                pass  # correct format
        elif np.shape(gamma_0)[0] == self.K - 1:  # not including two of the cutpoints
            if gamma_0[0] != np.NINF:  # not including negative infinity cutpoint
                if gamma_0[-1] != np.inf:
                    raise ValueError('The cutpoint paramter \gamma_K must be numpy.inf (got {}, expected {})'.format(
                        gamma_0[-1], np.inf))
                elif gamma_0[0] != 0.0:
                    raise ValueError('The cutpoint parameter \gamma_1 must be 0.0 (got {}, expected {})'.format(
                        gamma_0[0], 0.0))
                else:
                    gamma_0 = np.insert(gamma_0, np.NINF)
                    pass  # correct format
            elif gamma_0[1] != 0.0:  # Including \gamma_0 = np.NINF but not \gamma_1 = 0.0
                raise ValueError('The cutpoint parameter \gamma_1 must be 0.0 (got {}, expected {})'.format(
                    gamma_0[1], 0.0))
            else:
                if gamma_0[-1] == np.inf:
                    raise ValueError('Length of gamma_0 seems to be one less than it needs to be. Missing a cutpoint! ('
                                     'got {}, expected {})'.format(len(gamma_0), len(gamma_0) + 1))
                else:
                    gamma_0 = np.append(gamma_0, np.inf)
                    pass  # correct format
        elif np.shape(gamma_0)[0] == self.K + 1:  # including all of the cutpoints
            if gamma_0[0] != np.NINF:
                raise ValueError('The cutpoint parameter \gamma_0 must be numpy.NINF (got {}, expected {})'.format(
                    gamma_0[0], np.NINF))
            if gamma_0[1] != 0.0:
                raise ValueError('The cutpoint parameter \gamma_1 must be 0.0 (got {}, expected {})'.format(
                    gamma_0[1], 0.0))
            if gamma_0[-1] != np.inf:
                raise ValueError('The cutpoint parameter \gamma_K must be numpy.inf (got {}, expected {})'.format(
                    gamma_0[-1], np.inf))
            pass  # correct format
        else:
            raise ValueError('Could not recognise gamma_0 shape. (np.shape(gamma_0) was {})'.format(np.shape(gamma_0)))
        assert gamma_0[0] == np.NINF
        assert gamma_0[1] == 0.0
        assert gamma_0[-1] == np.inf
        if not np.all(gamma_0[2:-1] > 0):
            raise ValueError('The cutpoint parameters must be positive. (got {})'.format(gamma_0))
        assert np.shape(gamma_0)[0] == self.K + 1
        m_samples = []
        y_samples = []
        gamma_samples = []
        return m_0, y_0, gamma_0, m_samples, y_samples, gamma_samples

    def sample_metropolis_within_gibbs(self, m_0, y_0, gamma_0, sigma_gamma, steps, first_step=1):
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
        m, y, gamma_prev, m_samples, y_samples, gamma_samples = self._sample_initiate(m_0, y_0, gamma_0)
        precision_gamma = 1. / sigma_gamma
        i_gamma_k = np.add(self.t_train, 1)
        i_gamma_k_minus = self.t_train
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty gamma (K + 1, ) array to collect the upper cut-points for each class
            gamma = -1. * np.ones(self.K + 1)
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_K = +\infty
            gamma[0] = np.NINF
            gamma[1] = 0.0
            gamma[-1] = np.inf

            # Vector
            for k in range(2, self.K):  # TODO: Can be made into a C binding as in Andrius' code
                gamma_proposal = -np.inf
                while gamma_proposal <= gamma[k - 1] or gamma_proposal > gamma_prev[k + 1]:
                    gamma_proposal = norm.rvs(loc=gamma_prev[k], scale=sigma_gamma)
                gamma[k] = gamma_proposal
            #print('gamma_proposal', gamma)
            # Calculate acceptance probability
            num_2 = np.sum(np.log(
                    norm.cdf(np.multiply(precision_gamma, gamma_prev[3:self.K + 1] - gamma_prev[2:self.K]))
                    - norm.cdf(np.multiply(precision_gamma, gamma[1:self.K - 1] - gamma_prev[2:self.K]))
            ))
            den_2 = np.sum(np.log(
                    norm.cdf(np.multiply(precision_gamma, gamma[3:self.K + 1] - gamma[2:self.K]))
                    - norm.cdf(np.multiply(precision_gamma, gamma_prev[1:self.K - 1] - gamma[2:self.K]))
            ))
            # Needs a product over m
            gamma_k = gamma[i_gamma_k]
            gamma_prev_k = gamma_prev[i_gamma_k]
            gamma_k_minus = gamma[i_gamma_k_minus]
            gamma_prev_k_minus = gamma_prev[i_gamma_k_minus]
            num_1 = np.sum(np.log(norm.cdf(gamma_k - m) - norm.cdf(gamma_k_minus - m)))
            den_1 = np.sum(np.log(norm.cdf(gamma_prev_k - m) - norm.cdf(gamma_prev_k_minus - m)))
            #print('num_1', num_1)
            #print('den_1', den_1)
            alpha = np.exp(num_1 + num_2 - den_1 - den_2)
            #print('alpha = ', alpha)

            # # Scalar version
            # alpha = 0
            # for k in range(2, self.K):
            #     # Propose cutpoint parameter
            #     gamma_proposal = -np.inf
            #     while gamma_proposal <= gamma[k - 1] or gamma_proposal > gamma_prev[k + 1]:
            #         gamma_proposal = norm.rvs(loc=gamma_prev[k], scale=sigma_gamma)
            #         print('gamma_proposal', gamma_proposal)
            #     gamma[k] = gamma_proposal
            #     alpha += np.log(
            #             norm.cdf(precision_gamma * (gamma_prev[k + 1] - gamma_prev[k]))
            #             - norm.cdf(precision_gamma * (gamma[k - 1] - gamma_prev[k]))
            #     ) - np.log(
            #         norm.cdf(precision_gamma * (gamma[k + 1] - gamma[k]))
            #         - norm.cdf(precision_gamma * (gamma_prev[k - 1] - gamma[k]))
            #     )
            #
            # # Needs a product over m
            # gamma_k = gamma[i_gamma_k]
            # gamma_prev_k = gamma_prev[i_gamma_k]
            # gamma_k_minus = gamma[i_gamma_k_minus]
            # gamma_prev_k_minus = gamma_prev[i_gamma_k_minus]
            # print('gamma_k', np.shape(gamma_k), gamma_k[1])
            # print('gamma_k_minus', np.shape(gamma_k_minus), gamma_k_minus[1])
            # print('gamma_prev_k', np.shape(gamma_prev_k), gamma_prev_k[1])
            # print('m', m[1])
            # num_1 = np.sum(np.log(norm.cdf(gamma_k - m) - norm.cdf(gamma_k_minus - m)))
            # den_1 = np.sum(np.log(norm.cdf(gamma_prev_k - m) - norm.cdf(gamma_prev_k_minus - m)))
            # alpha = np.exp(alpha + num_1 - den_1)
            # print('alpha = ', alpha)

            if uniform.rvs(0, 1) < alpha:
                # Accept
                gamma_prev = gamma
                # Sample y from the usual full conditional
                # Empty y (N, ) matrix to collect y sample over
                y = -1. * np.ones(self.N)
                for n, m_n in enumerate(m):  # i in range N
                    # Class index, k, is the target class
                    k_true = self.t_train[n]
                    # Initiate yi at 0
                    y_n = np.NINF  # this is a trick for the next line
                    # Sample from the truncated Gaussian
                    while y_n > gamma[k_true + 1] or y_n <= gamma[k_true]:
                        # sample y
                        y_n = norm.rvs(loc=m_n, scale=1)
                    # Add sample to the Y vector
                    y[n] = y_n
            else:
                # Reject, and use previous \gamma, y sample
                gamma = gamma_prev
            # Calculate statistics, then sample other conditional
            # TODO: Explore if this needs a regularisation trick (Covariance matrices are poorly conditioned)
            # TODO: Factorize cholesky
            mean = self.cov @ y
            m = multivariate_normal.rvs(mean=mean, cov=self.cov)
            m_samples.append(m)
            y_samples.append(y)
            gamma_samples.append(gamma)
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)

    def sample(self, m_0, y_0, gamma_0, steps, first_step=1):
        """
        Sample from the posterior.

        Sampling occurs in Gibbs blocks over the parameters: y (auxilliaries), m (GP regression posterior means) and
        gamma (cutpoint parameters).

        :arg m_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type m_0: :class:`np.ndarray`.
        :arg y_0: (N, ) numpy.ndarray of the initial location of the sampler.
        :type y_0: :class:`np.ndarray`.
        :arg gamma_0: (K + 1, ) numpy.ndarray of the initial location of the sampler.
        :type gamma_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Gibbs samples. The acceptance rate for the Gibbs algorithm is 1.
        """
        m, y, gamma_prev, m_samples, y_samples, gamma_samples = self._sample_initiate(m_0, y_0, gamma_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            # Empty gamma (K + 1, ) array to collect the upper cut-points for each class
            gamma = -1. * np.ones(self.K + 1)
            uppers = -1. * np.ones(self.K - 2)
            locs = -1. * np.ones(self.K - 2)
            for k in range(1, self.K - 1):  # TODO change the index to the class.
                indeces = np.where(self.t_train == k)
                indeces2 = np.where(self.t_train == k + 1)
                if indeces2:
                    uppers[k - 1] = np.min(np.append(y[indeces2], gamma_prev[k + 2]))
                else:
                    uppers[k - 1] = gamma_prev[k + 2]
                if indeces:
                    locs[k - 1] = np.max(np.append(y[indeces], gamma_prev[k]))
                else:
                    locs[k - 1] = gamma_prev[k]
            # Fix \gamma_0 = -\infty, \gamma_1 = 0, \gamma_K = +\infty
            gamma[0] = np.NINF
            gamma[1] = 0.0
            gamma[2:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            gamma[-1] = np.inf
            # update gamma prev
            gamma_prev = gamma
            # Empty y (N, ) matrix to collect y sample over
            y = -1. * np.ones(self.N)
            for n, m_n in enumerate(m):  # i in range N
                # Class index, k, is the target class
                k_true = self.t_train[n]
                # Initiate yi at 0
                y_n = np.NINF  # this is a trick for the next line
                # Sample from the truncated Gaussian
                # print(gamma[k_true + 1])
                # print(gamma[k_true])
                # print(m_n)
                while y_n > gamma[k_true + 1] or y_n <= gamma[k_true]:
                    # sample y
                    y_n = norm.rvs(loc=m_n, scale=1)
                # Add sample to the Y vector
                y[n] = y_n
            # Calculate statistics, then sample other conditional
            # TODO: Explore if this needs a regularisation trick (Covariance matrices are poorly conditioned)
            # TODO: Can clearly factorize the Cholesky decomposition here
            mean = self.cov @ y
            m = multivariate_normal.rvs(mean=mean, cov=self.cov)
            # print(m, 'm')
            # print(y, 'y')
            # print(gamma, 'gamma')
            m_samples.append(m)
            y_samples.append(y)
            gamma_samples.append(gamma)
        return np.array(m_samples), np.array(y_samples), np.array(gamma_samples)

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
            gamma[1] = 0.0
            gamma[2:-1] = uniform.rvs(loc=locs, scale=uppers - locs)
            gamma[-1] = np.inf
            # Calculate the measurable function and append the resulting MC sample
            distribution_over_classes_samples.append(self._probit_likelihood(m, gamma))

        monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
        return monte_carlo_estimate

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


class PseudoMarginalOrdinalGP(EPOrdinalGP):
    """
    Pseudo-Marginal Bayesian Inference for Gaussian process ordinal regression.

    ref: Filippone, Maurizio & Girolami, Mark. (2014). Pseudo-Marginal Bayesian Inference for Gaussian Processes.
        IEEE Transactions on Pattern Analysis and Machine Intelligence. 10.1109/TPAMI.2014.2316530. 
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`PseudoMarginalOrdinal` sampler object.

        Inherits the :class:`EPOrdinalGP` estimator class.

        :returns: An :class:`PseudoMarginalOrdinalGP` object.
        """
        super().__init__(*args, **kwargs)

    def _prior_reparameterised(self, theta, indices):
        """
        A reparametrisation such that all of the hyperparameters can be sampled from the real line,
        therefore there is no transformation (thus no jacobian) when sampling from the proposal distribution.
        Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
        """
        # Do not update these hyperparameters by default
        noise_variance = None
        gamma = None
        scale = None
        varphi = None
        index = 0
        log_prior_theta = np.zeros(len(theta))
        if indices[0]:
            # Gamma prior is placed on the noise std - evaluate the prior pdf
            log_noise_std = theta[index]
            log_prior_pdf = norm.logpdf(
                log_noise_std,
                loc=self.noise_std_hyperparameters[0],
                scale=self.noise_std_hyperparameters[1])
            log_prior_theta[index] = log_prior_pdf
            noise_variance = np.exp(log_noise_std)**2
            # scale = scale_0
            if noise_variance < 1.0e-04:
                warnings.warn(
                    "WARNING: noise variance is very low - numerical stability"
                    " issues may arise (noise_variance={}).".format(
                        noise_variance))
            elif noise_variance > 1.0e3:
                warnings.warn(
                    "WARNING: noise variance is very large - numerical "
                    "stability issues may arise (noise_variance={}).".format(
                        noise_variance))
            index += 1
        if indices[1]:
            if gamma is None:
                # Get gamma from classifier
                gamma = self.gamma
            gamma[1] = theta[index]
            log_prior_pdf = norm.logpdf(
                theta[index],
                loc=self.gamma_hyperparameters[1, 0],
                scale=self.gamma_hyperparameters[1, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                if gamma is None:
                    # Get gamma from classifier
                    gamma = self.gamma
                gamma[j] = gamma[j-1] + np.exp(theta[index])
                log_prior_pdf = norm.pdf(
                    theta[index],
                    loc=self.gamma_hyperparameters[j, 0],
                    scale=self.gamma_hyperparameters[j, 1])
                log_prior_theta[index] = log_prior_pdf
                index += 1
        if indices[self.J]:
            scale_std = np.exp(theta[index])
            scale = scale_std**2
            log_prior_pdf = norm.logpdf(
                theta[index],
                loc=self.kernel.scale_hyperparameters[self.J, 0],
                scale=self.kernel.scale_hyperparameters[self.J, 1])
            log_prior_pdf[index] = log_prior_pdf
            index += 1
        if indices[self.J + 1]:
            if self.kernel._general and self.kernel._ARD:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and lengthscales parameter for each dimension and class
                raise ValueError("TODO")
            else:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and a single, shared lengthscale parameter
                varphi = np.exp(theta[index])
                log_prior_pdf = norm.logpdf(
                    theta[index],
                    loc=self.kernel.psi[0],
                    scale=self.kernel.psi[1])
                log_prior_theta[index] = log_prior_pdf
                index += 1
        # Update prior covariance
        print(varphi)
        self.hyperparameters_update(
            gamma=gamma, varphi=varphi, scale=scale,
            noise_variance=noise_variance)
        # intervals = self.gamma[2:self.J] - self.gamma[1:self.J - 1]
        return log_prior_theta 

    def _prior(self, theta, indices):
        """
        A priors defined over their usual domains, and so a transformation of random variables is used
        for sampling from proposal distrubutions defined over continuous domains.
        Hyperparameter priors assumed to be independent assumption so take the product of prior pdfs.
        """
        # Do not update these hyperparameters by default
        noise_variance = None
        gamma = None
        scale = None
        varphi = None
        log_prior_theta = np.zeros(len(theta))
        index = 0
        if indices[0]:
            # Gamma prior is placed on the noise std - evaluate the prior pdf
            noise_std = np.exp(theta[index])
            log_prior_pdf = gamma_.logpdf(
                noise_std,
                loc=self.noise_std_hyperparameters[0],
                scale=self.noise_std_hyperparameters[1])
            log_prior_theta[index] = log_prior_pdf
            # scale = scale_0
            if noise_variance < 1.0e-04:
                warnings.warn(
                    "WARNING: noise variance is very low - numerical stability"
                    " issues may arise (noise_variance={}).".format(
                        noise_variance))
            elif noise_variance > 1.0e3:
                warnings.warn(
                    "WARNING: noise variance is very large - numerical "
                    "stability issues may arise (noise_variance={}).".format(
                        noise_variance))
            index += 1
        if indices[1]:
            if gamma is None:
                # Get gamma from classifier
                gamma = self.gamma
            gamma[1] = theta[index]
            log_prior_pdf = norm.logpdf(
                theta[index],
                loc=self.gamma_hyperparameters[1, 0],
                scale=self.gamma_hyperparameters[1, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                if gamma is None:
                    # Get gamma from classifier
                    gamma = self.gamma
                gamma[j] = gamma[j-1] + np.exp(theta[index])
                log_prior_pdf = norm.logpdf(
                    theta[index],
                    loc=self.gamma_hyperparameters[j, 0],
                    scale=self.gamma_hyperparameters[j, 1])
                log_prior_theta[index] = log_prior_pdf
                index += 1
        if indices[self.J]:
            scale_std = np.exp(theta[index])
            scale = scale_std**2
            log_prior_pdf = norm.logpdf(
                    theta[index],
                    loc=self.gamma_hyperparameters[self.J, 0],
                    scale=self.gamma_hyperparameters[self.J, 1])
            log_prior_theta[index] = log_prior_pdf
            index += 1
        if indices[self.J + 1]:
            if self.kernel._general and self.kernel._ARD:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and lengthscales parameter for each dimension and class
                raise ValueError("TODO")
            else:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and a single, shared lengthscale parameter
                varphi = np.exp(theta[index])
                log_prior_pdf = gamma_.logpdf(
                    varphi,
                    a=self.kernel.psi[0],
                    scale=1./ self.kernel.psi[1])
                log_prior_theta[index] = log_prior_pdf
                index += 1
        print("VARPHI", varphi)

        # Update prior covariance
        self.hyperparameters_update(
            gamma=gamma, varphi=varphi, scale=scale,
            noise_variance=noise_variance)
        # intervals = self.gamma[2:self.J] - self.gamma[1:self.J - 1]
        return log_prior_theta

    def _proposal_initiate(self, theta, indices, proposal_cov):
        """TODO: check this code"""
        if np.shape(proposal_cov) == (len(theta),):
            # Independent elliptical Gaussian proposals with standard deviations equal to proposal_L_cov
            cov = np.diagonal(proposal_cov[np.ix_(indices)])
            L_cov = np.sqrt(proposal_cov)
        elif np.shape(proposal_cov) == ():
            # Independent spherical Gaussian proposals with standard deviation equal to proposal_L_cov
            L_cov = np.sqrt(proposal_cov)
        elif np.shape(proposal_cov) == (len(theta), len(theta)):
            # Multivariate Gaussian proposals with cholesky factor equal to proposal_L_cov
            # The proposal distribution is a multivariate Gaussian
            # Take the sub-matrix marginal Gaussian for the indices
            mask = np.outer(indices, indices)
            mask = np.array(mask, dtype=bool)
            cov = proposal_cov[np.ix_(mask)]
            (L_cov, _) = cho_factor(cov)
        else:
            raise ValueError("Unsupported dimensions of proposal_L_cov, got {},"
                " expected square matrix or 1D vector or scalar".format(np.shape(proposal_cov)))
        return L_cov

    def _proposal_reparameterised(self, theta, indices, L_cov):
        """independence assumption so take the product of prior pdfs."""
        # No need to update parameters as this is done in the prior
        z = np.random.normal(0, 1, len(theta))
        delta = np.dot(L_cov, z)
        theta = theta + delta
        # Since the variables have been reparametrised to be sampled over the real domain, then there is
        # no change of variables and no jacobian
        log_jacobian_theta = 0.0
        return theta, log_jacobian_theta

    def _proposal(self, theta, indices, L_cov):
        """independence assumption so take the product of prior pdfs."""
        # No need to update parameters as this is done in the prior
        z = np.random.normal(0, 1, len(theta))
        delta = np.dot(L_cov, z)
        theta = theta + delta
        index = 0
        log_jacobian_theta = np.zeros(len(theta))
        # Calculate the jacobian from the theorem of transformation of continuous random variables
        if indices[0]:
            # noise_std is sampled from the domain of log(noise_std) and so the jacobian is
            log_jacobian_theta[index] = -theta[index]  # -ve since jacobian is 1/\sigma
            index += 1
        if indices[1]:
            # gamma_1 is sampled from the domain of gamma_1, so jacobian is unity
            log_jacobian_theta[index] = 0.0
            index += 1
        for j in range(2, self.J):
            if indices[j]:
                # gamma_j is sampled from the domain of log(gamma_j - gamma_j-1) and so the jacobian is
                log_jacobian_theta[index] = -theta[index] # -ve since jacobian is 1/(gamma_j - gamma_j-1)
                index += 1
        if indices[self.J]:
            # scale is sampled from the domain of log(scale) and so the jacobian is
            log_jacobian_theta[index] = -theta[index]
            index += 1
        if indices[self.J + 1]:
            if self.kernel._general and self.kernel._ARD:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and lengthscales parameter for each dimension and class
                raise ValueError("TODO")
            else:
                # In this case, then there is a scale parameter, the first
                # cutpoint, the interval parameters,
                # and a single, shared lengthscale parameter
                # varphi is sampled from the domain of log(varphi) and so the jacobian is
                log_jacobian_theta[index] = -theta[index]
                index += 1
        return theta, log_jacobian_theta

    def _get_log_likelihood_vectorised(self, f):
        """Likelihood of ordinal regression. This is product of scalar normal cdf."""
        # TODO: probably need no upper bound on this for numerical stability purposes.
        calligraphic_Z, *_ = self._calligraphic_Z_vectorised(
                self.gamma, self.noise_std, f)
        return np.sum(np.log(calligraphic_Z), axis=1)  # since calligraphic_Z is (num_samples, N) and sum along N

    def _get_log_likelihood(self, f):
        """Get log p(t|f) log likelihood."""
        calligraphic_Z, *_ = self._calligraphic_Z(
                self.gamma, self.noise_std, f)
        return np.sum(np.log(calligraphic_Z))

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

    # def _weight(
    #     self, f_samp, prior_L_cov, half_log_det_prior_cov, posterior_L_cov, half_log_det_posterior_cov, posterior_mean):
    #     return (
    #         self._get_log_likelihood(f_samp)
    #         + self._log_multivariate_normal_pdf(
    #             f_samp, prior_L_cov, half_log_det_prior_cov)
    #         - self._log_multivariate_normal_pdf(
    #             f_samp, posterior_L_cov, half_log_det_posterior_cov, mean=posterior_mean)
    #     )

    def _weight(
        self, f_samp, prior_cov_inv, half_log_det_prior_cov, posterior_cov_inv, half_log_det_posterior_cov, posterior_mean):
        print(self._get_log_likelihood(f_samp))
        print(self._log_multivariate_normal_pdf(
                f_samp, prior_cov_inv, half_log_det_prior_cov))
        print(- self._log_multivariate_normal_pdf(
                f_samp, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean))
        return (
            self._get_log_likelihood(f_samp)
            + self._log_multivariate_normal_pdf(
                f_samp, prior_cov_inv, half_log_det_prior_cov)
            - self._log_multivariate_normal_pdf(
                f_samp, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean)
        )

    def _weight_vectorised(
        self, f_samps, prior_L_cov, prior_cov_inv, half_log_det_prior_cov, posterior_L_cov, posterior_cov_inv,
        half_log_det_posterior_cov, posterior_mean):
        # print(self._get_log_likelihood_vectorised(f_samps))
        # print(self._log_multivariate_normal_pdf_vectorised(
        #         f_samps, prior_L_cov, half_log_det_prior_cov))
        # print(self._log_multivariate_normal_pdf_vectorised(
        #         f_samps, posterior_L_cov, half_log_det_posterior_cov, mean=posterior_mean))

        print(self._get_log_likelihood_vectorised(f_samps))
        print(self._log_multivariate_normal_pdf_vectorised(
                f_samps, prior_cov_inv, half_log_det_prior_cov))
        print(- self._log_multivariate_normal_pdf_vectorised(
                f_samps, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean))

        log_ws = (self._get_log_likelihood_vectorised(f_samps)
            + self._log_multivariate_normal_pdf_vectorised(
                f_samps, prior_cov_inv, half_log_det_prior_cov)
            - self._log_multivariate_normal_pdf_vectorised(
                f_samps, posterior_cov_inv, half_log_det_posterior_cov, mean=posterior_mean))
        return log_ws

    def _importance_sampler_vectorised(
        self, num_importance_samples, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
        posterior_mean, posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov):
        """
        Sampling from an unbiased estimate of the marginal likelihood p(y|\theta) given the likelihood of the parameters
        p(y | f) and samples
        from an (unbiased) approximating distribution q(f|y, \theta).
        """
        # TODO
        zs = np.random.normal(0, 1, (num_importance_samples, self.N))
        #zs = np.random.normal(0, 1, (self.N, num_importance_samples))
        # f_samps = posterior_L_cov @ zs + posterior_mean
        f_samps = np.einsum('ij, kj -> ki', posterior_L_cov, zs) + posterior_mean
        # for i in range(num_importance_samples):
        #     plt.scatter(self.X_train, f_samps[i])
        # plt.show()

        log_ws = self._weight_vectorised(
            f_samps, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
            posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov, posterior_mean)

        #print(log_ws)

        max_log_ws = np.max(log_ws)
        log_sum_exp = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws)))

        return log_sum_exp - np.log(num_importance_samples)

    def _importance_sampler(
            self, num_importance_samples, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov):
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
            z = np.random.normal(0, 1, self.N)
            f_samp = posterior_mean + np.dot(posterior_L_cov, z)
            #plt.scatter(self.X_train, f_samp)
            log_ws[i] = self._weight(
                f_samp, prior_L_cov, half_log_det_prior_cov, posterior_L_cov,
                half_log_det_posterior_cov, posterior_mean)
        # print(log_ws)
        #plt.show()
        # Normalise the w vectors using the log-sum-exp operator
        max_log_ws = np.max(log_ws)
        log_sum_exp = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws)))
        return log_sum_exp - np.log(num_importance_samples)

    def _transition_operator(
            self, u, log_prior_u, log_jacobian_u, log_marginal_likelihood_u,
            indices, proposal_L_cov, num_importance_samples, reparameterised):
        "Transition operator for the metropolis step."
        # Evaluate priors and proposal conditionals
        if reparameterised:
            v, log_jacobian_v = self._proposal_reparameterised(u, indices, proposal_L_cov)
            log_prior_v = self._prior_reparameterised(v, indices)
        else:
            v, log_jacobian_v = self._proposal(u, indices, proposal_L_cov)
            log_prior_v = self._prior(v, indices)
        fx, gx, posterior_mean, posterior_cov = self.approximate_posterior(
            v, indices, first_step=1, write=False, verbose=False)
        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        # (posterior_L_cov, lower) = cho_factor(posterior_cov + self.jitter * np.eye(self.N))  # Doesn't seem to work for samples
        # half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))

        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        posterior_L_cov = np.linalg.cholesky(posterior_cov + self.jitter * np.eye(self.N))
        half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
        posterior_cov_inv = np.linalg.inv(posterior_cov + self.jitter * np.eye(self.N))
 
        # Since this was never performed in the EP posterior approximation
        # (prior_L_cov, lower) = cho_factor(self.K + self.jitter * np.eye(self.N))
        # prior_L_cov = np.linalg.cholesky(self.K + self.jitter * np.eye(self.N))
        # half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))

        prior_L_cov = np.linalg.cholesky(self.K + self.jitter * np.eye(self.N))
        half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))
        prior_cov_inv = np.linalg.inv(self.K + self.jitter * np.eye(self.N))

        log_marginal_likelihood_v = self._importance_sampler_vectorised(
            num_importance_samples, prior_L_cov, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, half_log_det_posterior_cov)

        print("HELLO")
        print(log_marginal_likelihood_v)

        log_marginal_likelihood_v = self._importance_sampler(
            num_importance_samples, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov)

        print(log_marginal_likelihood_v)
        assert 0
        # Log ratio
        log_a = (
            log_marginal_likelihood_v + np.sum(log_prior_v) + np.sum(log_jacobian_v)
            - log_marginal_likelihood_u - np.sum(log_prior_u) - np.sum(log_jacobian_u))
        log_A = np.minimum(0, log_a)
        threshold = np.random.uniform(low=0.0, high=1.0)
        if log_A > np.log(threshold):
            print(log_A, ">", np.log(threshold), " ACCEPT")
            return (v, log_prior_v, log_jacobian_v, log_marginal_likelihood_v)
        else:
            print(log_A, "<", np.log(threshold), " REJECT")
            return (u, log_prior_u, log_jacobian_u, log_marginal_likelihood_u)

    def _sample_initiate(
            self, theta_0, indices, proposal_cov, num_importance_samples, reparameterised):
        # Get starting point for the Markov Chain
        if theta_0 is None:
            theta_0 = self.get_theta(indices)
        # Get type of proposal density
        proposal_L_cov = self._proposal_initiate(theta_0, indices, proposal_cov)
        _, _, posterior_mean, posterior_cov = self.approximate_posterior(
            theta_0, indices, first_step=1, write=False, verbose=True)

        fx, gx, posterior_mean, posterior_cov = self.approximate_posterior(
            theta_0, indices, first_step=1, write=False, verbose=False)

        # perform cholesky decomposition since this was never performed in the EP posterior approximation
        posterior_L_cov = np.linalg.cholesky(posterior_cov + self.jitter * np.eye(self.N))
        half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
        posterior_cov_inv = np.linalg.inv(posterior_cov + self.jitter * np.eye(self.N))

        # (posterior_L_cov, lower) = cho_factor(posterior_cov + self.jitter * np.eye(self.N))
        # half_log_det_posterior_cov = np.sum(np.log(np.diag(posterior_L_cov)))
        # posterior_L_covT_inv = solve_triangular(
        #     posterior_L_cov.T, np.eye(self.N), lower=True)
        # posterior_cov_inv = solve_triangular(posterior_L_cov, posterior_L_covT_inv, lower=False)

        prior_L_cov = np.linalg.cholesky(self.K + self.jitter * np.eye(self.N))
        half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))
        prior_cov_inv = np.linalg.inv(self.K + self.jitter * np.eye(self.N))

        # Since this was never performed in the EP posterior approximation
        # (prior_L_cov, lower) = cho_factor(self.K + self.jitter * np.eye(self.N))
        # prior_L_covT_inv = solve_triangular(
        #     prior_L_cov.T, np.eye(self.N), lower=True)
        # prior_cov_inv = solve_triangular(prior_L_cov, prior_L_covT_inv, lower=False)
        # half_log_det_prior_cov = np.sum(np.log(np.diag(prior_L_cov)))

        log_marginal_likelihood_theta = self._importance_sampler_vectorised(
            num_importance_samples, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov
        )

        print(log_marginal_likelihood_theta)

        log_marginal_likelihood_theta = self._importance_sampler(
            num_importance_samples, prior_L_cov, prior_cov_inv, half_log_det_prior_cov,
            posterior_mean, posterior_L_cov, posterior_cov_inv, half_log_det_posterior_cov)

        print(log_marginal_likelihood_theta)
        assert 0
        # Evaluate priors and proposal conditionals
        if reparameterised:
            theta_0, log_jacobian_theta = self._proposal_reparameterised(theta_0, indices, proposal_L_cov)
            log_prior_theta = self._prior_reparameterised(theta_0, indices)
        else:
            theta_0, log_jacobian_theta = self._proposal(theta_0, indices, proposal_L_cov)
            log_prior_theta = self._prior(theta_0, indices)
        theta_samples = []
        return (
            theta_0, theta_samples, log_prior_theta,
            log_jacobian_theta, log_marginal_likelihood_theta, proposal_L_cov)

    def sample(
            self, indices, proposal_cov, steps, first_step=1, num_importance_samples=100, theta_0=None,
            reparameterised=True):
        (theta, theta_samples, log_prior_theta,
        log_jacobian_theta, log_marginal_likelihood_theta, proposal_L_cov) = self._sample_initiate(
            theta_0, indices, proposal_cov, num_importance_samples, reparameterised)
        for _ in trange(
            first_step, first_step + steps, desc="Psuedo-marginal Sampler Progress", unit="samples"):
            theta, log_prior_theta, log_jacobian_theta, log_marginal_likelihood_theta = self._transition_operator(
                theta, log_prior_theta, log_jacobian_theta, log_marginal_likelihood_theta, indices, proposal_L_cov,
                num_importance_samples, reparameterised)
            theta_samples.append(theta)
        return np.array(theta_samples)


class EllipticalSliceOrdinal(Sampler):
    """
    Elliptical Slice sampling
    """

    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)

    def _sample_initiate(self, parameter):
        """Initialise variables for the sample method."""
        parameter_samples = []
        return parameter, parameter_samples

    def ELLSS_transition_operator(self, L_K, N, y, f, log_likelihood, classifier):
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

    def _elliptical(self, xx, chol_Sigma, log_like_fn, cur_log_like, angle_range):
        """
        Elliptical slice Gaussian prior posterior update attributed to Iain Murray, September 2009.
`
        -------------------------------------------------------------------------------
        The standard MIT License for gppu_elliptical.m and other code in this
        distribution that was written by Iain Murray and/or Ryan P. Adams.
        http://www.opensource.org/licenses/mit-license.php
        -------------------------------------------------------------------------------
        Copyright (c) 2010 Iain Murray, Ryan P. Adams

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to
        deal in the Software without restriction, including without limitation the
        rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
        IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
        FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
        LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
        FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
        IN THE SOFTWARE.
        -------------------------------------------------------------------------------

        Here is the original docstring from the original matlab implementation by Iain Murray:
        % GPPU_ELLIPTICAL Gaussian prior posterior update - slice sample on random ellipses
        %
        % [xx, cur_log_like] = gppu_elliptical(xx, chol_Sigma, log_like_fn[, cur_log_like])
        %
        % A Dx1 vector xx with prior N(0, Sigma) is updated leaving the posterior
        % distribution invariant.
        %
        % Inputs:
        % xx Dx1 initial vector (can be any array with D elements)
        % chol_Sigma DxD chol(Sigma).Sigma is the prior covariance of xx
        % log_like_fn @ fn log_like_fn(xx) returns 1x1 log likelihood
        % cur_log_like 1x1 Optional: log_like_fn(xx) of initial vector.
        % You can omit this argument or pass [].
        % angle_range 1x1 Default 0: explore whole ellipse with break point at
        % first rejection.Set in (0, 2 * pi] to explore a bracket of
        % the specified width centred uniformly at randomly.
        %
        % Outputs:
        % xx Dx1(size matches input) perturbed vector
        % cur_log_like 1x1 log_like_fn(xx) of final vector
        %
        % See also: GPPU_UNDERRELAX, GPPU_LINESLICE
        %
        % Iain Murray, September 2009
        """
        N = len(xx)


