from abc import ABC, abstractmethod
from .kernels import Kernel
import pathlib
import numpy as np
from scipy.stats import norm, multivariate_normal
from tqdm import trange
from .utilities import sample_U
# from .utilities import log_heaviside_probit_likelihood


class Sampler(ABC):
    """
    Base class for samplers. This class allows users to define a classification problem, get predictions
    using a exact Bayesian inference.

    All samplers must define an init method, which may or may not inherit Sampler as a parent class using `super()`.
    All samplers that inherit Sampler define a number of methods that return the samples.
    All samplers must define a _sample_initiate method that is used to initate the sampler.
    All samplers must define an predict method can be  used to make predictions given test data.
    """

    @abstractmethod
    def __init__(self, X_train, t_train, kernel, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg X: The data vector.
        :type X: :class:`numpy.ndarray`
        :param t: The target vector.
        :type t: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
        :arg str write_path: Write path for outputs.
        ""

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
    Multinomial Probit regression using Gibbs sampling with GP priors. Inherits the sampler ABC
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.I = np.eye(self.K)  # Q: this eye is different to np.eye(K). Why use of both? Is this even used?
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.sigma = np.linalg.inv(np.eye(self.N) + self.C)
        self.cov = self.C @ self.sigma

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

        :param M_0: (N, K) numpy.ndarray of the initial location of the sampler.
        :type M_0: :class:`np.ndarray`.
        :arg :class:`numpy.ndarray` init: The initial location of the sampler in parameter space.
        :arg :class:`numpy.ndarray` steps: The number of steps in the sampler.
        :param varphi: (K, D) np.array for TODO: need this?

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
                y_n[k_true] = -1.0  # TODO: this is a bit hacky
                # Sample from the cone of the truncated multivariate Gaussian
                # TODO: find out a way to compare yi[k_true] to
                #  all the other values apart from itself, as that is the theory
                while y_n[k_true] < np.max(y_n):
                    # sample Y jointly
                    y_n = multivariate_normal.rvs(mean=m, cov=self.I)
                # Add sample to the Y vector
                Y[n, :] = y_n[:]
            # Calculate statistics, then sample other conditional
            # Empty M_T (K, N) matrix to collect m_k samples over
            M_T = -1. * np.ones((self.K, self.N))
            for k in range(self.K):
                mean = self.cov @ Y.T[k]
                m_k = multivariate_normal.rvs(mean=mean, cov=self.cov)
                # Add sample to the M vector
                M_T[k, :] = m_k
            M = M_T.T
            M_samples.append(M)
            Y_samples.append(Y)
        return np.array(M_samples), np.array(Y_samples)

    def expectation_wrt_u(self, m, n_samples):
        """
        Calculate distribution over classes.

        :arg m: is an (K, 1) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :type m: :class:`numpy.ndarray`
        """
        # Find matrix of coefficients
        I = np.eye(self.K)
        Lambda = np.tile(m, (self.K, 1))
        Lambda_T = Lambda.T
        # Symmetric matrix of differences
        differences = Lambda - Lambda_T
        # Take samples
        samples = []
        for i in range(n_samples):
            # Sample normal random variate
            u = norm.rvs(0, 1, self.K)
            U = np.tile(u, (self.K, 1))
            # TODO: Alternatively?
            # u = norm.rvs(0, 1, 1)
            # U = np.multiply(u, np.ones((K, K)))
            ones = np.ones((self.K, self.K))
            # TODO: consider doing a log-transform to do stable multiplication
            random_variables = np.add(U, differences)
            np.fill_diagonal(random_variables, 1)
            sample = np.prod(random_variables, axis=0)
            samples.append(sample)

        # TODO: check that this sum is correct
        distribution_over_classes = 1 / n_samples * np.sum(samples, axis=0)
        return distribution_over_classes

    def multidimensional_expectation_wrt_u(self, ms, n_samples):
        """
        Calculate distribution over classes for multiple m at the same time.

        :arg ms: An (N_test, K) array filled with m_k^{new_i, s} where s is the sample, k is the class indicator
        and i is the index of the test object.
        :type ms: :class:`numpy.ndarray`
        :arg int nsamples: Number of samples to take in the monte carlo estimate.

        :returns: Distribution over classes
        """
        # Find matrix of coefficients
        K = np.shape(ms)[1]
        I = np.eye(K)
        N_test = np.shape(ms)[0]
        ms = ms.reshape((N_test, 1, K))
        print(np.shape(ms), 'shape_ms')
        # Lambdas is an (n_test, K, K) stack of Lambda matrices
        Lambdas = np.tile(ms, (1, K, 1))
        print(np.shape(Lambdas), 'shape Lambdas')
        Lambdas_T = Lambdas.transpose((0, 2, 1))
        print(np.shape(Lambdas_T), 'shape LambdasT')
        # Symmetric matrices of differences
        differences = np.subtract(Lambdas_T, Lambdas)
        # Take samples
        sampless = []
        for i in range(n_samples):
            # U is (K, K)
            U = sample_U(K)
            # assume its okay to use the same random variable over all of these data points.
            # Us is (N_test, K, K)
            Us = np.tile(U, (N_test, 1, 1))
            random_variables = np.add(Us, differences)
            cum_dists = norm.cdf(random_variables, loc=0, scale=1)
            log_cum_dists = np.log(cum_dists)
            # Employ a for loop here as I couldn't work out the vectorised way
            # the operation isn't so expensive.
            for j in range(N_test):
                np.fill_diagonal(log_cum_dists[j], 0)
            # Sum across the elements of interest, which is the inner most rows (axis=2)
            # log samples will be (N_test, K) array of a (log) distribution sample for each test point
            log_samples = np.sum(log_cum_dists, axis=2)
            samples = np.exp(log_samples)
            sampless.append(samples)
        # Calculate the MC estimate, should be a (N_test, K) array, so sum is across the samples (axis=0)
        distributions_over_classes = 1 / n_samples * np.sum(sampless, axis=0)
        assert (np.shape(distributions_over_classes) == (N_test, K))
        assert 0 # TODO: this code is not tested.
        return distributions_over_classes

    def predict_vector(self, Y_samples, X_test):
        """
        Calculate the Gibbs prediction over classes given a vector of new data points, X_test.

        :arg Y_samples:
        :arg X_test:
        :return: A monte carlo estimate of the class probabilities.
        """
        X_new = np.append(X_test, self.X_train, axis=0)
        N_test = np.shape(X_test)[0]
        # Cs_new = simple_kernel_matrix(X_train, X_test, varphi, s) TODO: figure out wtf is going on here
        Cs_new = self.kernel.kernel_matrix(self.X_train, X_test)
        # cs_new = np.diagonal(simple_kernel_matrix(X_test, X_test, varphi, s))
        cs_new = np.diagonal(self.kernel.kernel_matrix(X_test, X_test))
        intermediate_matrix = self.sigma @ Cs_new  #  (N_train, N_test)
        # Assertion that Cs_new is (N, N_test)
        assert (np.shape(Cs_new) == (np.shape(self.X_train)[0], np.shape(X_test)[0]))
        K = np.shape(Y_samples[0])[1]
        n_posterior_samples = np.shape(Y_samples)[0]
        # Sample pmf over classes
        distribution_samples = []
        # For each sample
        for Y in Y_samples:
            # Initiate m with null values
            ms = -1. * np.ones((N_test, K))
            for k, y_k in enumerate(Y.T):
                # (1, N_test)
                # TODO: this could be expensive.
                mean_k = y_k.T @ intermediate_matrix
                for i in range(N_test):
                    var_ki = cs_new[i] - np.dot(Cs_new.T[i], intermediate_matrix.T[i])
                    ms[i, k] = norm.rvs(loc=mean_k[i], scale=var_ki)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            distribution_samples.append(self.multidimensional_expectation_wrt_u(ms, n_samples=1000))
        monte_carlo_estimates = (1. / n_posterior_samples) * np.sum(distribution_samples, axis=0)
        # TODO: Could also get a variance from the MC estimate
        return monte_carlo_estimates

    def predict(self, Y_samples, x_test, n_samples=1000):
        """
        Make gibbs prediction class of x_new given the posterior samples.

        :param Y_samples: The Gibbs samples of the latent variable Y.
        :param x_test: The new data point.
        :param n_samples: The number of samples in the monte carlo estimate.
        :return:
        """
        x_test = np.array([x_test])
        #X_new = np.append(self.X_train, x_test, axis=0)
        Cs_new = self.kernel.kernel_vector_matrix(x_test, self.X_train)
        #Cs_new = simple_kernel_vector_matrix(x_new, X, varphi, s)
        n_posterior_samples = np.shape(Y_samples)[0]
        # Sample pmf over classes
        distribution_over_classes_samples = []
        # For each sample
        for Y in Y_samples:
            m = -1. * np.ones(self.K)  # Initiate m with null values
            for k, y_k in enumerate(Y.T):
                mean_k = y_k.T @ self.sigma @ Cs_new
                var_k = self.kernel.kernel(x_test[0], x_test[0]) - Cs_new.T @ self.sigma @ Cs_new
                m[k] = norm.rvs(loc=mean_k, scale=var_k)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            distribution_over_classes_samples.append(self.expectation_wrt_u(m, n_samples))
        monte_carlo_estimate = (1. / n_posterior_samples) * np.sum(distribution_over_classes_samples, axis=0)
        # TODO: Could also get a variance from the MC estimate
        return monte_carlo_estimate


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
            raise ValueError("In the binomial case, t must contain only 1s and/or 0s (got {})".format(t))
        self.K = int(np.max(t_train) + 1)  # the number of classes
        self.t_train = t_train
        self.cov = np.linalg.inv(self.X_train_T @ self.X_train)  # From Mark's lecture notes

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
        :return: Array of n samples and acceptance rate.
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
            beta = multivariate_normal.rvs(mean=mean, cov=self.cov)
            beta_samples.append(beta)
            Y_samples.append(Y)
        beta_samples = np.array(beta_samples, dtype=np.float64)
        Y_samples = np.array(Y_samples, dtype=np.float64)
        return beta_samples, Y_samples

    def predict(self, beta_samples, x_test):
        """Make gibbs prediction class of x_new given the beta posterior samples."""
        f = [norm.cdf(np.dot(beta, x_test)) for beta in beta_samples]
        return sum(f) / len(beta_samples)


class InvalidKernel(Exception):
    """An invalid kernel has been passed to `Sampler`"""

    def __init__(self, kernel):
        """
        Construct the exception.

        :arg kernel: The object pass to :class:`Sampler` as the kernel
            argument.
        :rtype: :class:`InvalidKernel`
        """
        message = (
            f"{kernel} is not an instance of"
            "probit.kernels.Kernel"
        )

        super().__init__(message)
