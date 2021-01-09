from abc import ABC, abstractmethod
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
from scipy.stats import norm, uniform, multivariate_normal, expon
from tqdm import trange
from .utilities import (
    sample_Us, sample_U, function_u3, function_u2, sample_varphi, samples_varphi,
    matrix_of_differences, matrix_of_differencess)


class Estimator(ABC):
    """
    Base class for variational Bayes estimators. This class allows users to define a classification problem,
    get predictions using a approximate Bayesian inference.

    All estimators must define an init method, which may or may not inherit Sampler as a parent class using
        `super()`.
    All estimators that inherit Estimator define a number of methods that return the posterior estimates.
    All estimators must define a estimate method that can be used to estimate (converge to) the posterior.
    All estimators must define a _estimate_initiate method that is used to initiate estimate.
    All estimators must define an predict method can be  used to make predictions given test data.
    """

    @abstractmethod
    def __init__(self, X_train, t_train, kernel, sigma, tau, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :param t_train: (N, ) The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
        :arg sigma: The (K, ) (location/ scale) hyper-hyper-parameters that define psi prior.
            Not to be confused with `Sigma`, which is a covariance matrix.
        :arg tau: The (K, ) (location/ scale) hyper-hyper-parameters that define psi prior.
        :arg str write_path: Write path for outputs.

        :returns: A :class:`Estimator` object
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
        self.X_train = X_train
        self.X_train_T = X_train.T
        self.IN = np.eye(self.N)
        self.mean_0 = np.zeros(self.N)
        self.sigma = sigma
        self.tau = tau
        self.grid = np.ogrid[0, self.N]
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
    def _estimate_initiate(self):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """

    @abstractmethod
    def estimate(self):
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


class VariationBayesMultinomialGP(Estimator):
    """
    A Variational Bayes classifier. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.

    Multinomial Probit regression using Gibbs sampling with GP priors. 
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.I = np.eye(self.K)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(self.IN + self.C)
        self.cov = self.C @ self.Sigma

    def _ws_general_unnormalised(self, psi_tilde, n_samples, M_tilde):
        """
        Return the w values of the sample.

        2005 Page _ Eq.(_)
        TODO
        :arg psi_tilde: Posterior mean estimate of psi
        :arg M_tilde: ?
        """
        # Draw from varphi
        varphi_samples = samples_varphi(psi_tilde)
        for varphi_sample in varphi_samples:
            self.kernel.varphi = varphi_sample
            Cs = self.kernel.kernel_matrix(self.X)
            M_tilde = m_tilde[K]
            for i, C in enumerate(Cs):
                normal_pdf = multivariate_normal.pdf(M_tilde[i], mean=self.mean_0, cov=C)
        return normal_pdf

    def _phi_tilde(self, varphi):
        # TODO: Deal with varphi_k.
        return np.divide(np.add(self.sigma, 1), np.add(self.tau, varphi))

    def _varphi_tilde(self, varphi_samples, n_samples=1000):
        """
        Return the posterior mean estimate of varphi via importance sampling.

        2005 Page 9 Eq.(9)
        """
        sum = 0
        n_samples = np.shape(varphi_samples)[0]
        for i in range(n_samples):
            varphi_sample = varphi_samples[i]
            sum += varphi_sample * w(varphi_sample)
        return sum

    def _M_tilde(self, Y_tilde, varphi_tilde):
        """
        Return the posterior mean estimate of M.

        2005 Page 9 Eq.(8)

        :arg Y_tilde: (N, K) array
        :type Y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type Y_tilde: :class:`np.ndarray`
        """
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # calculate updated C and sigma
        # TODO: Do I want to update these in the class scope?
        C_tilde = self.kernel.kernel_matrix(self.X_train)  # (N, N)
        sigma_tilde = np.linalg.inv(self.IN + C_tilde)  # (N, N)
        return C_tilde @ sigma_tilde @ Y_tilde  # (N, K)

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements as defined on page 8 of the paper.

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        # The max of the GP vector m_k is t_n
        t = np.argmax(M_tilde, axis=1)
        # TODO: There is no need to do these sequentially. Do them at the same time.
        numerator_expectation = self._vector_expectation_p_m(function_u2, M_tilde, t, n_samples=1000)
        denominator_expectation = self._vector_expectation_p_m(function_u3, M_tilde, t, n_samples=1000)
        # Eq.(5)
        Y_tilde = M_tilde - np.divide(numerator_expectation, denominator_expectation)
        # Eq.(6)
        # This part of the differences sum must be 0, since sum is over j \neq i
        Y_tilde[self.grid, t] = M_tilde[self.grid, t]
        diff_sum = np.sum(Y_tilde - M_tilde, axis=1)
        Y_tilde[self.grid, t] = M_tilde[self.grid, t] - diff_sum
        return Y_tilde

    def _scalar_Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements as defined on page 8 of the paper.

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
            2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        Y_tilde = -1. * np.ones((self.N, self.K))
        # Not vectorised.
        for i in range(self.N):
            m_tilde_n = M_tilde[i, :]
            t_n = np.argmax(m_tilde_n)
            expectation_3 = self._expectation_p_m(function_u3, m_tilde_n, t_n, n_samples=1000)
            expectation_2 = self._expectation_p_m(function_u2, m_tilde_n, t_n, n_samples=1000)
            # Equation 5
            y_tilde_n = m_tilde_n - np.divide(expectation_2, expectation_3)
            # Equation 6 follows
            # This part of the differences sum must be 0, since sum is over j \neq i
            y_tilde_n[t_n] = m_tilde_n[t_n]
            diff_sum = np.sum(y_tilde_n - m_tilde_n)
            y_tilde_n[t_n] = m_tilde_n[t_n] - diff_sum
            Y_tilde[i, :] = y_tilde_n
        return Y_tilde

    def _vector_expectation_p_m_1_alt(self, M, n_samples=1000):
        """
        Calculate the Monte Carlo estimate of the expectation of a function of
            the M over the distribution p.

        numerator of 2005 Page 8 Eq.(5)

        :arg vector_function: is a function that outputs a vector value, (e.g. function_u2 which is the
            numerator of Eq.(5);
            or function_u3 which is the denominator of Eq.(5)) for options see probit.utilities.
        :arg M: An (N, K) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :arg n_samples: The number of samples to take.
        """
        t = np.argmax(M, axis=1)
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(M, self.K, self.N)  # (N, K, K) we will product across axis 2 (rows)
        vector_differences = differences[self.grid, :, t]  # (N, K)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume its okay to use the same random variables over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # sum is over j \neq k
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # sum is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0  # TODO: Test it.
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (n_samples, N, K)
        samples = np.exp(log_samples)
        ## TODO: from last time, make sure the dimensions of normal_pdfs and
        ## samples match for the elementwise multiplication
        # TODO tomorow: Tidy up function evaluation code, move anything that is reused into utilities.
        # Code for the non general case if possibly, although I think it is simpler than the Gibbs example.
        # Try on ordered data.

        # Take the sample U as a vector (N, K)
        # TODO: make sure that we are making the us constant that need to be. and taking the correct Us
        # TODO: from below equation 6 etc. \phi
        us = Us[:, :, 0]  # (N, K)
        assert us[0, 1] == us[0, 2]
        assert us[0, 0] == us[0, 1]
        normal_pdfs = norm.pdf(us - vector_differences, loc=0, scale=1)  # (N, K)
        # Find the elementwise product of these two vectors which returns a (N, K) array
        return np.multiply(normal_pdfs, )
        # axis 1 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=1)

    def _vector_expectation_p_m_2(self, M, n_samples=1000):
        """
        Calculate the Monte Carlo estimate of the expectation of a function of
            the M over the distribution p.

        This is the numerator of the rightmost term of 2005 Page 8 equation (5).

        :arg M: An (N, K) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :arg n_samples: The number of samples to take.
        """
        function_eval = function_u1_alt(difference, U, t_n)
        # Take the sample U as a vector (K, )
        u = U[:, 0]
        normal_pdf = norm.pdf(u - vector_difference, loc=0, scale=1)
        # Find the elementwise product of these two vectors which returns a (K, ) array
        return np.multiply(normal_pdf, function_eval)

    def _vector_expectation_p_m_3(self, M, n_samples=1000):
        """
        Calculate the Monte Carlo estimate of the expectation of a function of
            the M over the distribution p.

        denominator of 2005 Page 8 Eq.(5)

        :arg vector_function: is a function that outputs a vector value, (e.g. function_u2 which is the
            numerator of Eq.(5);
            or function_u3 which is the denominator of Eq.(5)) for options see probit.utilities.
        :arg M: An (N, K) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :arg n_samples: The number of samples to take.
        """
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(M, self.K, self.N)  # (N, K, K) we will product across axis 2 (rows)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume its okay to use the same random variables over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # sum is over j \neq k
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        t = np.argmax(M, axis=1)
        # sum is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0  # TODO: Test it.
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)
        samples = np.exp(log_samples)
        # axis 1 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=1)

    def _expectation_p_m(self, function, m_n, n_samples=1000):
        """
        Calculate the Monte Carlo estimate of the expectation of a function of
            the m_n, over the distribution p.

        e.g. 2005 Page 8 Eq.(5), 2005 Page _ Eq.(_)

        :arg function:i is a function, (e.g. function_u1(difference, U) which is the numerator of Eq.(5); or
            function_u2(difference, U) which is the denominator of Eq.(5)) for options see probit.utilities.
        :arg m_n: An (K, ) array filled with m_k^{new, s} where s is the sample, and k is the class indicator.
        :arg n_samples: The number of samples to take.
        """
        # Factored out calculations
        difference = matrix_of_differences(m_n)
        t_n = np.argmax(m_n)
        vector_difference = difference[:, t_n]
        K = len(m_n)
        # Take samples
        samples = []
        for i in range(n_samples):
            U = sample_U(K, different_across_classes=1)
            function_eval = function(difference, vector_difference, U, t_n, K)
            samples.append(function_eval)

        distribution_over_classes = 1 / n_samples * np.sum(samples, axis=0)
        return(distribution_over_classes)

    def expn_u_M_tilde(self, M_tilde, n_samples):
        """
        Return a sample estimate of a function of the r.v. u ~ N(0, 1)

        2005 Page _ Eq.(_)

        :param M_tilde: M_tilde,
        """

        def function(u, M_tilde):
            norm.pdf(u, loc=M)

        sum = 0
        for i in range(n_samples):
            sum += function(u, M_tilde)
        return sum / n_samples

    def M(self, sigma, Y_tilde):
        """
        Q(M) where Y_tilde is the expectation of Y with respect to the posterior
        component over Y.

        2005 Page _ Eq.(_)
        """
        M_tilde = sigma @ Y_tilde
        # The approximate posterior component Q(M) is a product of normal distributions over the classes
        # But how do I translate this function into code? Must need to take an expectation wrt to it
        return M_tilde

    def expn_M(self, sigma, m_tilde, n_samples):
        # Draw samples from the random variable, M ~ Q(M)
        # Use M to calculate f(M)
        # Take monte carlo estimate
        K = np.shape(m_tilde)[0]
        return None
