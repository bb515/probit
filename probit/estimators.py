from abc import ABC, abstractmethod
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
from scipy.stats import norm, multivariate_normal
from tqdm import trange
from .utilities import (
    sample_Us, sample_varphis, matrix_of_differencess)


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
    def __init__(self, X_train, t_train, kernel, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :param t_train: (N, ) The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
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
        self.mean_0 = np.zeros(self.N)
        if self.kernel.general_kernel:
            sigma = np.reshape(self.kernel.sigma, (self.K, 1))
            tau = np.reshape(self.kernel.tau, (self.K, 1))

            self.sigma = np.tile(sigma, (1, self.D))  # (K, D)
            self.tau = np.tile(tau, (1, self.D))  # (K, D)
        else:
            self.sigma = self.kernel.sigma
            self.tau = self.kernel.tau
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
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(self.IN + self.C)
        self.cov = self.C @ self.Sigma

    def _estimate_initiate(self):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """

    def estimate(self, M_0, steps, first_step=1):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
        """
        M_tilde, varphi_tilde, psi_tilde = self._estimate_initiate(M_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            Y_tilde = self._Y_tilde(M_tilde)
            M_tilde, Sigma_tilde, C_tilde = self._M_tilde(Y_tilde, varphi_tilde)
            varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde, n_samples=500)
            psi_tilde = self._psi_tilde(varphi_tilde)
        return M_tilde, Sigma_tilde, C_tilde, Y_tilde, varphi_tilde

    def _predict_vector_generalised(self, Sigma_tilde, C_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.

        This is the general case where there are hyperparameters varphi (K, D)
            for all dimensions and classes.

        :param Sigma_tilde:
        :param C_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        # X_new = np.append(X_test, self.X_train, axis=0)
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi  # TODO: test this.
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (K, N, N_test)
        # TODO: this is a bottleneck
        cs_news = [np.diag(self.kernel.kernel_matrix(X_test, X_test)[k]) for k in range(self.K)]  # (K, N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (K, N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (K, N_test, N)

        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (K, N_test, )

        # calculate M_tilde_new
        Y_tilde_T = np.transpose(Y_tilde)  # (K, N)
        # TODO: might be able to vectorise without the transpose
        M_tilde_new = np.empty((N_test, self.K))
        sigma_squared_tilde_new = np.empty((N_test, self.K))
        for k, y_k in enumerate(Y_tilde_T):
            M_tilde_new[:, k] = intermediate_vectors_T[k] @ y_k
            sigma_squared_tilde_new[:, k] = cs_news[k] - intermediate_scalars[k]
        # Take an expectation wrt to the rv u, use n_samples=1000 draws from p(u)
        return self._vector_expectation_wrt_u(M_tilde_new, sigma_squared_tilde_new, n_samples)
        M_tilde_new = Y_tilde_T @ intermediate_vectors  # (K, N) * (K, N, N_test)  # not sure correct.
        M_tilde_new = intermediate_vectors_T @ Y_tilde_T  #  (K, N_test, N) * (N, K) = (K, N_test, K)
        # Sample pmf over classes
        distribution_over_classes_sampless = []

        Y_T = np.reshape(Y.T, (self.K, self.N, 1))
        M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_T)
        M_new_tilde_T = np.reshape(M_new_tilde_T, (self.K, N_test))
        var_new_tilde = np.subtract(cs_news, intermediate_scalars)
        M = norm.rvs(loc=M_new_tilde_T.T, scale=var_new_tilde.T)


        for Y in Y_samples:
            # Initiate m with null values
            ms = np.empty((N_test, self.K))
            for k, y_k in enumerate(Y.T):
                mean_k = intermediate_vectors_T[k] @ y_k  # (N_test, )
                var_k = cs_news[k] - intermediate_scalars[k]  # (N_test, )
                ms[:, k] = norm.rvs(loc=mean_k, scale=var_k)
            # Take an expectation wrt the rv u, use n_samples=1000 draws from p(u)
            # TODO: How do we know that 1000 samples is enough to converge?
            #  Goes with root n_samples but depends on the estimator variance
            distribution_over_classes_sampless.append(self._vector_expectation_wrt_u(ms, n_samples))
        # TODO: Could also get a variance from the MC estimate.
        return (1. / n_posterior_samples) * np.sum(distribution_over_classes_sampless, axis=0)

    def _predict_vector(self, Sigma_tilde, C_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.

        :param Sigma_tilde:
        :param C_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        # X_new = np.append(X_test, self.X_train, axis=0)
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi  # TODO: test this.
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N_train, N_test)
        # TODO: this is a bottleneck
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = self.Sigma @ Cs_news  # (N_train, N_test)
        intermediate_vectors_T = intermediate_vectors.T
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        # Sample pmf over classes
        distribution_over_classes_sampless = []

        # calculate M_tilde_new
        Y_tilde_T = np.transpose(Y_tilde)  # (K, N)

        M_tilde_new = Y_tilde_T @ Sigma_tilde @ Cs_news


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

    def predict(self):
        """
        Return the samples

        This method should be implemented in every concrete sampler.
        """

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=500):
        """
        Return the w values of the sample on 2005 Page 9 Eq.(7).

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from varphi
        varphis = sample_varphis(psi_tilde, n_samples, general=self.kernel.general_kernel)  # (n_samples, K, D) in general, (n_samples, ) for ISO case. Depend on the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X, varphis)  # (n_samples, K, N, N) in general, (n_samples, N, N) for ISO case. Depend on the shape of psi_tilde.
        # Transpose M_tilde to become (n_samples, K, N) multivariate normal between m_k (N, ) and Cs_k over all samples
        M_tilde_T = np.transpose(M_tilde)  # (K, N)
        # TODO: begrudgingly using a nested for loop here, as I don't see the alternative,
        #  maybe I can calculate this by hand.
        ws = np.empty((n_samples, self.K))
        if self.kernel.general_kernel:
            for i in range(n_samples):
                    for k in range(self.K):
                        w_i_k = multivariate_normal.rvs(M_tilde_T[k], cov=Cs_samples[i, k])
                        # Add sample to the unnormalised w vectors
                        ws[i, k] = w_i_k
            # Normalise the w vectors
            denominator = np.sum(ws, axis=0)  # (K, )
            ws = np.divide(ws, denominator)  # (n_samples, K)
            ws = np.reshape(ws, (n_samples, self.K, 1))  # (n_samples, K, 1)
            ws = np.tile(ws, (1, 1, self.D))  # (n_samples, K, D)
        else:
            for i in range(n_samples):
                for k in range(self.K):
                    w_i_k = multivariate_normal.rvs(M_tilde_T[k], cov=Cs_samples[i])
                    # Add sample to the unnormalised w vectors
                    ws[i, k, :] = w_i_k
            # Normalise the w vectors
            denominator = np.sum(ws, axis=0)  # (K, )
            ws = np.divide(ws, denominator)  # (n_samples, K)
            # Since all length scale parameters are the same, these should be close within some tolerance
            # TODO: set the tolerance.
            assert np.allclose(ws, ws[0])
            # Take only one axis, since there is only one varphi
            ws = ws[:, 0]  # (n_samples, )
        element_prod = np.multiply(ws, varphis)
        return np.sum(element_prod, axis=0)

    def _psi_tilde(self, varphi_tilde):
        if self.kernel.general_kernel:
            sigma = np.reshape(self.sigma, (self.K, 1))
            tau = np.reshape(self.tau, (self.K, 1))

            sigma = np.tile(self.sigma, (1, self.D))  # (K, D)
            tau = np.tile(self.tau, (1, self.D))  # (K, D)

        else:
            return np.divide(np.add(1, self.sigma), np.add(self.tau, varphi_tilde))

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
        # Calculate updated C and sigma
        # TODO: Do I want to update these in the class scope?
        C_tilde = self.kernel.kernel_matrix(self.X_train)  # (N, N)
        Sigma_tilde = np.linalg.inv(self.IN + C_tilde)  # (N, N)
        return C_tilde @ Sigma_tilde @ Y_tilde, Sigma_tilde, C_tilde  # (N, K)

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements 2005 Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        # The max of the GP vector m_k is t_n
        t = np.argmax(M_tilde, axis=1)
        negative_P = self._negative_P(M_tilde, t, n_samples=1000)
        # Eq.(5)
        Y_tilde = np.subtract(M_tilde, negative_P)
        # Eq.(6)
        # This part of the differences sum must be 0, since sum is over j \neq i
        Y_tilde[self.grid, t] = M_tilde[self.grid, t]
        diff_sum = np.sum(Y_tilde - M_tilde, axis=1)
        Y_tilde[self.grid, t] = M_tilde[self.grid, t] - diff_sum
        return Y_tilde

    def _negative_P(self, M_tilde, t, n_samples=1000):
        """
        Estimate the rightmost term of 2005 Page 8 Eq.(5), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg n_samples: The number of samples to take.
        """
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(M_tilde, self.K, self.N)  # (N, K, K) we will product across axis 2 (rows)
        vector_differences = differences[self.grid, :, t]  # (N, K)
        vector_differencess = np.tile(vector_differences, (n_samples, 1, 1))  # (n_samples, N, K)
        vector_differencess = np.moveaxis(vector_differencess, 1, 0)  # (N, n_samples, K)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume it's okay to use the same samples of U over all of the data points
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
        # Take the samples us as matrices (n_samples, N, K)
        us = Us[:, :, 0]  # (n_samples, K)
        assert us[0, 1] == us[0, 2]
        assert us[0, 0] == us[0, 1]
        # Assume it's okay to use the same samples of U over all of the data points
        diff = np.add(us, vector_differences)
        normal_pdfs = norm.pdf(diff, loc=0, scale=1)  # (n_samples, N, K)
        normal_cdfs = norm.cdf(diff, loc=0, scale=1)  # (n_samples, N, K)
        # Find the elementwise product of these two samples of matrices
        element_prod_pdf = np.multiply(normal_pdfs, samples)
        element_prod_cdf = np.multiple(normal_cdfs, samples)
        # axis 0 is the n_samples samples, which is the monte-carlo sum of interest
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=0)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=0)
        return np.divde(element_prod_pdf, element_prod_cdf)  # (N, K)
