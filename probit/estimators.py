from abc import ABC, abstractmethod
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
from scipy.stats import norm, multivariate_normal
from tqdm import trange
from .utilities import (
    sample_Us, sample_varphis, matrix_of_differencess, matrix_of_valuess)


class Estimator(ABC):
    """
    Base class for variational Bayes estimators. This class allows users to define a classification problem,
    get predictions using an approximate Bayesian inference.

    All estimators must define an init method, which may or may not inherit Sampler as a parent class using
        `super()`.
    All estimators that inherit Estimator define a number of methods that return the posterior estimates.
    All estimators must define a :meth:`estimate` that can be used to estimate (converge to) the posterior.
    All estimators must define a :meth:`_estimate_initiate` that is used to initiate estimate.
    All estimators must define a :meth:`predict` can be used to make predictions given test data.
    """

    @abstractmethod
    def __init__(self, X_train, t_train, kernel, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete sampler.

        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :arg t_train: (N, ) The target vector.
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
        self.grid = np.ogrid[0:self.N]
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        if np.all(t_train >= 0):
            self.K = int(np.max(t_train) + 1)  # the number of classes (from 0 indexing)
        else:
            raise ValueError("t must contain only positive integer values (got {})").format(t_train)
        self.t_train = t_train
        if self.kernel.general_kernel:
            sigma = np.reshape(self.kernel.sigma, (self.K, 1))
            tau = np.reshape(self.kernel.tau, (self.K, 1))
            self.sigma = np.tile(sigma, (1, self.D))  # (K, D)
            self.tau = np.tile(tau, (1, self.D))  # (K, D)
        else:
            self.sigma = self.kernel.sigma
            self.tau = self.kernel.tau

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


class VBMultinomialGP(Estimator):
    """
    A Variational Bayes classifier. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma

    def _estimate_initiate(self, M_0):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """
        varphi_0 = np.ones(np.shape(self.kernel.varphi))
        psi_tilde = np.ones(np.shape(self.kernel.varphi))
        return M_0, varphi_0, psi_tilde

    def estimate(self, M_0, steps, first_step=1):
        """
        Estimating the posterior means are a 3 step iteration over M_tilde, varphi_tilde and psi_tilde
            Eq.(8), (9), (10), respectively.

        :param M_0: (N, K) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Posterior mean and covariance estimates.
        :rtype: (5, ) tuple of :class:`numpy.ndarrays`
        """
        M_tilde, varphi_tilde, psi_tilde = self._estimate_initiate(M_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            Y_tilde = self._Y_tilde(M_tilde)
            M_tilde, Sigma_tilde, C_tilde = self._M_tilde(Y_tilde, varphi_tilde)
            varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde, n_samples=5)  # TODO: Cythonize. Major bottleneck.
            psi_tilde = self._psi_tilde(varphi_tilde)
        return M_tilde, Sigma_tilde, C_tilde, Y_tilde, varphi_tilde

    def _vector_expectation_wrt_u(self, M_new_tilde, var_new_tilde, n_samples):
        """
        Calculate distribution over classes for M_new_tilde at the same time.

        :arg M_new_tilde: An (N_test, K) array filled with \tilde{m}_k^{new_i} where
            k is the class indicator and i is the index of the test object.
        :type M_new_tilde: :class:`numpy.ndarray`
        :param var_new_tilde: An (N_test, K) array filled with \tilde{sigma}_k^{new_i} where
           k is the class indicator and i is the index of the test object.
        :arg int n_samples: Number of samples to take in the monte carlo estimate.

        :returns: Distribution over classes
        """
        nu_new_tilde = np.sqrt(np.add(1, var_new_tilde))  # (N_test, K)
        N_test = np.shape(M_new_tilde)[0]
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(M_new_tilde, self.K, N_test)  # (N_test, K, K) we will product across axis 2 (rows)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N_test, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N_test, n_samples, K, K)
        # Assume its okay to use the same random variables over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        # Multiply nu by u
        nu_new_tildes = matrix_of_valuess(nu_new_tilde, self.K, N_test)
        # Find the transpose for the product across classes
        nu_new_tilde_Ts = nu_new_tildes.transpose((0, 2, 1))
        Us_nu_new_tilde_Ts = np.multiply(Us, nu_new_tildes)
        random_variables = np.add(Us_nu_new_tilde_Ts, differencess)
        random_variables = np.divide(random_variables, nu_new_tilde_Ts)
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

    def _predict_vector_generalised(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
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
        cs_news = [np.diag(self.kernel.kernel_matrix(X_test, X_test)[k]) for k in range(self.K)]  # (K, N_test)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (K, N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (K, N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (K, N_test)
        # Calculate M_tilde_new # TODO: test this.
        Y_tilde_T = np.reshape(Y_tilde.T, (self.K, self.N, 1))
        M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_tilde_T)
        M_new_tilde_T = np.reshape(M_new_tilde_T, (self.K, N_test))
        var_new_tilde_T = np.subtract(cs_news, intermediate_scalars)
        return self._vector_expectation_wrt_u(M_new_tilde_T.T, var_new_tilde_T.T, n_samples)

    def _predict_vector(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.
        :param Sigma_tilde:
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
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test)) # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors)  # (N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        # Calculate M_tilde_new # TODO: test this.
        Y_tilde_T = np.reshape(Y_tilde.T, (self.K, self.N, 1))
        M_new_tilde = np.matmul(intermediate_vectors_T, Y_tilde)  # (N_test, K)
        var_new_tilde = np.subtract(cs_news, intermediate_scalars)  # (N_test, )
        var_new_tilde = np.reshape(var_new_tilde, (N_test, 1))  # TODO: do in place shape changes - quicker(?) and memor
        var_new_tilde = np.tile(var_new_tilde, (1, self.K))  # (N_test, K)
        return self._vector_expectation_wrt_u(M_new_tilde, var_new_tilde, n_samples)

    def predict(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :param Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel.general_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            if vectorised:
                return self._predict_vector_generalised(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")
        else:
            if vectorised:
                return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _multivariate_normal_pdf(self, m_k, C):
        """
        Return a number proportional to the multivariate normal pdf of r.v. m_k given mean 0 and covariance C.

        :param m_k: 
        :param C: 
        :return:
        """
        epsilon = 1e-6 * self.IN
        C_tild = C + epsilon
        L = np.linalg.cholesky(C_tild)
        L_inv = np.linalg.inv(L)
        C_inv = L_inv.T @ L_inv
        # We don't need the constant term, since it cancels in the ratio
        # - TODO: in that case use a log transform here
        # i.e. np.exp(- self.N / 2 * np.log(2 * np.pi) - 0.5 * m_k @ C_inv @ m_k)
        return np.exp(- 0.5 * m_k @ C_inv @ m_k)

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=500):
        """
        Return the w values of the sample on 2005 Page 9 Eq.(7).

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from varphi
        # TODO: Get a domain error here with varphi_tilde blowing up?

        varphis = sample_varphis(psi_tilde, n_samples, general=self.kernel.general_kernel)  # (n_samples, K, D) in general, (n_samples, ) for ISO case. Depend on the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)  # (n_samples, K, N, N) in general, (n_samples, N, N) for ISO case. Depend on the shape of psi_tilde.
        # Transpose M_tilde to become (n_samples, K, N) multivariate normal between m_k (N, ) and Cs_k over all samples
        M_tilde_T = np.transpose(M_tilde)  # (K, N)
        # TODO: begrudgingly using a nested for loop here, as I don't see the alternative,
        #  maybe I can calculate this by hand.
        ws = np.empty((n_samples, self.K))  # TODO: probably the wrong size.
        if self.kernel.general_kernel:
            for i in range(n_samples):
                    for k in range(self.K):
                        # Instead of using multivariate_normal.pdf we must use something like linalg cholesky, or
                        # nugget regularisation.
                        w_i_k = self._multivariate_normal_pdf(M_tilde_T[k], Cs_samples[i, k])
                        print(w_i_k)
                        # Add sample to the unnormalised w vectors
                        ws[i, k] = w_i_k
            # Normalise the w vectors
            normalising_constant = np.sum(ws, axis=0)  # (K,)
            ws = np.divide(ws, normalising_constant)  # (n_samples, K)
            ws = np.reshape(ws, (n_samples, self.K, 1))  # (n_samples, K, 1)
            ws = np.tile(ws, (1, 1, self.D))  # (n_samples, K, D)
        else:
            for i in range(n_samples):
                for k in range(self.K):
                    w_i_k = self._multivariate_normal_pdf(M_tilde_T[k], Cs_samples[i])
                    # Add sample to the unnormalised w vectors
                    ws[i, k, :] = w_i_k
            # Normalise the w vectors
            denominator = np.sum(ws, axis=0)  # (K, )
            ws = np.divide(ws, denominator)  # (n_samples, K)
            # Since all length scale parameters are the same, these should be close within some tolerance
            # TODO: set the tolerance. Actually im not so sure about this. Check what I meant by the above.
            assert np.allclose(ws[:, 0], ws[:, 1])
            # Take only one axis, since there is only one varphi
            ws = ws[:, 0]  # (n_samples, )
        element_prod = np.multiply(ws, varphis)
        return np.sum(element_prod, axis=0)

    def _psi_tilde(self, varphi_tilde):
        #_psi_tilde = np.divide(np.add(1, self.sigma), np.add(self.tau, varphi_tilde))
        #print('\n', _psi_tilde, 1./ _psi_tilde, '\n')  # TODO: for debugging
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
        assert np.shape(Y_tilde) == (self.N, self.K)
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # Calculate updated C and sigma
        # TODO: Do I want to update these in the class scope?
        C_tilde = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
        Sigma_tilde = np.linalg.inv(np.add(self.IN, C_tilde))  # (K, N, N)
        M_tilde = np.empty((self.N, self.K))
        for k, y_k in enumerate(Y_tilde.T):
            M_tilde[:, k] = C_tilde[k] @ Sigma_tilde[k] @ y_k
        return M_tilde, Sigma_tilde, C_tilde  # (N, K)

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements 2005 Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        # t = np.argmax(M_tilde, axis=1)  # The max of the GP vector m_k is t_n TODO: this is incorrect.
        negative_P = self._negative_P(M_tilde, self.t_train, n_samples=1000)  # TODO: correct version.
        # Eq.(5)
        Y_tilde = np.subtract(M_tilde, negative_P)
        # Eq.(6)
        # This part of the differences sum must be 0, since sum is over j \neq i
        Y_tilde[self.grid, self.t_train] = M_tilde[self.grid, self.t_train]
        diff_sum = np.sum(Y_tilde - M_tilde, axis=1)
        Y_tilde[self.grid, self.t_train] = M_tilde[self.grid, self.t_train] - diff_sum
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
        # vector_differences = differences[self.grid, :, t]  # (N, K)
        # vector_differencess = np.tile(vector_differences, (n_samples, 1, 1))  # (n_samples, N, K)
        # vector_differencess = np.moveaxis(vector_differencess, 1, 0)  # (N, n_samples, K)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume it's okay to use the same samples of U over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)  # (N, n_samples, K, K)
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (N, n_samples, K, K)
        log_cum_dists = np.log(cum_dists)  # (N, n_samples, K, K)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[self.grid, :, :, t]  # (N, n_samples, K)  # TODO: not sure if this works, could be the wrong values.
        log_M_nk_M_nt_pdfs = norm.pdf(random_variables[self.grid, :, :, t])  # (N, n_samples, K)  # TODO: not sure if this works
        # product is over j \neq k
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # product is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0  # TODO: Test it.
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (N, n_samples, K)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        # TODO: subtract these using log.
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte Carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf)  # (N, K)

        # # Superseded
        # samples = np.exp(log_samples)
        # # Take the samples us as matrices (n_samples, N, K)
        # us = Us[:, :, 0]  # (n_samples, K)  # TODO: definately an issue here.
        # print(us)
        # assert us[0, 1] == us[0, 2]
        # assert us[0, 0] == us[0, 1]
        # # Assume it's okay to use the same samples of U over all of the data points
        # diff = np.add(us, vector_differences)
        # normal_pdfs = norm.pdf(diff, loc=0, scale=1)  # (n_samples, N, K)
        # normal_cdfs = norm.cdf(diff, loc=0, scale=1)  # (n_samples, N, K)
        # # Find the elementwise product of these two samples of matrices
        # element_prod_pdf = np.multiply(normal_pdfs, samples)
        # element_prod_cdf = np.multiply(normal_cdfs, samples)
        # # axis 0 is the n_samples samples, which is the monte-carlo sum of interest
        # element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=0)
        # element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=0)
        # return np.divde(element_prod_pdf, element_prod_cdf)  # (N, K)


class VBMultinomialOrderedGP(Estimator):
    """
    A Variational Bayes classifier for ordered likelihood. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference. It is for the ordered likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, cutpoint, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :arg cutpoint: The (K +1, ) array of cutpoint parameters \bm{gamma}.
        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma
        if self.kernel.general_kernel:
            raise ValueError('The kernel must not be ARD type (kernel.general_kernel=1),'
                             ' but ISO type (kernel.general_kernel=0). (got {}, expected)'.format(
                self.kernel.general_kernel, 0))
        if not all(
                cutpoint[i] <= cutpoint[i + 1]
                for i in range(self.N)):
            raise CutpointValueError(cutpoint)
        if np.any(cutpoint < 0):
            raise ValueError('cutpoint array must have *non negative* elements (got {}).'.format(cutpoint))
        if cutpoint[0] != 0.0:
            raise ValueError('The first cutpoint parameter must be 0.0 (got {}, expected {})'.format(cutpoint[0], 0.0))
        if cutpoint[-1] not in [np.infty, np.inf]:
            raise ValueError('The last cutpoint parameter must be numpy.inf (got {}, expected {})'.format(cutpoint[-1], np.inf))
        self.cutpoint = cutpoint

    def _estimate_initiate(self, m_0):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.
        """
        varphi_0 = np.ones(np.shape(self.kernel.varphi))
        psi_tilde = np.ones(np.shape(self.kernel.varphi))
        return m_0, varphi_0, psi_tilde

    def estimate(self, m_0, steps, first_step=1):
        """
        Estimating the posterior means are a 3 step iteration over M_tilde, varphi_tilde and psi_tilde
            Eq.(8), (9), (10), respectively.

        :param m_0: (N, ) numpy.ndarray of the initial location of the posterior mean.
        :type m_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Posterior mean and covariance estimates.
        :rtype: (5, ) tuple of :class:`numpy.ndarrays`
        """
        m_tilde, varphi_tilde, psi_tilde = self._estimate_initiate(m_0)
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            y_tilde = self._y_tilde(m_tilde)
            m_tilde, Sigma_tilde, C_tilde = self._m_tilde(y_tilde, varphi_tilde)
            varphi_tilde = self._varphi_tilde(m_tilde, psi_tilde, n_samples=5)  # TODO: Cythonize. Major bottleneck.
            psi_tilde = self._psi_tilde(varphi_tilde)
        return m_tilde, Sigma_tilde, C_tilde, y_tilde, varphi_tilde

    def _predict_vector(self, Sigma_tilde, y_tilde, varphi_tilde, X_test):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.
        :param Sigma_tilde:
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
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        c_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ C_news  # (N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors)  # (N_test, N)
        intermediate_scalars = np.sum(np.multiply(C_news, intermediate_vectors), axis=0)  # (N_test, )
        # Calculate m_tilde_new # TODO: test this.
        m_new_tilde = np.dot(intermediate_vectors_T, y_tilde)  # (N_test, N) (N, ) = (N_test, )
        var_new_tilde = np.subtract(c_news, intermediate_scalars)  # (N_test, )
        var_new_tilde = np.reshape(var_new_tilde, (N_test, 1))  # TODO: do in place shape changes - quicker(?) and memor
        predictive_distributions = np.empty(N_test, self.K)
        for n in range(N_test):
            for k in range(self.K):
                gamma_k = self.gamma[k + 1]
                gamma_k_minus_1 = self.gamma[k]
                m_n = m_new_tilde[n]
                predictive_distributions[n, k] = (norm.pdf(gamma_k_minus_1 - m_n) - norm.pdf(gamma_k - m_n)) / (
                        norm.cdf(gamma_k - m_n) - norm.cdf(gamma_k_minus_1 - m_n))
        return predictive_distributions  # (N_test, K)

    def predict(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :param Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel.general_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            return ValueError("ARD kernel may not be used in the ordered likelihood estimator.")
        else:
            if vectorised:
                return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _multivariate_normal_pdf(self, m_k, C):
        """
        Return a number proportional to the multivariate normal pdf of r.v. m_k given mean 0 and covariance C.

        :param m_k:
        :param C:
        :return:
        """
        epsilon = 1e-6 * self.IN
        C_tild = C + epsilon
        L = np.linalg.cholesky(C_tild)
        L_inv = np.linalg.inv(L)
        C_inv = L_inv.T @ L_inv
        # We don't need the constant term, since it cancels in the ratio
        # - TODO: in that case use a log transform here
        # i.e. np.exp(- self.N / 2 * np.log(2 * np.pi) - 0.5 * m_k @ C_inv @ m_k)
        return np.exp(- 0.5 * m_k @ C_inv @ m_k)

    def _varphi_tilde(self, m_tilde, psi_tilde, n_samples=500):
        """
        Return the w values of the sample on 2005 Page 9 Eq.(7).

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from varphi
        # TODO: Get a domain error here with varphi_tilde blowing up? Probably because of mutlivariate normal.
        varphis = sample_varphis(psi_tilde, n_samples, general=self.kernel.general_kernel)  # (n_samples, )
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)  # (n_samples, N, N)
        # TODO: begrudgingly using a for loop here, as I don't see the alternative,
        #  maybe I can calculate this by hand.
        ws = np.empty((n_samples, ))
        for i in range(n_samples):
            ws[i, :] = self._multivariate_normal_pdf(m_tilde, Cs_samples[i])
        # Normalise the w vectors
        denominator = np.sum(ws, axis=0)  # ()
        ws = np.divide(ws, denominator)  # (n_samples, )
        return np.dot(ws, varphis)

    def _psi_tilde(self, varphi_tilde):
        return np.divide(np.add(1, self.sigma), np.add(self.tau, varphi_tilde))

    def _m_tilde(self, y_tilde, varphi_tilde):
        """
        Return the posterior mean estimate of m.

        2020 Page 4 Eq.(14)

        :arg y_tilde: (N, K) array
        :type y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type y_tilde: :class:`np.ndarray`
        """
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # Calculate updated C and sigma
        C_tilde = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
        Sigma_tilde = np.linalg.inv(np.add(self.IN, C_tilde))  # (K, N, N)
        m_tilde = C_tilde @ Sigma_tilde @ y_tilde
        return m_tilde, Sigma_tilde, C_tilde  # (N, K)

    def _y_tilde(self, m_tilde):
        """
        Calculate Y_tilde elements 2021 Page 3 Eq.(11).

        :arg M_tilde: The posterior expectations for M (N, ).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, ) containing \tilde(y)_{n} values.
        """
        p = self._p(m_tilde, self.t_train, n_samples=1000)
        # Eq. (11)
        y_tilde = np.add(m_tilde, p)
        return y_tilde

    def _p(self, m_tilde, t, n_samples=1000):
        """
        Estimate the rightmost term of 2021 Page 3 Eq.(11), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg n_samples: The number of samples to take.
        """
        p = np.empty(self.N)
        for n in range(self.N):
            gamma_k = self.gamma[t[n] + 1]
            gamma_k_minus_1 = self.gamma[t[n]]
            m_n = m_tilde[n]
            p[n] = (norm.pdf(gamma_k_minus_1 - m_n) - norm.pdf(gamma_k - m_n)) / (
                    norm.cdf(gamma_k - m_n) - norm.cdf(gamma_k_minus_1 - m_n))
        return p  # (N, )


class CutpointValueError(Exception):
    """An invalid cutpoint argument was used to construct the classifier model."""

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