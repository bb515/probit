from abc import ABC, abstractmethod
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
import random
from scipy.stats import norm, multivariate_normal
from tqdm import trange
from .utilities import (
    sample_Us, sample_varphis,
    matrix_of_differencess, matrix_of_valuess, matrix_of_VB_differencess, matrix_of_VB_differences,
    unnormalised_log_multivariate_normal_pdf, vectorised_unnormalised_log_multivariate_normal_pdf,
    vectorised_multiclass_unnormalised_log_multivariate_normal_pdf)


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
        if t_train.dtype != int:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        else:
            self.t_train = t_train
        self.K = int(np.max(self.t_train) + 1)  # the number of classes (from -1 +1 indexing)
        if self.kernel.ARD_kernel:
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

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=3000, vectorised=True, numerical_stability=True):
        """
        Return the posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).

        This is the same for all multinomial categorical estimators, and so can live in the Abstract Base Class.

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
            Default 3000.
        :arg bool vectorised: If `True` then it will do calculations in a more efficient, vectorised way using
            `numpy.einsum()`, if `False` then it will do a scalar version of the calculations. Default `True`.
        :arg bool numerical_stability: If `True` then it will do calculations using log-sum-exp trick, c.f.
            https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ since normalising vectors of log likelihoods can
            result in under- or overflow. log-sum-exp trick resolves this issue. Default `True`.
        :return: The posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).
        """
        # Vector draw from varphi
        # (n_samples, K, D) in general and ARD, (n_samples, ) for single shared kernel and ISO case. Depends on the
        # shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)  # Note that this resets the kernel varphi
        log_varphis = np.log(varphis)
        # (n_samples, K, N, N) in Multinomial general and ARD, (n_samples, N, N) single shared kernel and
        # ISO case, which will depend on the shape of the hyperhyperparameters psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)
        # Nugget regularisation for numerical stability. 1e-5 or 1e-6 typically used - important to keep the same
        Cs_samples = np.add(Cs_samples, 1e-8 * np.eye(self.N))
        if numerical_stability is True:
            if self.kernel.general_kernel and self.kernel.ARD_kernel:
                if vectorised is True:
                    log_ws = vectorised_multiclass_unnormalised_log_multivariate_normal_pdf(
                        M_tilde, mean=None, covs=Cs_samples)
                elif vectorised is False:
                    M_tilde_T = np.transpose(M_tilde)  # (K, N)
                    log_ws = np.empty((n_samples, self.K))
                    for i in range(n_samples):
                            for k in range(self.K):
                                # This fails if M_tilde and varphi are not initialised correctly
                                # Add sample to the unnormalised w vectors
                                log_ws[i, k] = unnormalised_log_multivariate_normal_pdf(
                                    M_tilde_T[k], mean=None, cov=Cs_samples[i, k])
                log_ws = np.reshape(log_ws, (n_samples, self.K, 1))
            else:
                # TODO: Does it matter I only use a single class to evaluate M_tilde_T[0]?
                if vectorised is True:
                    log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(M_tilde[:, 0], mean=None, covs=Cs_samples)
                elif vectorised is False:
                    log_ws = np.empty((n_samples, ))
                    for i in range(n_samples):
                        # This fails if M_tilde and varphi are not initialised correctly
                        # Add sample to the unnormalised w vectors
                        log_ws[i] = unnormalised_log_multivariate_normal_pdf(M_tilde[:, 0], mean=None, cov=Cs_samples[i])
            # Normalise the w vectors in a numerically stable fashion
            max_log_ws = np.max(log_ws, axis=0)  # (K, )  or ()
            log_normalising_constant = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws), axis=0))
            log_ws = np.subtract(log_ws, log_normalising_constant)  # (n_samples, K) or (n_samples,)
            element_prod = np.add(log_ws, log_varphis)
            element_prod = np.exp(element_prod)
            magic_number = self.D  # Also could be self.K?
            return magic_number * np.sum(element_prod, axis=0)
        elif numerical_stability is False:
            if self.kernel.general_kernel and self.kernel.ARD_kernel:
                ws = np.empty((n_samples, self.K))
                for i in range(n_samples):
                    for k in range(self.K):
                        # Add sample to the unnormalised w vectors
                        ws[i, k] = multivariate_normal.pdf(M_tilde[:, k], mean=None, cov=Cs_samples[i, k])
                ws = np.reshape(ws, (n_samples, self.K, 1))
            else:
                ws = np.empty((n_samples,))
                for i in range(n_samples):
                    ws[i] = multivariate_normal.pdf(M_tilde[:, 0], mean=None, cov=Cs_samples[i])
            normalising_constant = np.sum(ws, axis=0)  # (K, 1) or ()
            ws = np.divide(ws, normalising_constant)  # (n_samples, K, 1) or (n_samples,)
            # (n_samples, K, 1) * (n_samples, K, D) or (n_samples,) * (n_samples,)
            element_prod = np.multiply(ws, varphis)
            return np.sum(element_prod, axis=0)

    def _psi_tilde(self, varphi_tilde):
        """
        Return the posterior mean estimate of the hyperhyperparameters psi 2005 Page 9 Eq.(10).

        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :arg varphi_tilde: Posterior mean estimate of varphi.
        :return: The posterior mean estimate of the hyperhyperparameters psi 2005 Page 9 Eq.(10).
        """
        return np.divide(np.add(1, self.sigma), np.add(self.tau, varphi_tilde))

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
        # Assume its okay to use the same random variables over all N_test data points
        Us = sample_Us(self.K, n_samples, different_across_classes=True)  # (n_samples, K, K)
        # Multiply nu by u
        nu_new_tildes = matrix_of_valuess(nu_new_tilde, self.K, N_test)  # (N_test, K, K)
        nu_new_tildess = np.tile(nu_new_tildes, (n_samples, 1, 1, 1))  # (n_samples, N_test, K, K)
        nu_new_tildess = np.moveaxis(nu_new_tildess, 1, 0)  # (N_test, n_samples, K, K)
        # # Find the transpose (for the product across classes)
        # nu_new_tilde_Ts = nu_new_tildes.transpose((0, 2, 1))  # (N_test, K, K)
        # Find the transpose (for the product across classes)
        nu_new_tilde_Tss = nu_new_tildess.transpose((0, 1, 3, 2))  # (N_test, n_samples, K, K)
        Us_nu_new_tilde_Ts = np.multiply(Us, nu_new_tildess)  # TODO: do we actually need to use transpose here?
        random_variables = np.add(Us_nu_new_tilde_Ts, differencess)  # (N_test, n_samples, K, K)
        random_variables = np.divide(random_variables, nu_new_tilde_Tss)  # TODO: do we actually need to use transpose here?
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # Fill diagonals with 0
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # axis 0 is the N_test objects,
        # axis 1 is the n_samples samples, axis 3 is then the row index, which is the product of cdfs of interest
        log_samples = np.sum(log_cum_dists, axis=3)
        samples = np.exp(log_samples)
        # axis 1 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=1)  # (N_test, K)

    def _predict_vector_general(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Make variational Bayes prediction over classes of X_test.

        This is the general case where there are hyperparameters varphi (K, D) for all dimensions and classes.
        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :param Sigma_tilde:
        :param C_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y (N, K).
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi  # TODO: test this.
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (K, N, N_test)
        # TODO: this is a bottleneck - write a specialised kernel function - wonder how GPFlow does it
        covariances_new = self.kernel.kernel_matrix(X_test, X_test)
        cs_news = [np.diag(covariances_new[k]) for k in range(self.K)]  # (K, N_test)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (K, N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (K, N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (K, N_test)
        # Calculate M_tilde_new
        # TODO: could just use Y_tilde instead of Y_tilde_T? It would be better just to store Y in memory as (K, N)
        Y_tilde_T = np.reshape(Y_tilde.T, (self.K, self.N, 1))
        M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_tilde_T)  # (K, N_test, 1)?
        M_new_tilde_T = np.reshape(M_new_tilde_T, (self.K, N_test))  # (K, N_test)
        ## This was tested to be the slower option because matmul is an optimised sum and multiply.
        # M_new_tilde_T_2 = np.sum(np.multiply(Y_tilde_T, intermediate_vectors), axis=1)  # (K, N_test)
        # assert np.allclose(M_new_tilde_T, M_new_tilde_T_2)
        var_new_tilde_T = np.subtract(cs_news, intermediate_scalars)
        return self._vector_expectation_wrt_u(M_new_tilde_T.T, var_new_tilde_T.T, n_samples)

    def _predict_vector(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Return posterior predictive prediction over classes of X_test.

        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :param Sigma_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (N, N_test)
        # TODO: Generalises to (K, N, N)?
        intermediate_vectors_T = np.transpose(intermediate_vectors)  # (N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        # Calculate M_tilde_new # TODO: test this.
        # Y_tilde_T = np.reshape(Y_tilde.T, (self.K, self.N, 1))
        M_new_tilde = np.matmul(intermediate_vectors_T, Y_tilde)  # (N_test, K)
        var_new_tilde = np.subtract(cs_news, intermediate_scalars)  # (N_test, )
        var_new_tilde = np.reshape(var_new_tilde, (N_test, 1))  # TODO: do in place shape changes - quicker(?) and
        # TODO: less memory - take a look at the binary case as that has in place shape changes.
        var_new_tilde = np.tile(var_new_tilde, (1, self.K))  # (N_test, K)
        return self._vector_expectation_wrt_u(M_new_tilde, var_new_tilde, n_samples)


class VBBinomialGP(Estimator):
    """
    A binary Variational Bayes classfier. Inherits the Estimator ABC.

    This class allows users to define a binary classification problem, get predictions
    using approximate Bayesian inference.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`VBBinomialGP` estimator object.

        :returns: An :class:`VBBinomialGP` estimator object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        if self.K != 2:
            raise ValueError("t_train must only contain +1 or -1 class labels, got {}".format(self.t_train))
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma

    def _estimate_initiate(self, M_0, varphi_0=None, psi_0=None):
        """
        Initialise the sampler.

        :arg M_0: Intialisation of posterior mean estimates.
        :arg varphi_0: Initialisation of hyperparameter posterior mean estimates. If `None` then initialised to ones,
            default `None`.
        :arg psi_0: Initialisation of hyperhyperparameter posterior mean estimates. If `None` then initialised to ones,
            default `None`.
        :return: Containers for the mean estimates of parameters and hyperparameters.
        :rtype: (3,) tuple
        """
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        return M_0, varphi_0, psi_0

    def estimate(self, M_0, steps, varphi_0=None, psi_0=None, first_step=1, fix_hyperparameters=False):
        """
        Estimate the posterior means.

        This is a one step iteration over M_tilde (equation 11).

        :param M_0: (N, K) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg int first_step: The first step. Useful for burn in algorithms.

        :return: Posterior mean and covariance estimates.
        :rtype: (5, ) tuple of :class:`numpy.ndarrays`
        """
        M_tilde, varphi_tilde, psi_tilde = self._estimate_initiate(M_0, varphi_0, psi_0)

        for _ in trange(first_step, first_step + steps,
                        desc="Estimator progress", unit="iterations"):
            Y_tilde = self._Y_tilde(M_tilde)
            M_tilde = self._M_tilde(Y_tilde, varphi_tilde)
            if fix_hyperparameters is False:
                varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde, vectorised=True)  # TODO: Cythonize. Major bottleneck.
                psi_tilde = self._psi_tilde(varphi_tilde)
                print("varphi_tilde = ", varphi_tilde, "psi_tilde = ", psi_tilde)
        return M_tilde, self.Sigma, self.C, Y_tilde, varphi_tilde, psi_tilde

    def _predict_vector(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test):
        """
        Make prediction over binary classes of X_test.

        :param Sigma_tilde:
        :param C_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: An (standard and analytical) estimate of the (binary) class probabilities.
        """
        # N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck: need to make a vectorised version of this in kernel functions
        cs_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test,)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (N, N_test)
        intermediate_scalars = np.sum(np.multiply(Cs_news, intermediate_vectors), axis=0)  # (N_test,)
        # Calculate M_tilde_new # TODO: test this. Results look wrong to me for binary case.
        M_new_tilde = intermediate_vectors.T @ Y_tilde  # (N_test,)
        var_new_tilde = np.subtract(cs_news, intermediate_scalars)  # (N_test,)
        nu_new_tilde = np.sqrt(np.add(1, var_new_tilde))  # (N_test,)
        random_variables = np.divide(M_new_tilde, nu_new_tilde)
        return norm.cdf(random_variables)

    def predict(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test):
        """
        Return the posterior predictive distribution over classes.

        :param Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :param X_test: The new data points, array like (N_test, D).
        :return: A Monte Carlo estimate of the class probabilities.
        """
        return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test)

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=1000, vectorised=True, numerical_stability=True):
        """
        Return the posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).

        This is different from the corresponding multinomial categorical estimator function in the Abstract Base Class.

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
            Default 3000.
        :arg bool vectorised: If `True` then it will do calculations in a more efficient, vectorised way using
            `numpy.einsum()`, if `False` then it will do a scalar version of the calculations. Default `True`.
        :arg bool numerical_stability: If `True` then it will do calculations using log-sum-exp trick, c.f.
            https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ since normalising vectors of log likelihoods can
            result in under- or overflow. log-sum-exp trick resolves this issue. Default `True`.
        :return: The posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).
        """
        # TODO: There seems to be an algorithmic error in the function. Numerically it should be stable.
        # TODO: This wasn't derived for the binary case so would perhaps need a separate derivation.
        # Vector draw from varphi. In the binary case we always have a single and shared covariance function.
        # (n_samples, D) in ARD, (n_samples, ) for ISO case. Depends on the shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)
        log_varphis = np.log(varphis)
        # (n_samples, N, N) in ARD, (n_samples, N, N) for ISO case. Depends on the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)
        #print("Cs_samples[0]", Cs_samples[0])
        # Nugget regularisation for numerical stability. 1e-5 or 1e-6 typically used - important to keep the same
        Cs_samples = np.add(Cs_samples, 1e-12 * np.eye(self.N))
        if numerical_stability is True:
            if vectorised is True:
                log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(M_tilde, mean=None, covs=Cs_samples)
            elif vectorised is False:
                log_ws = np.empty((n_samples,))
                for i in range(n_samples):
                    # This fails if M_tilde and varphi need to be initialised correctly
                    # Add sample to the unnormalised w vectors
                    log_ws[i] = unnormalised_log_multivariate_normal_pdf(M_tilde, mean=None, cov=Cs_samples[i])
            # Normalise the w vectors using the log-sum-exp operator
            max_log_ws = np.max(log_ws)
            log_normalising_constant = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws), axis=0))
            log_ws = np.subtract(log_ws, log_normalising_constant)
            element_prod = np.add(log_varphis, log_ws)
            element_prod = np.exp(element_prod)
            return np.sum(element_prod, axis=0)
        elif numerical_stability is False:
            # This fails if M_tilde and varphi are not initialised correctly due to numerical instabilities
            #   (overflow/underflow)
            ws = np.empty((n_samples, ))
            for i in range(n_samples):
                # TODO: Does it matter I only use a single class to evaluate M_tilde_T[0]?
                ws[i] = multivariate_normal.pdf(M_tilde, mean=None, cov=Cs_samples[i])
            normalising_constant = np.sum(ws, axis=0)  # (1,)
            ws = np.divide(ws, normalising_constant)  # (n_samples,)
            magic_number = self.D
            return magic_number * np.dot(ws, varphis)  # (1,)

    def _varphi_tilde_SS(self, M_tilde, psi_tilde, n_samples=10):
        """
        Return the posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).

        This is different from the corresponding multinomial categorical estimator function in the Abstract Base Class.

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
            Default 3000.
        :return: The posterior mean estimate of the hyperparameters varphi 2005 Page 9 Eq.(9).
        """
        # TODO: Numerical instabilities here. Seems to be a problem with underfitting in this function.
        # Vector draw from varphi. In the binary case we always have a single and shared covariance function.
        # (n_samples, D) in ARD, (n_samples, ) for ISO case. Depends on the shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)
        # (n_samples, N, N) in ARD, (n_samples, N, N) for ISO case. Depends on the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)
        #print("Cs_samples[0]", Cs_samples[0])
        # Nugget regularisation for numerical stability. 1e-5 or 1e-6 typically used - important to keep the same
        Cs_samples = np.add(Cs_samples, 1e-5 * np.eye(self.N))
        ws = np.empty((n_samples,))
        for i in range(n_samples):
            # This fails if M_tilde and varphi need to be initialised correctly
            # Add sample to the unnormalised w vectors
            ws[i] = multivariate_normal.pdf(M_tilde, mean=None, cov=Cs_samples[i])
        # Normalise the w vectors
        normalising_constant = np.sum(ws, axis=0)  # ()
        ws = np.divide(ws, normalising_constant)  # (n_samples,)
        element_prod = np.multiply(varphis, ws)
        return np.sum(element_prod, axis=0)

    def _M_tilde(self, Y_tilde, varphi_tilde):
        """
        Return the posterior mean estimate of M.

        2005 Page 10 Eq.(11)

        :arg Y_tilde: (N,) array
        :type Y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type Y_tilde: :class:`np.ndarray`
        """
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # Calculate updated C and sigma
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (N, N)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))  # (N, N)
        self.cov = self.C @ self.Sigma
        return self.cov @ Y_tilde

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements 2005 Page 10 Eq.(11)

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page 10 Eq.(11)
        :return: Y_tilde (N,) containing \tilde(y)_{n} values.
        """
        return np.add(M_tilde, self._P(M_tilde, self.t_train))

    def _P(self, M_tilde, t_train):
        """
        Estimate the P of 2005 Page 10 Eq.(11), which can be obtained analytically derived from straightforward results
        for corrections to the mean of a Gaussian due to truncation.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg n_samples: The number of samples to take.
        """
        return np.divide(
            np.multiply(t_train, norm.pdf(M_tilde)), norm.cdf(np.multiply(t_train, M_tilde))
        )


class VBMultinomialSparseGP(Estimator):
    """
    A Sparse Variational Bayes classifier via ADF with optional informative samping. Inherits the Estimator ABC.

    This class allows users to define a classification problem, get predictions using approximate Bayesian inference of
    a sparse approximation to a Gaussian process.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`VBMultinomialSparseGP` estimator object.

        :returns: An :class:`VBMultinomialSparseGP` object.
        """
        super().__init__(*args, **kwargs)
        # All indeces are initially in the inactive set
        self.inactive_set = list(range(self.N))
        # For random inducing point selection
        random.shuffle(self.inactive_set)
        self.active_set = []
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma

    def _estimate_initiate(self, steps, M_0, M_ADF_0=None, s_ADF_0=None, varphi_0=None, psi_0=None):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.

        :arg M_0: Initialisation of posterior mean estimates.
        :arg M_ADF_0: Initialisation of ADF inverse covariance updates (S, N, K). If `None` then initialised to zeros,
            default `None`.
        :arg s_ADF_0: Initialisation of ADF scalar update (K, N). If `None` then initialised to zeros,
            default `None`.
        :arg varphi_0: Initialisation of hyperparameter posterior mean estimates. If `None` then initialised to ones,
            default `None`.
        :arg psi_0: Initialisation of hyperhyperparameter posterior mean estimates. If `None` then initialised to ones,
            default `None`.
        :return: Containers for the mean estimates of parameters and hyperparameters.
        :rtype: (6,) tuple.
        """
        if steps > self.N:
            raise ValueError("There are more update steps, which each requires a new inducing data point, than"
                             " data points themselves. Steps must be equal to or less than the number of data points!"
                             " (expected <={}, got {})".format(self.N, steps))
        if M_ADF_0 is None:
            M_ADF_0 = np.zeros((steps, self.N, self.K))
        if s_ADF_0 is None:
            s_ADF_0 = np.ones((self.K, self.N))
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        # Since the first ADF approximation before data-update is simply the GP prior
        Sigma_ADF_0 = self.cov
        Ms = []
        Ys = []
        M_ADFs = []
        s_ADFs = []
        varphis = []
        psis = []
        bounds = []
        Sigma_ADFs = []
        containers = (Ms, Ys, Sigma_ADFs, M_ADFs, s_ADFs, varphis, psis, bounds)
        return M_0, Sigma_ADF_0, M_ADF_0, s_ADF_0, varphi_0, psi_0, containers

    def estimate(self, M_0, steps, M_ADF_0=None, s_ADF_0=None, varphi_0=None, psi_0=None,
                 first_step=0, informative_selection=False, fix_hyperparameters=True, write=False):
        """
        Estimating the posterior means via the ADF approximations via  4 step iteration over Sigma_ADF, s_ADF, M_ADF,
        M_tilde_ADF from Eq.(12), (13), (14), (15) respectively.
        The hyperparameters are fixed by default, but full approximate inference over the hyperparmaeters can be done by
        setting fix_hyperparameters=False, so the iteration will includei updates of the posterior means varphi_tilde
        and psi_tilde Eq.(9), (10).

        :param M_0: (N, K) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :arg M_ADF_0: Initialisation of ADF inverse covariance updates (S, N, K). If `None` then initialised to
            zeros, default `None`.
        :arg s_ADF_0: Initialisation of ADF scalar update of the posterior variance (K, N). If
            `None` then initialised to zeros, default `None`.
        :param varphi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type varphi_0: :class:`np.ndarray`.
        :param psi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type psi_0: :class:`np.ndarray`.
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool fix_hyperparamters: If set to "True" will fix varphi to initial values. If set to "False" will
            do posterior mean updates of the hyperparameters. Default "False".
        :arg bool informative_selection: If set to "True" will use the informative selection crtieria for the new
            point which joins the active set.

        :return: Posterior mean and covariance estimates.
        :rtype: (9, ) tuple of :class:`numpy.ndarrays` of the approximate posterior means, other statistics and
        tuple of lists of per-step evolution of those statistics.
        """
        M_tilde, Sigma_ADF, M_ADF, s_ADF, varphi_tilde, psi_tilde, containers = self._estimate_initiate(
            steps, M_0, M_ADF_0, s_ADF_0, varphi_0, psi_0)
        Ms, Ys, Sigma_ADFs, M_ADFs, s_ADFs, varphis, psis, bounds = containers
        for step in trange(first_step, first_step + steps,
                        desc="Sparse GP priors estimator progress", unit="iterations"):
            # Check if there are any points left in the inactive set
            if len(self.active_set) != self.N:
                # Policy to choose new data point
                Y_tilde, calligraphic_Z, negative_p = self._Y_tilde(M_tilde)
                #print('Y_tilde', Y_tilde)
                _, index = self.new_point(informative_selection, Sigma_ADF, Y_tilde, varphi_tilde)
                C_n = self.C[:, index, :]
                #print('C_n', C_n)
                s_ADF, Sigma_ADF, Sigma_ADF_n = self._s_ADF(s_ADF, Sigma_ADF, M_ADF, C_n, index)  # (), (N, K)
                #print('s_ADF', s_ADF)
                M_ADF = self._M_ADF(M_ADF, s_ADF, Sigma_ADF_n, index, step)
                #print('M_ADF', M_ADF)
                M_tilde = self._M_tilde(M_tilde, s_ADF, negative_p, Sigma_ADF_n, index,
                                                        fix_hyperparameters, varphi_tilde)
                #print('M_tilde', M_tilde)
                if fix_hyperparameters is False:
                    varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde,
                                                      n_samples=1000)  # TODO: Cythonize. Major bottleneck.
                    psi_tilde = self._psi_tilde(varphi_tilde)
                if write:
                    # Calculate bound TODO: work out if calculating the bound makes sense.
                    # TODO: won't be able to calculate the lower bound if it is not positive definite.
                    #bound = self.variational_lower_bound(self.N, self.K, M_tilde, Sigma_ADF, self.C, calligraphic_Z)
                    #print("bound", bound)
                    Ms.append(np.linalg.norm(M_tilde))
                    Ys.append(np.linalg.norm(Y_tilde))
                    Sigma_ADFs.append(Sigma_ADF)
                    M_ADFs.append(np.linalg.norm(M_ADF))  # TODO: Don't store a (S, N, K) for each step.
                    s_ADFs.append(np.linalg.norm(s_ADF))
                    if fix_hyperparameters is False:
                        varphis.append(varphi_tilde)
                        psis.append(psi_tilde)
                    # bounds.append(bound)
            else:
                # Need to stop the iteration since there are no more points in the inactive set to include
                # TODO: have already handled this error by limiting the number of steps I can take.
                break
        containers = (Ms, Ys, M_ADFs, s_ADFs, varphis, psis, bounds)
        return M_tilde, M_ADF, s_ADF, Sigma_ADF, self.C, Y_tilde, varphi_tilde, psi_tilde, containers

    def new_point(self, informative_selection, Sigma_ADF=None, Y_tilde=None, varphi_tilde=None):
        """
        Return a new point based on some policy.

        If informative_selection is true, then Sigma_ADF, Y_tilde and varphi_tilde must be supplied.
        :return: x_new
        """
        if informative_selection:
            return self.informative_new_point(Sigma_ADF, Y_tilde, varphi_tilde)
        else:
            return self.random_new_point()

    def random_new_point(self):
        """
        Return a random point from inactive set and add it to the active set.

        :return x_new: The the new inducing point.
        :return index: The index of X_train of the new inducing point.
        """
        # inactive_set has been randomly shuffled, so we can pop from the end
        index = self.inactive_set.pop()
        x_new = [self.X_train[index]]
        self.active_set.append(index)
        return x_new, index

    def informative_new_point(self, Sigma_ADF, Y_tilde, varphi_tilde):
        """
        Return a random point from inactive set and add it to the active set.

        :return x_new: The the new inducing point.
        :return index: The index of X_train of the new inducing point.
        """
        Z = self.predict(
            Sigma_ADF, Y_tilde, varphi_tilde, self.X_train[self.inactive_set],
            n_samples=1000, vectorised=True)  # (N_inactive, K)
        Z = Z[self.t_train[self.inactive_set]]  # (N_inactive,)
        # Don't need to keep inactive set and active set there (same information, but prevents search?).
        # TODO: Try not to pop from the middle of a list, isn't pythonic or fast.
        index = self.inactive_set.pop(np.argmax(Z))  # Gets index and removes from the list
        x_new = [self.X_train[index]]
        # Add the new index to the active set
        self.active_set.append(index)
        return x_new, index

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
        if self.kernel.ARD_kernel and self.kernel.general_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            if vectorised:
                return self._predict_vector_general(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")
        else:
            if vectorised:
                return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    # def _predict_vector_general(self, Sigma_ARD, Y_tilde, varphi_tilde, X_test, n_samples=1000):
    #     """
    #     Make sparse GP prediction over classes of X_test.
    #
    #     This is the general case where there are hyperparameters varphi (K, D) for all dimensions and classes.
    #     This is the same for all categorical estimators, and so can live in the Abstract Base Class.
    #
    #     :param Sigma_ARD:
    #     :param C_tilde:
    #     :param Y_tilde: The posterior mean estimate of the latent variable Y (N, K).
    #     :param varphi_tilde:
    #     :param X_test: The new data points, array like (N_test, D).
    #     :param n_samples: The number of samples in the Monte Carlo estimate.
    #     :return: A Monte Carlo estimate of the class probabilities.
    #     """
    #     N_test = np.shape(X_test)[0]
    #     # Update the kernel with new varphi  # TODO: test this.
    #     self.kernel.varphi = varphi_tilde
    #     # Cs_news[:, i] is Cs_new for X_test[i]
    #     Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (K, N, N_test)
    #     # TODO: this is a bottleneck - write a specialised kernel function - wonder how GPFlow does it
    #     covariances_new = self.kernel.kernel_matrix(X_test, X_test)
    #     cs_news = [np.diag(covariances_new[k]) for k in range(self.K)]  # (K, N_test)
    #     # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
    #     intermediate_vectors = Sigma_tilde @ Cs_news  # (K, N, N_test)
    #     intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (K, N_test, N)
    #     intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (K, N_test)
    #     # Calculate M_tilde_new
    #     # TODO: could just use Y_tilde instead of Y_tilde_T? It would be better just to store Y in memory as (K, N)
    #     Y_tilde_T = np.reshape(Y_tilde.T, (self.K, self.N, 1))
    #     M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_tilde_T)  # (K, N_test, 1)?
    #     M_new_tilde_T = np.reshape(M_new_tilde_T, (self.K, N_test))  # (K, N_test)
    #     ## This was tested to be the slower option because matmul is an optimised sum and multiply.
    #     # M_new_tilde_T_2 = np.sum(np.multiply(Y_tilde_T, intermediate_vectors), axis=1)  # (K, N_test)
    #     # assert np.allclose(M_new_tilde_T, M_new_tilde_T_2)
    #     var_new_tilde_T = np.subtract(cs_news, intermediate_scalars)
    #     return self._vector_expectation_wrt_u(M_new_tilde_T.T, var_new_tilde_T.T, n_samples)

    def _s_ADF(self, s_ADF, Sigma_ADF, M_ADF, C_n, index):
        """
        Return the update of the ADF scalar update (K, N).

        2005 Page 12 Eq.(12), (13).

        :arg s_ADF: ADF scalar update of the posterior variance (K, N).
        :arg Sigma_ADF: The ADF approximation to the GP posterior covariance.
        :arg M_ADF: The ADF inverse covariance update (S, N, K).
        :arg C_n: (K, N) The covariance vector for the new inducing point.
        :returns: updated s_ADF (K, N) and Sigma_ADF_n (N, K)
        """
        # TODO: an issue here.
        # Probably want to calculate the covariance column on the fly, depending on choice of x.
        M_ADF_n = M_ADF[:, index, :]  # (S, K)
        M_ADF_T = M_ADF.transpose(2, 1, 0)  # (K, N, S)
        M_ADF_n_T = M_ADF_n.T  # (K, S)
        Sigma_ADF_n2 = C_n - np.einsum('ijk, ik -> ij', M_ADF_T, M_ADF_n_T)
        # This was checked as correct
        Sigma_ADF_n = C_n - np.einsum('jik,jk->ki', M_ADF, M_ADF_n)  # (K, N) <- (K, N) - einsum((S, N, K), (S, K))
        assert np.allclose(Sigma_ADF_n, Sigma_ADF_n2)
        Sigma_ADF[:, :, index] = Sigma_ADF_n  # (K, N) <- (K, N)
        #Sigma_ADF[:, index, :] = Sigma_ADF_n  # (K, N) <- (K, N)
        precisions = 1 / (1 + s_ADF[:, index])
        #print("precisions", precisions)
        precisions = np.reshape(precisions, (self.K, 1))
        # TODO: posterior variance should be non-negative??
        update = np.multiply(
            np.diagonal(
                np.einsum('...i, ...j', Sigma_ADF_n, Sigma_ADF_n), axis1=1, axis2=2),
            precisions)
        #print("update", update)
        return s_ADF - update, Sigma_ADF, Sigma_ADF_n  # (K, N), (N, K)

    def _M_ADF(self, M_ADF, s_ADF, Sigma_ADF_n, index, step):
        """
        Return the update of the ADF vector updates (S, N, K)
        :arg s_ADF: ADF scalar update of the posterior variance (K, N).
        :arg Sigma_ADF_n: (K, N)
        :arg index: The index of self.t_train and self.X_train of the new inducing point, n.
        :return: M_ADF (S, N, K)
        """
        standard_deviations = 1/np.sqrt(1 + s_ADF[:, index])  # (K, )
        standard_deviations = np.reshape(standard_deviations, (1, self.K))
        Sigma_ADF_n_T = Sigma_ADF_n.T
        # (S, N, K) = (K,) (N, K)
        # (N, ) = (1,) (N, ) : K times
        M_ADF[step, :, :] = np.multiply(Sigma_ADF_n_T, standard_deviations)
        return M_ADF

    def _negative_p_n(self, M_tilde_n, t_n, n_samples=3000):
        """
        Estimate the rightmost term of 2005 Page 8 Eq.(5), a ratio of Monte Carlo estimates of the expectation of
            functions of M wrt to the distribution p.

        :arg M_tilde_n: The current estimate of posterior expectations for M_n (K,), where M_n is the regression
            variable for the inducing point, n.
        :arg t_n: The target class for the inducing point, n.
        :arg n_samples: The number of Monte-Carlo samples to take in approximating the expectations.
        """
        differences = matrix_of_VB_differences(M_tilde_n, self.K, t_n)  # (K, K)
        differences = np.tile(differences, (n_samples, 1, 1))  # (n_samples, K, K)
        # Assume it's okay to use the same sample of U over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=False)  # (n_samples, K, K)
        random_variables = np.add(Us, differences)  # (n_samples, K, K) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (n_samples, K, K)
        log_cum_dists = np.log(cum_dists)  # (n_samples, K, K)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[:, t_n, :]  # (n_samples, K)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[:, t_n, :]))  # (n_samples, K)
        # Product is over j \neq tn=i
        log_cum_dists[:, :, t_n] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=2)  # TODO: not sure if needed for the variational bound
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=0)
        # product is over j \neq k
        log_cum_dists[:, range(self.K), range(self.K)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=2)
        log_samples = np.sum(log_cum_dists, axis=2)  # (n_samples, K)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, K)

    def _negative_p(self, M_tilde, t, n_samples=3000):
        """
        Estimate the rightmost term of 2005 Page 8 Eq.(5), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg t: The target vector.
        :arg n_samples: The number of samples to take.
        """
        # Find matrix of differences
        #differences = matrix_of_differencess(M_tilde, self.K, self.N)  # TODO: confirm this is wrong
        # we will product across axis 2 (rows)
        differences = matrix_of_VB_differencess(M_tilde, self.K, self.N, t, self.grid)  # (N, K, K)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume it's okay to use the same samples of U over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=False)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)  # (N, n_samples, K, K) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (N, n_samples, K, K)
        log_cum_dists = np.log(cum_dists)  # (N, n_samples, K, K)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[self.grid, :, t, :]  # (N, n_samples, K)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[self.grid, :, t, :]))  # (N, n_samples, K)
        # product is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=3)  # TODO: not sure if correct
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=1)
        # product is over j \neq k
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (N, n_samples, K)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte Carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, K)

    def _M_tilde(self, M_tilde, s_ADF, negative_p, Sigma_ADF_n, index, fix_hyperparameters, varphi_tilde):
        """
        TODO: This doesn't generalise to the binary case.

        :arg M_tilde:
        :arg s_ADF:
        :arg Sigma_ADF_n:
        :arg negative_P:
        :arg index: The index of self.t_train and self.X_train of the new inducing point, n.
        :return:
        """
        if fix_hyperparameters is False:
            # TODO: This option doesn't really make sense in this setting.
            # Update hyperparameters and GP posterior covariance with new values
            self.kernel.varphi = varphi_tilde
            # Calculate updated C and sigma
            self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
            self.Sigma = np.linalg.inv(np.add(self.IN, self.C))  # (K, N, N)
            self.cov = self.C @ self.Sigma  # (K, N, N) @ (K, N, N) = (K, N, N)
        # t_n = self.t_train[index]
        # M_tilde_n = M_tilde[index]
        # negative_p_n, calligraphic_Z = self._negative_P(M_tilde_n, t_n)
        negative_p_n = negative_p[index, :]
        weight = negative_p_n / (1 + s_ADF[:, index])  # (K,)
        weight = np.reshape(weight, (1, self.K))  # (K, 1)
        return M_tilde - np.multiply(Sigma_ADF_n.T, weight)
        # return M_tilde - np.multiply(Sigma_ADF_n, weight), Y_tilde, calligraphic_Z
        # return M_tilde - (negative_p_n)/(1 + s_ADF[:, index]) * Sigma_ADF_n  # (K,) * (K, N) -> (K, N)
        # return M_tilde + (Y_tilde[index, :] - M_tilde[index, :])/(1 + s_ADF[:, index]) * Sigma_ADF_n

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements 2005 Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        # t = np.argmax(M_tilde, axis=1)  # The max of the GP vector m_k is t_n this would be incorrect.
        negative_p, calligraphic_Z = self._negative_p(M_tilde, self.t_train, n_samples=3000)  # TODO: correct version.
        Y_tilde = np.subtract(M_tilde, negative_p)  # Eq.(5)
        Y_tilde[self.grid, self.t_train] = 0
        Y_tilde[self.grid, self.t_train] = np.sum(M_tilde - Y_tilde, axis=1)
        return Y_tilde, calligraphic_Z, negative_p


class VBMultinomialGP(Estimator):
    """
    A Variational Bayes classifier. Inherits the Estimator ABC.

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`VBMultinomialGP` estimator object.

        :returns: An :class:`VBMultinomialGP` object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma

    def _estimate_initiate(self, M_0, varphi_0=None, psi_0=None):
        """
        Initialise the sampler.

        This method should be implemented in every concrete sampler.

        :arg M_0: Intialisation of posterior mean estimates.
        :return: Containers for the mean estimates of parameters and hyperparameters.
        :rtype: (4,) tuple.
        """
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        Ms = []
        Ys = []
        varphis = []
        psis = []
        bounds = []
        containers = (Ms, Ys, varphis, psis, bounds)
        return M_0, varphi_0, psi_0, containers

    def estimate(self, M_0, steps, varphi_0=None, psi_0=None, first_step=1, fix_hyperparameters=False, write=False):
        """
        Estimating the posterior means via a 3 step iteration over M_tilde, varphi_tilde and psi_tilde from
            Eq.(8), (9), (10), respectively. Unless the hyperparameters are fixed, so the iteration is 1 step
            over M_tilde.

        :param M_0: (N, K) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the sampler.
        :param varphi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type varphi_0: :class:`np.ndarray`.
        :param psi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type psi_0: :class:`np.ndarray`.
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool fix_hyperparamters: If set to "True" will fix varphi to initial values. If set to "False" will
            do posterior mean updates of the hyperparameters. Default "False".

        :return: Posterior mean and covariance estimates.
        :rtype: (7, ) tuple of :class:`numpy.ndarrays` of the approximate posterior means, other statistics and
        tuple of lists of per-step evolution of those statistics.
        """
        M_tilde, varphi_tilde, psi_tilde, containers = self._estimate_initiate(M_0, varphi_0, psi_0)
        Ms, Ys, varphis, psis, bounds = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors estimator progress", unit="iterations"):
            Y_tilde, calligraphic_Z = self._Y_tilde(M_tilde)
            M_tilde = self._M_tilde(Y_tilde, varphi_tilde, fix_hyperparameters)
            if fix_hyperparameters is False:
                varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde, n_samples=1000)  # TODO: Cythonize. Major bottleneck.
                psi_tilde = self._psi_tilde(varphi_tilde)
            if write:
                # Calculate bound
                bound = self.variational_lower_bound(self.N, self.K, M_tilde, self.Sigma, self.C, calligraphic_Z)
                Ms.append(np.linalg.norm(M_tilde))
                Ys.append(np.linalg.norm(Y_tilde))
                if fix_hyperparameters is False:
                    varphis.append(varphi_tilde)
                    psis.append(psi_tilde)
                bounds.append(bound)
        containers = (Ms, Ys, varphis, psis, bounds)
        return M_tilde, self.Sigma, self.C, Y_tilde, varphi_tilde, psi_tilde, containers

    def _M_tilde(self, Y_tilde, varphi_tilde, fix_hyperparameters):
        """
        Return the posterior mean estimate of M.

        2005 Page 9 Eq.(8)

        :arg Y_tilde: (N, K) array
        :type Y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type Y_tilde: :class:`np.ndarray`
        """
        if fix_hyperparameters is False:
            # Update hyperparameters and GP posterior covariance with new values
            self.kernel.varphi = varphi_tilde
            print(varphi_tilde, self.kernel.varphi, 'varphi')
            # Calculate updated C and sigma
            self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
            print(self.C, 'C', np.linalg.cond(self.C))
            self.Sigma = np.linalg.inv(np.add(self.IN, self.C))  # (K, N, N)
            self.cov = self.C @ self.Sigma  # (K, N, N) @ (K, N, N) = (K, N, N)
        # TODO: Maybe just keep Y_tilde in this shape in memory.
        Y_tilde_reshape = Y_tilde.T.reshape(self.K, self.N, 1)  # (K, N, 1) required for np.multiply
        M_tilde_T = self.cov @ Y_tilde_reshape  # (K, N, 1)
        return M_tilde_T.reshape(self.K, self.N).T  # (N, K)

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements 2005 Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, K).
        :type M_tilde: :class:`numpy.ndarray`
        2005 Page _ Eq.(_)
        :return: Y_tilde (N, K) containing \tilde(y)_{nk} values.
        """
        # t = np.argmax(M_tilde, axis=1)  # The max of the GP vector m_k is t_n this would be incorrect.
        negative_P, calligraphic_Z = self._negative_P(M_tilde, self.t_train, n_samples=3000)  # TODO: correct version.
        Y_tilde = np.subtract(M_tilde, negative_P)  # Eq.(5)
        Y_tilde[self.grid, self.t_train] = 0
        Y_tilde[self.grid, self.t_train] = np.sum(M_tilde - Y_tilde, axis=1)
        # SS
        # # This part of the differences sum must be 0, since sum is over j \neq i
        # Y_tilde[self.grid, self.t_train] = M_tilde[self.grid, self.t_train]  # Eq.(6)
        # diff_sum = np.sum(Y_tilde - M_tilde, axis=1)
        # Y_tilde[self.grid, self.t_train] = M_tilde[self.grid, self.t_train] - diff_sum
        return Y_tilde, calligraphic_Z

    def _negative_P(self, M_tilde, t, n_samples=3000):
        """
        Estimate the rightmost term of 2005 Page 8 Eq.(5), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg t: The target vector.
        :arg n_samples: The number of samples to take.
        """
        # Find matrix of differences
        #differences = matrix_of_differencess(M_tilde, self.K, self.N)  # TODO: confirm this is wrong
        # we will product across axis 2 (rows)
        differences = matrix_of_VB_differencess(M_tilde, self.K, self.N, t, self.grid)  # (N, K, K)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, K, K)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, K, K)
        # Assume it's okay to use the same samples of U over all of the data points
        Us = sample_Us(self.K, n_samples, different_across_classes=False)  # (n_samples, K, K)
        random_variables = np.add(Us, differencess)  # (N, n_samples, K, K) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (N, n_samples, K, K)
        log_cum_dists = np.log(cum_dists)  # (N, n_samples, K, K)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[self.grid, :, t, :]  # (N, n_samples, K)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[self.grid, :, t, :]))  # (N, n_samples, K)
        # product is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=3)  # TODO: not sure if correct
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=1)
        # product is over j \neq k
        log_cum_dists[:, :, range(self.K), range(self.K)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (N, n_samples, K)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte Carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, K)
        # Superceded
        # M_nk_M_nt_cdfs = cum_dists[self.grid, :, :, t]  # (N, n_samples, K)
        # M_nk_M_nt_pdfs = norm.pdf(random_variables[self.grid, :, t, :])
        # cum_dists[:, :, range(self.K), range(self.K)] = 1
        # cum_dists[self.grid, :, :, t] = 1
        # samples = np.prod(cum_dists, axis=3)
        # element_prod_pdf = np.multiply(M_nk_M_nt_pdfs, samples)
        # element_prod_cdf = np.multiply(M_nk_M_nt_cdfs, samples)
        # element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        # element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)

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
        if self.kernel.ARD_kernel and self.kernel.general_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            if vectorised:
                return self._predict_vector_general(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")
        else:
            if vectorised:
                return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def variational_lower_bound(self, N, K, M,  Sigma, C, calligraphic_Z, numerically_stable=True):
        """
        Calculate the variational lower bound of the log marginal likelihood.

        :arg M_tilde:
        :arg Sigma_tilde:
        :arg C_tilde:
        :arg calligraphic_Z:
        :arg bool numerically_stable:
        """
        if numerically_stable is True:
            if self.kernel.general_kernel:
                C = C + 1e-8 * np.eye(N)
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_inv = L_inv.transpose((0, 2, 1)) @ L_inv  # (K, N, N) (K, N, N) -> (K, N, N)
                L_Sigma = np.linalg.cholesky(Sigma)
                half_log_det_C = np.trace(np.log(L), axis1=1, axis2=2)  # TODO: check
                half_log_det_Sigma = np.trace(np.log(L_Sigma), axis1=1, axis2=2)
                summation = np.einsum('ijk, ki->ij', C_inv, M)  # (K, N, N) @ (N, K) -> (K, N)
                summation = np.einsum('ij, ji-> ', summation, M)  # (K, N) (N, K) -> ()
                bound = (
                    - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                    + (N * K / 2) - (np.sum(np.trace(Sigma, axis1=1, axis2=2)) / 2)
                    - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma, axis1=1, axis2=2)) / 2)
                    - np.sum(half_log_det_C) + np.sum(half_log_det_Sigma)
                    + np.sum(np.log(calligraphic_Z))
                )
            else:
                # Case when Sigma is (N, N)
                C = C + 1e-8 * np.eye(N)
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_inv = L_inv.T @ L_inv  # (N, N) (N, N) -> (N, N)
                L_Sigma = np.linalg.cholesky(Sigma)
                half_log_det_C = np.trace(np.log(L))
                half_log_det_Sigma = np.trace(np.log(L_Sigma))
                summation = np.einsum('ik, k->i', C_inv, M)
                summation = np.dot(M, summation)
                one = - np.trace(Sigma) / 2
                two = - np.trace(C_inv @ Sigma) / 2
                three = - half_log_det_C
                four = half_log_det_Sigma
                five = np.sum(np.log(calligraphic_Z))
                print("one ", one)
                print("two ", two)
                print("three ", three)
                print("four ", four)
                print("five ", five)
                bound = (
                        - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * K / 2) - np.trace(Sigma) / 2
                        - (summation / 2) - (np.trace(C_inv @ Sigma) / 2)
                        - half_log_det_C + half_log_det_Sigma
                        + np.sum(np.log(calligraphic_Z))
                )
            print('bound = ', bound)
            return bound
        elif numerically_stable is False:
            C = C + 1e-4 * np.eye(N)
            C_inv = np.linalg.inv(C)
            M_T = M.T
            intermediate_vectors = np.empty((N, K))

            if self.kernel.general_kernel:
                for k in range(K):
                    intermediate_vectors[:, k] = C_inv[k] @ M_T[k]
                summation = np.sum(np.multiply(intermediate_vectors, M))
                # Case when Sigma is (K, N, N)
                bound = (
                        - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * K / 2) - (np.sum(np.trace(Sigma, axis1=1, axis2=2)) / 2)
                        - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma, axis1=1, axis2=2)) / 2)
                        - (np.sum(np.log(np.linalg.det(C_inv))) / 2) + (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                        + np.sum(np.log(calligraphic_Z))
                )
            else:
                # Case when Sigma is (N, N)
                summation = np.sum(np.multiply(M, C_inv @ M))
                one = - (np.sum(np.trace(Sigma)) / 2)
                two = - (np.sum(np.trace(C_inv @ Sigma)) / 2)
                three = - (np.sum(np.log(np.linalg.det(C))) / 2)
                four = (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                five = np.sum(np.log(calligraphic_Z))
                print("one ", one)
                print("two ", two)
                print("three ", three)
                print("four ", four)
                print("five ", five)
                bound = (
                        - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * K / 2) - (np.sum(np.trace(Sigma)) / 2)
                        - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma)) / 2)
                        - (np.sum(np.log(np.linalg.det(C))) / 2) + (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                        + np.sum(np.log(calligraphic_Z))
                )

            print('bound = ', bound)
            return bound


class VBMultinomialOrderedGP(Estimator):
    """
    A Variational Bayes classifier for ordered likelihood. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference. It is for the ordered likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :arg cutpoint: The (K +1, ) array of cutpoint parameters \bm{gamma}.
        :type cutpoint: :class:`numpy.ndarray`
        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma
        if self.kernel.ARD_kernel:
            raise ValueError('The kernel must not be ARD type (kernel.ARD_kernel=1),'
                             ' but ISO type (kernel.ARD_kernel=0). (got {}, expected)'.format(
                self.kernel.ARD_kernel, 0))
        if self.kernel.general_kernel:
            raise ValueError('The kernel must not be general type (kernel.general_kernel=1),'
                             ' but simple type (kernel.general_kernel=0). (got {}, expected)'.format(
                self.kernel.general_kernel, 0))

    def _estimate_initiate(self, m_0, gamma_0, varphi_0=None, psi_0=None):
        """
        Initialise the estimator.
        """
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        ys = []
        ms = []
        varphis = []
        psis = []
        bounds = []
        containers = (ms, ys, varphis, psis, bounds)
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
                        [gamma_0[0], gamma_0[-1]], [np.inf, np.NINF]))
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
        if not all(
                gamma_0[i] <= gamma_0[i + 1]
                for i in range(self.K)):
            raise CutpointValueError(gamma_0)

        return m_0, gamma_0, varphi_0, psi_0, containers

    def estimate(self, m_0, gamma_0, steps, varphi_0=None, psi_0=None,
                 first_step=1, fix_hyperparameters=False, write=False):
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
        m_tilde, gamma, varphi_tilde, psi_tilde, containers = self._estimate_initiate(
            m_0, gamma_0, varphi_0, psi_0)
        ms, ys, varphis, psis, bounds = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            y_tilde, calligraphic_Z = self._y_tilde(m_tilde, gamma)
            m_tilde = self._m_tilde(y_tilde, varphi_tilde)
            if not fix_hyperparameters:
                varphi_tilde = self._varphi_tilde(m_tilde, psi_tilde, n_samples=1000)
                psi_tilde = self._psi_tilde(varphi_tilde)
            if write:
                bound = self.variational_lower_bound(
                    self.N, self.K, m_tilde, self.Sigma_tilde, self.C_tilde, calligraphic_Z, numerically_stable=True)
                ms.append(m_tilde)
                ys.append(y_tilde)
                varphis.append(varphi_tilde)
                psis.append(psi_tilde)
                bounds.append(bound)
        containers = (ms, ys, varphis, psis, bounds)
        return gamma, m_tilde, self.Sigma_tilde, self.C_tilde, y_tilde, varphi_tilde, containers

    def _predict_vector(self, gamma, Sigma_tilde, y_tilde, varphi_tilde, X_test):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.
        :param Sigma_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
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
        predictive_distributions = np.empty((N_test, self.K))
        # TODO: vectorise
        for n in range(N_test):
            for k in range(self.K):
                gamma_k = gamma[k + 1]
                gamma_k_minus_1 = gamma[k]
                var = var_new_tilde[n]
                m_n = m_new_tilde[n]
                predictive_distributions[n, k] = (
                        norm.cdf((gamma_k - m_n) / var) - norm.cdf((gamma_k_minus_1 - m_n) / var)
                )
        return predictive_distributions  # (N_test, K)

    def predict(self, gamma, Sigma_tilde, Y_tilde, varphi_tilde, X_test, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :param Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel.ARD_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            raise ValueError('For the ordered likelihood estimator, the kernel must not be ARD type'
                             ' (kernel.ARD_kernel=1), but ISO type (kernel.ARD_kernel=0). (got {}, expected)'.format(
                self.kernel.ARD_kernel, 0))
        else:
            if vectorised:
                return self._predict_vector(gamma, Sigma_tilde, Y_tilde, varphi_tilde, X_test)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _varphi_tilde(self, m_tilde, psi_tilde, n_samples=10, vectorised=True):
        """
        Return the w values of the sample on 2005 Page 9 Eq.(7).

        :arg m_tilde: Posterior mean estimate of M_tilde.
        :arg psi_tilde: Posterior mean estimate of psi.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from
        # (n_samples, K, D) in general and ARD, (n_samples, ) for single shared kernel and ISO case. Depends on the
        # shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)  # (n_samples, )
        log_varphis = np.log(varphis)
        # (n_samples, K, N, N) in general and ARD, (n_samples, N, N) for single shared kernel and ISO case. Depends on
        # the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)  # (n_samples, N, N)
        Cs_samples = np.add(Cs_samples, 1e-5 * np.eye(self.N))
        if vectorised:
            log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(m_tilde, mean=None, covs=Cs_samples)
        else:
            log_ws = np.empty((n_samples,))
            # Scalar version
            for i in range(n_samples):
                log_ws[i] = unnormalised_log_multivariate_normal_pdf(m_tilde, mean=None, cov=Cs_samples[i])
            # Normalise the w vectors
        max_log_ws = np.max(log_ws)
        log_normalising_constant = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws), axis=0))
        log_ws = np.subtract(log_ws, log_normalising_constant)
        element_prod = np.add(log_varphis, log_ws)
        element_prod = np.exp(element_prod)
        magic_number = 2.0
        print("varphi_tilde", magic_number * np.sum(element_prod, axis=0))
        return magic_number * np.sum(element_prod, axis=0)

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
        self.C_tilde = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
        self.Sigma_tilde = np.linalg.inv(np.add(self.IN, self.C_tilde))  # (K, N, N)
        return self.C_tilde @ self.Sigma_tilde @ y_tilde  # (N, K)

    def _y_tilde(self, m_tilde, gamma):
        """
        Calculate Y_tilde elements 2021 Page 3 Eq.(11).

        :arg M_tilde: The posterior expectations for M (N, ).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, ) containing \tilde(y)_{n} values.
        """
        p, calligraphic_z = self._p(m_tilde, gamma)
        # Eq. (11)
        y_tilde = np.add(m_tilde, p)
        return y_tilde, calligraphic_z

    def _p(self, m_tilde, gamma):
        """
        Estimate the rightmost term of 2021 Page 3 Eq.(11), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg n_samples: The number of samples to take.
        """
        # TODO: check this maths is correct or if it is out by one error
        gamma_ks = gamma[self.t_train + 1]
        gamma_k_minus_1s = gamma[self.t_train]
        calligraphic_z = norm.cdf(gamma_ks - m_tilde) - norm.cdf(gamma_k_minus_1s - m_tilde)
        p = (norm.pdf(gamma_k_minus_1s - m_tilde) - norm.pdf(gamma_ks - m_tilde)) / calligraphic_z
        return p, calligraphic_z  # (N, ) (N, )

    def variational_lower_bound(self, N, K, M, Sigma, C, calligraphic_Z, numerically_stable=True):
        """
        Calculate the variational lower bound of the log marginal likelihood.

        :arg M_tilde:
        :arg Sigma_tilde:
        :arg C_tilde:
        :arg calligraphic_Z:
        :arg bool numerically_stable:
        """
        if numerically_stable is True:
            # Will always only have one GP, Sigma is (N, N)
            C = C + 1e-8 * np.eye(N)
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_inv = L_inv.T @ L_inv  # (N, N) (N, N) -> (N, N)
            L_Sigma = np.linalg.cholesky(Sigma)
            half_log_det_C = np.trace(np.log(L))
            half_log_det_Sigma = np.trace(np.log(L_Sigma))
            summation = np.einsum('ik, k->i', C_inv, M)
            summation = np.dot(M, summation)
            # one = - np.trace(Sigma) / 2
            # two = - np.trace(C_inv @ Sigma) / 2
            # three = - half_log_det_C
            # four = half_log_det_Sigma
            # five = np.sum(np.log(calligraphic_Z))
            # print("one ", one)
            # print("two ", two)
            # print("three ", three)
            # print("four ", four)
            # print("five ", five)
            bound = (
                    - (N * np.log(2 * np.pi) / 2)
                    + (N / 2) - np.trace(Sigma) / 2
                    - (summation / 2) - (np.trace(C_inv @ Sigma) / 2)
                    - half_log_det_C + half_log_det_Sigma
                    + np.sum(np.log(calligraphic_Z))
            )
        elif numerically_stable is False:
            # Will always only have one GP, Sigma is (N, N)
            C = C + 1e-4 * np.eye(N)
            C_inv = np.linalg.inv(C)
            summation = np.sum(np.multiply(M, C_inv @ M))
            # one = - (np.sum(np.trace(Sigma)) / 2)
            # two = - (np.sum(np.trace(C_inv @ Sigma)) / 2)
            # three = - (np.sum(np.log(np.linalg.det(C))) / 2)
            # four = (np.sum(np.log(np.linalg.det(Sigma))) / 2)
            # five = np.sum(np.log(calligraphic_Z))
            # print("one ", one)
            # print("two ", two)
            # print("three ", three)
            # print("four ", four)
            # print("five ", five)
            bound = (
                    - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                    + (N * K / 2) - (np.sum(np.trace(Sigma)) / 2)
                    - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma)) / 2)
                    - (np.sum(np.log(np.linalg.det(C))) / 2) + (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                    + np.sum(np.log(calligraphic_Z))
            )
        print('bound = ', bound)
        return bound


class VBMultinomialOrderedGPTemp(Estimator):
    """
    A Variational Bayes classifier for ordered likelihood. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference. It is for the ordered likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`Gibbs_GP` sampler object.

        :arg cutpoint: The (K +1, ) array of cutpoint parameters \bm{gamma}.
        :type cutpoint: :class:`numpy.ndarray`
        :returns: An :class:`Gibbs_GP` object.
        """
        super().__init__(*args, **kwargs)
        self.IN = np.eye(self.N)
        self.C = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.Sigma = np.linalg.inv(np.add(self.IN, self.C))
        self.cov = self.C @ self.Sigma
        if self.kernel.ARD_kernel:
            raise ValueError('The kernel must not be ARD type (kernel.ARD_kernel=1),'
                             ' but ISO type (kernel.ARD_kernel=0). (got {}, expected)'.format(
                self.kernel.ARD_kernel, 0))
        if self.kernel.general_kernel:
            raise ValueError('The kernel must not be general type (kernel.general_kernel=1),'
                             ' but simple type (kernel.general_kernel=0). (got {}, expected)'.format(
                self.kernel.general_kernel, 0))

    def _estimate_initiate(self, m_0, gamma_0, varphi_0=None, psi_0=None):
        """
        Initialise the estimator.
        """
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        ys = []
        ms = []
        varphis = []
        psis = []
        bounds = []
        containers = (ms, ys, varphis, psis, bounds)
        # Treat user parsing of cutpoint parameters with just the upper cutpoints for each class
        if np.shape(gamma_0)[0] == self.K - 1:  # not including any of the fixed cutpoints: -\infty, \infty
            gamma_0 = np.append(gamma_0, np.inf)  # append the infinity cutpoint
            gamma_0 = np.insert(gamma_0, np.NINF)  # insert the negative infinity cutpoint at index 0
            pass  # correct format
        elif np.shape(gamma_0)[0] == self.K:  # not including one of the infinity cutpoints
            if gamma_0[-1] != np.inf:
                if gamma_0[0] != np.NINF:
                    raise ValueError('The last cutpoint parameter must be numpy.inf, or the first cutpoint parameter'
                                     ' must be numpy.NINF (got {}, expected {})'.format(
                        [gamma_0[0], gamma_0[-1]], [np.inf, np.NINF]))
                else:  # gamma_0[0] is negative infinity
                    gamma_0.append(np.inf)
                    pass  # correct format
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
        # if not np.all(gamma_0[2:-1] > 0):
        #     raise ValueError('The cutpoint parameters must be positive. (got {})'.format(gamma_0))
        assert np.shape(gamma_0)[0] == self.K + 1
        if not all(
                gamma_0[i] <= gamma_0[i + 1]
                for i in range(self.K)):
            raise CutpointValueError(gamma_0)

        return m_0, gamma_0, varphi_0, psi_0, containers

    def estimate(self, m_0, gamma_0, steps, varphi_0=None, psi_0=None,
                 first_step=1, fix_hyperparameters=False, write=False):
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
        m_tilde, gamma, varphi_tilde, psi_tilde, containers = self._estimate_initiate(
            m_0, gamma_0, varphi_0, psi_0)
        ms, ys, varphis, psis, bounds = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples"):
            y_tilde, calligraphic_Z = self._y_tilde(m_tilde, gamma)
            m_tilde = self._m_tilde(y_tilde, varphi_tilde)
            if not fix_hyperparameters:
                varphi_tilde = self._varphi_tilde(m_tilde, psi_tilde, n_samples=1000)
                psi_tilde = self._psi_tilde(varphi_tilde)
            if write:
                bound = self.variational_lower_bound(
                    self.N, self.K, m_tilde, self.Sigma_tilde, self.C_tilde, calligraphic_Z, numerically_stable=True)
                ms.append(m_tilde)
                ys.append(y_tilde)
                varphis.append(varphi_tilde)
                psis.append(psi_tilde)
                bounds.append(bound)
        containers = (ms, ys, varphis, psis, bounds)
        bound = self.variational_lower_bound(
            self.N, self.K, m_tilde, self.Sigma_tilde, self.C_tilde, calligraphic_Z, numerically_stable=True)
        return gamma, m_tilde, self.Sigma_tilde, self.C_tilde, y_tilde, varphi_tilde, bound, containers

    def _predict_vector(self, gamma, Sigma_tilde, y_tilde, varphi_tilde, X_test):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.
        :param Sigma_tilde:
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde:
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
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
        predictive_distributions = np.empty((N_test, self.K))
        # TODO: vectorise
        for n in range(N_test):
            for k in range(self.K):
                gamma_k = gamma[k + 1]
                gamma_k_minus_1 = gamma[k]
                var = var_new_tilde[n]
                m_n = m_new_tilde[n]
                predictive_distributions[n, k] = (
                        norm.cdf((gamma_k - m_n) / var) - norm.cdf((gamma_k_minus_1 - m_n) / var)
                )
        return predictive_distributions  # (N_test, K)

    def predict(self, gamma, Sigma_tilde, Y_tilde, varphi_tilde, X_test, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :param Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :param Y_tilde: The posterior mean estimate of the latent variable Y.
        :param varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :param X_test: The new data points, array like (N_test, D).
        :param n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel.ARD_kernel:
            # This is the general case where there are hyper-parameters
            # varphi (K, D) for all dimensions and classes.
            raise ValueError('For the ordered likelihood estimator, the kernel must not be ARD type'
                             ' (kernel.ARD_kernel=1), but ISO type (kernel.ARD_kernel=0). (got {}, expected)'.format(
                self.kernel.ARD_kernel, 0))
        else:
            if vectorised:
                return self._predict_vector(gamma, Sigma_tilde, Y_tilde, varphi_tilde, X_test)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _varphi_tilde(self, m_tilde, psi_tilde, n_samples=10, vectorised=True):
        """
        Return the w values of the sample on 2005 Page 9 Eq.(7).

        :arg m_tilde: Posterior mean estimate of M_tilde.
        :arg psi_tilde: Posterior mean estimate of psi.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from
        # (n_samples, K, D) in general and ARD, (n_samples, ) for single shared kernel and ISO case. Depends on the
        # shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)  # (n_samples, )
        log_varphis = np.log(varphis)
        # (n_samples, K, N, N) in general and ARD, (n_samples, N, N) for single shared kernel and ISO case. Depends on
        # the shape of psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)  # (n_samples, N, N)
        Cs_samples = np.add(Cs_samples, 1e-5 * np.eye(self.N))
        if vectorised:
            log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(m_tilde, mean=None, covs=Cs_samples)
        else:
            log_ws = np.empty((n_samples,))
            # Scalar version
            for i in range(n_samples):
                log_ws[i] = unnormalised_log_multivariate_normal_pdf(m_tilde, mean=None, cov=Cs_samples[i])
            # Normalise the w vectors
        max_log_ws = np.max(log_ws)
        log_normalising_constant = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws), axis=0))
        log_ws = np.subtract(log_ws, log_normalising_constant)
        element_prod = np.add(log_varphis, log_ws)
        element_prod = np.exp(element_prod)
        magic_number = 2.0
        print("varphi_tilde", magic_number * np.sum(element_prod, axis=0))
        return magic_number * np.sum(element_prod, axis=0)

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
        self.C_tilde = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (K, N, N)
        self.Sigma_tilde = np.linalg.inv(np.add(self.IN, self.C_tilde))  # (K, N, N)
        return self.C_tilde @ self.Sigma_tilde @ y_tilde  # (N, K)

    def _y_tilde(self, m_tilde, gamma):
        """
        Calculate Y_tilde elements 2021 Page 3 Eq.(11).

        :arg M_tilde: The posterior expectations for M (N, ).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, ) containing \tilde(y)_{n} values.
        """
        p, calligraphic_z = self._p(m_tilde, gamma)
        # Eq. (11)
        y_tilde = np.add(m_tilde, p)
        return y_tilde, calligraphic_z

    def _p(self, m_tilde, gamma):
        """
        Estimate the rightmost term of 2021 Page 3 Eq.(11), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg M_tilde: The posterior expectations for M (N, K).
        :arg n_samples: The number of samples to take.
        """
        # TODO: check this maths is correct or if it is out by one error
        gamma_ks = gamma[self.t_train + 1]
        gamma_k_minus_1s = gamma[self.t_train]
        calligraphic_z = norm.cdf(gamma_ks - m_tilde) - norm.cdf(gamma_k_minus_1s - m_tilde)
        p = (norm.pdf(gamma_k_minus_1s - m_tilde) - norm.pdf(gamma_ks - m_tilde)) / calligraphic_z
        return p, calligraphic_z  # (N, ) (N, )

    def variational_lower_bound(self, N, K, M, Sigma, C, calligraphic_Z, numerically_stable=True):
        """
        Calculate the variational lower bound of the log marginal likelihood.

        :arg M_tilde:
        :arg Sigma_tilde:
        :arg C_tilde:
        :arg calligraphic_Z:
        :arg bool numerically_stable:
        """
        if numerically_stable is True:
            # Will always only have one GP, Sigma is (N, N)
            C = C + 1e-8 * np.eye(N)
            L = np.linalg.cholesky(C)
            L_inv = np.linalg.inv(L)
            C_inv = L_inv.T @ L_inv  # (N, N) (N, N) -> (N, N)
            L_Sigma = np.linalg.cholesky(Sigma)
            half_log_det_C = np.trace(np.log(L))
            half_log_det_Sigma = np.trace(np.log(L_Sigma))
            summation = np.einsum('ik, k->i', C_inv, M)
            summation = np.dot(M, summation)
            # one = - np.trace(Sigma) / 2
            # two = - np.trace(C_inv @ Sigma) / 2
            # three = - half_log_det_C
            # four = half_log_det_Sigma
            # five = np.sum(np.log(calligraphic_Z))
            # print("one ", one)
            # print("two ", two)
            # print("three ", three)
            # print("four ", four)
            # print("five ", five)
            bound = (
                    - (N * np.log(2 * np.pi) / 2)
                    + (N / 2) - np.trace(Sigma) / 2
                    - (summation / 2) - (np.trace(C_inv @ Sigma) / 2)
                    - half_log_det_C + half_log_det_Sigma
                    + np.sum(np.log(calligraphic_Z))
            )
        elif numerically_stable is False:
            # Will always only have one GP, Sigma is (N, N)
            C = C + 1e-4 * np.eye(N)
            C_inv = np.linalg.inv(C)
            summation = np.sum(np.multiply(M, C_inv @ M))
            # one = - (np.sum(np.trace(Sigma)) / 2)
            # two = - (np.sum(np.trace(C_inv @ Sigma)) / 2)
            # three = - (np.sum(np.log(np.linalg.det(C))) / 2)
            # four = (np.sum(np.log(np.linalg.det(Sigma))) / 2)
            # five = np.sum(np.log(calligraphic_Z))
            # print("one ", one)
            # print("two ", two)
            # print("three ", three)
            # print("four ", four)
            # print("five ", five)
            bound = (
                    - (N * K * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                    + (N * K / 2) - (np.sum(np.trace(Sigma)) / 2)
                    - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma)) / 2)
                    - (np.sum(np.log(np.linalg.det(C))) / 2) + (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                    + np.sum(np.log(calligraphic_Z))
            )
        print('bound = ', bound)
        return bound


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