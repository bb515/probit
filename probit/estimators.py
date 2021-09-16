from abc import ABC, abstractmethod
from operator import pos
from .kernels import Kernel, InvalidKernel
import pathlib
import numpy as np
import random
from scipy.stats import norm, multivariate_normal
from tqdm import trange
import warnings
import math
import matplotlib.pyplot as plt
from .utilities import (
    sample_Us, sample_varphis,
    matrix_of_differencess, matrix_of_valuess, matrix_of_VB_differencess, matrix_of_VB_differences,
    unnormalised_log_multivariate_normal_pdf, vectorised_unnormalised_log_multivariate_normal_pdf,
    vectorised_multiclass_unnormalised_log_multivariate_normal_pdf,
    fromb_t1, fromb_t2, fromb_t3, fromb_t4, fromb_t5,
    fromb_t1_vector, fromb_t2_vector, fromb_t3_vector, fromb_t4_vector,
    fromb_t5_vector)


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
    def __init__(self, X_train, t_train, kernel, J, write_path=None):
        """
        Create an :class:`Sampler` object.

        This method should be implemented in every concrete Estimator.

        :arg X_train: (N, D) The data vector.
        :type X_train: :class:`numpy.ndarray`
        :arg t_train: (N, ) The target vector.
        :type t_train: :class:`numpy.ndarray`
        :arg kernel: The kernel to use, see :mod:`probit.kernels` for options.
        :arg J: The number of (ordinal) classes.
        :arg str write_path: Write path for outputs.

        :returns: A :class:`Estimator` object
        """
        if not (isinstance(kernel, Kernel)):
            raise InvalidKernel(kernel)
        else:
            self.kernel = kernel
        if write_path is None:
            self.write_path = pathlib.Path()
        else:
            self.write_path = pathlib.Path(write_path)
        self.D = np.shape(X_train)[1]
        self.N = np.shape(X_train)[0]
        self.X_train = X_train
        self.X_train_T = X_train.T
        # 20/07 removed self.K here since it is not general. It changes during each initialisation.
        self.grid = np.ogrid[0:self.N]
        if np.all(np.mod(t_train, 1) == 0):
            t_train = t_train.astype(int)
        else:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        if t_train.dtype != int:
            raise ValueError("t must contain only integer values (got {})".format(t_train))
        else:
            self.t_train = t_train
        self.J = J
        # Needed for vectorised functions
        self.grid = np.ogrid[0:self.N]  # All the indices for indexing sets of self.t_train
        self.where_t_0 = np.where(self.t_train==0)[0]
        self.where_t_Jminus1 = np.where(self.t_train==self.J-1)[0]
        self.where_t_neither = np.setxor1d(self.grid, self.where_t_0)
        self.where_t_neither = np.setxor1d(
            self.where_t_neither, self.where_t_Jminus1)
        self.where_t_not0 = np.concatenate(
            (self.where_t_neither, self.where_t_Jminus1))
        self.where_t_notJminus1 = np.concatenate(
            (self.where_t_0, self.where_t_neither))
        if self.kernel._ARD:
            sigma = np.reshape(self.kernel.sigma, (self.J, 1))
            tau = np.reshape(self.kernel.tau, (self.J, 1))
            self.sigma = np.tile(sigma, (1, self.D))  # (J, D)
            self.tau = np.tile(tau, (1, self.D))  # (J, D)
        else:
            self.sigma = self.kernel.sigma
            self.tau = self.kernel.tau
        self.jitter = 1e-8  # default jitter (See GPML by Williams et al. for a good explanation of jitter)
        self.upper_bound = 6.0
        self.upper_bound2 = 18.0

    @abstractmethod
    def _estimate_initiate(self):
        """
        Initialise the Estimator.

        This method should be implemented in every concrete Estimator.
        """

    @abstractmethod
    def estimate(self):
        """
        Return the samples

        This method should be implemented in every concrete Estimator.
        """

    @abstractmethod
    def predict(self):
        """
        Return the samples

        This method should be implemented in every concrete Estimator.
        """

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=3000, vectorised=True, numerical_stability=True):
        """
        Return the posterior mean estimate of the hyperparameters varphi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(9).

        This is the same for all multinomial categorical estimators, and so can live in the Abstract Base Class.

        :arg M_tilde: Posterior mean estimate of M_tilde.
        :type M_tilde: :class:`numpy.ndarray`
        :arg psi_tilde: Posterior mean estimate of psi.
        :type psi_tilde: :class:`numpy.ndarray`
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in Girolami and
            Rogers Page 13 Default 3000.
        :arg bool vectorised: If `True` then it will do calculations in a more efficient, vectorised way using
            `numpy.einsum()`, if `False` then it will do a scalar version of the calculations. Default `True`.
        :arg bool numerical_stability: If `True` then it will do calculations using log-sum-exp trick, c.f.
            https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ since normalising vectors of log likelihoods can
            result in under- or overflow. log-sum-exp trick resolves this issue. Default `True`.
        :return: The posterior mean estimate of the hyperparameters varphi Girolami and Rogers Page 9 Eq.(9).
        """
        # Vector draw from varphi
        # (n_samples, J, D) in general and ARD, (n_samples, ) for single shared kernel and ISO case. Depends on the
        # shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)  # Note that this resets the kernel varphi
        log_varphis = np.log(varphis)
        # (n_samples, J, N, N) in Multinomial general and ARD, (n_samples, N, N) single shared kernel and
        # ISO case, which will depend on the shape of the hyperhyperparameters psi_tilde.
        Cs_samples = self.kernel.kernel_matrices(self.X_train, self.X_train, varphis)
        # Nugget regularisation for numerical stability. 1e-5 or 1e-6 typically used - important to keep the same
        Cs_samples = np.add(Cs_samples, self.jitter * np.eye(self.N))
        if numerical_stability is True:
            if self.kernel._general and self.kernel._ARD:
                if vectorised is True:
                    log_ws = vectorised_multiclass_unnormalised_log_multivariate_normal_pdf(
                        M_tilde, mean=None, covs=Cs_samples)
                elif vectorised is False:
                    M_tilde_T = np.transpose(M_tilde)  # (J, N)
                    log_ws = np.empty((n_samples, self.J))
                    for i in range(n_samples):
                            for i in range(self.J):
                                # This fails if M_tilde and varphi are not initialised correctly
                                # Add sample to the unnormalised w vectors
                                log_ws[i, k] = unnormalised_log_multivariate_normal_pdf(
                                    M_tilde_T[k], mean=None, cov=Cs_samples[i, k])
                log_ws = np.reshape(log_ws, (n_samples, self.J, 1))
            else:
                # TODO: Does it matter I only use a single class to evaluate M_tilde_T[0]?
                if vectorised is True:
                    log_ws = vectorised_unnormalised_log_multivariate_normal_pdf(
                        M_tilde[:, 0], mean=None, covs=Cs_samples)
                elif vectorised is False:
                    log_ws = np.empty((n_samples, ))
                    for i in range(n_samples):
                        # This fails if M_tilde and varphi are not initialised correctly
                        # Add sample to the unnormalised w vectors
                        log_ws[i] = unnormalised_log_multivariate_normal_pdf(
                            M_tilde[:, 0], mean=None, cov=Cs_samples[i])
            # Normalise the w vectors in a numerically stable fashion  #TODO: Instead, use scipy's implementation
            max_log_ws = np.max(log_ws, axis=0)  # (J, )  or ()
            log_normalising_constant = max_log_ws + np.log(np.sum(np.exp(log_ws - max_log_ws), axis=0))
            log_ws = np.subtract(log_ws, log_normalising_constant)  # (n_samples, J) or (n_samples,)
            element_prod = np.add(log_ws, log_varphis)
            element_prod = np.exp(element_prod)
            magic_number = self.D  # Also could be self.J?
            return magic_number * np.sum(element_prod, axis=0)
        elif numerical_stability is False:
            if self.kernel._general and self.kernel._ARD:
                ws = np.empty((n_samples, self.J))
                for i in range(n_samples):
                    for j in range(self.J):
                        # Add sample to the unnormalised w vectors
                        ws[i, k] = multivariate_normal.pdf(M_tilde[:, k], mean=None, cov=Cs_samples[i, k])
                ws = np.reshape(ws, (n_samples, self.J, 1))
            else:
                ws = np.empty((n_samples,))
                for i in range(n_samples):
                    ws[i] = multivariate_normal.pdf(M_tilde[:, 0], mean=None, cov=Cs_samples[i])
            normalising_constant = np.sum(ws, axis=0)  # (J, 1) or ()
            ws = np.divide(ws, normalising_constant)  # (n_samples, J, 1) or (n_samples,)
            # (n_samples, J, 1) * (n_samples, J, D) or (n_samples,) * (n_samples,)
            element_prod = np.multiply(ws, varphis)
            return np.sum(element_prod, axis=0)

    def _psi_tilde(self, varphi_tilde):
        """
        Return the posterior mean estimate of the hyperhyperparameters psi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(10).

        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :arg varphi_tilde: Posterior mean estimate of varphi.
        :type varphi_tilde: :class:`numpy.ndarray`
        :return: The posterior mean estimate of the hyperhyperparameters psi Girolami and Rogers Page 9 Eq.(10).
        """
        return np.divide(np.add(1, self.sigma), np.add(self.tau, varphi_tilde))

    def _vector_expectation_wrt_u(self, M_new_tilde, var_new_tilde, n_samples):
        """
        Calculate distribution over classes for M_new_tilde at the same time.

        :arg M_new_tilde: An (N_test, J) array filled with \tilde{m}_k^{new_i} where
            k is the class indicator and i is the index of the test object.
        :type M_new_tilde: :class:`numpy.ndarray`
        :arg var_new_tilde: An (N_test, J) array filled with \tilde{sigma}_k^{new_i} where
           k is the class indicator and i is the index of the test object.
        :arg int n_samples: Number of samples to take in the monte carlo estimate.

        :returns: Distribution over classes
        """
        nu_new_tilde = np.sqrt(np.add(1, var_new_tilde))  # (N_test, J)
        N_test = np.shape(M_new_tilde)[0]
        # Find antisymmetric matrix of differences
        differences = matrix_of_differencess(M_new_tilde, self.J, N_test)  # (N_test, J, J) we will product across axis 2 (rows)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N_test, J, J)
        differencess = np.moveaxis(differencess, 1, 0)  # (N_test, n_samples, J, J)
        # Assume its okay to use the same random variables over all N_test data points
        Us = sample_Us(self.J, n_samples, different_across_classes=True)  # (n_samples, J, J)
        # Multiply nu by u
        nu_new_tildes = matrix_of_valuess(nu_new_tilde, self.J, N_test)  # (N_test, J, J)
        nu_new_tildess = np.tile(nu_new_tildes, (n_samples, 1, 1, 1))  # (n_samples, N_test, J, J)
        nu_new_tildess = np.moveaxis(nu_new_tildess, 1, 0)  # (N_test, n_samples, J, J)
        # # Find the transpose (for the product across classes)
        # nu_new_tilde_Ts = nu_new_tildes.transpose((0, 2, 1))  # (N_test, J, J)
        # Find the transpose (for the product across classes)
        nu_new_tilde_Tss = nu_new_tildess.transpose((0, 1, 3, 2))  # (N_test, n_samples, J, J)
        Us_nu_new_tilde_Ts = np.multiply(Us, nu_new_tildess)  # TODO: do we actually need to use transpose here?
        random_variables = np.add(Us_nu_new_tilde_Ts, differencess)  # (N_test, n_samples, J, J)
        random_variables = np.divide(random_variables, nu_new_tilde_Tss)  # TODO: do we actually need to use transpose here?
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)
        log_cum_dists = np.log(cum_dists)
        # Fill diagonals with 0
        log_cum_dists[:, :, range(self.J), range(self.J)] = 0
        # axis 0 is the N_test objects,
        # axis 1 is the n_samples samples, axis 3 is then the row index, which is the product of cdfs of interest
        log_samples = np.sum(log_cum_dists, axis=3)
        samples = np.exp(log_samples)
        # axis 1 is the n_samples samples, which is the monte-carlo sum of interest
        return 1. / n_samples * np.sum(samples, axis=1)  # (N_test, J)

    def _predict_vector_general(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Make variational Bayes prediction over classes of X_test.

        This is the general case where there are hyperparameters varphi (J, D) for all dimensions and classes.
        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :arg Sigma_tilde:
        :arg C_tilde:
        :arg Y_tilde: The posterior mean estimate of the latent variable Y (N, J).
        :arg varphi_tilde:
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi  # TODO: test this.
        self.kernel.varphi = varphi_tilde
        # Cs_news[:, i] is Cs_new for X_test[i]
        Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (J, N, N_test)
        # TODO: this is a bottleneck - write a specialised kernel function - wonder how GPFlow does it
        covariances_new = self.kernel.kernel_matrix(X_test, X_test)
        cs_news = [np.diag(covariances_new[k]) for j in range(self.J)]  # (J, N_test)
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = Sigma_tilde @ Cs_news  # (J, N, N_test)
        intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (J, N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (J, N_test)
        # Calculate M_tilde_new
        # TODO: could just use Y_tilde instead of Y_tilde_T? It would be better just to store Y in memory as (J, N)
        Y_tilde_T = np.reshape(Y_tilde.T, (self.J, self.N, 1))
        M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_tilde_T)  # (J, N_test, 1)?
        M_new_tilde_T = np.reshape(M_new_tilde_T, (self.J, N_test))  # (J, N_test)
        ## This was tested to be the slower option because matmul is an optimised sum and multiply.
        # M_new_tilde_T_2 = np.sum(np.multiply(Y_tilde_T, intermediate_vectors), axis=1)  # (J, N_test)
        # assert np.allclose(M_new_tilde_T, M_new_tilde_T_2)
        var_new_tilde_T = np.subtract(cs_news, intermediate_scalars)
        return self._vector_expectation_wrt_u(M_new_tilde_T.T, var_new_tilde_T.T, n_samples)

    def _predict_vector(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000):
        """
        Return posterior predictive prediction over classes of X_test.

        This is the same for all categorical estimators, and so can live in the Abstract Base Class.

        :arg Sigma_tilde:
        :arg Y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde:
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
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
        # TODO: Generalises to (J, N, N)?
        intermediate_vectors_T = np.transpose(intermediate_vectors)  # (N_test, N)
        intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(0)  # (N_test, )
        # Calculate M_tilde_new # TODO: test this.
        # Y_tilde_T = np.reshape(Y_tilde.T, (self.J, self.N, 1))
        M_new_tilde = np.matmul(intermediate_vectors_T, Y_tilde)  # (N_test, J)
        var_new_tilde = np.subtract(cs_news, intermediate_scalars)  # (N_test, )
        var_new_tilde = np.reshape(var_new_tilde, (N_test, 1))  # TODO: do in place shape changes - quicker(?) and
        # TODO: less memory - take a look at the binary case as that has in place shape changes.
        var_new_tilde = np.tile(var_new_tilde, (1, self.J))  # (N_test, J)
        return self._vector_expectation_wrt_u(M_new_tilde, var_new_tilde, n_samples)

    def _g(self, x):
        """
        Polynomial part of a series expansion for log survival function for a normal random variable. With the third
        term, for x>4, this is accurate to three decimal places.
        The third term becomes significant when sigma is large. 
        """
        return -1. / x**2 + 5/ (2 * x**4) - 37 / (3 *  x**6)

    def _calligraphic_Z_tails(self, z1, z2):
        """Series expansion at infinity. Even for z1, z2 >= 4 this is accurate to three decimal places."""
        return 1/np.sqrt(2 * np.pi) * (
        1 / z1 * np.exp(-0.5 * z1**2 + self._g(z1)) - 1 / z2 * np.exp(-0.5 * z2**2 + self._g(z2)))

    def _calligraphic_Z_far_tails(self, z):
        """Prevents overflow at large z."""
        return 1 / (z * np.sqrt(2 * np.pi)) * np.exp(-0.5 * z**2 + self._g(z))

    def _objective_gradient_initiate(self, gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients):
        """
        Evaluate container for gradient of the objective function.

        :arg gamma_0:
        :type gamma_0:
        :arg varphi_0:
        :type varphi_0:
        :arg noise_variance_0:
        :type noise_variance_0:
        :arg scale_0:
        :type scale_0:
        :arg bool calculate_all_gradients:
        """
        # Initiate or reset gx
        if self.kernel._general and self.kernel._ARD:
            raise ValueError("TODO")
        # noise_var, b_1, [\Delta^1, \Delta^2, ..., \Delta^(J-2)], scale, kernel specific hyperparameters
        gx_0 = np.zeros((1 + 1 + (self.J - 2) + 1 + self.kernel.num_hyperparameters,))
        if calculate_all_gradients is True:
            # Optimize all hyperparameters
            gx_0[:] = 1
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is None
                and scale_0 is not None):
            # Optimize only varphi and noise variance
            gx_0[0] = 1
            gx_0[-self.kernel.num_hyperparameters:] = 1
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is not None
                and scale_0 is None):
            # Optimize only varphi and scale
            gx_0[self.J] = 1
            gx_0[-self.kernel.num_hyperparameters:] = 1
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is None
                and scale_0 is not None):
            # Optimize only varphi
            gx_0[-self.kernel.num_hyperparameters:] = 1
        elif (gamma_0 is not None
                and noise_variance_0 is None
                and varphi_0 is not None
                and scale_0 is not None):
            # Optimize only noise variance
            gx_0[0] = 1
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is None):
            # Optimize only scale
            gx_0[self.J] = 1
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is not None):
            # Optimize only first two threshold parameters
            gx_0[1] = 1
            gx_0[2] = 1
        indices = np.where(gx_0 != 0)
        return gx_0, indices

    def _grid_over_hyperparameters_initiate(
        self, gamma_0, varphi_0, noise_variance_0, scale_0, res, range_x1, range_x2, J, calculate_all_gradients):
        """
        Initiate metadata and hyperparameters for plotting the objective function surface over hyperparameters.

        :arg gamma_0:
        :type gamma_0:
        :arg varphi_0:
        :type varphi_0:
        :arg noise_variance_0:
        :type noise_variance_0:
        :arg scale_0:
        :type scale_0:
        :arg int res:
        :arg range_x1:
        :type range_x1:
        :arg range_x2:
        :type range_x2:
        :arg int J:
        """
        # Infer the hyperparameter space to grid over
        if (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is None
                and scale_0 is not None
                ):
            # Grid over log varphi and log noise_std
            xlabel = r"$\sigma$"
            ylabel = r"$\varphi$"
            xscale = "log"
            yscale = "log"
            x1s = np.logspace(range_x1[0], range_x1[1], res)
            x2s = np.logspace(range_x2[0], range_x2[1], res)
            xx, yy = np.meshgrid(x1s, x2s)
            Phi_new = np.dstack((xx, yy))
            Phi_new = Phi_new.reshape((len(x1s) * len(x2s), 2))
            fxs = np.empty(len(Phi_new))
            gxs = np.empty((len(Phi_new), 2))
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is not None
                and scale_0 is None):
            # Grid over log varphi and log scale
            xlabel = r"$s$"
            ylabel = r"$\varphi$"
            xscale = "log"
            yscale = "log"
            x1s = np.logspace(range_x1[0], range_x1[1], res)
            x2s = np.logspace(range_x2[0], range_x2[1], res)
            xx, yy = np.meshgrid(x1s, x2s)
            Phi_new = np.dstack((xx, yy))
            Phi_new = Phi_new.reshape((len(x1s) * len(x2s), 2))
            fxs = np.empty(len(Phi_new))
            gxs = np.empty((len(Phi_new), 2))
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is None
                and scale_0 is not None):
            # Grid over log varphi only
            xlabel = r"$\varphi$"
            ylabel = None
            xscale = "log"
            yscale = None
            x1s = np.logspace(range_x1[0], range_x1[1], res)
            x2s = None
            xx = None
            yy = None
            Phi_new = x1s
            fxs = np.empty(len(Phi_new))
            gxs = np.empty(len(Phi_new))
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is not None):
            # Grid first two gamma variables
            xlabel = r"$\gamma_{1}$"
            ylabel = r"$\gamma_{2} - \gamma{1}$"
            xscale = "linear"
            yscale = "log"
            x1s = np.linspace(range_x1[0], range_x1[1], res)
            x2s = np.logspace(range_x2[0], range_x2[1], res)
            xx, yy = np.meshgrid(x1s, x2s)
            Phi_new = np.dstack((xx, yy))
            Phi_new = Phi_new.reshape((len(x1s) * len(x2s), 2))
            fxs = np.empty(len(Phi_new))
            gxs = np.empty((len(Phi_new),2))
        elif (gamma_0 is not None
                and noise_variance_0 is None
                and varphi_0 is not None
                and scale_0 is not None):
            # Grid over log noise_std only
            xlabel = r"$\sigma$"
            ylabel = None
            xscale = "log"
            yscale = None
            x1s = np.logspace(range_x1[0], range_x1[1], res)
            x2s = None
            xx = None
            yy = None
            Phi_new = x1s
            fxs = np.empty(len(Phi_new))
            gxs = np.empty(len(Phi_new))
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is None):
            # Grid over log scale only
            xlabel = r"$s$"
            ylabel = None
            xscale = "log"
            yscale = None
            x1s = np.logspace(range_x1[0], range_x1[1], res)
            x2s = None
            xx = None
            yy = None
            Phi_new = x1s
            fxs = np.empty(len(Phi_new))
            gxs = np.empty(len(Phi_new))
        else:
            raise ValueError(
                "Could not determine what objective surface you wanted to plot from the input arguments"
                " (got gamma_0={}, varphi_0={}, noise_variance_0={}, scale_0={})".format(
                gamma_0, varphi_0, noise_variance_0, scale_0))
        intervals = gamma_0[2:J] - gamma_0[1:J - 1]
        # Initiate gradient container as boolean array with True assigned to those indices we calculate gradients for
        gx_0, indices = self._objective_gradient_initiate(gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients)
        return (
            x1s, x2s, xlabel, ylabel, xscale, yscale, xx, yy,
            Phi_new, fxs, gxs, intervals, gx_0, indices)

    def _grid_over_hyperparameters_update(self, phi, gamma_0, varphi_0, noise_variance_0, scale_0):
        """
        Update the hyperparameters, phi.

        :arg kernel:
        :type kernel:
        :arg phi: The updated values of the hyperparameters.
        :type phi:
        """
        if (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is None
                and scale_0 is not None):
            noise_std = phi[0]
            noise_variance = noise_std**2
            varphi = phi[1]
            # Update kernel parameters, update prior and posterior covariance
            self._hyperparameters_update(varphi=varphi, noise_variance=noise_variance)
            gamma = gamma_0
            scale = scale_0
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is not None
                and scale_0 is None):
            scale = phi[0]
            varphi = phi[1]
            # Update kernel parameters, update prior and posterior covariance
            self._hyperparameters_update(varphi=varphi, scale=scale)
            noise_variance = noise_variance_0
            noise_std = np.sqrt(noise_variance)
            gamma = gamma_0
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is None
                and scale_0 is not None):
            varphi = phi
            # Update kernel parameters,update prior and posterior covariance - noise_variance stays the same
            self._hyperparameters_update(varphi=varphi)
            gamma = gamma_0
            noise_variance = noise_variance_0
            noise_std = np.sqrt(noise_variance)
            scale = scale_0
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is not None):
            gamma = gamma_0
            gamma[1] = phi[0]
            gamma[2] = phi[1] + phi[0]
            # No update of prior and posterior covariance
            noise_variance = noise_variance_0
            noise_std = np.sqrt(noise_variance)
            varphi = varphi_0
            scale = scale_0
        elif (gamma_0 is not None
                and noise_variance_0 is None
                and varphi_0 is not None
                and scale_0 is not None):
            noise_std = phi
            noise_variance = noise_std**2
            # Update posterior covariance - varphi stays the same
            self._hyperparameters_update(noise_variance=noise_variance)
            varphi = varphi_0
            scale = scale_0
            gamma = gamma_0
            noise_std = np.sqrt(noise_variance)
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is None):
            scale = phi
            # Update posterior covariance - varphi stays the same
            self._hyperparameters_update(scale=scale)
            gamma = gamma_0
            noise_variance = noise_variance_0
            varphi = varphi_0
            noise_std = np.sqrt(noise_variance)
        return gamma, varphi, noise_variance, noise_std, scale

    def _update_prior(self):
        """Update prior covariances."""
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        self.partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(self.X_train, self.X_train)
        self.partial_K_scale = self.kernel.kernel_partial_derivative_scale(self.X_train, self.X_train)

    def _calligraphic_Z(self, gamma, noise_std, m, upper_bound=None, upper_bound2=None, verbose=False):
        """
        Return the normalising constants for the truncated normal distribution in a numerically stable manner.

        :arg gamma: The cutpoints.
        :type gamma: :class:`numpy.array`
        :arg float noise_std: The noise standard deviation.
        :arg m: The mean vector.
        :type m: :class:`numpy.ndarray`
        :arg float upper_bound: The threshold of the normal z value for which the pdf is close enough to zero.
        :arg float upper_bound2: The threshold of the normal z value for which the pdf is close enough to zero. 
        :arg bool numerical_stability: If set to true, will calculate in a numerically stable way. If set to false,
            will calculate in a faster, but less numerically stable way.
        :returns: (
            calligraphic_Z,
            norm_pdf_z1s, norm_pdf_z2s,
            norm_cdf_z1s, norm_cdf_z2s,
            gamma_1s, gamma_2s,
            z1s, z2s)
        :rtype: tuple (
            :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, :class:`numpy.ndarray`)
        """
        gamma_1s = gamma[self.t_train]
        gamma_2s = gamma[self.t_train + 1]
        # Otherwise
        z1s = (gamma_1s - m) / noise_std
        z2s = (gamma_2s - m) / noise_std
        norm_pdf_z1s = norm.pdf(z1s)
        norm_pdf_z2s = norm.pdf(z2s)
        norm_cdf_z1s = norm.cdf(z1s)
        norm_cdf_z2s = norm.cdf(z2s)
        calligraphic_Z = norm_cdf_z2s - norm_cdf_z1s
        if upper_bound is not None:
            # Using series expansion approximations
            indices1 = np.where(z1s > upper_bound)
            indices2 = np.where(z2s < -upper_bound)
            indices = np.union1d(indices1, indices2)
            z1_indices = z1s[indices]
            z2_indices = z2s[indices]
            calligraphic_Z[indices] = self._calligraphic_Z_tails(z1_indices, z2_indices)
            if upper_bound2 is not None:
                indices = np.where(z1s > upper_bound2)
                z1_indices = z1s[indices]
                calligraphic_Z[indices] = self._calligraphic_Z_far_tails(z1_indices)
                indices = np.where(z2s < -upper_bound2)
                z2_indices = z2s[indices]
                calligraphic_Z[indices] = self._calligraphic_Z_far_tails(-z2_indices)
        if verbose is True:
            number_small_densities = len(calligraphic_Z[calligraphic_Z < self.EPS])
            if number_small_densities != 0:
                warnings.warn(
                    "calligraphic_Z (normalising constants for truncated normal) "
                    "must be greater than tolerance={} (got {}): SETTING to Z_ns[Z_ns<tolerance]=tolerance\n"
                            "z1s={}, z2s={}".format(
                    self.EPS, calligraphic_Z, z1s, z2s))
                for i, value in enumerate(calligraphic_Z):
                    if value < 0.01:
                        print("call_Z={}, z1 = {}, z2 = {}".format(value, z1s[i], z2s[i]))
        return calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, norm_cdf_z1s, norm_cdf_z2s, gamma_1s, gamma_2s


class VBBinomialGP(Estimator):
    """
    # TODO put in noise_variance
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
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        if self.J != 2:
            raise ValueError("t_train must only contain +1 or -1 class labels, got {}".format(self.t_train))
        # TODO: swap this notation
        self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))
        self.cov = self.K @ self.Sigma

    def _estimate_initiate(self, M_0, varphi_0=None, psi_0=None):
        """
        Initialise the Estimator.

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

        :arg M_0: (N, J) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`.
        :arg int steps: The number of steps in the Estimator.
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
        return M_tilde, self.Sigma, self.K, Y_tilde, varphi_tilde, psi_tilde

    def _predict_vector(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test):
        """
        Make prediction over binary classes of X_test.

        :arg Sigma_tilde:
        :arg C_tilde:
        :arg Y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde:
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
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

        :arg Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :arg Y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :return: A Monte Carlo estimate of the class probabilities.
        """
        return self._predict_vector(Sigma_tilde, Y_tilde, varphi_tilde, X_test)

    def _varphi_tilde(self, M_tilde, psi_tilde, n_samples=1000, vectorised=True, numerical_stability=True):
        """
        Return the posterior mean estimate of the hyperparameters varphi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(9).

        This is different from the corresponding multinomial categorical estimator function in the Abstract Base Class.

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in Girolami and
            Rogers Page 13 Default 3000.
        :arg bool vectorised: If `True` then it will do calculations in a more efficient, vectorised way using
            `numpy.einsum()`, if `False` then it will do a scalar version of the calculations. Default `True`.
        :arg bool numerical_stability: If `True` then it will do calculations using log-sum-exp trick, c.f.
            https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/ since normalising vectors of log likelihoods can
            result in under- or overflow. log-sum-exp trick resolves this issue. Default `True`.
        :return: The posterior mean estimate of the hyperparameters varphi in Girolami and Rogers Page 9 Eq.(9).
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
        Return the posterior mean estimate of the hyperparameters varphi.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(9).

        This is different from the corresponding multinomial categorical estimator function in the Abstract Base Class.

        :arg psi_tilde: Posterior mean estimate of psi.
        :arg M_tilde: Posterior mean estimate of M_tilde.
        :arg int n_samples: The int number of samples for the importance sampling estimate, 500 is used in Girolami and
         Rogers Page 13. Default 3000.
        :return: The posterior mean estimate of the hyperparameters varphi Girolami and Rogers Page 9 Eq.(9).
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

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 10 Eq.(11)

        :arg Y_tilde: (N,) array
        :type Y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type Y_tilde: :class:`np.ndarray`
        """
        # Update the varphi with new values
        self.kernel.varphi = varphi_tilde
        # Calculate updated C and sigma
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (N, N)
        self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))  # (N, N)
        self.cov = self.K @ self.Sigma
        return self.cov @ Y_tilde

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 10 Eq.(11)

        :arg M_tilde: The posterior expectations for M (N, J).
        :type M_tilde: :class:`numpy.ndarray`
        Girolami and Rogers Page 10 Eq.(11)
        :return: Y_tilde (N,) containing \tilde(y)_{n} values.
        """
        return np.add(M_tilde, self._P(M_tilde, self.t_train))

    def _P(self, M_tilde, t_train):
        """
        Estimate the P, which can be obtained analytically derived from straightforward results for corrections to the
        mean of a Gaussian due to truncation.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 10 Eq.(11)

        :arg M_tilde: The posterior expectations for M (N, J).
        :arg n_samples: The number of samples to take.
        """
        return np.divide(
            np.multiply(t_train, norm.pdf(M_tilde)), norm.cdf(np.multiply(t_train, M_tilde))
        )


class VBMultinomialSparseGP(Estimator):
    """
    # TODO: This is likely deprecated by more general EP approach.
    # TODO: put in noise_variance parameter.
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
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        super().__init__(*args, **kwargs)
        # All indices are initially in the inactive set
        self.inactive_set = list(range(self.N))
        # For random inducing point selection
        random.shuffle(self.inactive_set)
        self.active_set = []
        self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))
        self.cov = self.K @ self.Sigma

    def _estimate_initiate(self, steps, M_0, M_ADF_0=None, s_ADF_0=None, varphi_0=None, psi_0=None):
        """
        Initialise the Estimator.

        :arg M_0: Initialisation of posterior mean estimates.
        :arg M_ADF_0: Initialisation of ADF inverse covariance updates (S, N, J). If `None` then initialised to zeros,
            default `None`.
        :arg s_ADF_0: Initialisation of ADF scalar update (J, N). If `None` then initialised to zeros,
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
            M_ADF_0 = np.zeros((steps, self.N, self.J))
        if s_ADF_0 is None:
            s_ADF_0 = np.ones((self.J, self.N))
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

        :arg M_0: (N, J) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`
        :arg int steps: The number of steps in the Estimator.
        :arg M_ADF_0: Initialisation of ADF inverse covariance updates (S, N, J). If `None` then initialised to
            zeros, default `None`.
        :arg s_ADF_0: Initialisation of ADF scalar update of the posterior variance (J, N). If
            `None` then initialised to zeros, default `None`.
        :arg varphi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type varphi_0: :class:`np.ndarray`
        :arg psi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type psi_0: :class:`np.ndarray`
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
                        desc="Sparse GP priors Estimator progress", unit="iterations"):
            # Check if there are any points left in the inactive set
            if len(self.active_set) != self.N:
                # Policy to choose new data point
                Y_tilde, calligraphic_Z, negative_p = self._Y_tilde(M_tilde)
                #print('Y_tilde', Y_tilde)
                _, index = self.new_point(informative_selection, Sigma_ADF, Y_tilde, varphi_tilde)
                C_n = self.K[:, index, :]
                #print('C_n', C_n)
                s_ADF, Sigma_ADF, Sigma_ADF_n = self._s_ADF(s_ADF, Sigma_ADF, M_ADF, C_n, index)  # (), (N, J)
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
                    #bound = self.variational_lower_bound(self.N, self.J, M_tilde, Sigma_ADF, self.K, calligraphic_Z)
                    #print("bound", bound)
                    Ms.append(np.linalg.norm(M_tilde))
                    Ys.append(np.linalg.norm(Y_tilde))
                    Sigma_ADFs.append(Sigma_ADF)
                    M_ADFs.append(np.linalg.norm(M_ADF))  # TODO: Don't store a (S, N, J) for each step.
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
        return M_tilde, M_ADF, s_ADF, Sigma_ADF, self.K, Y_tilde, varphi_tilde, psi_tilde, containers

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
            n_samples=1000, vectorised=True)  # (N_inactive, J)
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

        :arg Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :arg Y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde: The posterior mean estimate of the hyperparameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD and self.kernel._general:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
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
    #     This is the general case where there are hyperparameters varphi (J, D) for all dimensions and classes.
    #     This is the same for all categorical estimators, and so can live in the Abstract Base Class.
    #
    #     :arg Sigma_ARD:
    #     :arg C_tilde:
    #     :arg Y_tilde: The posterior mean estimate of the latent variable Y (N, J).
    #     :arg varphi_tilde:
    #     :arg X_test: The new data points, array like (N_test, D).
    #     :arg n_samples: The number of samples in the Monte Carlo estimate.
    #     :return: A Monte Carlo estimate of the class probabilities.
    #     """
    #     N_test = np.shape(X_test)[0]
    #     # Update the kernel with new varphi  # TODO: test this.
    #     self.kernel.varphi = varphi_tilde
    #     # Cs_news[:, i] is Cs_new for X_test[i]
    #     Cs_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (J, N, N_test)
    #     # TODO: this is a bottleneck - write a specialised kernel function - wonder how GPFlow does it
    #     covariances_new = self.kernel.kernel_matrix(X_test, X_test)
    #     cs_news = [np.diag(covariances_new[k]) for j in range(self.J)]  # (J, N_test)
    #     # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
    #     intermediate_vectors = Sigma_tilde @ Cs_news  # (J, N, N_test)
    #     intermediate_vectors_T = np.transpose(intermediate_vectors, (0, 2, 1))  # (J, N_test, N)
    #     intermediate_scalars = (np.multiply(Cs_news, intermediate_vectors)).sum(1)  # (J, N_test)
    #     # Calculate M_tilde_new
    #     # TODO: could just use Y_tilde instead of Y_tilde_T? It would be better just to store Y in memory as (J, N)
    #     Y_tilde_T = np.reshape(Y_tilde.T, (self.J, self.N, 1))
    #     M_new_tilde_T = np.matmul(intermediate_vectors_T, Y_tilde_T)  # (J, N_test, 1)?
    #     M_new_tilde_T = np.reshape(M_new_tilde_T, (self.J, N_test))  # (J, N_test)
    #     ## This was tested to be the slower option because matmul is an optimised sum and multiply.
    #     # M_new_tilde_T_2 = np.sum(np.multiply(Y_tilde_T, intermediate_vectors), axis=1)  # (J, N_test)
    #     # assert np.allclose(M_new_tilde_T, M_new_tilde_T_2)
    #     var_new_tilde_T = np.subtract(cs_news, intermediate_scalars)
    #     return self._vector_expectation_wrt_u(M_new_tilde_T.T, var_new_tilde_T.T, n_samples)

    def _s_ADF(self, s_ADF, Sigma_ADF, M_ADF, C_n, index):
        """
        Return the update of the ADF scalar update (J, N).

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 12 Eq.(12), (13).

        :arg s_ADF: ADF scalar update of the posterior variance (J, N).
        :arg Sigma_ADF: The ADF approximation to the GP posterior covariance.
        :arg M_ADF: The ADF inverse covariance update (S, N, J).
        :arg C_n: (J, N) The covariance vector for the new inducing point.
        :returns: updated s_ADF (J, N) and Sigma_ADF_n (N, J)
        """
        # TODO: an issue here.
        # Probably want to calculate the covariance column on the fly, depending on choice of x.
        M_ADF_n = M_ADF[:, index, :]  # (S, J)
        M_ADF_T = M_ADF.transpose(2, 1, 0)  # (J, N, S)
        M_ADF_n_T = M_ADF_n.T  # (J, S)
        Sigma_ADF_n2 = C_n - np.einsum('ijk, ik -> ij', M_ADF_T, M_ADF_n_T)
        # This was checked as correct
        Sigma_ADF_n = C_n - np.einsum('jik,jk->ki', M_ADF, M_ADF_n)  # (J, N) <- (J, N) - einsum((S, N, J), (S, J))
        assert np.allclose(Sigma_ADF_n, Sigma_ADF_n2)
        Sigma_ADF[:, :, index] = Sigma_ADF_n  # (J, N) <- (J, N)
        #Sigma_ADF[:, index, :] = Sigma_ADF_n  # (J, N) <- (J, N)
        precisions = 1 / (1 + s_ADF[:, index])
        #print("precisions", precisions)
        precisions = np.reshape(precisions, (self.J, 1))
        # TODO: posterior variance should be non-negative??
        update = np.multiply(
            np.diagonal(
                np.einsum('...i, ...j', Sigma_ADF_n, Sigma_ADF_n), axis1=1, axis2=2),
            precisions)
        #print("update", update)
        return s_ADF - update, Sigma_ADF, Sigma_ADF_n  # (J, N), (N, J)

    def _M_ADF(self, M_ADF, s_ADF, Sigma_ADF_n, index, step):
        """
        Return the update of the ADF vector updates (S, N, J)
        :arg s_ADF: ADF scalar update of the posterior variance (J, N).
        :arg Sigma_ADF_n: (J, N)
        :arg index: The index of self.t_train and self.X_train of the new inducing point, n.
        :return: M_ADF (S, N, J)
        """
        standard_deviations = 1/np.sqrt(1 + s_ADF[:, index])  # (J, )
        standard_deviations = np.reshape(standard_deviations, (1, self.J))
        Sigma_ADF_n_T = Sigma_ADF_n.T
        # (S, N, J) = (J,) (N, J)
        # (N, ) = (1,) (N, ) : J times
        M_ADF[step, :, :] = np.multiply(Sigma_ADF_n_T, standard_deviations)
        return M_ADF

    def _negative_p_n(self, M_tilde_n, t_n, n_samples=3000):
        """
        Calculate ratio of Monte Carlo estimates of the expectation of functions of M wrt to the distribution p.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        The rightmost term of Page 8 Eq.(5).

        :arg M_tilde_n: The current estimate of posterior expectations for M_n (J,), where M_n is the regression
            variable for the inducing point, n.
        :arg t_n: The target class for the inducing point, n.
        :arg n_samples: The number of Monte-Carlo samples to take in approximating the expectations.
        """
        differences = matrix_of_VB_differences(M_tilde_n, self.J, t_n)  # (J, J)
        differences = np.tile(differences, (n_samples, 1, 1))  # (n_samples, J, J)
        # Assume it's okay to use the same sample of U over all of the data points
        Us = sample_Us(self.J, n_samples, different_across_classes=False)  # (n_samples, J, J)
        random_variables = np.add(Us, differences)  # (n_samples, J, J) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (n_samples, J, J)
        log_cum_dists = np.log(cum_dists)  # (n_samples, J, J)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[:, t_n, :]  # (n_samples, J)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[:, t_n, :]))  # (n_samples, J)
        # Product is over j \neq tn=i
        log_cum_dists[:, :, t_n] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=2)  # TODO: not sure if needed for the variational bound
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=0)
        # product is over j \neq k
        log_cum_dists[:, range(self.J), range(self.J)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=2)
        log_samples = np.sum(log_cum_dists, axis=2)  # (n_samples, J)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, J)

    def _negative_p(self, M_tilde, t, n_samples=3000):
        """
        Calculate ratio of Monte Carlo estimates of the expectation of functions of M wrt to the distribution p.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        The rightmost term of Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, J).
        :arg t: The target vector.
        :arg n_samples: The number of samples to take.
        """
        # Find matrix of differences
        #differences = matrix_of_differencess(M_tilde, self.J, self.N)  # TODO: confirm this is wrong
        # we will product across axis 2 (rows)
        differences = matrix_of_VB_differencess(M_tilde, self.J, self.N, t, self.grid)  # (N, J, J)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, J, J)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, J, J)
        # Assume it's okay to use the same samples of U over all of the data points
        Us = sample_Us(self.J, n_samples, different_across_classes=False)  # (n_samples, J, J)
        random_variables = np.add(Us, differencess)  # (N, n_samples, J, J) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (N, n_samples, J, J)
        log_cum_dists = np.log(cum_dists)  # (N, n_samples, J, J)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[self.grid, :, t, :]  # (N, n_samples, J)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[self.grid, :, t, :]))  # (N, n_samples, J)
        # product is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=3)  # TODO: not sure if correct
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=1)
        # product is over j \neq k
        log_cum_dists[:, :, range(self.J), range(self.J)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (N, n_samples, J)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte Carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, J)

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
            self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (J, N, N)
            self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))  # (J, N, N)
            self.cov = self.K @ self.Sigma  # (J, N, N) @ (J, N, N) = (J, N, N)
        # t_n = self.t_train[index]
        # M_tilde_n = M_tilde[index]
        # negative_p_n, calligraphic_Z = self._negative_P(M_tilde_n, t_n)
        negative_p_n = negative_p[index, :]
        weight = negative_p_n / (1 + s_ADF[:, index])  # (J,)
        weight = np.reshape(weight, (1, self.J))  # (J, 1)
        return M_tilde - np.multiply(Sigma_ADF_n.T, weight)
        # return M_tilde - np.multiply(Sigma_ADF_n, weight), Y_tilde, calligraphic_Z
        # return M_tilde - (negative_p_n)/(1 + s_ADF[:, index]) * Sigma_ADF_n  # (J,) * (J, N) -> (J, N)
        # return M_tilde + (Y_tilde[index, :] - M_tilde[index, :])/(1 + s_ADF[:, index]) * Sigma_ADF_n

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, J).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, J) containing \tilde(y)_{nk} values.
        """
        # t = np.argmax(M_tilde, axis=1)  # The max of the GP vector m_k is t_n this would be incorrect.
        negative_p, calligraphic_Z = self._negative_p(M_tilde, self.t_train, n_samples=3000)  # TODO: correct version.
        Y_tilde = np.subtract(M_tilde, negative_p)  # Eq.(5)
        Y_tilde[self.grid, self.t_train] = 0
        Y_tilde[self.grid, self.t_train] = np.sum(M_tilde - Y_tilde, axis=1)
        return Y_tilde, calligraphic_Z, negative_p


class VBMultinomialGP(Estimator):
    """
    # TODO: put in noise_variance. I think I will just not support noise variance here.
    A Variational Bayes classifier. Inherits the Estimator ABC.

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, noise_variance=None, *args, **kwargs):
        """
        Create an :class:`VBMultinomialGP` estimator object.

        :returns: An :class:`VBMultinomialGP` object.
        """
        if noise_variance is not None:
            raise ValueError("noise_variance is not supported for this classifier. (expected `None`, got {})".format(
                noise_variance))
        super().__init__(*args, **kwargs)
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        # TODO: Swap this notation
        self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))
        self.cov = self.K @ self.Sigma

    def _estimate_initiate(self, M_0, varphi_0=None, psi_0=None):
        """
        Initialise the Estimator.

        :arg M_0: Intialisation of posterior mean estimates.
        :return: Containers for the mean estimates of parameters and hyperparameters.
        :rtype: (4,) tuple
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

        :arg M_0: (N, J) numpy.ndarray of the initial location of the posterior mean.
        :type M_0: :class:`np.ndarray`
        :arg int steps: The number of steps in the Estimator.
        :arg varphi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type varphi_0: :class:`np.ndarray`
        :arg psi_0: (L, M) numpy.ndarray of the initial location of the posterior mean.
        :type psi_0: :class:`np.ndarray`
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
                        desc="GP priors Estimator progress", unit="iterations"):
            Y_tilde, calligraphic_Z = self._Y_tilde(M_tilde)
            M_tilde = self._M_tilde(Y_tilde, varphi_tilde, fix_hyperparameters)
            if fix_hyperparameters is False:
                varphi_tilde = self._varphi_tilde(M_tilde, psi_tilde, n_samples=1000)  # TODO: Cythonize. Major bottleneck.
                psi_tilde = self._psi_tilde(varphi_tilde)
            if write:
                # Calculate bound
                bound = self.variational_lower_bound(self.N, self.J, M_tilde, self.Sigma, self.K, calligraphic_Z)
                Ms.append(np.linalg.norm(M_tilde))
                Ys.append(np.linalg.norm(Y_tilde))
                if fix_hyperparameters is False:
                    varphis.append(varphi_tilde)
                    psis.append(psi_tilde)
                bounds.append(bound)
        containers = (Ms, Ys, varphis, psis, bounds)
        return M_tilde, self.Sigma, self.K, Y_tilde, varphi_tilde, psi_tilde, containers

    def _M_tilde(self, Y_tilde, varphi_tilde, fix_hyperparameters):
        """
        Return the posterior mean estimate of M.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(8)

        :arg Y_tilde: (N, J) array
        :type Y_tilde: :class:`np.ndarray`
        :arg varphi_tilde: array whose size depends on the kernel.
        :type Y_tilde: :class:`np.ndarray`
        """
        if fix_hyperparameters is False:
            # Update hyperparameters and GP posterior covariance with new values
            self.kernel.varphi = varphi_tilde
            print(varphi_tilde, self.kernel.varphi, 'varphi')
            # Calculate updated C and sigma
            self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)  # (J, N, N)
            print(self.K, 'C', np.linalg.cond(self.K))
            self.Sigma = np.linalg.inv(np.add(np.eye(self.N), self.K))  # (J, N, N)
            self.cov = self.K @ self.Sigma  # (J, N, N) @ (J, N, N) = (J, N, N)
        # TODO: Maybe just keep Y_tilde in this shape in memory.
        Y_tilde_reshape = Y_tilde.T.reshape(self.J, self.N, 1)  # (J, N, 1) required for np.multiply
        M_tilde_T = self.cov @ Y_tilde_reshape  # (J, N, 1)
        return M_tilde_T.reshape(self.J, self.N).T  # (N, J)

    def _Y_tilde(self, M_tilde):
        """
        Calculate Y_tilde elements.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 8 Eq.(5).

        :arg M_tilde: The posterior expectations for M (N, J).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, J) containing \tilde(y)_{nk} values.
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
        Estimate a ratio of Monte Carlo estimates of the expectation of a functions of M wrt to the distribution p.

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        the rightmost term of 2005 Page 8 Eq.(5),

        :arg M_tilde: The posterior expectations for M (N, J).
        :arg t: The target vector.
        :arg n_samples: The number of samples to take.
        """
        # Find matrix of differences
        #differences = matrix_of_differencess(M_tilde, self.J, self.N)  # TODO: confirm this is wrong
        # we will product across axis 2 (rows)
        differences = matrix_of_VB_differencess(M_tilde, self.J, self.N, t, self.grid)  # (N, J, J)
        differencess = np.tile(differences, (n_samples, 1, 1, 1))  # (n_samples, N, J, J)
        differencess = np.moveaxis(differencess, 1, 0)  # (N, n_samples, J, J)
        # Assume it's okay to use the same samples of U over all of the data points
        Us = sample_Us(self.J, n_samples, different_across_classes=False)  # (n_samples, J, J)
        random_variables = np.add(Us, differencess)  # (N, n_samples, J, J) Note that it is \prod_k u + m_ni - m_nk
        cum_dists = norm.cdf(random_variables, loc=0, scale=1)  # (N, n_samples, J, J)
        log_cum_dists = np.log(cum_dists)  # (N, n_samples, J, J)
        # Store values for later
        log_M_nk_M_nt_cdfs = log_cum_dists[self.grid, :, t, :]  # (N, n_samples, J)
        log_M_nk_M_nt_pdfs = np.log(norm.pdf(random_variables[self.grid, :, t, :]))  # (N, n_samples, J)
        # product is over j \neq tn=i
        log_cum_dists[self.grid, :, :, t] = 0
        calligraphic_Z = np.sum(log_cum_dists, axis=3)  # TODO: not sure if correct
        calligraphic_Z = np.exp(calligraphic_Z)
        calligraphic_Z = 1. / n_samples * np.sum(calligraphic_Z, axis=1)
        # product is over j \neq k
        log_cum_dists[:, :, range(self.J), range(self.J)] = 0
        # Sum across the elements of the log product of interest (rows, so axis=3)
        log_samples = np.sum(log_cum_dists, axis=3)  # (N, n_samples, J)
        log_element_prod_pdf = np.add(log_M_nk_M_nt_pdfs, log_samples)
        log_element_prod_cdf = np.add(log_M_nk_M_nt_cdfs, log_samples)
        element_prod_pdf = np.exp(log_element_prod_pdf)
        element_prod_cdf = np.exp(log_element_prod_cdf)
        # Monte Carlo estimate: Sum across the n_samples (axis=1)
        element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)
        return np.divide(element_prod_pdf, element_prod_cdf), calligraphic_Z  # (N, J)
        # Superceded
        # M_nk_M_nt_cdfs = cum_dists[self.grid, :, :, t]  # (N, n_samples, J)
        # M_nk_M_nt_pdfs = norm.pdf(random_variables[self.grid, :, t, :])
        # cum_dists[:, :, range(self.J), range(self.J)] = 1
        # cum_dists[self.grid, :, :, t] = 1
        # samples = np.prod(cum_dists, axis=3)
        # element_prod_pdf = np.multiply(M_nk_M_nt_pdfs, samples)
        # element_prod_cdf = np.multiply(M_nk_M_nt_cdfs, samples)
        # element_prod_pdf = 1. / n_samples * np.sum(element_prod_pdf, axis=1)
        # element_prod_cdf = 1. / n_samples * np.sum(element_prod_cdf, axis=1)

    def predict(self, Sigma_tilde, Y_tilde, varphi_tilde, X_test, n_samples=1000, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :arg Y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD and self.kernel._general:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
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

    def variational_lower_bound(self, N, J, M,  Sigma, C, calligraphic_Z, numerical_stability=True):
        """
        Calculate the variational lower bound of the log marginal likelihood.

        # TODO: Calculating det(C) is numerically instable. Need to use the trick given in GPML.

        :arg M_tilde:
        :arg Sigma_tilde:
        :arg C_tilde:
        :arg calligraphic_Z:
        :arg bool numerical_stability:
        """
        if numerical_stability is True:
            if self.kernel._general:
                C = C + self.jitter * np.eye(N)
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
                C_inv = L_inv.transpose((0, 2, 1)) @ L_inv  # (J, N, N) (J, N, N) -> (J, N, N)
                L_Sigma = np.linalg.cholesky(Sigma)
                half_log_det_C = np.trace(np.log(L), axis1=1, axis2=2)  # TODO: check
                half_log_det_Sigma = np.trace(np.log(L_Sigma), axis1=1, axis2=2)
                summation = np.einsum('ijk, ki->ij', C_inv, M)  # (J, N, N) @ (N, J) -> (J, N)
                summation = np.einsum('ij, ji-> ', summation, M)  # (J, N) (N, J) -> ()
                bound = (
                    - (N * J * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                    + (N * J / 2) - (np.sum(np.trace(Sigma, axis1=1, axis2=2)) / 2)
                    - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma, axis1=1, axis2=2)) / 2)
                    - np.sum(half_log_det_C) + np.sum(half_log_det_Sigma)
                    + np.sum(np.log(calligraphic_Z))
                )
            else:
                # Case when Sigma is (N, N)
                C = C + self.jitter * np.eye(N)
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
                        - (N * J * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * J / 2) - np.trace(Sigma) / 2
                        - (summation / 2) - (np.trace(C_inv @ Sigma) / 2)
                        - half_log_det_C + half_log_det_Sigma
                        + np.sum(np.log(calligraphic_Z))
                )
            print('bound = ', bound)
            return bound
        elif numerical_stability is False:
            C = C + 1e-4 * np.eye(N)
            C_inv = np.linalg.inv(C)
            M_T = M.T
            intermediate_vectors = np.empty((N, J))

            if self.kernel._general:
                for j in range(J):
                    intermediate_vectors[:, k] = C_inv[k] @ M_T[k]
                summation = np.sum(np.multiply(intermediate_vectors, M))
                # Case when Sigma is (J, N, N)
                bound = (
                        - (N * J * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * J / 2) - (np.sum(np.trace(Sigma, axis1=1, axis2=2)) / 2)
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
                        - (N * J * np.log(2 * np.pi) / 2) + (N * np.log(2 * np.pi) / 2)
                        + (N * J / 2) - (np.sum(np.trace(Sigma)) / 2)
                        - (summation / 2) - (np.sum(np.trace(C_inv @ Sigma)) / 2)
                        - (np.sum(np.log(np.linalg.det(C))) / 2) + (np.sum(np.log(np.linalg.det(Sigma))) / 2)
                        + np.sum(np.log(calligraphic_Z))
                )

            print('bound = ', bound)
            return bound


class VBOrderedGP(Estimator):
    """
    TODO: On 05/08 Tried to generalise the grid_over_hyperparameters to optionally grid over s.
    TODO: On 03/08 Tried to find and fix the numerical stability bug.
    TODO: On 20/07 applied refactoring to including noise variance parameter.
    TODO: On 20/07 applied refactoring to be able to call __init__ method at each hyperparameter update and it work.
    TODO: Why do I need to initiate it with a noise variance? Can't this be done as estimate time? A: init self.Sigma
    INPROGRESS: Change cov to Sigma, and Sigma to cov. Since Sigma should be reserved for posterior covariance.
    A Variational Bayes classifier for ordered likelihood. Inherits the Estimator ABC

    This class allows users to define a classification problem, get predictions
    using approximate Bayesian inference. It is for the ordered likelihood.

    For this a :class:`probit.kernels.Jernel` is required for the Gaussian Process.
    """

    def __init__(self, noise_variance, *args, **kwargs):
        """
        Create an :class:`VBOrderedGP` Estimator object.

        :returns: An :class:`VBMultinimoalOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        if self.kernel._ARD:
            raise ValueError('The kernel must not be ARD type (kernel._ARD=1),'
                             ' but ISO type (kernel._ARD=0). (got {}, expected)'.format(
                self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError('The kernel must not be general type (kernel._general=1),'
                             ' but simple type (kernel._general=0). (got {}, expected)'.format(
                self.kernel._general, 0))
        self.grid = np.ogrid[0:self.N]  # All the indices for indexing sets of self.t_train
        self.EPS = 0.000001  # Acts as a machine tolerance
        # Threshold of single sided standard deviations that normal cdf can be approximated to 0 or 1
        # More than this + redundancy leads to numerical instability due to catestrophic cancellation
        # Less than this leads to a poor approximation due to series expansion at infinity truncation
        # Good values found between 4 and 6
        self.upper_bound = 6
        # More than this + redundancy leads to numerical instability due to overflow
        # Less than this results in poor approximation due to neglected probability mass in the tails
        # Good values found between 18 and 30
        self.upper_bound2 = 30
        #self.jitter = 1e-8  #TODO # Regularisation for matrix inversion (see GPML by Williams et al. for a good explanation)
        self.jitter = 1e-6
        if noise_variance is None:
            self.noise_variance = 1.0
        else:
            self.noise_variance = noise_variance
        self.noise_std = np.sqrt(self.noise_variance)
        self._update_prior()
        self._update_posterior()
        # TODO: might need cholesky factor later on. 20/07 depricated Sigma_tilde since self.Sigma and self.K were not
        # used.
        # TODO: 15/08 refactored so that partial derivatives are generated here in initiation

    def _update_posterior(self):
        """Update posterior covariances."""
        # Only need to store Sigma_div_var and cov (cov to avoid matrix multiplication at prediction)
        L_cov = np.linalg.cholesky(self.noise_variance * np.eye(self.N) + self.K)
        L_K = np.linalg.cholesky(self.K + self.jitter * np.eye(self.N))
        self.log_det_K = 2 * np.sum(np.log(np.diag(L_K)))
        self.log_det_cov = -2 * np.sum(np.log(np.diag(L_cov)))
        cov = np.linalg.inv(L_cov)  # The only inverse required per posterior update, O(N^3) complexity
        self.cov = cov.T @ cov  # O(N^2) memory
        self.Sigma_div_var = self.K @ self.cov  # O(N^2) memory
        self.Sigma = self.Sigma_div_var * self.noise_variance
        self.partial_Sigma_div_var = self.noise_variance * self.cov @ self.partial_K_varphi @ self.cov

    def _hyperparameters_update(self, varphi=None, scale=None, noise_variance=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior covariances.

        :arg varphi:
        :arg s:
        :arg noise_variance:
        """
        if varphi is not None or scale is not None:
            self.kernel.hyperparameter_update(varphi=varphi, scale=scale)
            # Update prior covariance
            self._update_prior()
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)
        # Update posterior covariance
        self._update_posterior()

    def _estimate_initiate(self, m_0, dm_0, gamma, varphi_0, psi_0):
        """
        Initialise the estimator.

        :arg m_0: The initial state of the approximate posterior mean (N,). If `None` then initialised to
            zeros, default `None`. 
        :type m_0: :class:`numpy.ndarray`
        :arg gamma:  The (J + 1, ) array of cutpoint parameters \bm{gamma}.
        :type gamma: :class:`numpy.ndarray`
        :arg varphi_0: Initialisation of hyperparameter posterior mean estimates. If `None` then initialised
            to ones, default `None`.
        :type varphi_0: :class:`numpy.ndarray` or float
        :arg psi_0: Initialisation of the mean estimate of the parameter controlling the hyperparameter prior.
            If `None` then initialised to ones, default `None`.
        :type psi_0: :class:`numpy.ndarray` or float
        :arg float noise_variance_0:
        """
        if m_0 is None:
            m_0 = np.random.rand(self.N)  # TODO: is there a good reason for this?
            # m_0 = np.zeros(self.N)
        if dm_0 is None:
            dm_0 = None  # Do not track derivatives
        else:
            dm_0 = np.zeros(self.N)
        if varphi_0 is None:
            varphi_0 = np.ones(np.shape(self.kernel.varphi))
        if psi_0 is None:
            psi_0 = np.ones(np.shape(self.kernel.varphi))
        ys = []
        ms = []
        varphis = []
        psis = []
        fxs = []
        containers = (ms, ys, varphis, psis, fxs)
        # TODO: cutpoints are hyperparameters that are fixed, not variables, so can keep them in class scope
        # Treat user parsing of cutpoint parameters with just the upper cutpoints for each class
        # Convert gamma to numpy array
        gamma = np.array(gamma)
        if np.shape(gamma)[0] == self.J - 1:  # not including any of the fixed cutpoints: -\infty, \infty
            gamma = np.append(gamma, np.inf)  # append the infinity cutpoint
            gamma = np.insert(gamma, np.NINF)  # insert the negative infinity cutpoint at index 0
            pass  # correct format
        elif np.shape(gamma)[0] == self.J:  # not including one of the infinity cutpoints
            if gamma[-1] != np.inf:
                if gamma[0] != np.NINF:
                    raise ValueError('The last cutpoint parameter must be numpy.inf, or the first cutpoint parameter'
                                     ' must be numpy.NINF (got {}, expected {})'.format(
                        [gamma[0], gamma[-1]], [np.inf, np.NINF]))
                else:  # gamma[0] is negative infinity
                    gamma.append(np.inf)
                    pass  # correct format
            else:
                gamma = np.insert(gamma, np.NINF)
                pass  # correct format
        elif np.shape(gamma)[0] == self.J + 1:  # including all of the cutpoints
            if gamma[0] != np.NINF:
                raise ValueError('The cutpoint parameter \gamma must be numpy.NINF (got {}, expected {})'.format(
                    gamma[0], np.NINF))
            if gamma[-1] != np.inf:
                raise ValueError('The cutpoint parameter \gamma_J must be numpy.inf (got {}, expected {})'.format(
                    gamma[-1], np.inf))
            pass  # correct format
        else:
            raise ValueError('Could not recognise gamma shape. (np.shape(gamma) was {})'.format(np.shape(gamma)))
        assert gamma[0] == np.NINF
        assert gamma[-1] == np.inf
        assert np.shape(gamma)[0] == self.J + 1
        if not all(
                gamma[i] <= gamma[i + 1]
                for i in range(self.J)):
            raise CutpointValueError(gamma)
        return m_0, dm_0, gamma, varphi_0, psi_0, containers

    def estimate(self, steps, gamma, varphi_tilde_0=None, psi_tilde_0=None, noise_variance=None, m_tilde_0=None, dm_tilde_0=None,
                    first_step=1, fix_hyperparameters=False, write=False, track_derivatives=False):
        """
        Estimating the posterior means are a 3 step iteration over m_tilde, varphi_tilde and psi_tilde
            Eq.(8), (9), (10), respectively or, optionally, just an iteration over m_tilde.

        :arg int steps: The number of iterations the Estimator takes.
        :arg gamma: The (J + 1, ) array of cutpoint parameters \bm{gamma}.
        :type gamma: :class:`numpy.ndarray`
        :arg varphi_tilde_0: Initialisation of hyperparameter posterior mean estimates. If `None` then initialised
            to ones, default `None`.
        :type varphi_tilde_0: :class:`numpy.ndarray` or float
        :arg psi_tilde_0: Initialisation of the mean estimate of the parameter controlling the hyperparameter prior.
            If `None` then initialised to ones, default `None`.
        :type psi_tilde_0: :class:`numpy.ndarray` or float
        :arg float noise_variance: Initialisation of noise variance. If `None` then initialised to one, default `None`.
        :arg m_tilde_0: The initial state of the approximate posterior mean (N,). If `None` then initialised to
            zeros, default `None`.
        :type m_tilde_0: :class:`numpy.ndarray`
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool fix_hyperparameters: Boolean variable that sets if the posterior means of the hyperparameters are
            fixed. If `True` then `varphi_tilde` and `psi_tilde` will remain at their initial values and not be updated.
        :arg bool write: Boolean variable to store and write arrays of interest. If set to "True", the method will
            output non-empty containers of evolution of the statistics over the steps. If set to "False", statistics
            will not be written and those containers will remain empty.
        :return: Posterior mean and covariance estimates.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate posterior means, other statistics and
            tuple of lists of per-step evolution of those statistics.
        """
        m_tilde, dm_tilde, gamma, varphi_tilde, psi_tilde, containers = self._estimate_initiate(
            m_tilde_0, dm_tilde_0, gamma, varphi_tilde_0, psi_tilde_0)
        ms, ys, varphis, psis, fxs = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples", disable=True):
            # Eq. (11)
            p = self._p(m_tilde, gamma, self.noise_std, numerically_stable=True)
            y_tilde = self._y_tilde(p, m_tilde, gamma, self.noise_std)
            m_tilde = self._m_tilde(y_tilde, self.Sigma_div_var)
            # plt.scatter(self.X_train, y_tilde)
            # plt.title("y_tilde")
            # plt.show()
            # plt.scatter(self.X_train, m_tilde)
            # plt.title("m_tilde")
            # plt.show()
            if dm_tilde is not None:
                dm_tilde = self._dm_tilde(dm_tilde, y_tilde, sigma_dp, self.partial_Sigma_div_var, self.Sigma_div_var)
                #dm2 = noise_variance * self.cov_ @ self.partial_K_varphi @ self.cov_ @ y_tilde
                #plt.scatter(self.X_train, dm_tilde)
                #plt.scatter(self.X_train, dm2)
                #plt.title("dm_tilde")
                #plt.show()
            if not fix_hyperparameters:
                varphi_tilde = self._varphi_tilde(m_tilde, psi_tilde, n_samples=1000)
                psi_tilde = self._psi_tilde(varphi_tilde)
                self._hyperparameters_update(varphi=varphi_tilde)
            if write:
                calligraphic_Z, *_ = self._calligraphic_Z(
                    gamma, self.noise_std, m_tilde)
                fx = self.objective(self.N, m_tilde, y_tilde, self.Sigma_div_var, self.cov, self.K, calligraphic_Z, noise_variance, self.log_det_K, self.log_det_cov)
                ms.append(m_tilde)
                ys.append(y_tilde)
                varphis.append(varphi_tilde)
                psis.append(psi_tilde)
                fxs.append(fx)
        containers = (ms, ys, varphis, psis, fxs)
        return m_tilde, dm_tilde, y_tilde, p, varphi_tilde, containers

    def _predict_vector(self, gamma, cov, y_tilde, varphi_tilde, noise_variance, X_test):
        """
        Make variational Bayes prediction over classes of X_test given the posterior samples.
        :arg Sigma_tilde:
        :arg y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde:
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        N_test = np.shape(X_test)[0]
        # Update the kernel
        self._hyperparameters_update(varphi=varphi_tilde)
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        c_news = np.diag(self.kernel.kernel_matrix(X_test, X_test)) # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]
        intermediate_vectors = cov @ C_news  # (N, N_test)
        intermediate_scalars = np.sum(np.multiply(C_news, intermediate_vectors), axis=0)  # (N_test, )
        # Calculate m_tilde_new # TODO: test this.
        posterior_predictive_m = np.einsum('ij, i -> j', intermediate_vectors, y_tilde)  # (N, N_test) (N, ) = (N_test,)
        # plt.scatter(self.X_train, y_tilde)
        # plt.plot(X_test, posterior_predictive_m)
        # plt.hlines(gamma[[1, 2]], -0.5, 1.5)
        # plt.show()
        posterior_var = c_news - intermediate_scalars
        posterior_std = np.sqrt(posterior_var)
        posterior_predictive_var = posterior_var + noise_variance  # (N_test, )
        posterior_predictive_std = np.sqrt(posterior_predictive_var)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(gamma[k + 1], posterior_predictive_m), posterior_predictive_std)
            Z2 = np.divide(np.subtract(gamma[k], posterior_predictive_m), posterior_predictive_std)
            predictive_distributions[:, k] = norm.cdf(Z1) - norm.cdf(Z2)
        return predictive_distributions, posterior_predictive_m, posterior_std # (N_test, J), (N_test,) (N_test,)

    def predict(self, gamma, cov_, y_tilde, varphi_tilde, noise_variance, X_test, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg Sigma_tilde: The posterior mean estimate of the marginal posterior covariance.
        :arg y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError('For the ordered likelihood estimator, the kernel must not be ARD type'
                             ' (kernel._ARD=1), but ISO type (kernel._ARD=0). (got {}, expected)'.format(
                self.kernel._ARD, 0))
        else:
            if vectorised:
                return self._predict_vector(gamma, cov_, y_tilde, varphi_tilde, noise_variance, X_test)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _varphi_tilde(self, m_tilde, psi_tilde, n_samples=10, vectorised=True):
        """
        Return the w values of the sample

        Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial Probit Regression with Gaussian Process
        Priors," in Neural Computation, vol. 18, no. 8, pp. 1790-1817, Aug. 2006, doi: 10.1162/neco.2006.18.8.1790.2005
        Page 9 Eq.(9).

        :arg m_tilde: Posterior mean estimate of M_tilde.
        :arg psi_tilde: Posterior mean estimate of psi.
        :arg int n_samples: The number of samples for the importance sampling estimate, 500 is used in 2005 Page 13.
        """
        # Vector draw from
        # (n_samples, J, D) in general and _ARD, (n_samples, ) for single shared kernel and ISO case. Depends on the
        # shape of psi_tilde.
        varphis = sample_varphis(psi_tilde, n_samples)  # (n_samples, )
        log_varphis = np.log(varphis)
        # (n_samples, J, N, N) in general and _ARD, (n_samples, N, N) for single shared kernel and ISO case. Depends on
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

    def _m_tilde(self, y_tilde, Sigma_div_var):
        """
        Return the posterior mean estimate of m.

        2021 Page Eq.()

        :arg y_tilde: (N,) array
        :type y_tilde: :class:`np.ndarray`
        """
        return Sigma_div_var @ y_tilde  # (N, J)

    def _dm_tilde(self, dm_tilde, y_tilde, sigma_dp, partial_Sigma_div_var, Sigma_div_var):
        """
        Return the derivative wrt varphi of the posterior mean estimate of m.
        """
        return partial_Sigma_div_var @ y_tilde + Sigma_div_var @ (np.eye(self.N) + np.diag(sigma_dp)) @ dm_tilde

    def _y_tilde(self, p, m_tilde, gamma, noise_std):
        """
        Calculate Y_tilde elements 2021 Page Eq.().

        :arg M_tilde: The posterior expectations for M (N, ).
        :type M_tilde: :class:`numpy.ndarray`
        :return: Y_tilde (N, ) containing \tilde(y)_{n} values.
        """
        # y_tilde = np.add(m_tilde, noise_std * p)
        # for i, value in enumerate(y_tilde):
        #     gamma_k = gamma[self.t_train[i]]
        #     gamma_kplus1 = gamma[self.t_train[i] + 1]
        #     m_i = m_tilde[i]
        #     z1 = (gamma_k - m_i) / noise_std
        #     z2 = (gamma_kplus1 - m_i) / noise_std
        #     if (value < gamma_k) or (value > gamma_kplus1):
        #         print("gamma_k, y, gamma_kplus1=[{}, {}, {}], z1 = {}, z2 = {}, m={}, i={}".format(
        #             gamma_k, value, gamma_kplus1, z1, z2, m_i, i))
        #     if (value == np.inf) or (value == -np.inf):
        #         print("gamma_k, y, gamma_kplus1=[{}, {}, {}], z1 = {}, z2 = {}, m={}, i={}".format(
        #             gamma_k, value, gamma_kplus1, z1, z2, m_i, i))
        return np.add(m_tilde, noise_std * p)

    def _dp(self, m, gamma, noise_std, numerically_stable=True):
        """
        Estimate the rightmost term of 2021 partial bound partial log sigma.
        """
        (calligraphic_Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s,
        *_) = self._calligraphic_Z(
            gamma, noise_std, m)
        sigma_dp = (z1s * norm_pdf_z1s - z2s * norm_pdf_z2s) / calligraphic_Z
        # Need to deal with the tails to prevent catestrophic cancellation
        indices1 = np.where(z1s > self.upper_bound)
        indices2 = np.where(z2s < -self.upper_bound)
        indices = np.union1d(indices1, indices2)
        z1_indices = z1s[indices]
        z2_indices = z2s[indices]
        sigma_dp[indices] = self._dp_tails(z1_indices, z2_indices)
        # Define the derivative for when z2 and z1 values take a value of (+/-)infinity respectively
        indices = np.where(z1s==-np.inf)
        sigma_dp[indices] = - z2s[indices] * norm_pdf_z2s[indices] / calligraphic_Z[indices]
        indices = np.intersect1d(indices, indices2)
        sigma_dp[indices] = self._dp_far_tails(z2s[indices])
        indices = np.where(z2s==np.inf)
        sigma_dp[indices] = z1s[indices] * norm_pdf_z1s[indices] / calligraphic_Z[indices]
        indices = np.intersect1d(indices, indices1)
        sigma_dp[indices] = self._dp_far_tails(z1s[indices])
        # Finally, get the far tails for the non-infinity case to prevent overflow
        if numerically_stable is True:
            indices = np.where(z1s > self.upper_bound2)
            z1_indices = z1s[indices]
            sigma_dp[indices] = self._dp_far_tails(z1_indices)
            indices = np.where(z2s < -self.upper_bound2)
            z2_indices = z2s[indices]
            sigma_dp[indices] = self._dp_far_tails(z2_indices)
        return sigma_dp

    def _p(self, m, gamma, noise_std, numerically_stable=True):
        """
        Estimate the rightmost term of 2021 Page Eq.(), a ratio of Monte Carlo estimates of the expectation of a
            functions of M wrt to the distribution p.

        :arg m: The current posterior mean estimate.
        :type m: :class:`numpy.ndarray`
        :arg gamma: The threshold parameters.
        :type gamma: :class:`numpy.ndarray`
        :arg float noise_std: The noise standard deviation.
        :returns: (p, calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s)
        :rtype: tuple (
            :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`,
            :class:`numpy.ndarray`, :class:`numpy.ndarray`, :class:`numpy.ndarray`)
        """
        (calligraphic_Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s,
        *_) = self._calligraphic_Z(
            gamma, noise_std, m)
        p = (norm_pdf_z1s - norm_pdf_z2s) / calligraphic_Z
        # TODO: clean up sigma_dp, which was depricated.
        # sigma_dp = (z1s * norm_pdf_z1s - z2s * norm_pdf_z2s) / calligraphic_Z
        # Need to deal with the tails to prevent catestrophic cancellation
        indices1 = np.where(z1s > self.upper_bound)
        indices2 = np.where(z2s < -self.upper_bound)
        indices = np.union1d(indices1, indices2)
        z1_indices = z1s[indices]
        z2_indices = z2s[indices]
        p[indices] = self._p_tails(z1_indices, z2_indices)
        # sigma_dp[indices] = self._dp_tails(z1_indices, z2_indices)
        # Define the derivative for when z2 and z1 values take a value of (+/-)infinity respectively
        # indices = np.where(z1s==-np.inf)
        # sigma_dp[indices] = - z2s[indices] * norm_pdf_z2s[indices] / calligraphic_Z[indices]
        # indices = np.intersect1d(indices, indices2)
        # sigma_dp[indices] = self._dp_far_tails(z2s[indices])
        # indices = np.where(z2s==np.inf)
        # sigma_dp[indices] = z1s[indices] * norm_pdf_z1s[indices] / calligraphic_Z[indices]
        # indices = np.intersect1d(indices, indices1)
        # sigma_dp[indices] = self._dp_far_tails(z1s[indices])
        # Finally, get the far tails for the non-infinity case to prevent overflow
        if numerically_stable is True:
            indices = np.where(z1s > self.upper_bound2)
            z1_indices = z1s[indices]
            p[indices] = self._p_far_tails(z1_indices)
            # sigma_dp[indices] = self._dp_far_tails(z1_indices)
            indices = np.where(z2s < -self.upper_bound2)
            z2_indices = z2s[indices]
            p[indices] = self._p_far_tails(z2_indices)
            # sigma_dp[indices] = self._dp_far_tails(z2_indices)
        # sigma_dp -= p**2
        #plt.scatter(m, sigma_dp/noise_std)
        #plt.scatter(m, p)
        #plt.scatter(self.X_train, p)
        #plt.show()
        return p # , sigma_dp

    def _dp_tails(self, z1, z2):
        """Series expansion at infinity."""
        return (
            z1 * np.exp(-0.5 * z1**2) - z2 * np.exp(-0.5 * z2**2)) / (
                1 / z1 * np.exp(-0.5 * z1**2)* np.exp(self._g(z1)) - 1 / z2 * np.exp(-0.5 * z2**2) * np.exp(self._g(z2)))

    def _dp_far_tails(self, z):
        """Prevents overflow at large z."""
        return z**2 * np.exp(-self._g(z))

    def _p_tails(self, z1, z2):
        """Series expansion at infinity. Even for z1, z2 >= 4 this is accurate to three decimal places."""
        return (
            np.exp(-0.5 * z1**2) - np.exp(-0.5 * z2**2)) / (
                1 / z1 * np.exp(-0.5 * z1**2)* np.exp(self._g(z1))
                - 1 / z2 * np.exp(-0.5 * z2**2) * np.exp(self._g(z2)))

    def _p_far_tails(self, z):
        """Prevents overflow at large z."""
        return z * np.exp(-self._g(z))

    def objective(
        self, N, m, y, Sigma_div_var, cov, K, calligraphic_Z, noise_variance, log_det_K, log_det_cov, numerical_stability=True, verbose=True):
        """
        Calculate fx, the variational lower bound of the log marginal likelihood.

        .. math::
                \mathcal{F(\Phi)} =,

            where :math:`F(\Phi)` is the variational lower bound of the log marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg int N: The number of datapoints.
        :arg m: The posterior mean.
        :type m: :class:`numpy.ndarray`
        :arg y: The posterior mean.
        :type y: :class:`numpy.ndarray`
        :arg Sigma_div_var: The posterior covariance.
        :type Sigma_div_var: :class:`numpy.ndarray`
        :arg cov:
        :type cov: :class:`numpy.ndarray`
        :arg K: The prior covariance.
        :type K: :class:`numpy.ndarray`
        :arg float noise_variance: The noise variance.
        :arg float log_det_K: The log determinant of the prior covariance.
        :arg float log_det_cov: The log determinant of (a factor in) the posterior covariance.
        :arg calligraphic_Z: The array of normalising constants.
        :type calligraphic_Z: :class:`numpy.ndarray`
        :arg bool numerical_stability: If the function is evaluated in a numerically stable way, default `True`. `False`
            is NOT recommended as often np.linalg.det(C) returns a value 0.0.
        :return: fx
        :rtype: float
        """
        trace_Sigma_div_var = np.trace(Sigma_div_var)
        trace_K_inv_Sigma = noise_variance * np.trace(cov)
        log_det_Sigma = log_det_K + N * np.log(noise_variance) + log_det_cov 
        one = - trace_Sigma_div_var / 2
        two = - log_det_K / 2
        three = - trace_K_inv_Sigma / 2
        four = - y.T @ Sigma_div_var @ cov @ y * (1. / 2)
        five = log_det_Sigma / 2
        six = N / 2
        seven = np.sum(calligraphic_Z)
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

        # ## SS
        # trace_Sigma = np.trace(Sigma)
        # if numerical_stability is True:
        #     if K_chol is None:
        #         K_chol = np.linalg.cholesky(K + self.jitter * np.eye(self.N))
        #     if K_inv is None:
        #         K_chol_inv = np.linalg.inv(K_chol)
        #         # NOTE this is mostly the same as np.linalg.inv(C + self.jitter * np.eye(self.N))
        #         # within a small tolerance, but with a few outliers. It may not be stable, but it may be faster
        #         K_inv = K_chol_inv.T @ K_chol_inv
        #     log_det_K = 2. * np.sum(np.log(np.diag(K_chol)))
        #     trace_K_inv_Sigma = N - np.einsum('ij, ji ->', self.cov, K)
        #     Sigma_tilde = self.Sigma + self.jitter * np.eye(self.N)
        #     Sigma_chol = np.linalg.cholesky(Sigma_tilde)
        #     log_det_Sigma = 2. * np.sum(np.log(np.diag(Sigma_chol)))
        # one = - trace_Sigma / (2 * noise_variance)
        # two = - log_det_K / 2
        # three = - trace_K_inv_Sigma / 2
        # four = -m.T @ K_inv @ m / 2
        # five = log_det_Sigma / 2
        # six = N / 2.
        # seven = np.sum(calligraphic_Z)
        # fx = one + two + three + four + five + six  + seven
        # if verbose:
        #     print("one ", one)
        #     print("two ", two)
        #     print("three ", three)
        #     print("four ", four)  # Sometimes largest contribution
        #     print("five ", five)
        #     print("six ", six)
        #     print("seven ", seven)
        #     print('fx = {}'.format(fx))
        return -fx

    def objective_gradient(
            self, gx, intervals, gamma, varphi, noise_variance, noise_std, m, dm, y, p, cov, partial_K_varphi, N,
            calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
            numerical_stability=True, verbose=True, plot=True):
        """
        Calculate gx, the jacobian of the variational lower bound of the log marginal likelihood at the VB equilibrium,

        .. math::
                \mathcal{\frac{\partial F(\Phi)}{\partial \Phi}}

            where :math:`F(\Phi)` is the variational lower bound of the log marginal likelihood at the EP equilibrium,
            :math:`\Phi` is the set of hyperparameters, :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg intervals: The vector of the first cutpoint and the intervals between cutpoints for
            unconstrained optimisation of the cutpoint parameters.
        :type intervals: :class:`numpy.ndarray`
        :arg varphi: The lengthscale parameters.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance:
        :arg float noise_std:
        :arg m: The posterior mean.
        :type m: :class:`numpy.ndarray`
        :arg cov: An intermediate matrix in calculating the posterior covariance, Sigma.
        :type cov: :class:`numpy.ndarray`
        :arg Sigma: The posterior covariance.
        :type Sigma: :class:`numpy.ndarray`
        :arg K_inv: The inverse of the prior covariance.
        :type K_inv: :class:`numpy.ndarray`
        :arg calligraphic_Z: The array of normalising constants.
        :type calligraphic_Z: :class:`numpy.ndarray`
        :arg bool numerical_stability: If the function is evaluated in a numerically stable way, default `True`.
        :return: fx
        :rtype: float
        """
        # For gx[0] -- ln\sigma  # TODO: currently seems analytically incorrect
        if gx[0]:
            one = N - noise_variance * np.trace(cov)
            sigma_dp = self._dp(m, gamma, noise_std, numerically_stable=True)
            two = - (1. / noise_std) * np.sum(sigma_dp)
            if verbose:
                print("one ", one)
                print("two ", two)
                print("gx_sigma = ", one + two)
            gx[0] = one + two
        # For gx[1] -- \b_1
        if gx[1]:
            # TODO: treat these with numerical stability, or fix them
            intermediate_vector_1s = np.divide(norm_pdf_z1s, calligraphic_Z)
            intermediate_vector_2s = np.divide(norm_pdf_z2s, calligraphic_Z)
            indices = np.where(self.t_train == 0)
            gx[1] += np.sum(intermediate_vector_1s[indices])
            for j in range(2, self.J):
                indices = np.where(self.t_train == k - 1)
                gx[k - 1] -= np.sum(intermediate_vector_2s[indices])
                gx[k] += np.sum(intermediate_vector_1s[indices])
            # gx[self.J] -= 0  # Since J is number of classes
            gx[1:self.J] /= noise_std
            # For gx[2:self.J] -- ln\Delta^r
            gx[2:self.J] *= intervals
            if verbose:
                print(gx[2:self.J])
        # For gx[self.J] -- s
        if gx[self.J]:
            raise ValueError("TODO")
        # For kernel parameters
        if gx[self.J + 1]:
            if self.kernel._general and self.kernel._ARD:
                raise ValueError("TODO")
            else:
                if numerical_stability is True:
                    # Using matrix inversion Lemma
                    one = (varphi / 2) * y.T @ cov @ partial_K_varphi @ cov @ y
                    two = - (varphi / 2) * np.einsum('ij, ji ->', partial_K_varphi, cov)
                    three = - varphi * y.T @ cov @ cov @ partial_K_varphi @ cov @ y
                    gx[self.J + 1] = one + two + three
                    if verbose:
                        print("one", one)
                        print("two", two)
                        print("gx = {}".format(gx[self.J + 1]))
                # Update gx[-1], the partial derivative of the lower bound wrt to the lengthscale
        return -gx  # TODO: Correct direction for negative log likelihood minimisation

    def grid_over_hyperparameters(
            self, range_x1, range_x2, res, gamma_0=None, varphi_0=None, noise_variance_0=None, scale_0=None,
            calculate_all_gradients=False, m_0=None, write=False, verbose=False):
        """
        Return meshgrid values of fx and directions of gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs - the rule is that if any of the
        variables are None, then those are the variables to grid over. We can only visualise these surfaces for
        maximum of 2 variables, so the number of combinations is Mc2 + Mc1 where M is the total no. of hyperparameters.

        Special cases are frequent: log and non log variables. 2 axis vs 1 axis objective function, calculate
        new Gram matrix or not. So the simplest way is to combinate manually.
        """
        steps = 100
        (
            x1s, x2s, xlabel, ylabel, xscale, yscale, xx, yy,
            Phi_new, fxs, gxs, intervals, gx_0, indices) = self._grid_over_hyperparameters_initiate(
                gamma_0, varphi_0, noise_variance_0, scale_0, res, range_x1, range_x2, self.J, calculate_all_gradients)
        print("indeces", indices)
        error = np.inf
        fx_old = np.inf
        for i, phi in enumerate(Phi_new):
            gamma, varphi, noise_variance, noise_std, scale = self._grid_over_hyperparameters_update(phi, gamma_0, varphi_0, noise_variance_0, scale_0)
            # Reset error and posterior mean
            iteration = 0
            error = np.inf
            fx_old = np.inf
            # TODO: reset m_0 is None?
            # Convergence is sometimes very fast so this may not be necessary
            while error / steps > self.EPS:
                iteration += 1
                (m_0, dm_0, y, p, *_) = self.estimate(
                    steps, gamma, varphi_tilde_0=varphi, noise_variance=noise_variance, m_tilde_0=m_0,
                    first_step=1, fix_hyperparameters=True, write=False)
                calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, *_ = self._calligraphic_Z(
                    gamma, noise_std, m_0)
                fx = self.objective(self.N, m_0, y, self.Sigma_div_var, self.cov, self.K, calligraphic_Z, noise_variance, self.log_det_K, self.log_det_cov)
                error = np.abs(fx_old - fx)  # TODO: usually converges pretty fast and anyway this is redundant.
                fx_old = fx
                if 1:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(Phi_new)))
            gx = self.objective_gradient(
                gx_0.copy(), intervals, gamma, varphi, noise_variance, noise_std, m_0, dm_0, y, p,
                self.cov, self.partial_K_varphi, self.N, calligraphic_Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
                numerical_stability=True, verbose=True)
            print(gx)
            print(gx[indices])
            fxs[i] = fx
            gxs[i] = gx[indices]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
            print("gamma={}, varphi={}, noise_variance={}, scale={}, fx={}, gx={}".format(
                gamma, varphi, noise_variance, scale, fx, gxs[i]))
        if x2s is not None:
            return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale
        # # return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        # if (gamma_0 is not None
        #         and varphi_0 is None
        #         and noise_variance_0 is None):
        #     return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        # elif (gamma_0 is not None
        #         and varphi_0 is None
        #         and noise_variance_0 is not None
        #         and scale_0 is None):
        #     return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        # elif (gamma_0 is not None
        #         and noise_variance_0 is not None
        #         and varphi_0 is None):
        #     return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale
        # elif (gamma_0 is not None
        #         and noise_variance_0 is not None
        #         and varphi_0 is not None
        #         and scale_0 is not None):
        #     return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        # elif (gamma_0 is not None
        #         and noise_variance_0 is None
        #         and varphi_0 is not None):
        #     return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale
        # elif (gamma_0 is not None
        #         and noise_variance_0 is not None
        #         and varphi_0 is not None
        #         and scale_0 is None):
        #     return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def _hyperparameter_training_step_initialise(self, theta):
        """
        Initialise the hyperparameter training step.

        :arg theta: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}} \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\varphi}],

            where :math:`\sigma` is the noise standard deviation, :math:`\b_{1}` is the first cutpoint,
            :math:`\Delta_{l}` is the :math:`l`th cutpoint interval, :math:`\varphi` is the single shared lengthscale
            parameter or vector of parameters in which there are in the most general case J * D parameters.
        :type theta: :class:`numpy.ndarray`
        :return: (gamma, noise_variance) the updated cutpoints and noise variance.
        :rtype: (2,) tuple
        """
        noise_std = np.exp(theta[0])
        noise_variance = noise_std**2
        if noise_variance < 1.0e-04:
            warnings.warn("WARNING: noise variance is very low - numerical stability issues may arise "
                "(noise_variance={}).".format(noise_variance))
        elif noise_variance > 1.0e3:
            warnings.warn("WARNING: noise variance is very large - numerical stability issues may arise "
                "(noise_variance={}).".format(noise_variance))
        gamma = np.empty((self.J + 1,))  # including all of the cutpoints
        gamma[0] = np.NINF
        gamma[-1] = np.inf
        gamma[1] = theta[1]
        for i in range(2, self.J):
            gamma[i] = gamma[i - 1] + np.exp(theta[i])
        scale_std = np.exp(theta[self.J])
        scale = scale_std**2
        if self.kernel._general and self.kernel._ARD:
            # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
            # and lengthscales parameter for each dimension and class
            varphi = np.exp(np.reshape(theta[self.J:self.J + self.J * self.D], (self.J, self.D)))
        else:
            # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
            # and a single, shared lengthscale parameter
            varphi = np.exp(theta[self.J])
        # Update prior covariance
        self._hyperparameters_update(varphi=varphi, noise_variance=noise_variance)
        return gamma, varphi, noise_variance, scale

    def hyperparameter_training_step(
            self, theta, gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients, posterior_mean_0=None, Sigma_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=True):
        """
        TODO: Needs completing. Probably is best to precalculate gx_0 outside this method.
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :arg steps:
        :arg posterior_mean_0:
        :arg Sigma_0:
        :arg mean_EP_0:
        :arg precision_EP_0:
        :arg amplitude_EP_0:
        :arg varphi_0:
        :arg psi_0:
        :arg grad_Z_wrt_cavity_mean_0:
        :arg first_step:
        :arg fix_hyperparameters:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        steps = self.N
        error = np.inf
        iteration = 0
        gx, indices = self._objective_gradient_initiate(gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients)
        # Update prior covariance, get hyperparameters from theta
        gamma, varphi, noise_variance, scale = self._hyperparameter_training_step_initialise(theta, gx, calculate_all_gradients)
        posterior_mean = posterior_mean_0
        Sigma = Sigma_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        intervals = gamma[2:self.J] - gamma[1:self.J - 1]
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
             precision_EP, amplitude_EP, containers) = self.estimate(
                steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, first_step=first_step, write=write)
            if verbose:
                print("({}), error={}".format(iteration, error))
        weights, precision_EP, Lambda_cholesky, Lambda = self.compute_EP_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        if write:
            (posterior_means, Sigmas, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_marginal_likelihoods) = containers
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            gamma, Sigma, precision_EP,
            posterior_mean, noise_variance)
        # t1, t2, t3, t4, t5 = self.compute_integrals(
        #     gamma, Sigma, precision_EP, posterior_mean, noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
        gx = self.objective_gradient(
            gx, intervals, gamma, self.kernel.varphi, noise_variance, t2, t3, t4, t5, Lambda, weights)
        # Only some variables are being optimized over
        gx = gx[indices]
        if verbose:
            print(repr(gamma), ",")
            print(self.kernel.varphi, ",")
            print(noise_variance, ",")
            print("\nfunction_eval={}\n jacobian_eval={}".format(
                fx, gx))
        else:
            print("gamma={}, noise_variance={}, varphi={}\nfunction_eval={}".format(
                gamma, noise_variance, self.kernel.varphi, fx))
        return fx, gx


class EPOrderedGP(Estimator):
    """
    An Expectation Propagation classifier for ordered likelihood. Inherits the Estimator ABC.

    Expectation propagation algorithm as written in Appendix B
    Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.

    This class allows users to define a classification problem and get predictions
    using approximate Bayesian inference. It is for ordered likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian Process.
    """
    def __init__(self, *args, **kwargs):
        """
        Create an :class:`EPOrderedGP` Estimator object.

        :returns: An :class:`EPOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
        if self.kernel._ARD:
            raise ValueError('The kernel must not be _ARD type (kernel._ARD=1),'
                             ' but ISO type (kernel._ARD=0). (got {}, expected)'.format(
                self.kernel._ARD, 0))
        if self.kernel._general:
            raise ValueError('The kernel must not be general type (kernel._general=1),'
                             ' but simple type (kernel._general=0). (got {}, expected)'.format(
                self.kernel._general, 0))
        self.grid = np.ogrid[0:self.N] # All indices for sequential message passing
        self.EPS = 0.001  # Acts as a machine tolerance
        # Threshold of single sided standard deviations that normal cdf can be approximated to 0 or 1
        self.upper_bound = 4 

    def _estimate_initiate(
            self, gamma, varphi, noise_variance=None, posterior_mean_0=None, Sigma_0=None,
            mean_EP_0=None, precision_EP_0=None, amplitude_EP_0=None):
        """
        Initialise the Estimator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Estimator.
        :arg gamma: The (J + 1, ) array of cutpoint parameters \bm{gamma}.
        :type gamma: :class:`numpy.ndarray`
        :arg varphi: Initialisation of hyperparameters.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance: The variance of the noise model. If `None` then initialised to one, default `None`.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If `None` then initialised to zeros,
            default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg Sigma_0: The initial state of the posterior covariance (N,). If `None` then initialised to prior
            covariance, default `None`.
        :type Sigma_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,). If `None` then initialised to zeros,
            default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site) variance (N,). If `None` then initialised to zeros,
            default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site) amplitudes (N,). If `None` then initialised to ones,
            default `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg psi_0: Initialisation of hyperhyperparameters. If `None` then initialised to ones, default `None`.
        :type psi_0: :class:`numpy.ndarray` or float
        :arg grad_Z_wrt_cavity_mean_0: Initialisation of the EP weights, which are gradients of the approximate marginal
            likelihood wrt to the 'cavity distribution mean'. If `None` then initialised to zeros, default `None`.
        :type grad_Z_wrt_cavity_mean_0: :class:`numpy.ndarray`
        :return: Containers for the mean estimates of parameters and hyperparameters.
        :rtype: (12,) tuple.
        """
        # Treat user parsing of cutpoint parameters with just the upper cutpoints for each class
        if np.shape(gamma)[0] == self.J - 1:  # not including any of the fixed cutpoints: -\infty, \infty
            gamma = np.append(gamma, np.inf)  # append the infinity cutpoint
            gamma = np.insert(gamma, np.NINF)  # insert the negative infinity cutpoint at index 0
            pass  # correct format
        elif np.shape(gamma)[0] == self.J:  # not including one of the infinity cutpoints
            if gamma[-1] != np.inf:
                if gamma[0] != np.NINF:
                    raise ValueError('The last cutpoint parameter must be numpy.inf, or the first cutpoint parameter'
                                     ' must be numpy.NINF (got {}, expected {})'.format(
                        [gamma[0], gamma[-1]], [np.inf, np.NINF]))
                else:  # gamma[0] is negative infinity
                    gamma.append(np.inf)
                    pass  # correct format
            else:
                gamma = np.insert(gamma, np.NINF)
                pass  # correct format
        elif np.shape(gamma)[0] == self.J + 1:  # including all of the cutpoints
            if gamma[0] != np.NINF:
                raise ValueError('The cutpoint parameter \gamma must be numpy.NINF (got {}, expected {})'.format(
                    gamma[0], np.NINF))
            if gamma[-1] != np.inf:
                raise ValueError('The cutpoint parameter \gamma_J must be numpy.inf (got {}, expected {})'.format(
                    gamma[-1], np.inf))
            pass  # correct format
        else:
            raise ValueError('Could not recognise gamma shape. (gamma={}, np.shape(gamma) was {}, J = {})'.format(
                gamma, np.shape(gamma), self.J))
        assert gamma[0] == np.NINF
        assert gamma[-1] == np.inf
        assert np.shape(gamma)[0] == self.J + 1
        if not all(
                gamma[i] <= gamma[i + 1]
                for i in range(self.J)):
            raise CutpointValueError(gamma)
        if Sigma_0 is None:
            # Since the first EP approximation before data-update is simply the GP prior
            Sigma_0 = self.K
        if mean_EP_0 is None:
            mean_EP_0 = np.zeros((self.N,))  # TODO: If there is no need to store in the memory the steps, then why would that be the case in ADF
        if precision_EP_0 is None:
            precision_EP_0 = np.zeros((self.N,))
        if amplitude_EP_0 is None:
            amplitude_EP_0 = np.ones((self.N,))
        if posterior_mean_0 is None:
            posterior_mean_0 = (Sigma_0 @ np.diag(precision_EP_0)) @ mean_EP_0
        if noise_variance is None:
            noise_variance = 1.0
        error = 0.0
        grad_Z_wrt_cavity_mean_0 = np.empty((self.N,))  # Initialisation does not matter.
        posterior_means = []
        Sigmas = []
        mean_EPs = []
        amplitude_EPs = []
        precision_EPs = []
        approximate_marginal_likelihoods = []
        containers = (posterior_means, Sigmas, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_marginal_likelihoods)
        return (gamma, noise_variance, posterior_mean_0, Sigma_0, mean_EP_0,
                precision_EP_0, amplitude_EP_0, grad_Z_wrt_cavity_mean_0, containers, error)

    def estimate(self, steps, gamma, varphi, noise_variance, posterior_mean_0=None, Sigma_0=None, mean_EP_0=None,
                 precision_EP_0=None, amplitude_EP_0=None, first_step=1, fix_hyperparameters=True, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal likelihood) via Expectation propagation
        iteration as written in Appendix B
        Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.

        EP does not attempt to learn a posterior distribution over hyperparameters, but instead tries to approximate
        the joint posterior given some hyperparameters (which have to be optimised during model selection).

        :arg int steps: The number of iterations the Estimator takes.
        :arg gamma: The (J + 1, ) array of cutpoint parameters \bm{gamma}.
        :type gamma: :class:`numpy.ndarray`
        :arg varphi: Initialisation of hyperparameter posterior mean estimates. If `None` then initialised to ones,
            default `None`.
        :type varphi: :class:`numpy.ndarray` or float
        :arg float noise_variance: Initialisation of noise variance. If `None` then initialised to one, default `None`.
        :arg posterior_mean_0: The initial state of the approximate posterior mean (N,). If `None` then initialised to
            zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg Sigma_0: The initial state of the posterior covariance (N, N). If `None` then initialised to prior
            covariance, default `None`.
        :type Sigma_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,). If `None` then initialised to zeros,
            default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site) variance (N,). If `None` then initialised to
            zeros, default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site) amplitudes (N,). If `None` then initialised to
            ones, default `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool fix_hyperparameters: Must be `True`, since the hyperparameter approximate posteriors are
            of the hyperparameters are not calculated in this EP approximation.
        :arg bool write: Boolean variable to store and write arrays of interest. If set to "True", the method will
            output non-empty containers of evolution of the statistics over the steps. If set to "False", statistics
            will not be written and those containers will remain empty.
        :return: Posterior mean and covariance estimates.
        :rtype: (8, ) tuple of :class:`numpy.ndarrays` of the approximate posterior means, other statistics and
            tuple of lists of per-step evolution of those statistics.
        """
        if fix_hyperparameters is False:
            return ValueError("fix_hyperparameters must be True (got False, expected True)")
        (gamma, noise_variance, posterior_mean, Sigma, mean_EP, precision_EP,
         amplitude_EP, grad_Z_wrt_cavity_mean, containers, error) = self._estimate_initiate(
            gamma, varphi, noise_variance, posterior_mean_0, Sigma_0, mean_EP_0, precision_EP_0, amplitude_EP_0)
        (posterior_means, Sigmas, mean_EPs, precision_EPs, amplitude_EPs,
         approximate_log_marginal_likelihoods) = containers
        for step in trange(first_step, first_step + steps,
                        desc="EP GP priors Estimator Progress", unit="iterations", disable=True):
            index = self.new_point(step, random_selection=False)
            # Find the mean and variance of the leave-one-out posterior distribution Q^{\backslash i}(\bm{f})
            (Sigma, Sigma_nn, posterior_mean, cavity_mean_n, cavity_variance_n, mean_EP_n_old,
             precision_EP_n_old, amplitude_EP_n_old) = self._remove(
                index, Sigma, posterior_mean, mean_EP, precision_EP, amplitude_EP)
            # Tilt/ moment match
            (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n, grad_Z_wrt_cavity_mean_n, grad_Z_wrt_cavity_mean,
             posterior_mean, posterior_mean_n_new, posterior_covariance_n_new, z1, z2, nu_n) = self._include(
                index, posterior_mean, cavity_mean_n, cavity_variance_n, gamma, noise_variance, grad_Z_wrt_cavity_mean)
            diff = precision_EP_n - precision_EP_n_old
            if (
                    np.abs(diff) > self.EPS
                    and Z_n > self.EPS
                    and precision_EP_n > 0.0
                    and posterior_covariance_n_new > 0.0
            ):
                # Update posterior mean and rank-1 covariance
                Sigma, posterior_mean = self._update(
                    index, mean_EP_n_old, Sigma, Sigma_nn, precision_EP_n_old, grad_Z_wrt_cavity_mean_n,
                    posterior_mean_n_new, posterior_mean, posterior_covariance_n_new, diff)
                # Update EP parameters
                precision_EP[index] = precision_EP_n
                mean_EP[index] = mean_EP_n
                amplitude_EP[index] = amplitude_EP_n
                error += (diff**2
                          + (mean_EP_n - mean_EP_n_old)**2
                          + (amplitude_EP_n - amplitude_EP_n_old)**2)
                if write:
                    # approximate_log_marginal_likelihood = self._approximate_log_marginal_likelihood(
                    # Sigma, precision_EP, mean_EP)
                    posterior_means.append(posterior_mean)
                    Sigmas.append(Sigma)
                    mean_EPs.append(mean_EP)
                    precision_EPs.append(precision_EP)
                    amplitude_EPs.append(amplitude_EP)
                    # approximate_log_marginal_likelihood.append(approximate_marginal_log_likelihood)
            else:
                if precision_EP_n < 0.0 or posterior_covariance_n_new < 0.0:
                    print("Skip {} update z1={}, z2={}, nu={} p_new={}, p_old={}.\n".format(
                        index, z1, z2, nu_n, precision_EP_n, precision_EP_n_old))
        containers = (posterior_means, Sigmas, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_log_marginal_likelihoods)
        return error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP, precision_EP, amplitude_EP, containers

    def new_point(self, step, random_selection=True):
        """
        Return a new point based on some policy.

        :arg int step: The current iteration step.
        :arg bool random_selection: If random_selection is true, then returns a random point from the ordering.
            Otherwise, it returns a sequential point. Default `True`.
        :return: index
        """
        if random_selection:
            return random.randint(0, self.N-1)
        else:
            return step % self.N

    def _remove(self, index, Sigma, posterior_mean, mean_EP, precision_EP, amplitude_EP):
        """
        Calculate the product of approximate posterior factors with the current index removed.

        This is called the cavity distribution, "a bit like leaving a hole in the dataset".
        :arg int index: The index of the current site (the index of the datapoint that is "left out").
        :arg Sigma: The current posterior covariance estimate (N, N).
        :type Sigma: :class:`numpy.ndarray`
        :arg posterior_mean: The state of the approximate posterior mean (N,).
        :type posterior_mean: :class:`numpy.ndarray`
        :arg mean_EP: The state of the individual (site) mean (N,).
        :type mean_EP: :class:`numpy.ndarray`
        :arg precision_EP: The state of the individual (site) variance (N,).
        :type precision_EP: :class:`numpy.ndarray`
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :type amplitude_EP: :class:`numpy.ndarray`
        :returns: A (8,) tuple containing cavity mean and variance, and old site states.
        """
        mean_EP_n_old = mean_EP[index]
        diag_Sigma = np.diag(Sigma)  # (N,)
        Sigma_nn = diag_Sigma[index]  # Variance of the latent function at our x_index
        precision_EP_n_old = precision_EP[index]
        amplitude_EP_n_old = amplitude_EP[index]
        posterior_mean_n = posterior_mean[index]
        if Sigma_nn > 0:
            cavity_variance_n = Sigma_nn / (1 - Sigma_nn * precision_EP_n_old)
            if cavity_variance_n > 0:
                cavity_mean_n = posterior_mean_n + cavity_variance_n * precision_EP_n_old * (
                        posterior_mean_n - mean_EP_n_old)
            else:
                raise ValueError("cavity_variance_n must be non-negative (got {})".format(cavity_variance_n))
        else:
            raise ValueError("Sigma_nn must be non-negative (got {})".format(Sigma_nn))
        return (Sigma, Sigma_nn, posterior_mean, cavity_mean_n, cavity_variance_n,
                mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old)

    def _include(self, index, posterior_mean, cavity_mean_n, cavity_variance_n, gamma, noise_variance,
                 grad_Z_wrt_cavity_mean, numerically_stable=True):
        """
        Update the approximate posterior by incorporating the message p(t_i|m_i) into Q^{\i}(\bm{f}).

        Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)

        This includes one true-observation likelihood, and 'tilts' the approximation towards the true posterior.
        It updates the approximation to the true posterior by minimising a moment-matching KL divergence between the
        tilted distribution and the posterior distribution. This gives us an approximate posterior in the approximating
        family. The update to Sigma is a rank-1 update (see the outer product of two 1d vectors), and so it essentially
        constructs a piecewise low rank approximation to the GP posterior covariance matrix, until convergence
        (by which point it will no longer be low rank).

        :arg int index: The index of the current site (the index of the datapoint that is "left out").
        :arg posterior_mean: The state of the approximate posterior mean (N,).
        :type posterior_mean: :class:`numpy.ndarray`
        :arg float cavity_mean_n: The cavity mean of the current site.
        :arg float cavity_variance_n: The cavity variance of the current site.
        :arg gamma: The (J + 1, ) array of cutpoint parameters \bm{gamma}.
        :type gamma: :class:`numpy.ndarray`
        :arg float noise_variance: Initialisation of noise variance. If `None` then initialised to one, default `None`.
        :arg grad_Z_wrt_cavity_mean: The gradient of the log normalising constant with respect to
            the cavity mean (The EP "weights").
        :type grad_Z_wrt_cavity_mean: :class:`numpy.ndarray`
        :arg bool numerically_stable: Option to employ a technically more numerically stable implementation. When
            "True", will resort to thresholding at the cost of accuracy, when "False" will use an experimental,
            potentially more accurate, but less stable implementation.
        :returns: A (11,) tuple containing cavity mean and variance, and old site states.
        """
        variance = cavity_variance_n + noise_variance
        std_dev = np.sqrt(variance)
        target = self.t_train[index]
        # TODO: implement this method in other implementations.<<<
        if numerically_stable:
            # Compute Z
            norm_cdf_z2 = 0.0
            norm_cdf_z1 = 1.0
            norm_pdf_z1 = 0.0
            norm_pdf_z2 = 0.0
            z1 = 0.0
            z2 = 0.0
            if target == 0:
                z1 = (gamma[target + 1] - cavity_mean_n) / std_dev
                z1_abs = np.abs(z1)
                if z1_abs > self.upper_bound:
                    z1 = np.sign(z1) * self.upper_bound
                Z_n = norm.cdf(z1) - norm_cdf_z2
                norm_pdf_z1 = norm.pdf(z1)
            elif target == self.J - 1:
                z2 = (gamma[target] - cavity_mean_n) / std_dev
                z2_abs = np.abs(z2)
                if z2_abs > self.upper_bound:
                    z2 = np.sign(z2) * self.upper_bound
                Z_n = norm_cdf_z1 - norm.cdf(z2)
                norm_pdf_z2 = norm.pdf(z2)
            else:
                z1 = (gamma[target + 1] - cavity_mean_n) / std_dev
                z2 = (gamma[target] - cavity_mean_n) / std_dev
                Z_n = norm.cdf(z1) - norm.cdf(z2)
                norm_pdf_z1 = norm.pdf(z1)
                norm_pdf_z2 = norm.pdf(z2)
            if Z_n < self.EPS:
                if np.abs(np.exp(-0.5*z1**2 + 0.5*z2**2) - 1.0) > self.EPS**2:
                    grad_Z_wrt_cavity_mean_n = (z1 * np.exp(-0.5*z1**2 + 0.5*z2**2) - z2**2) / (
                        ((np.exp(-0.5 * z1 ** 2) + 0.5 * z2 ** 2) - 1.0) * variance
                    )
                    grad_Z_wrt_cavity_variance_n = (-1.0 + (z1**2 + 0.5 * z2**2) - z2**2)/(
                        ((np.exp(-0.5*z1**2 + 0.5 * z2**2) - 1.0) * 2.0 * variance)
                    )
                    grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
                    nu_n = grad_Z_wrt_cavity_mean_n_2 - 2.0 * grad_Z_wrt_cavity_variance_n
                else:
                    grad_Z_wrt_cavity_mean_n = 0.0
                    grad_Z_wrt_cavity_mean_n_2 = 0.0
                    grad_Z_wrt_cavity_variance_n = -(1.0 - self.EPS)/(2.0 * variance)
                    nu_n = (1.0 - self.EPS) / variance
                    warnings.warn(
                        "Z_n must be greater than tolerance={} (got {}): SETTING to Z_n to approximate value\n"
                        "z1={}, z2={}".format(
                            self.EPS, Z_n, z1, z2))
                if nu_n >= 1.0 / variance:
                    nu_n = (1.0 - self.EPS) / variance
                if nu_n <= 0.0:
                    nu_n = self.EPS * variance
            else:
                grad_Z_wrt_cavity_variance_n = (- z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (2.0 * variance * Z_n)  # beta
                grad_Z_wrt_cavity_mean_n = (- norm_pdf_z1 + norm_pdf_z2) / (std_dev * Z_n)  # alpha/gamma
                grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
                nu_n = grad_Z_wrt_cavity_mean_n_2 - 2.0 * grad_Z_wrt_cavity_variance_n
        else:
            z1 = (gamma[target + 1] - cavity_mean_n) / std_dev
            z2 = (gamma[target] - cavity_mean_n) / std_dev
            norm_cdf_z1 = norm.cdf(z1)
            norm_cdf_z2 = norm.cdf(z2)
            norm_pdf_z1 = norm.pdf(z1)
            norm_pdf_z2 = norm.pdf(z2)
            Z_n = norm_cdf_z1 - norm_cdf_z2
            if not Z_n > self.EPS:
                warnings.warn("Z_n must be greater than tolerance={} (got {}): SETTING to Z_n=tolerance\n"
                              "z1={}, z2={}".format(
                    self.EPS, Z_n, z1, z2))
                Z_n = self.EPS
            if math.isinf(z1) and math.isinf(z2):
                grad_Z_wrt_cavity_variance_n = 0.0
            elif math.isinf(z2):
                grad_Z_wrt_cavity_variance_n = (- z1 * norm_pdf_z1) / (2.0 * variance * Z_n)
            elif math.isinf(z1):
                grad_Z_wrt_cavity_variance_n = (z2 * norm_pdf_z2) / (2.0 * variance * Z_n)
            else:
                grad_Z_wrt_cavity_variance_n = (- z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (2.0 * variance * Z_n)  # beta
            grad_Z_wrt_cavity_mean_n = (- norm_pdf_z1 + norm_pdf_z2) / (std_dev * Z_n)  # alpha/gamma
            grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n ** 2
            nu_n = grad_Z_wrt_cavity_mean_n_2 - 2.0 * grad_Z_wrt_cavity_variance_n
            if nu_n >= 1.0 / variance:
                warnings.warn("nu_n must be greater than 1. / variance (got {}): "
                              "SETTING nu_n=(1.0 - self.EPS) / variance = {}\n"
                              "z1={}, z2={}".format(
                    nu_n, (1.0 - self.EPS) / variance, z1, z2))
                nu_n = (1.0 - self.EPS) / variance
            if nu_n <= 0.0:
                warnings.warn("nu_n must be greater than zero (got {}): SETTING nu_n=tolerance={}\n"
                              "z1={}, z2={}".format(
                    nu_n, self.EPS, z1, z2))
                #nu_n = self.EPS * variance
                nu_n = self.EPS
        # Update alphas
        grad_Z_wrt_cavity_mean[index] = grad_Z_wrt_cavity_mean_n
        if math.isnan(grad_Z_wrt_cavity_mean_n):
            print("cavity_mean_n", cavity_mean_n)
            print("cavity_variance_n", cavity_variance_n)
            print("target", target)
            print("z1", z1)
            print("z2", z2)
            print("Z_n", Z_n)
            print("norm_pdf_z1", "norm_pdf_z2", norm_pdf_z1, norm_pdf_z2)
            print("beta", grad_Z_wrt_cavity_variance_n)
            print("alpha", grad_Z_wrt_cavity_mean_n)
            raise ValueError("grad_Z_wrt_cavity_mean is nan (got {})".format(grad_Z_wrt_cavity_mean_n))
        if math.isnan(grad_Z_wrt_cavity_variance_n):
            print("cavity_mean_n", cavity_mean_n)
            print("cavity_variance_n", cavity_variance_n)
            print("target", target)
            print("z1", z1)
            print("z2", z2)
            print("Z_n", Z_n)
            print("norm_pdf_z1", "norm_pdf_z2", norm_pdf_z1, norm_pdf_z2)
            print("beta", grad_Z_wrt_cavity_variance_n)
            print("alpha", grad_Z_wrt_cavity_mean_n)
            raise ValueError("grad_Z_wrt_cavity_variance is nan (got {})".format(grad_Z_wrt_cavity_variance_n))
        if nu_n <= 0:
            print("cavity_mean_n", cavity_mean_n)
            print("cavity_variance_n", cavity_variance_n)
            print("target", target)
            print("z1", z1)
            print("z2", z2)
            print("Z_n", Z_n)
            print("norm_pdf_z1", "norm_pdf_z2", norm_pdf_z1, norm_pdf_z2)
            print("beta", grad_Z_wrt_cavity_variance_n)
            print("alpha", grad_Z_wrt_cavity_mean_n)
            raise ValueError("nu_n must be positive (got {})".format(nu_n))
        if nu_n > 1.0 / variance + self.EPS:
            print("cavity_mean_n", cavity_mean_n)
            print("cavity_variance_n", cavity_variance_n)
            print("target", target)
            print("target", target)
            print("z1", z1)
            print("z2", z2)
            print("Z_n", Z_n)
            print("norm_pdf_z1", "norm_pdf_z2", norm_pdf_z1, norm_pdf_z2)
            print("beta", grad_Z_wrt_cavity_variance_n)
            print("alpha", grad_Z_wrt_cavity_mean_n)
            raise ValueError("nu_n must be less than 1.0 / (cavity_variance_n + noise_variance) = {}, got {}".format(
                1.0 / variance, nu_n))
        # hnew = loomean + loovar * alpha;
        posterior_mean_n_new = cavity_mean_n + cavity_variance_n * grad_Z_wrt_cavity_mean_n
        # cnew = loovar - loovar * nu * loovar;
        posterior_covariance_n_new = cavity_variance_n - cavity_variance_n**2 * nu_n
        # pnew = nu / (1.0 - loovar * nu);
        precision_EP_n = nu_n / (1.0 - cavity_variance_n * nu_n)
        # print("posterior_mean_n_new", posterior_mean_n_new)
        # print("nu_n", nu_n)
        # print("precision_EP_n", precision_EP_n)
        # mnew = loomean + alpha / nu;
        mean_EP_n = cavity_mean_n + grad_Z_wrt_cavity_mean_n / nu_n
        # snew = Zi * sqrt(loovar * pnew + 1.0)*exp(0.5 * alpha * alpha / nu);
        amplitude_EP_n = Z_n * np.sqrt(cavity_variance_n * precision_EP_n + 1.0) * np.exp(
            0.5 * grad_Z_wrt_cavity_mean_n_2 / nu_n)
        return (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
                grad_Z_wrt_cavity_mean_n, grad_Z_wrt_cavity_mean, posterior_mean,
                posterior_mean_n_new, posterior_covariance_n_new, z1, z2, nu_n)

    def _update(self, index, mean_EP_n_old, Sigma, Sigma_nn, precision_EP_n_old, grad_Z_wrt_cavity_mean_n,
                posterior_mean_n_new, posterior_mean, posterior_covariance_n_new, diff):
        """
        Update the posterior mean and covariance.

        Projects the tilted distribution on to an approximating family, giving us a projection onto the approximating
        family. The update for the t_n is a rank-1 update. Constructs a low rank approximation to the GP posterior
        covariance matrix.

        :arg int index: The index of the current likelihood (the index of the datapoint that is "left out").
        :arg float mean_EP_n_old: The state of the individual (site) mean (N,).
        :arg Sigma: The current posterior covariance estimate (N, N).
        :type Sigma: :class:`numpy.ndarray`
        :arg float Sigma_nn: The current site posterior covariance estimate.
        :arg float precision_EP_n_old: The state of the individual (site) variance (N,).
        :arg float grad_Z_wrt_cavity_mean_n: The gradient of the log normalising constant with respect to
            the site cavity mean (The EP "weight").
        :arg float posterior_mean_n_new: The state of the site approximate posterior mean.
        :arg float posterior_covariance_n_new: The state of the site approximate posterior variance.
        :arg float diff: The differance between precision_EP_n and precision_EP_n_old.
        :returns: The updated posterior mean and covariance estimates.
        :rtype: tuple (`numpy.ndarray`, `numpy.ndarray`)
        """
        # rho = diff/(1+diff*Aii);
        rho = diff / (1 + diff * Sigma_nn)
		# eta = (alpha+epinvvar*(postmean-epmean))/(1.0-Aii*epinvvar) ;
        eta = (grad_Z_wrt_cavity_mean_n + precision_EP_n_old * (posterior_mean[index] - mean_EP_n_old)) / (
                1.0 - Sigma_nn * precision_EP_n_old)
        # ai[i] = Retrieve_Posterior_Covariance (i, index, settings) ;
        a_n = Sigma[:, index]  # The index'th column of Sigma
        ##a_n = Sigma[index, :]
        # postcov[j]-=rho*ai[i]*ai[j] ;
        Sigma = Sigma - rho * np.outer(a_n, a_n)
        # postmean+=eta*ai[i];
        posterior_mean += eta * a_n
        # assert(fabs((settings->alpha+index)->pair->postmean-alpha->hnew)<EPS)
        if np.abs(posterior_covariance_n_new - Sigma[index, index]) > self.EPS:
            raise ValueError("np.abs(posterior_covariance_n_new - Sigma[index, index]) must be less than some "
                             "tolerance. Got (posterior_covariance_n_new={}, Sigma_index_index={}, diff={})".format(
                posterior_covariance_n_new, Sigma[index, index], posterior_covariance_n_new - Sigma[index, index]))
        # assert(fabs((settings->alpha+index)->postcov[index]-alpha->cnew)<EPS)
        if np.abs(posterior_mean_n_new - posterior_mean[index]) > self.EPS:
            raise ValueError("np.abs(posterior_mean_n_new - posterior_mean[index]) must be less than some tolerance."
                             " Got (posterior_mean_n_new={}, posterior_mean_index={}, diff={})".format(
                posterior_mean_n_new, posterior_mean[index], posterior_mean_n_new - posterior_mean[index]))
        return Sigma, posterior_mean

    def _approximate_log_marginal_likelihood(self, Sigma, precision_EP, amplitude_EP, mean_EP, numerical_stability):
        """
        Calculate the approximate log marginal likelihood. TODO: need to test this.

        :arg Sigma: The approximate posterior covariance.
        :arg mean_EP: The state of the individual (site) mean (N,).
        :arg precision_EP: The state of the individual (site) variance (N,).
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :arg bool numerical_stability: If the calculation is made in a numerically stable manner.
        """
        precision_matrix = np.diag(precision_EP)
        inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
        log_amplitude_EP = np.log(amplitude_EP)
        intermediate_vector = np.multiply(mean_EP, precision_EP)
        B = intermediate_vector.T @ Sigma @ intermediate_vector - intermediate_vector.T @ mean_EP
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

    def _predict_vector(self, gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda):
        """
        Make EP prediction over classes of X_test given the posterior samples.
        :arg gamma:
        :arg Sigma:
        :arg posterior_mean:
        :arg varphi:
        :arg X_test: The new data points, array like (N_test, D).
        :arg Lambda: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        # error = 0.0
        # absolute_error = 0.0
        Pi_inv = np.diag(1. / precision_EP)
        # Lambda = np.linalg.inv(np.add(Pi_inv, self.K))  # (N, N)
        Lambda_chol = np.linalg.cholesky(np.add(Pi_inv, self.K))  # The result of using the EP point precisions is that it is overcondident: we'd rathe
        Lambda_chol_inv = np.linalg.inv(Lambda_chol)
        Lambda = Lambda_chol_inv.T @ Lambda_chol_inv
        N_test = np.shape(X_test)[0]
        # Update the kernel with new varphi
        self.kernel.varphi = varphi
        # C_news[:, i] is C_new for X_test[i]
        C_news = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        # TODO: this is a bottleneck
        c_news = np.diag(self.kernel.kernel_matrix(X_test, X_test))  # (N_test, )
        # intermediate_vectors[:, i] is intermediate_vector for X_test[i]

        # Superceded
        # var_new = c_news - intermediate_scalars + noise_variance
        # print(var_new)
        # std_new = np.sqrt(var_new)
        # mean_new = np.einsum('ij, i -> j', intermediate_vectors, mean_EP)
        # predictive_distributions = np.empty((N_test, self.J))
        # for j in range(self.J):
        #     Z1 = np.divide(np.subtract(gamma[k + 1], mean_new), std_new)
        #     Z2 = np.divide(np.subtract(gamma[k], mean_new), std_new)
        #     predictive_distributions[:, k] = norm.cdf(Z1) - norm.cdf(Z2)
        # return predictive_distributions  # (N_test, J)

        intermediate_vectors = Lambda @ C_news  # (N, N_test)
        intermediate_scalars = np.einsum('ij, ij -> j', C_news, intermediate_vectors)
        posterior_var = c_news - intermediate_scalars
        posterior_pred_var = posterior_var + noise_variance
        posterior_std = np.sqrt(posterior_var)
        posterior_pred_std = np.sqrt(posterior_pred_var)
        posterior_pred_mean = np.einsum('ij, i -> j', intermediate_vectors, mean_EP)
        predictive_distributions = np.empty((N_test, self.J))
        for j in range(self.J):
            Z1 = np.divide(np.subtract(gamma[k + 1], posterior_pred_mean), posterior_pred_std)
            Z2 = np.divide(np.subtract(gamma[k], posterior_pred_mean), posterior_pred_std)
            predictive_distributions[:, k] = norm.cdf(Z1) - norm.cdf(Z2)
        return predictive_distributions, posterior_pred_mean, posterior_std  # (N_test, J), (N_test,), (N_test,)

    def predict(self, gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda, vectorised=True):
        """
        Return the posterior predictive distribution over classes.

        :arg Sigma: The EP posterior covariance estimate.
        :arg y_tilde: The posterior mean estimate of the latent variable Y.
        :arg varphi_tilde: The posterior mean estimate of the hyper-parameters varphi.
        :arg X_test: The new data points, array like (N_test, D).
        :arg n_samples: The number of samples in the Monte Carlo estimate.
        :return: A Monte Carlo estimate of the class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError('For the ordered likelihood estimator, the kernel must not be _ARD type'
                             ' (kernel._ARD=1), but ISO type (kernel._ARD=0). (got {}, expected)'.format(
                self.kernel._ARD, 0))
        else:
            if vectorised:
                return self._predict_vector(gamma, Sigma, mean_EP, precision_EP, varphi, noise_variance, X_test, Lambda)
            else:
                return ValueError("The scalar implementation has been superseded. Please use "
                                  "the vector implementation.")

    def _hyperparameter_training_step_initialise(self, theta, gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients):
        """
        Initialise the hyperparameter training step.

        :arg theta: The set of (log-)hyperparameters
            .. math::
                [\log{\sigma} \log{b_{1}} \log{\Delta_{1}} \log{\Delta_{2}} ... \log{\Delta_{J-2}} \log{\varphi}],
            or
            .. math::
                [\log{\varphi}],
            or
            .. math::
                [\log(\sigma), \log{\varphi}],
            where :math:`\sigma` is the noise standard deviation, :math:`\b_{1}` is the first cutpoint,
            :math:`\Delta_{l}` is the :math:`l`th cutpoint interval, :math:`\varphi` is the single shared lengthscale
            parameter or vector of parameters in which there are in the most general case J * D parameters.
        :type theta: :class:`numpy.ndarray`
        :return: (gamma, noise_variance) the updated cutpoints and noise variance.
        :rtype: (2,) tuple
        """
        if calculate_all_gradients:
            # Optimize all hyperparameters
            noise_std = np.exp(theta[0])
            noise_variance = noise_std**2
            if noise_variance < 1.0e-04:
                warnings.warn("WARNING: noise variance is very low - numerical stability issues may arise "
                            "(noise_variance={}).".format(noise_variance))
            elif noise_variance > 1.0e3:
                warnings.warn("WARNING: noise variance is very large - numerical stability issues may arise "
                            "(noise_variance={}).".format(noise_variance))
            gamma = np.empty((self.J + 1,))  # including all of the cutpoints
            gamma[0] = np.NINF
            gamma[-1] = np.inf
            gamma[1] = theta[1]
            for i in range(2, self.J):
                gamma[i] = gamma[i - 1] + np.exp(theta[i])
            scale_std = np.exp(theta[self.J])
            scale = scale_std**2
            if self.kernel._general and self.kernel._ARD:
                # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
                # and lengthscales parameter for each dimension and class
                varphi = np.exp(np.reshape(theta[self.J:self.J + self.J * self.D], (self.J, self.D)))
            else:
                # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
                # and a single, shared lengthscale parameter
                varphi = np.exp(theta[self.J])
            # Update prior covariance
            self.kernel.hyperparameter_update(varphi=varphi)
            self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
            # TODO: Calculate partial_K_varphi here?
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is None
                and scale_0 is not None):
            # Optimize only varphi and noise variance
            gamma = gamma_0
            scale = scale_0
            raise ValueError("TODO")
        elif (gamma_0 is not None
                and varphi_0 is None
                and noise_variance_0 is not None
                and scale_0 is None):
            # Optimize only varphi and scale
            gamma = gamma_0
            noise_variance = noise_variance_0
            raise ValueError("TODO")
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is None
                and scale_0 is not None):
            # Optimize only varphi
            if self.kernel._general and self.kernel._ARD:
                # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
                # and lengthscales parameter for each dimension and class
                varphi = np.exp(np.reshape(theta[0:self.J + self.J * self.D], (self.J, self.D)))
            else:
                # In this case, then there is a scale parameter, the first cutpoint, the interval parameters,
                # and a single, shared lengthscale parameter
                varphi = np.exp(theta[0])
            # Update prior covariance
            self.kernel.hyperparameter_update(varphi=varphi)
            self.K = self.kernel.kernel_matrix(self.X_train, self.X_train)
            gamma = gamma_0
            noise_variance = noise_variance_0
            scale = scale_0
        elif (gamma_0 is not None
                and noise_variance_0 is None
                and varphi_0 is not None
                and scale_0 is not None):
            # Optimize only noise variance
            gamma = gamma_0
            varphi = varphi_0
            scale = scale_0
            raise ValueError("TODO")
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is None):
            # Optimize only scale
            gamma = gamma_0
            varphi = varphi_0
            noise_variance = noise_variance_0
            raise ValueError("TODO")
        elif (gamma_0 is not None
                and noise_variance_0 is not None
                and varphi_0 is not None
                and scale_0 is not None):
            # Optimize only first two threshold parameters
            gamma = gamma_0
            varphi = varphi_0
            noise_variance = noise_variance_0
            scale = scale_0
            raise ValueError("TODO")
        return gamma, varphi, noise_variance, scale

    def _hyperparameters_update(self, varphi=None, scale=None, noise_variance=None):
        """
        Reset kernel hyperparameters, generating new prior and posterior covariances.

        :arg varphi:
        :arg s:
        :arg noise_variance:
        """
        if varphi is not None or scale is not None:
            self.kernel.hyperparameter_update(varphi=varphi, scale=scale)
            # Update prior covariance
            self._update_prior()
        # Initalise the noise variance
        if noise_variance is not None:
            self.noise_variance = noise_variance
            self.noise_std = np.sqrt(noise_variance)

    def grid_over_hyperparameters(
            self, range_x1, range_x2, res, gamma_0=None, varphi_0=None, noise_variance_0=None, scale_0=None, calculate_all_gradients=False,
            posterior_mean_0=None, Sigma_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=False):
        """
        Return meshgrid values of fx and directions of gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs.
        """
        steps = self.N  # Seems to work well in practice.
        (
            x1s, x2s, xlabel, ylabel, xscale, yscale, xx, yy,
        Phi_new, fxs, gxs, intervals, gx_0, indices) = self._grid_over_hyperparameters_initiate(
            gamma_0, varphi_0, noise_variance_0, scale_0, res, range_x1, range_x2, self.J, calculate_all_gradients)
        for i, phi in enumerate(Phi_new):
            gamma, varphi, noise_variance, noise_std, scale = self._grid_over_hyperparameters_update(phi, gamma_0, varphi_0, noise_variance_0, scale_0)
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            Sigma = Sigma_0
            mean_EP = mean_EP_0
            precision_EP = precision_EP_0
            amplitude_EP = amplitude_EP_0
            while error / steps > self.EPS**2:
                iteration += 1
                (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
                 precision_EP, amplitude_EP, containers) = self.estimate(
                    steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma,
                    mean_EP_0=mean_EP, precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP,
                    first_step=first_step, write=write)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(Phi_new)))
            weights, precision_EP, Lambda_cholesky, Lambda = self.compute_EP_weights(
                precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
            t1, t2, t3, t4, t5 = self.compute_integrals_vector(
                gamma, Sigma, precision_EP, posterior_mean, noise_variance)
            # t1, t2, t3, t4, t5 = self.compute_integrals(
            #     gamma, Sigma, precision_EP, posterior_mean, noise_variance)
            fx = self.objective(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi, noise_variance, t2, t3, t4, t5, Lambda, weights)
            gxs[i] = gx[indices]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(varphi, noise_variance, fx))
        if x2s is not None:
            return fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy, xlabel, ylabel, xscale, yscale
        else:
            return fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale

    def hyperparameter_training_step(
            self, theta, gamma_0, varphi_0, noise_variance_0, scale_0,
            posterior_mean_0=None, Sigma_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False, verbose=True):
        """
        Optimisation routine for hyperparameters.

        :arg theta: (log-)hyperparameters to be optimised.
        :arg gamma_0:
        :type gamma_0:
        :arg varphi_0:
        :type varphi_0:
        :arg noise_variance_0:
        :type noise_variance_0:
        :arg scale_0:
        :type scale_0:
        :arg steps:
        :arg posterior_mean_0:
        :arg Sigma_0:
        :arg mean_EP_0:
        :arg precision_EP_0:
        :arg amplitude_EP_0:
        :arg varphi_0:
        :arg psi_0:
        :arg grad_Z_wrt_cavity_mean_0:
        :arg first_step:
        :arg fix_hyperparameters:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        steps = self.N
        error = np.inf
        iteration = 0
        # Update prior covariance and get hyperparameters from theta
        gamma, varphi, noise_variance, scale = self._hyperparameter_training_step_initialise(
            theta, gamma_0, varphi_0, noise_variance_0, scale_0, calculate_all_gradients)
        posterior_mean = posterior_mean_0
        Sigma = Sigma_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        intervals = gamma[2:self.J] - gamma[1:self.J - 1]
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, Sigma, mean_EP,
             precision_EP, amplitude_EP, containers) = self.estimate(
                steps, gamma, varphi, noise_variance, posterior_mean_0=posterior_mean, Sigma_0=Sigma, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP, amplitude_EP_0=amplitude_EP, first_step=first_step, write=write)
            if verbose:
                print("({}), error={}".format(iteration, error))
        weights, precision_EP, Lambda_cholesky, Lambda = self.compute_EP_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        if write:
            (posterior_means, Sigmas, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_marginal_likelihoods) = containers
        # Try optimisation routine
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            gamma, Sigma, precision_EP, posterior_mean, noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights)
        gx, indices = self._objective_gradient_initiate(gamma_0, varphi_0, noise_variance_0, scale_0)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, noise_variance, t2, t3, t4, t5, Lambda, weights)
        if verbose:
            print(repr(gamma), ",")
            print(self.kernel.varphi, ",")
            print(noise_variance, ",")
            print("\nfunction_eval={}\n jacobian_eval={}".format(
                fx, gx))
        else:
            print("gamma={}, noise_variance={}, varphi={}\nfunction_eval={}".format(
                gamma, noise_variance, self.kernel.varphi, fx))
        return fx, gx

    def compute_integrals_vector(
            self, gamma, Sigma, precision_EP, posterior_mean, noise_variance):
        """Computethe integrals required for the gradient evaluation."""
        # calculate gamma_t and gamma_tplus1 here
        noise_std = np.sqrt(noise_variance) * np.sqrt(2)  # TODO
        gamma_t = gamma[self.t_train]
        gamma_tplus1 = gamma[self.t_train + 1] 
        posterior_covariance = np.diag(Sigma)
        mean_t = (posterior_mean[self.where_t_not0]
            * noise_variance + posterior_covariance[self.where_t_not0]
            * gamma_t[self.where_t_not0]) / (
                noise_variance + posterior_covariance[self.where_t_not0])
        mean_tplus1 = (posterior_mean[self.where_t_notJminus1]
            * noise_variance + posterior_covariance[self.where_t_notJminus1]
            * gamma_tplus1[self.where_t_notJminus1]) / (
                noise_variance + posterior_covariance[self.where_t_notJminus1])
        sigma = np.sqrt(
            (noise_variance * posterior_covariance) / (
            noise_variance + posterior_covariance))
        sigma_t_not0 = sigma[self.where_t_not0]
        sigma_t_notJminus1 = sigma[self.where_t_notJminus1]
        a_t = mean_t - 5.0 * sigma_t_not0
        b_t = mean_t + 5.0 * sigma_t_not0
        h_t = b_t - a_t
        a_tplus1 = mean_tplus1 - 5.0 * sigma_t_notJminus1
        b_tplus1 = mean_tplus1 + 5.0 * sigma_t_notJminus1
        h_tplus1 = b_tplus1 - a_tplus1
        y_0 = np.zeros((20, self.N))
        y_t_not0 = np.zeros((20, len(self.where_t_not0)))
        y_t_notJminus1 = np.zeros((20, len(self.where_t_notJminus1)))
        t2 = np.zeros((self.N,))
        t3 = np.zeros((self.N,))
        t4 = np.zeros((self.N,))
        t5 = np.zeros((self.N,))
        t2[self.where_t_not0] = fromb_t2_vector(
                y_t_not0.copy(), mean_t, sigma_t_not0,
                a_t, b_t, h_t,
                posterior_mean[self.where_t_not0],
                posterior_covariance[self.where_t_not0],
                gamma_t[self.where_t_not0], gamma_tplus1[self.where_t_not0],
                noise_variance, noise_std, self.EPS)
        t3[self.where_t_notJminus1] = fromb_t3_vector(
                y_t_notJminus1.copy(), mean_tplus1, sigma_t_notJminus1,
                a_tplus1, b_tplus1,
                h_tplus1, posterior_mean[self.where_t_notJminus1],
                posterior_covariance[self.where_t_notJminus1],
                gamma_t[self.where_t_notJminus1],
                gamma_tplus1[self.where_t_notJminus1],
                noise_variance, noise_std, self.EPS)
        t4[self.where_t_notJminus1] = fromb_t4_vector(
                y_t_notJminus1.copy(), mean_tplus1, sigma_t_notJminus1,
                a_tplus1, b_tplus1,
                h_tplus1, posterior_mean[self.where_t_notJminus1],
                posterior_covariance[self.where_t_notJminus1],
                gamma_t[self.where_t_notJminus1],
                gamma_tplus1[self.where_t_notJminus1],
                noise_variance, noise_std, self.EPS),
        t5[self.where_t_not0] = fromb_t5_vector(
                y_t_not0.copy(), mean_t, sigma_t_not0,
                a_t, b_t, h_t,
                posterior_mean[self.where_t_not0],
                posterior_covariance[self.where_t_not0],
                gamma_t[self.where_t_not0], gamma_tplus1[self.where_t_not0],
                noise_variance, noise_std, self.EPS) 
        return (
            fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_covariance,
                gamma_t, gamma_tplus1,
                noise_std, self.EPS),
            t2,
            t3,
            t4,
            t5
        )

    def compute_integrals(
        self, gamma, Sigma, precision_EP, posterior_mean, noise_variance):
        """Compute the integrals required for the gradient evaluation."""
        # Call EP routine to find posterior distribution
        t1 = np.empty((self.N,))
        t2 = np.empty((self.N,))
        t3 = np.empty((self.N,))
        t4 = np.empty((self.N,))
        t5 = np.empty((self.N,))
        # Compute integrals - expensive for loop
        for i in range(self.N):
            t1[i] = fromb_t1(
                posterior_mean[i], Sigma[i, i], self.t_train[i], self.J,
                gamma, noise_variance, self.EPS)
            t2[i] = fromb_t2(
                posterior_mean[i], Sigma[i, i], self.t_train[i], self.J,
                gamma, noise_variance, self.EPS)
            t3[i] = fromb_t3(
                posterior_mean[i], Sigma[i, i], self.t_train[i], self.J,
                gamma, noise_variance, self.EPS)
            t4[i] = fromb_t4(
                posterior_mean[i], Sigma[i, i], self.t_train[i], self.J,
                gamma, noise_variance, self.EPS)
            t5[i] = fromb_t5(
                posterior_mean[i], Sigma[i, i], self.t_train[i], self.J,
                gamma, noise_variance, self.EPS)
        return t1, t2, t3, t4, t5

    def objective(self, precision_EP, posterior_mean, t1, Lambda_cholesky, Lambda, weights):
        """
        Calculate fx, the variational lower bound of the log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\Phi)} =,

            where :math:`F(\Phi)` is the variational lower bound of the log marginal likelihood at the EP equilibrium,
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
            self, gx, intervals, varphi, noise_variance, t2, t3, t4, t5, Lambda, weights):
        """
        TODO: This is a bottleneck: vectorise it.
        Calculate gx, the jacobian of the variational lower bound of the log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\Phi)}{\partial \Phi}}

            where :math:`F(\Phi)` is the variational lower bound of the log marginal likelihood at the EP equilibrium,
            :math:`\Phi` is the set of hyperparameters, :math:`h`, :math:`\Pi`, :math:`K`. #TODO

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
        # Update gx
        # For gx[0] -- ln\sigma
        if gx[0]:
            gx[0] = np.sum(t5 - t4)
            # gx[0] *= -0.5 * noise_variance  # This is not what is written in the paper. It is a typo in Chu code
            gx[0] *= np.sqrt(noise_variance)
        # For gx[1] -- \b_1
        if gx[1]:
            gx[1] = np.sum(t3 - t2)
        # For gx[2] -- ln\Delta^r
        for j in range(2, self.J):
            if gx[j]:
                targets = self.t_train[self.grid]
                gx[j] = np.sum(t3[targets == j - 1])
                gx[j] -= np.sum(t2[targets == self.J - 1])
                gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                gx[j] *= intervals[j - 2]
        # For gx[self.J] -- s
        if gx[self.J]:
            # TODO: Need to check this is correct: is it directly analogous to gradient wrt log varphi?
            partial_K_s = self.kernel.kernel_partial_derivative_s(self.X_train, self.X_train)
            # VC * VC * a' * partial_K_varphi * a / 2
            gx[self.J] = varphi * 0.5 * weights.T @ partial_K_s @ weights  # That's wrong. not the same calculation.
            # equivalent to -= varphi * 0.5 * np.trace(Lambda @ partial_K_varphi)
            gx[self.J] -= varphi * 0.5 * np.sum(np.multiply(Lambda, partial_K_s))
            # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
            # gx[self.J] -= 0.1 * varphi
            gx[self.J] *= 2.0  # since varphi = kappa / 2
        # For gx[self.J + 1] -- varphi
        if gx[self.J + 1]:
            partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(self.X_train, self.X_train)
            if self.kernel._general and self.kernel._ARD:
                raise ValueError("TODO")
            else:
                # VC * VC * a' * partial_K_varphi * a / 2
                gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights  # That's wrong. not the same calculation.
                # equivalent to -= varphi * 0.5 * np.trace(Lambda @ partial_K_varphi)
                gx[self.J + 1] -= varphi * 0.5 * np.sum(np.multiply(Lambda, partial_K_varphi))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[self.J] -= 0.1 * varphi
        return -gx

    def approximate_evidence(self, mean_EP, precision_EP, amplitude_EP, Sigma):
        """
        TODO: check and return line could be at risk of overflow
        Compute the approximate evidence at the EP solution.

        :return:
        """
        intermediate_vector = np.multiply(mean_EP, precision_EP)
        B = intermediate_vector.T @ Sigma @ intermediate_vector - np.multiply(
            intermediate_vector, mean_EP)
        Pi_inv = np.diag(1. / precision_EP)
        return (
            np.prod(
                amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
                / np.sqrt(np.linalg.det(np.add(Pi_inv, self.K))))

    def compute_EP_weights(
        self, precision_EP, mean_EP, grad_Z_wrt_cavity_mean):
        """
        Compute regression weights, and check that they are in equilibrium with
        the gradients of Z wrt cavity means.

        A matrix inverse is always required to evaluate fx.
        """
        if np.any(precision_EP == 0.0):
            # TODO: Only check for equilibrium if it has been updated in this swipe
            warnings.warn("Some sample has not been updated.\n")
            precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        Pi_inv = np.diag(1. / precision_EP)
        # Cholesky factorisation only
        L = np.linalg.cholesky(np.add(Pi_inv, self.K))
        # Inverse at each hyperparameter update -- O(N^3)
        L_inv = np.linalg.inv(L)
        Lambda = L_inv.T @ L_inv  # (N, N)
        weights = Lambda @ mean_EP
        if np.any(
            np.abs(weights - grad_Z_wrt_cavity_mean) > np.sqrt(self.EPS)):
            warnings.warn("Fatal error: the weights are not in equilibrium wit"
                "h the gradients".format(
                    weights, grad_Z_wrt_cavity_mean))
        return weights, precision_EP, L, Lambda


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