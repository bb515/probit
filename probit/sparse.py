"""
Sparse GPs

TODO: Laplace currently doesn't work.
"""
from probit.approximators import VBOrdinalGP, EPOrdinalGP, LaplaceOrdinalGP
#import enum
#from .kernels import Kernel, InvalidKernel
#import pathlib
#import random
from tqdm import trange
import warnings
#import math
import matplotlib.pyplot as plt
import numpy as np
from .utilities import (
    #read_array,
    #norm_z_pdf, norm_cdf,
    truncated_norm_normalising_constant)
    #p, dp)
# NOTE Usually the numba implementation is not faster
# from .numba.utilities import (
#     fromb_t1_vector, fromb_t2_vector,
#     fromb_t3_vector, fromb_t4_vector, fromb_t5_vector)
#from scipy.stats import norm
from scipy.linalg import cho_solve, cho_factor, solve_triangular
#from .utilities import (
#    sample_varphis,
#    fromb_t1_vector, fromb_t2_vector, fromb_t3_vector, fromb_t4_vector,
#    fromb_t5_vector)


class SparseVBOrdinalGP(VBOrdinalGP):
    """
    A sparse GP classifier for ordinal likelihood using the Variational Bayes
    (VB) approximation.
 
    Inherits the VBOrdinalGP class. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference and approximate prior using the Nystrom approximation. It is for
    the ordinal likelihood. For this a :class:`probit.kernels.Kernel` is
    required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SparseVBOrdinalGP"

    def __init__(
            self, M, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SparseVBOrderedGP` Approximator object.

        :arg M: The number of basis functions.

        :returns: A :class:`SparseVBOrderedGP` object.
        """
        super().__init__(*args, **kwargs)
        # self.EPS = 1e-8
        # self.EPS_2 = self.EPS**2
        # self.jitter = 1e-10
        # Choose inducing points
        self.M = M
        inducing_idx = np.random.randint(self.X_train.shape[0], size=self.M)
        self.Z = self.X_train[inducing_idx, :]
        # Initiate hyperparameters
        self._update_nystrom_prior()

    def _update_nystrom_prior(self):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        warnings.warn("Updating prior covariance with Nyström approximation")
        self.Kuu = self.kernel.kernel_matrix(self.Z, self.Z)
        # self.Kff = self.kernel.kernel_diagonal(self.X_train, self.X_train)
        self.Kfu = self.kernel.kernel_matrix(self.X_train, self.Z)
        warnings.warn("Done updating prior covariance with Nyström approximation")

        # For f = K m
        L_cov  = self.Kuu + self.jitter * np.eye(self.N)
        L_cov, _ = cho_factor(L_cov)
        L_covT_inv = solve_triangular(
            L_cov.T, np.eye(self.M), lower=True
        )
        self.Kuu_inv = solve_triangular(L_cov, L_covT_inv, lower=False)

        # Approximation of the Gram matrix
        self.Kff = self.Kfu @ self.K_inv @ self.Kfu.T

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
        """
        self.hyperparameters_update(
            cutpoints=cutpoints, varphi=varphi,
            variance=variance, noise_variance=noise_variance)
        if varphi_hyperparameters is not None:
            self.kernel.update_hyperparameter(
                varphi_hyperparameters=varphi_hyperparameters)
        # Update posterior covariance
        warnings.warn("Updating posterior covariance.")
        self._update_nystrom_posterior()
        warnings.warn("Done updating posterior covariance.")

    def _update_posterior(self):
        """
        Update posterior covariances.

        The reference for the derivation of these equations are found in.
        https://gpflow.github.io/GPflow/2.5.1/notebooks/theory/SGPR_notes.html

        We first apply te Woodbury identity to the effective covariance matrix:
        .. math::
                (\sigma^{2}I + K_{nm}K_{mm}^{-1}K_{mn})^-1 = 1./\sigma^{2}I - 1./sigma^{2} K_{nm}(\sigma^{2}K_{mm} + K_{mn}K_{nm})^{-1}K_{mn}

            where :math:`K_{mm}`, :math:`K_{nm}`, :math:`K_{mn}` represent the kernel
                evaluated at the datapoints :math:`X`, the inducing points :math:`Z`, and
                between the data and the inducing points respectively. 
        
        Now, to obtain a better conditioned matrix for inversion, we rotate by :math:`L`, where :math:`LL^{T} = K_{mm}`.

        """
        (L_cov, lower) = cho_factor(self.Kuu, lower=True)

        A = solve_triangular(L_cov, self.Kuf, lower=True) / self.noise_std

        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), 

        # TODO: Is this really the best cholesky to take. What are the eigenvalues?
        # are they bounded?
        # Note that this scipy implementation returns an upper triangular matrix
        # whereas numpy, tf, scipy.cholesky return a lower triangular,
        # then the position of the matrix transpose in the code would change.
        # This can be Nystromized.
        (self.L_cov, self.lower) = cho_factor(
            self.noise_variance * self.Kuu + self.Kfu.T @ self.Kfu)




        # (A + UBV)^-1 = A^-1 - A^-1U(B^-1 + VA^-1U)^-1VA
        # 
        L_covT_inv = solve_triangular(
            self.L_cov.T, np.eye(self.M), lower=True)
        cov = solve_triangular(self.L_cov, L_covT_inv, lower=False)
        self.cov = 1./self.noise_variance * (np.eye(self.N) + self.Kfu @ cov @ self.Kfu.T)

        # Also log_det_K has a Woodbury formula


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
        self.trace_posterior_cov_div_var = np.einsum(
            'ij, ij -> ', self.K, self.cov)


class SparseLaplaceOrdinalGP(LaplaceOrdinalGP):
    """
    A sparse GP classifier for ordinal likelihood using the Laplace
    approximation.
 
    Inherits the LaplaceOrdinalGP class. This class allows users to define a
    classification problem, get predictions using approximate Bayesian
    inference and approximate prior using the Nystrom approximation. It is for
    the ordinal likelihood. For this a :class:`probit.kernels.Kernel` is
    required for the Gaussian Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class from
        the string.
        """
        return "SparseLaplaceOrdinalGP"

    def __init__(
            self, M, *args, **kwargs):
            #cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`SparseLaplaceOrderedGP` Approximator object.

        :arg M: The number of basis functions.

        :returns: A :class:`SparseLaplaceOrderedGP` object.
        """
        print("M", M)
        super().__init__(*args, **kwargs)
        # self.EPS = 1e-8
        # self.EPS_2 = self.EPS**2
        # self.jitter = 1e-10
        # Choose inducing points
        self.M = M
        inducing_idx = np.random.randint(self.X_train.shape[0], size=self.M)
        self.Z = self.X_train[inducing_idx, :]
        # Initiate hyperparameters
        self._update_nystrom_prior()

        # Need to change data?

    def _update_nystrom_prior(self):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        warnings.warn("Updating prior covariance with Nyström approximation")
        self.Kuu = self.kernel.kernel_matrix(self.Z, self.Z)
        # self.Kff = self.kernel.kernel_diagonal(self.X_train, self.X_train)
        self.Kfu = self.kernel.kernel_matrix(self.X_train, self.Z)
        warnings.warn("Done updating prior covariance with Nyström approximation")

        # For f = K m
        L_cov  = self.Kuu + self.jitter * np.eye(self.M)
        L_cov, _ = cho_factor(L_cov)
        L_covT_inv = solve_triangular(
            L_cov.T, np.eye(self.M), lower=True
        )
        self.Kuu_inv = solve_triangular(L_cov, L_covT_inv, lower=False)

        # Approximation of the Gram matrix
        self.Kff = self.Kfu @ self.Kuu_inv @ self.Kfu.T

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
            posterior_mean_0[self.indices_where_0] = self.cutpoints_tplus1s[self.indices_where_0]
        error = 0.0
        weight_0 = np.empty(self.N)
        inverse_variance_0 = np.empty(self.N)
        posterior_means = []
        posterior_precisions = []
        containers = (posterior_means, posterior_precisions)
        return (weight_0, inverse_variance_0, posterior_mean_0, containers, error)

    def approximateSS(self, steps, posterior_mean_0=None, first_step=1, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Laplace approximation via Newton-Raphson iteration as
        written in
        Appendix A Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.
        Uses the Nystrom approximation as written in
        Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to
        Speed Up Kernel Machines. In T. K. Leen, T. G. Dietterich, & V. Tresp
        (Eds.), Advances in Neural Information Processing Systems 13 (NIPS 2000)
        (pp. 682-688). MIT Press.
        http://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

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
        plt.scatter(self.X_train, posterior_mean)
        plt.show()
        for _ in trange(first_step, first_step + steps,
                        desc="Laplace GP priors Approximator Progress",
                        unit="iterations", disable=True):
            (Z, norm_pdf_z1s, norm_pdf_z2s,
                z1s, z2s,
                norm_cdf_z1s, norm_cdf_z2s
                ) = truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS, upper_bound=self.upper_bound)
                #upper_bound2=self.upper_bound2, numerically_stable=True)  # TODO Turn this off!
            weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / self.noise_std
            z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
            z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
            precision  = weight**2 + (
                z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
                ) / Z / self.noise_variance

            L_cov = self.Kuu + self.Kfu.T @ np.diag(precision) @ self.Kfu + self.jitter * np.eye(self.M)
            L_cov, _ = cho_factor(L_cov)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.M), lower=True
            )
            inv_cov = solve_triangular(L_cov, L_covT_inv, lower=False)

            m = - self.K @ weight + posterior_mean  # - weight?

            t1 = -m + self.Kfu @ inv_cov @ self.Kfu.T @ m * precision

            posterior_mean += t1
            plt.scatter(self.X_train, posterior_mean)
            plt.show()

            error = np.abs(max(t1.min(), t1.max(), key=abs))
            if write is True:
                posterior_means.append(posterior_mean)
                posterior_precisions.append(posterior_precisions)
        containers = (posterior_means, posterior_precisions)
        # Initiation becomes harder if not working in terms of f.
        # Ambiguity over which K to use is removed.
        # Cost is amortized.
        return error, weight, posterior_mean, containers

    def approximate(self, steps, posterior_mean_0=None, first_step=1, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Laplace approximation via Newton-Raphson iteration as
        written in
        Appendix A Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning
        Research. 6. 1019-1041.
        Uses the Nystrom approximation as written in
        Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to
        Speed Up Kernel Machines. In T. K. Leen, T. G. Dietterich, & V. Tresp
        (Eds.), Advances in Neural Information Processing Systems 13 (NIPS 2000)
        (pp. 682-688). MIT Press.
        http://papers.nips.cc/paper/1866-using-the-nystrom-method-to-speed-up-kernel-machines.pdf

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
        for _ in trange(first_step, first_step + steps,
                        desc="Laplace GP priors Approximator Progress",
                        unit="iterations", disable=True):
            (Z, norm_pdf_z1s, norm_pdf_z2s,
                z1s, z2s,
                norm_cdf_z1s, norm_cdf_z2s
                ) = truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                self.K @ posterior_mean, self.EPS , upper_bound=self.upper_bound)
                # upper_bound2=self.upper_bound2)  # , numerically_stable=True)  # TODO Turn this off!
            weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / self.noise_std
            z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
            z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
            precision  = weight**2 + (
                z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
                ) / Z / self.noise_variance

            print(weight)
            print(precision)

            L_cov = self.Kuu + self.Kfu.T @ np.diag(precision) @ self.Kfu  + self.jitter * np.eye(self.M)
            L_cov, _ = cho_factor(L_cov)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.M), lower=True
            )
            inv_cov = solve_triangular(L_cov, L_covT_inv, lower=False)

            m = - weight + posterior_mean

            t1 = -m + self.Kfu @ inv_cov @ self.Kfu.T @ m * precision

            posterior_mean += t1

            plt.scatter(self.X_train, self.K @ posterior_mean)
            plt.show()

            error = np.abs(max(t1.min(), t1.max(), key=abs))
            if write is True:
                posterior_means.append(posterior_mean)
                posterior_precisions.append(posterior_precisions)
        containers = (posterior_means, posterior_precisions)
        # Initiation becomes harder if not working in terms of f.
        # Ambiguity over which K to use is removed.
        # Cost is amortized.
        posterior_mean = self.K @ posterior_mean
        return error, weight, posterior_mean, containers
