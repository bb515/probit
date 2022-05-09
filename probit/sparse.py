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
    truncated_norm_normalising_constant,
    p)  # , dp)
# NOTE Usually the numba implementation is not faster
# from .numba.utilities import (
#     fromb_t1_vector, fromb_t2_vector,
#     fromb_t3_vector, fromb_t4_vector, fromb_t5_vector)
from scipy.stats import norm
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
        # Choose inducing points
        self.M = M
        super().__init__(*args, **kwargs)
        # self.EPS = 1e-8
        # self.EPS_2 = self.EPS**2
        # self.jitter = 1e-10

    def _update_prior(self):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        inducing_idx = np.random.choice(
            self.X_train.shape[0], size=self.M, replace=False)
        self.Z = self.X_train[inducing_idx, :]
        warnings.warn(
            "Updating prior covariance with Nyström approximation")
        self.Kdiag = self.kernel.kernel_prior_diagonal(self.X_train)
        self.Kuu = self.kernel.kernel_matrix(self.Z, self.Z)
        self.Kuf = self.kernel.kernel_matrix(self.Z, self.X_train)
        warnings.warn(
            "Done updating prior covariance with Nyström approximation")
        # For f = K m

    # def hyperparameters_update(
    #     self, cutpoints=None, varphi=None, variance=None, noise_variance=None,
    #     varphi_hyperparameters=None):
    #     """
    #     Reset kernel hyperparameters, generating new prior and posterior
    #     covariances. Note that hyperparameters are fixed parameters of the
    #     approximator, not variables that change during the estimation. The strange
    #     thing is that hyperparameters can be absorbed into the set of variables
    #     and so the definition of hyperparameters and variables becomes
    #     muddled. Since varphi can be a variable or a parameter, then optionally
    #     initiate it as a parameter, and then intitate it as a variable within
    #     :meth:`approximate`. Problem is, if it changes at approximate time, then a
    #     hyperparameter update needs to be called.

    #     :arg cutpoints: (J + 1, ) array of the cutpoints.
    #     :type cutpoints: :class:`numpy.ndarray`.
    #     :arg varphi: The kernel hyper-parameters.
    #     :type varphi: :class:`numpy.ndarray` or float.
    #     :arg variance:
    #     :type variance:
    #     :arg float noise_variance: The noise variance.
    #     :type noise_variance:
    #     """
    #     self._hyperparameters_update(
    #         cutpoints=cutpoints, varphi=varphi,
    #         variance=variance, noise_variance=noise_variance)
    #     if varphi_hyperparameters is not None:
    #         self.kernel.update_hyperparameter(
    #             varphi_hyperparameters=varphi_hyperparameters)
    #     # Update posterior covariance
    #     warnings.warn("Updating posterior covariance.")
    #     self._update_posterior()
    #     warnings.warn("Done updating posterior covariance.")

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
        # Seems to be necessary to take this cho factor
        (L, lower) = cho_factor(
            self.Kuu + self.jitter * np.eye(self.M), lower=True)
        A = solve_triangular(L, self.Kuf, lower=True) / self.noise_std
        AAT = A @ A.T
        B = AAT + np.eye(self.M)  # cache qinv

        # This is not correct.
        # (LB, lower) = cho_factor(B, lower=True)
        # LBTinv = solve_triangular(LB, np.eye(self.M), lower=True)  # not sure if this is correct
        # Binv = solve_triangular(LB, LBTinv, lower=True)  # not sure if this is correct

        (LB, lower) = cho_factor(B)
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)  # not sure if this is correct
        Binv = solve_triangular(LB, LBTinv, lower=False)  # not sure if this is correct

        self.posterior_cov_div_var = A.T @ Binv @ A
        self.trace_posterior_cov_div_var = np.trace(
            self.posterior_cov_div_var)
        self.trace_cov = 1./self.noise_variance * (
            self.N + self.trace_posterior_cov_div_var)

        trace_k = np.sum(self.Kdiag) / self.noise_variance
        trace_q = np.trace(AAT)
        trace = trace_k - trace_q
        half_logdet_b = np.sum(np.log(np.diag(LB)))
        log_noise_variance = self.N * np.log(self.noise_variance)

        self.log_det_cov = -half_logdet_b + 0.5 * log_noise_variance + 0.5 * trace

    def approximate(
            self, steps, posterior_mean_0=None, first_step=1,
            write=False):
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
                posterior_mean, self.cutpoints_ts, self.cutpoints_tplus1s,
                self.noise_std, self.EPS, self.upper_bound, self.upper_bound2)
            y = self._y(
                p_, posterior_mean, self.noise_std)
            posterior_mean, nu = self._posterior_mean(
                    y, self.posterior_cov_div_var, self.noise_variance)
            if self.kernel.varphi_hyperhyperparameters is not None:
                # Posterior mean update for kernel hyperparameters
                # Kernel hyperparameters are variables here
                # TODO maybe this shouldn't be performed at every step.
                varphi = self._varphi(
                    posterior_mean, self.kernel.varphi_hyperparameters,
                    n_samples=10)
                varphi_hyperparameters = self._varphi_hyperparameters(
                    self.kernel.varphi)
                self.hyperparameters_update(
                    varphi=varphi,
                    varphi_hyperparameters=varphi_hyperparameters)
                print("varphi = ", self.kernel.varphi)
                print("varphihyper = ", self.kernel.varphi_hyperparameters)
            if write:
                Z, *_ = truncated_norm_normalising_constant(
                    self.cutpoints_ts, self.cutpoints_tplus1s,
                    self.noise_std, posterior_mean, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean, nu, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance,
                    self.log_det_cov)
                ms.append(posterior_mean)
                ys.append(y)
                if self.kernel.varphi_hyperparameters is not None:
                    varphis.append(self.kernel.varphi)
                    psis.append(self.kernel.varphi_hyperparameters)
                fxs.append(fx)
        containers = (ms, ys, varphis, psis, fxs)
        return posterior_mean, nu, y, p, containers

    def _posterior_mean(self, y, posterior_cov_div_var, noise_variance):
        # TODO: should be able to calculate this with a solve instead using LB and A.
        posterior_mean = posterior_cov_div_var @ y
        nu = 1./noise_variance  * (y - posterior_mean)
        return posterior_mean, nu

    def objective_gradient(
            self, gx, intervals, cutpoints_ts, cutpoints_tplus1s, varphi,
            noise_variance, noise_std,
            m, nu, posterior_cov_div_var, trace_posterior_cov_div_var,
            trace_cov, N, Z, norm_pdf_z1s, norm_pdf_z2s, indices,
            numerical_stability=True, verbose=False):
        """TODO"""
        return gx

    # This prediction can probably supercede others
    # def predict(
    #         self, cutpoints, posterior_inv_cov, nu,
    #         varphi, noise_variance, X_test, vectorised=True):
    #     """
    #     Return the posterior predictive distribution over classes.

    #     :arg cutpoints: (J + 1, ) array of the cutpoints.
    #     :type cutpoints: :class:`numpy.ndarray`.
    #     :arg posterior_inv_cov: The approximate posterior inverse covariance.
    #         Array like (N, N).
    #     :type posterior_inv_cov: :class:`numpy.ndarray`.
    #     :arg posterior_mean: The approximate posterior mean of the latent
    #         variable. Array like (N,)
    #     :type posterior_mean: :class:`numpy.ndarray`.
    #     :arg varphi: The kernel hyper-parameters.
    #     :type varphi: :class:`numpy.ndarray` or float.
    #     :arg float noise_variance: The noise variance. 
    #     :arg X_test: The new data points, array like (N_test, D).
    #     :arg n_samples: The number of samples in the Monte Carlo estimate.
    #     :return: A Monte Carlo estimate of the class probabilities.
    #     """
    #     if self.kernel._ARD:
    #         # This is the general case where there are hyper-parameters
    #         # varphi (J, D) for all dimensions and classes.
    #         raise ValueError(
    #             "For the ordinal likelihood approximator,the kernel "
    #             "must not be _ARD type (kernel._ARD=1), but"
    #             " ISO type (kernel._ARD=0). (got {}, expected)".format(
    #                 self.kernel._ARD, 0))
    #     else:
    #         if vectorised:
    #             return self._predict_vector(
    #                 cutpoints, nu, varphi, noise_variance, X_test)
    #         else:
    #             return ValueError(
    #                 "The scalar implementation has been "
    #                 "superseded. Please use "
    #                 "the vector implementation.")

    # def _predict_vector(
    #     self, cutpoints, nu, varphi, noise_variance, X_test):
    #     """
    #     Make posterior prediction over ordinal classes of X_test.

    #     :arg cutpoints: (J + 1, ) array of the cutpoints.
    #     :type cutpoints: :class:`numpy.ndarray`.
    #     :arg cov: A covariance matrix used in calculation of posterior
    #         predictions. (\sigma^2I + K)^{-1} Array like (N, N).
    #     :type cov: :class:`numpy.ndarray`
    #     :arg posterior_mean: The posterior mean. Array like (N,).
    #     :type posterior_mean: :class:`numpy.ndarray`
    #     :arg varphi: The kernel hyper-parameters.
    #     :type varphi: :class:`numpy.ndarray` or float.
    #     :arg float noise_variance: The noise variance.
    #     :arg X_test: The new data points, array like (N_test, D).
    #     :return: A Monte Carlo estimate of the class probabilities.
    #     :rtype tuple: ((N_test, J), (N_test,), (N_test,))
    #     """
    #     # Seems to be necessary to take this cho factor
    #     (L, lower) = cho_factor(
    #         self.Kuu + self.jitter * np.eye(self.M), lower=True)
    #     A = solve_triangular(L, self.Kuf, lower=True) / self.noise_std
    #     B = A @ A.T + np.eye(self.M)
    #     (LB, lower) = cho_factor(B, lower=True)

    #     N_test = np.shape(X_test)[0]
    #     # Update the kernel with new varphi
    #     self.kernel.varphi = varphi
    #     Kus = self.kernel.kernel_matrix(self.Z, X_test)  # (M, N_test)
    #     err = self.noise_variance * nu  # TODO: i think this is meant to be prior error! not posterior_mean error
    #     Aerr = A @ err
    #     c = solve_triangular(LB, Aerr, lower=True)
    #     tmp1 = solve_triangular(L, Kus, lower=True)
    #     tmp2 = solve_triangular(LB, tmp1, lower=True)
    #     posterior_pred_mean = tmp2.T @ c
    #     posterior_var = self.kernel.kernel_prior_diagonal(X_test)\
    #             + np.sum(tmp2**2, axis=0) - np.sum(tmp1**2, axis=0)
    #     posterior_pred_var = posterior_var + noise_variance
 
    #     posterior_std = np.sqrt(posterior_var)
    #     posterior_pred_std = np.sqrt(posterior_pred_var)
    #     predictive_distributions = np.empty((N_test, self.J))
    #     for j in range(self.J):
    #         Z1 = np.divide(np.subtract(
    #             cutpoints[j + 1], posterior_pred_mean), posterior_pred_std)
    #         Z2 = np.divide(
    #             np.subtract(cutpoints[j],
    #             posterior_pred_mean), posterior_pred_std)
    #         predictive_distributions[:, j] = norm.cdf(Z1) - norm.cdf(Z2)
    #     return predictive_distributions, posterior_pred_mean, posterior_std

    def approximate_posterior(
            self, theta, indices, steps=None, first_step=1, max_iter=2,
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
        posterior_mean = None
        # Convergence is sometimes very fast so this may not be necessary
        while error / steps > self.EPS and iteration < max_iter:
            iteration += 1
            (posterior_mean, nu, *_) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=write)
            (Z,
            norm_pdf_z1s,
            norm_pdf_z2s,
            *_ )= truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS)
            fx = self.objective(
                self.N, posterior_mean, nu, self.trace_cov,
                self.trace_posterior_cov_div_var, Z,
                self.noise_variance, self.log_det_cov)
            error = np.abs(fx_old - fx)
            fx_old = fx
            if verbose:
                print("({}), error={}".format(iteration, error))
        gx = self.objective_gradient(
                gx.copy(), intervals, self.cutpoints_ts,
                self.cutpoints_tplus1s,
                self.kernel.varphi, self.noise_variance, self.noise_std,
                posterior_mean, nu, self.posterior_cov_div_var,
                self.trace_posterior_cov_div_var, self.trace_cov,
                self.N, Z, norm_pdf_z1s, norm_pdf_z2s, indices,
                numerical_stability=True, verbose=False)
        gx = gx[indices_where]
        if verbose:
            print(
                "cutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints,
                    self.noise_variance,
                    self.kernel.varphi, fx))
        if calculate_posterior_cov == 0:
            return fx, gx, posterior_mean, (
                self.noise_variance * self.posterior_cov_div_var, False)
        elif calculate_posterior_cov == 2:
            return fx, gx, posterior_mean, (
                1./self.noise_variance * np.eye(np.shape(self.X_train)[0])
                - 1./self.noise_variance * self.posterior_cov_div_var, True)
        elif calculate_posterior_cov is None:
            return fx, gx


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
        self.M = M
        super().__init__(*args, **kwargs)
        # self.EPS = 1e-8
        # self.EPS_2 = self.EPS**2
        # self.jitter = 1e-10

    def _update_prior(self):
        """
        Update prior covariances with Nyström approximation.

        :arg M: Number of inducing inputs.

        """
        inducing_idx = np.random.choice(
            self.X_train.shape[0], size=self.M, replace=False)
        self.Z = self.X_train[inducing_idx, :]
        warnings.warn(
            "Updating prior covariance with Nyström approximation")
        self.Kdiag = self.kernel.kernel_prior_diagonal(self.X_train)
        self.Kuu = self.kernel.kernel_matrix(self.Z, self.Z)
        self.Kuf = self.kernel.kernel_matrix(self.Z, self.X_train)
        warnings.warn(
            "Done updating prior covariance with Nyström approximation")

    def _update_posterior(self, Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
            noise_std, noise_variance, nu):
        weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
        z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
        z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
        precision  = weight**2 + (
            z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
            ) / Z / noise_variance

        m = - weight + nu

        # Seems to be necessary to take this cho factor
        (L, lower) = cho_factor(
            self.Kuu + self.jitter * np.eye(self.M), lower=True)
        A = solve_triangular(L, self.Kuf, lower=True)
        ALambdaAT = A @ np.diag(precision) @ A.T
        B = ALambdaAT + np.eye(self.M)  # cache qinv
        (LB, lower) = cho_factor(B)
        # Perhaps not necessary to do this solve
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)
        Binv = solve_triangular(LB, LBTinv, lower=False)

        t1 = - m + (A @ Binv @ A.T @ m) * precision
        nu += t1
        error = np.abs(max(t1.min(), t1.max(), key=abs))

        # self.posterior_cov_div_var = A.T @ Binv @ A
        # self.trace_posterior_cov_div_var = np.trace(
        #     self.posterior_cov_div_var)
        # self.trace_cov = 1./self.noise_variance * (
        #     self.N + self.trace_posterior_cov_div_var)
        # trace_k = np.sum(self.Kdiag) / self.noise_variance
        # trace_q = np.trace(AAT)
        # trace = trace_k - trace_q
        # half_logdet_b = np.sum(np.log(np.diag(LB)))
        # log_noise_variance = self.N * np.log(self.noise_variance)
        # self.log_det_cov = -half_logdet_b + 0.5 * log_noise_variance + 0.5 * trace
        return error, t1, nu

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
            error, weight, posterior_mean = self._update_posterior(
                Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
                self.noise_std, self.noise_variance)
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


            # Seems to be necessary to take this cho factor
            (L, lower) = cho_factor(
                self.Kuu + self.jitter * np.eye(self.M), lower=True)
            A = solve_triangular(L, self.Kuf, lower=True) / self.noise_std
            AAT = A @ A.T
            B = AAT + np.eye(self.M)  # cache qinv

            # This is not correct.
            # (LB, lower) = cho_factor(B, lower=True)
            # LBTinv = solve_triangular(LB, np.eye(self.M), lower=True)  # not sure if this is correct
            # Binv = solve_triangular(LB, LBTinv, lower=True)  # not sure if this is correct

            (LB, lower) = cho_factor(B)
            LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)  # not sure if this is correct
            Binv = solve_triangular(LB, LBTinv, lower=False)  # not sure if this is correct

            self.posterior_cov_div_var = A.T @ Binv @ A
            self.trace_posterior_cov_div_var = np.trace(
                self.posterior_cov_div_var)
            self.trace_cov = 1./self.noise_variance * (
                self.N + self.trace_posterior_cov_div_var)

            trace_k = np.sum(self.Kdiag) / self.noise_variance
            trace_q = np.trace(AAT)
            trace = trace_k - trace_q
            half_logdet_b = np.sum(np.log(np.diag(LB)))
            log_noise_variance = self.N * np.log(self.noise_variance)

            self.log_det_cov = -half_logdet_b + 0.5 * log_noise_variance + 0.5 * trace


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
