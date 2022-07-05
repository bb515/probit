"""
Sparse GPs

TODO: Laplace currently doesn't work.
"""
from probit.approximators import PEPGP, VBGP, LaplaceGP
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
    read_array,
    read_scalar,
    #norm_z_pdf,
    norm_cdf,
    truncated_norm_normalising_constant,
    p)  # , dp)
# NOTE Usually the numba implementation is not faster
# from .numba.utilities import (
#     fromb_t1_vector, fromb_t2_vector,
#     fromb_t3_vector, fromb_t4_vector, fromb_t5_vector)
from scipy.linalg import cho_solve, cho_factor, solve_triangular
#from .utilities import (
#    sample_varphis,
#    fromb_t1_vector, fromb_t2_vector, fromb_t3_vector, fromb_t4_vector,
#    fromb_t5_vector)


class SparseVBGP(VBGP):
    """
    A sparse GP classifier for ordinal likelihood using the Variational Bayes
    (VB) approximation.
 
    Inherits the VBGP class. This class allows users to define a
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
        return "SparseVBGP"

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

    def _load_cached_prior(self):
        """
        Load cached prior covariances with Nyström approximation.
        """
        self.M = read_scalar(self.read_path, "M")
        self.Z = read_array(self.read_path, "Z")
        self.Kdiag = read_array(self.read_path, "Kdiag")
        self.Kuu = read_array(self.read_path, "Kuu")
        self.Kfu = read_array(self.read_path, "Kfu")

    def _update_prior(self):
        """
        Update prior covariances with inducing points.
        """
        inducing_idx = np.random.choice(
            self.X_train.shape[0], size=self.M, replace=False)
        self.Z = self.X_train[inducing_idx, :]
        warnings.warn(
            "Updating prior covariance with Nyström approximation")
        self.Kdiag = self.kernel.kernel_prior_diagonal(self.X_train)
        self.Kuu = self.kernel.kernel_matrix(self.Z, self.Z)
        self.Kfu = self.kernel.kernel_matrix(self.X_train, self.Z)
        warnings.warn(
            "Done updating prior covariance with Nyström approximation")

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
        Linv = solve_triangular(L, np.eye(self.M), lower=True)
        LinvT = Linv.T
        A = solve_triangular(L, self.Kfu.T, lower=True) / self.noise_std
        AAT = A @ A.T
        B = AAT + np.eye(self.M)  # cache qinv
        (LB, lower) = cho_factor(B)
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)
        Binv = solve_triangular(LB, LBTinv, lower=False)
        tmp = np.eye(self.M) - Binv
        # For prediction, will need a matrix, call it cov
        self.cov = LinvT @ tmp @ Linv
        self.cov2 = LinvT @ Binv @ A / self.noise_std
        # cov2 = solve_triangular(L.T, Binv, lower=True)
        # self.cov2 = cov2 @ A / self.noise_std

        # SS
        # self.posterior_cov_div_var = A.T @ Binv @ A
        # self.trace_posterior_cov_div_var = np.trace(
        #     self.posterior_cov_div_var)
        # self.trace_cov = 1./self.noise_variance * (
        #     self.N + self.trace_posterior_cov_div_var)

        # If keeping track of the weights only is the goal, then it is maybe better to store something other than posterior_cov_div_var.
        # trace_k = np.sum(self.Kdiag) / self.noise_variance
        # trace_q = np.trace(AAT)
        # trace = trace_k - trace_q
        # half_logdet_b = np.sum(np.log(np.diag(LB)))
        # log_noise_variance = self.N * np.log(self.noise_variance)

        # self.log_det_cov = -half_logdet_b + 0.5 * log_noise_variance + 0.5 * trace

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
        posterior_mean, containers = self._approximate_initiate(
            posterior_mean_0)
        posterior_means, gs, varphis, psis, fxs = containers
        for _ in trange(first_step, first_step + steps,
                        desc="GP priors Sampler Progress", unit="samples",
                        disable=True):
            p_ = p(
                posterior_mean, self.cutpoints_ts, self.cutpoints_tplus1s,
                self.noise_std, self.EPS, self.upper_bound, self.upper_bound2)
            g = self._g(
                p_, posterior_mean, self.noise_std)
            posterior_mean, weight = self._posterior_mean(
                    g, self.cov2, self.Kfu)
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
            if write:
                Z, *_ = truncated_norm_normalising_constant(
                    self.cutpoints_ts, self.cutpoints_tplus1s,
                    self.noise_std, posterior_mean, self.EPS)
                fx = self.objective(
                    self.N, posterior_mean, g, self.trace_cov,
                    self.trace_posterior_cov_div_var, Z,
                    self.noise_variance,
                    self.log_det_cov)
                posterior_means.append(posterior_mean)
                gs.append(g)
                if self.kernel.varphi_hyperparameters is not None:
                    varphis.append(self.kernel.varphi)
                    psis.append(self.kernel.varphi_hyperparameters)
                fxs.append(fx)
        containers = (posterior_means, gs, varphis, psis, fxs)
        return posterior_mean, weight, g, p, containers

    def _posterior_mean(self, g, cov, Kfu):
        weight = cov @ g
        return Kfu @ weight, weight

    def _subset_regressors(
            self, Kus):
        """
        Predict using the predictive variance from the subset regressors
        approximate method.
        """
        (L, lower) = cho_factor(
            self.Kuu + self.jitter * np.eye(self.M), lower=True)
        Linv = solve_triangular(L, np.eye(self.M), lower=True)
        A = solve_triangular(L, self.Kfu.T, lower=True) / self.noise_std
        AAT = A @ A.T
        B = AAT + np.eye(self.M)  # cache qinv
        (LB, lower) = cho_factor(B)
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)
        Binv = solve_triangular(LB, LBTinv, lower=False)
        return np.einsum('ij, ij -> j', Kus, Linv.T @ Binv @ Linv @ Kus)

    def _projected_process(
            self, Kus, Kss, temp):
        """
        Predict using the predictive variance from the projected process
        approximate method.

        This follows the GPFlow predictive distribution.
        """
        return Kss - np.einsum('ij, ij -> j', Kus, temp)

    def _predict(
            self, X_test, cov, weight, cutpoints, noise_variance,
            projected_process):
        """
        Predict using the predictive variance from the subset of regressors
        algorithm.
        """
        N_test = np.shape(X_test)[0]
        Kss = self.kernel.kernel_prior_diagonal(X_test)
        # Kfs = self.kernel.kernel_matrix(self.X_train, X_test)  # (N, N_test)
        Kus = self.kernel.kernel_matrix(self.Z, X_test)  # (M, N_test)
        temp = cov @ Kus
        if projected_process:
            posterior_variance = self._projected_process(
                Kus, Kss, temp)
        else:
            posterior_variance = self._subset_regressors(
                Kus)
        posterior_std = np.sqrt(posterior_variance)
        posterior_pred_mean = Kus.T @ weight
        posterior_pred_variance = posterior_variance + noise_variance
        posterior_pred_std = np.sqrt(posterior_pred_variance)
        return (
            self._ordinal_predictive_distributions(
                posterior_pred_mean, posterior_pred_std, N_test, cutpoints),
                posterior_pred_mean, posterior_std)

    def predict(
            self, X_test, cov, f, reparametrised=True, whitened=False,
            projected_process=True):
        """
        Return the posterior predictive distribution over classes.

        :arg X_test: The new data points, array like (N_test, D).
        :type X_test: :class:`numpy.ndarray`.
        :arg cov: The approximate
            covariance-posterior-inverse-covariance matrix. Array like (N, N).
        :type cov: :class:`numpy.ndarray`.
        :arg f: Array like (N,).
        :type f: :class:`numpy.ndarray`.
        :arg bool reparametrised: Boolean variable that is `True` if f is
            reparameterised, and `False` if not.
        :arg bool whitened: Boolean variable that is `True` if f is whitened,
            and `False` if not.
        :return: The ordinal class probabilities.
        """
        if self.kernel._ARD:
            # This is the general case where there are hyper-parameters
            # varphi (J, D) for all dimensions and classes.
            raise ValueError(
                "For the ordinal likelihood approximator,the kernel "
                "must not be _ARD type (kernel._ARD=1), but"
                " ISO type (kernel._ARD=0). (got {}, expected)".format(
                    self.kernel._ARD, 0))
        else:
            if whitened is True:
                raise NotImplementedError("Not implemented.")
            elif reparametrised is True:
                return self._predict(
                    X_test, cov, weight=f,
                    cutpoints=self.cutpoints,
                    noise_variance=self.noise_variance,
                    projected_process=projected_process)
            else:
                raise NotImplementedError("Not implemented.")

    def objective(self, K, weight, Z):
        return - weight.T @ K @ weight / 2 + np.sum(Z)

    # def objective_gradient(
    #         self, gx, intervals, cutpoints_ts, cutpoints_tplus1s, varphi,
    #         noise_variance, noise_std,
    #         m, weight, posterior_cov_div_var, trace_posterior_cov_div_var,
    #         trace_cov, N, Z, norm_pdf_z1s, norm_pdf_z2s, indices,
    #         numerical_stability=True, verbose=False):
    #     """TODO"""
    #     return gx

    def approximate_posterior(
            self, theta, indices, steps=None, first_step=1, max_iter=2,
            return_reparameterised=False, verbose=False):
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
            (posterior_mean, weight, *_) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=False)
            (Z,
            norm_pdf_z1s,
            norm_pdf_z2s,
            *_ )= truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS)
            if self.kernel.varphi_hyperhyperparameters is not None:
                fx = self.objective(
                    self.Kuu, weight, Z)
            else:
                # Only terms in f matter
                fx = -0.5 * weight.T @ self.Kuu @ weight + np.sum(Z)
            error = np.abs(fx_old - fx)
            fx_old = fx
            if verbose:
                print("({}), error={}".format(iteration, error))
        gx = 0
        # gx = self.objective_gradient(
        #         gx.copy(), intervals, self.cutpoints_ts,
        #         self.cutpoints_tplus1s,
        #         self.kernel.varphi, self.noise_variance, self.noise_std,
        #         posterior_mean, weight, self.posterior_cov_div_var,
        #         self.trace_posterior_cov_div_var, self.trace_cov,
        #         self.N, Z, norm_pdf_z1s, norm_pdf_z2s, indices,
        #         numerical_stability=True, verbose=False)
        # gx = gx[indices_where]
        if verbose:
            print(
                "cutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints,
                    self.noise_variance,
                    self.kernel.varphi,
                    fx))
        if return_reparameterised is True:
            return fx, gx, weight, (
                self.cov, False)
        elif return_reparameterised is False:
            raise ValueError(
                "Not implemented because this is a sparse implementation")
        elif return_reparameterised is None:
            return fx, gx


class SparseLaplaceGP(LaplaceGP):
    """
    A sparse GP classifier for ordinal likelihood using the Laplace
    approximation.
 
    Inherits the LaplaceGP class. This class allows users to define a
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
        return "SparseLaplaceGP"

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

    def _load_cached_prior(self):
        """
        Load cached prior covariances with Nyström approximation.
        """
        self.M = read_scalar(self.read_path, "M")
        self.Z = read_array(self.read_path, "Z")
        self.Kdiag = read_array(self.read_path, "Kdiag")
        self.Kuu = read_array(self.read_path, "Kuu")
        self.Kfu = read_array(self.read_path, "Kfu")

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
        self.Kfu = self.kernel.kernel_matrix(self.X_train, self.Z)
        warnings.warn(
            "Done updating prior covariance with Nyström approximation")

    def _update_posterior(self, Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
            noise_std, noise_variance, posterior_mean):
        weight = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
        z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
        z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
        precision  = weight**2 + (
            z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
            ) / Z / noise_variance
        # Seems to be necessary to take this cho factor
        (L, lower) = cho_factor(
            self.Kuu + self.jitter * np.eye(self.M), lower=True)
        A = solve_triangular(L, self.Kfu.T, lower=True)  # TODO can this be done by not transposing.
        ALambdaAT = A @ np.diag(precision) @ A.T
        B = ALambdaAT + np.eye(self.M)  # cache qinv
        (LB, lower) = cho_factor(B)
        # Perhaps not necessary to do this solve
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)
        Binv = solve_triangular(LB, LBTinv, lower=False)
        self.posterior_cov_div_var = A.T @ Binv @ A
        posterior_mean_new = self.posterior_cov_div_var @ (
            weight + posterior_mean * precision)
        t1 = posterior_mean - posterior_mean_new
        error = np.abs(max(t1.min(), t1.max(), key=abs))
        return error, weight, posterior_mean_new

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
        posterior_means = []
        posterior_precisions = []
        containers = (posterior_means, posterior_precisions)
        return (posterior_mean_0, containers, error)

    def compute_weights(
            self, posterior_mean):
        """
        Compute the regression weights required for the objective evaluation
        and its gradients.

        A matrix inverse is always required to evaluate the objective.

        :arg posterior_mean:
        """
        # Numerically stable calculation of ordinal likelihood!
        (Z,
        norm_pdf_z1s, norm_pdf_z2s,
        z1s, z2s, *_) = truncated_norm_normalising_constant(
            self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
            posterior_mean, self.EPS,
            upper_bound=self.upper_bound,
            upper_bound2=self.upper_bound2)
        w1 = norm_pdf_z1s / Z
        w2 = norm_pdf_z2s / Z
        z1s = np.nan_to_num(z1s, copy=True, posinf=0.0, neginf=0.0)
        z2s = np.nan_to_num(z2s, copy=True, posinf=0.0, neginf=0.0)
        g1 = z1s * w1
        g2 = z2s * w2
        v1 = z1s * g1
        v2 = z2s * g2
        q1 = z1s * v1
        q2 = z2s * v2
        weight = (w1 - w2) / self.noise_std
        precision = weight**2 + (g2 - g1) / self.noise_variance

        # Seems to be necessary to take this cho factor
        (L, lower) = cho_factor(
            self.Kuu + self.jitter * np.eye(self.M), lower=True)
        A = solve_triangular(L, self.Kfu.T, lower=True)
        ALambdaAT = A @ np.diag(precision) @ A.T
        B = ALambdaAT + np.eye(self.M)  # cache qinv
        (LB, lower) = cho_factor(B)
        # Perhaps not necessary to do this solve
        LBTinv = solve_triangular(LB.T, np.eye(self.M), lower=True)
        Binv = solve_triangular(LB, LBTinv, lower=False)
        cov = np.diag(precision) \
            - np.diag(precision) @ (A.T @ Binv @ A) @ np.diag(precision)
        return weight, precision, w1, w2, g1, g2, v1, v2, q1, q2, LB, cov, Z

    def approximate_posterior(
            self, theta, indices, steps=None,
            posterior_mean_0=None,
            return_reparameterised=False, first_step=1,
            verbose=False):
        """
        Newton-Raphson.

        :arg theta: (log-)hyperparameters to be optimised.
        :type theta:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return:
        """
        # Update prior covariance and get hyperparameters from theta
        (intervals, steps, error, iteration, indices_where,
                gx) = self._hyperparameter_training_step_initialise(
            theta, indices, steps)
        posterior_mean = posterior_mean_0
        while error / steps > self.EPS_2 and iteration < 10:  # TODO is this overkill?
            iteration += 1
            (error, weight, posterior_mean, containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                first_step=first_step, write=False)
            if verbose:
                print("({}), error={}".format(iteration, error))
        (weight, precision,
        w1, w2, g1, g2, v1, v2, q1, q2,
        L_cov, cov, Z) = self.compute_weights(
            posterior_mean)
        fx = self.objective(weight, posterior_mean, precision, L_cov, Z)
        if self.kernel._general and self.kernel._ARD:
            gx = np.zeros(1 + self.J - 1 + 1 + self.J * self.D)
        else:
            gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = gx[np.nonzero(indices)]
        if verbose:

            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (
                cov, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (
                self.noise_variance * self.posterior_cov_div_var, False)
        elif return_reparameterised is None:
            return fx, gx

    def approximate(
            self, steps, posterior_mean_0=None, first_step=1, write=False):
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
        (posterior_mean, containers, error) = self._approximate_initiate(
            posterior_mean_0)
        (posterior_means, posterior_precisions) = containers
        for _ in trange(first_step, first_step + steps,
                        desc="Laplace GP priors Approximator Progress",
                        unit="iterations", disable=True):
            (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s, *_
                    ) = truncated_norm_normalising_constant(
                self.cutpoints_ts, self.cutpoints_tplus1s, self.noise_std,
                posterior_mean, self.EPS, upper_bound=self.upper_bound)
                #upper_bound2=self.upper_bound2, numerically_stable=True)  # TODO Turn this off!
            error, weight, posterior_mean = self._update_posterior(
                Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
                self.noise_std, self.noise_variance, posterior_mean)
            if write is True:
                posterior_means.append(posterior_mean)
                posterior_precisions.append(posterior_precisions)
        containers = (posterior_means, posterior_precisions)
        return error, weight, posterior_mean, containers


class SparsePEPGP(PEPGP):
    """
    Notation changes from Bui 2017:
    m -> posterior_mean
    V -> posterior_cov

    A GP classifier for ordinal likelihood using the Power Expectation
    Propagation (PEP) approximation.

    TODO: must be able to inherit some of EP?
    Inherits the Approximator ABC.

    Power Expectation propagation algorithm as written in section 3.4 and G.2
    of Bui, Thang D. and Yan, Josiah and Turner, Richard E. A Unifying
    Framework for Gaussian Process Pseudo-Point Approximations Using Power
    Expectation Propagation. Journal of Machine Learning Research.
    18. 1532-4435

    This class allows users to define a classification problem and get
    predictions using approximate Bayesian inference. It is for ordinal
    likelihood.

    For this a :class:`probit.kernels.Kernel` is required for the Gaussian
    Process.
    """
    def __repr__(self):
        """
        Return a string representation of this class, used to import the class
        from the string.
        """
        return "PEPGP"

    def __init__(
        self, cutpoints, noise_variance=1.0, *args, **kwargs):
        # cutpoints_hyperparameters=None, noise_std_hyperparameters=None, *args, **kwargs):
        """
        Create an :class:`PEPGP` Approximator object.

        :arg cutpoints: (J + 1, ) array of the cutpoints.
        :type cutpoints: :class:`numpy.ndarray`.
        :arg float noise_variance: Initialisation of noise variance. If `None`
            then initialised to 1.0, default `None`.

        :returns: An :class:`PEPGP` object.
        """
        super().__init__(*args, **kwargs)
        # if cutpoints_hyperparameters is not None:
        #     warnings.warn("cutpoints_hyperparameters set as {}".format(cutpoints_hyperparameters))
        #     self.cutpoints_hyperparameters = cutpoints_hyperparameters
        # else:
        #     self.cutpoints_hyperparameters = None
        # if noise_std_hyperparameters is not None:
        #     warnings.warn("noise_std_hyperparameters set as {}".format(noise_std_hyperparameters))
        #     self.noise_std_hyperparameters = noise_std_hyperparameters
        # else:
        #     self.noise_std_hyperparameters = None
        self.EPS = 1e-4  # perhaps too large
        # self.EPS = 1e-6  # Decreasing EPS will lead to more accurate solutions but a longer convergence time.
        self.EPS_2 = self.EPS**2
        self.jitter = 1e-10
        # Initiate hyperparameters
        self.hyperparameters_update(cutpoints=cutpoints, noise_variance=noise_variance)

    def update_pep_variables(self, Kuuinv, posterior_mean, posterior_cov):
        """TODO: collapse"""
        return Kuuinv @ posterior_mean, Kuuinv @ (Kuuinv - posterior_cov)

    def compute_posterior(self, Kuu, gamma, beta):
        """TODO: collapse"""
        return Kuu @ gamma, Kuu - Kuu @ (beta @ Kuu)

    def run_pep_sequential(self, indices):
        pass

    def approximate(
        self, steps, alpha=0.8, minibatch_size=40, posterior_mean_0=None,
        posterior_cov_0=None, mean_EP_0=None, precision_EP_0=None,
        amplitude_EP_0=None, first_step=1, write=False):
        """
        If no steps are provided, should run for one epoch
        """
        (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, grad_Z_wrt_cavity_mean,
                permutation, containers, error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)



    def _approximate_initiate(
            self, posterior_mean_0=None, posterior_cov_0=None,
            mean_EP_0=None, precision_EP_0=None, amplitude_EP_0=None):
        """
        Initialise the Approximator.

        Need to make sure that the prior covariance is changed!

        :arg int steps: The number of steps in the Approximator.
        :arg posterior_mean_0: The initial state of the posterior mean (N,). If
             `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N,). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros,
            default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site)
            amplitudes (N,). If `None` then initialised to ones,
            default `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg psi_0: Initialisation of hyperhyperparameters. If `None`
            then initialised to ones, default `None`.
        :type psi_0: :class:`numpy.ndarray` or float
        :arg grad_Z_wrt_cavity_mean_0: Initialisation of the EP weights,
            which are gradients of the approximate marginal
            likelihood wrt to the 'cavity distribution mean'. If `None`
            then initialised to zeros, default `None`.
        :type grad_Z_wrt_cavity_mean_0: :class:`numpy.ndarray`
        :return: Containers for the approximate posterior means of parameters
            and hyperparameters.
        :rtype: (12,) tuple.
        """
        if posterior_cov_0 is None:
            # The first EP approximation before data-update is the GP prior cov
            posterior_cov_0 = self.K
        if mean_EP_0 is None:
            mean_EP_0 = np.zeros((self.N,))
        if precision_EP_0 is None:
            precision_EP_0 = np.zeros((self.N,))
        if amplitude_EP_0 is None:
            amplitude_EP_0 = np.ones((self.N,))
        if posterior_mean_0 is None:
            posterior_mean_0 = (
                posterior_cov_0 @ np.diag(precision_EP_0)) @ mean_EP_0
        error = 0.0
        grad_Z_wrt_cavity_mean_0 = np.zeros(self.N)  # Initialisation
        posterior_means = []
        posterior_covs = []
        mean_EPs = []
        amplitude_EPs = []
        precision_EPs = []
        approximate_marginal_likelihoods = []
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_marginal_likelihoods)
        return (posterior_mean_0, posterior_cov_0, mean_EP_0,
                precision_EP_0, amplitude_EP_0, grad_Z_wrt_cavity_mean_0,
                containers, error)


    def approximate(
            self, steps, alpha=0.8, minibatch_size=40, posterior_mean_0=None,
            posterior_cov_0=None, mean_EP_0=None, precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, write=False):
        """
        Estimating the posterior means and posterior covariance (and marginal
        likelihood) via Expectation propagation iteration as written in
        Appendix B Chu, Wei & Ghahramani, Zoubin. (2005). Gaussian Processes
        for Ordinal Regression.. Journal of Machine Learning Research. 6.
        1019-1041.

        EP does not attempt to learn a posterior distribution over
        hyperparameters, but instead tries to approximate
        the joint posterior given some hyperparameters. The hyperparameters
        have to be optimized with model selection step.

        :arg int steps: The number of iterations the Approximator takes.
        :arg posterior_mean_0: The initial state of the approximate posterior
            mean (N,). If `None` then initialised to zeros, default `None`.
        :type posterior_mean_0: :class:`numpy.ndarray`
        :arg posterior_cov_0: The initial state of the posterior covariance
            (N, N). If `None` then initialised to prior covariance,
            default `None`.
        :type posterior_cov_0: :class:`numpy.ndarray`
        :arg mean_EP_0: The initial state of the individual (site) mean (N,).
            If `None` then initialised to zeros, default `None`.
        :type mean_EP_0: :class:`numpy.ndarray`
        :arg precision_EP_0: The initial state of the individual (site)
            variance (N,). If `None` then initialised to zeros, default `None`.
        :type precision_EP_0: :class:`numpy.ndarray`
        :arg amplitude_EP_0: The initial state of the individual (site)
            amplitudes (N,). If `None` then initialised to ones, default
            `None`.
        :type amplitude_EP_0: :class:`numpy.ndarray`
        :arg int first_step: The first step. Useful for burn in algorithms.
        :arg bool fix_hyperparameters: Must be `True`, since the hyperparameter
            approximate posteriors are of the hyperparameters are not
            calculated in this EP approximation.
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
        (posterior_mean, posterior_cov, mean_EP, precision_EP,
                amplitude_EP, grad_Z_wrt_cavity_mean, containers,
                error) = self._approximate_initiate(
            posterior_mean_0, posterior_cov_0, mean_EP_0, precision_EP_0,
            amplitude_EP_0)
        (posterior_means, posterior_covs, mean_EPs, precision_EPs,
            amplitude_EPs, approximate_log_marginal_likelihoods) = containers
        for step in trange(first_step, first_step + steps,
                        desc="EP GP approximator progress",
                        unit="iterations", disable=True):
            index = self.new_point(step, random_selection=True)
            target = self.t_train[index]
            # Find the mean and variance of the leave-one-out
            # posterior distribution Q^{\backslash i}(\bm{f})
            (posterior_mean_n, posterior_variance_n, cavity_mean_n,
            cavity_variance_n, mean_EP_n_old,
            precision_EP_n_old, amplitude_EP_n_old) = self._remove(
                posterior_cov[index, index], posterior_mean[index],
                mean_EP[index], precision_EP[index], amplitude_EP[index])
            # Tilt/ moment match
            (mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n, posterior_mean_n_new,
            posterior_covariance_n_new, z1, z2, nu_n) = self._include(
                target, cavity_mean_n, cavity_variance_n,
                self.cutpoints[target], self.cutpoints[target + 1],
                self.noise_variance)
            # Update EP weight (alpha)
            grad_Z_wrt_cavity_mean[index] = grad_Z_wrt_cavity_mean_n
            #print(grad_Z_wrt_cavity_mean)
            diff = precision_EP_n - precision_EP_n_old
            if (np.abs(diff) > self.EPS
                    and Z_n > self.EPS
                    and precision_EP_n > 0.0
                    and posterior_covariance_n_new > 0.0):
                # Update posterior mean and rank-1 covariance
                posterior_cov, posterior_mean = self._update(
                    index, mean_EP_n_old, posterior_cov,
                    posterior_mean_n, posterior_variance_n,
                    precision_EP_n_old, grad_Z_wrt_cavity_mean_n,
                    posterior_mean_n_new, posterior_mean,
                    posterior_covariance_n_new, diff)
                # Update EP parameters
                precision_EP[index] = precision_EP_n
                mean_EP[index] = mean_EP_n
                amplitude_EP[index] = amplitude_EP_n
                error += (diff**2
                          + (mean_EP_n - mean_EP_n_old)**2
                          + (amplitude_EP_n - amplitude_EP_n_old)**2)
                if write:
                    # approximate_log_marginal_likelihood = \
                    # self._approximate_log_marginal_likelihood(
                    # posterior_cov, precision_EP, mean_EP)
                    posterior_means.append(posterior_mean)
                    posterior_covs.append(posterior_cov)
                    mean_EPs.append(mean_EP)
                    precision_EPs.append(precision_EP)
                    amplitude_EPs.append(amplitude_EP)
                    # approximate_log_marginal_likelihood.append(
                    #   approximate_marginal_log_likelihood)
            else:
                if precision_EP_n < 0.0 or posterior_covariance_n_new < 0.0:
                    print(
                        "Skip {} update z1={}, z2={}, nu={} p_new={},"
                        " p_old={}.\n".format(
                        index, z1, z2, nu_n,
                        precision_EP_n, precision_EP_n_old))
        containers = (posterior_means, posterior_covs, mean_EPs, precision_EPs,
                      amplitude_EPs, approximate_log_marginal_likelihoods)
        return (
            error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP, containers)
        # TODO: are there some other inputs missing here?
        # error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
        #  precision_EP, amplitude_EP, *_

    def new_point(self, step, random_selection=True):
        """
        Return a new point based on some policy.

        :arg int step: The current iteration step.
        :arg bool random_selection: If random_selection is true, then returns
            a random point from the ordering.
            Otherwise, it returns a sequential point. Default `True`.
        :return: index
        """
        if random_selection:
            return random.randint(0, self.N-1)
        else:
            # If steps < N, then some points never get updated (becomes ADF)
            return step % self.N

    def _remove(
            self, posterior_variance_n, posterior_mean_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old):
        """
        Calculate the product of approximate posterior factors with the current
        index removed.

        This is called the cavity distribution,
        "a bit like leaving a hole in the dataset".

        :arg float posterior_variance_n: Variance of latent function at index.
        :arg float posterior_mean_n: The state of the approximate posterior
            mean.
        :arg float mean_EP_n: The state of the individual (site) mean.
        :arg precision_EP_n: The state of the individual (site) variance.
        :arg amplitude_EP_n: The state of the individual (site) amplitudes.
        :returns: A (8,) tuple containing cavity mean and variance, and old
            site states.
        """
        if posterior_variance_n > 0:
            cavity_variance_n = posterior_variance_n / (
                1 - posterior_variance_n * precision_EP_n_old)
            if cavity_variance_n > 0:
                cavity_mean_n = (posterior_mean_n
                    + cavity_variance_n * precision_EP_n_old * (
                        posterior_mean_n - mean_EP_n_old))
            else:
                raise ValueError(
                    "cavity_variance_n must be non-negative (got {})".format(
                        cavity_variance_n))
        else:
            raise ValueError(
                "posterior_cov_nn must be non-negative (got {})".format(
                    posterior_variance_n))
        return (
            posterior_mean_n, posterior_variance_n,
            cavity_mean_n, cavity_variance_n,
            mean_EP_n_old, precision_EP_n_old, amplitude_EP_n_old)

    def _assert_valid_values(self, nu_n, variance, cavity_mean_n,
            cavity_variance_n, target, z1, z2, Z_n, norm_pdf_z1, norm_pdf_z2,
            grad_Z_wrt_cavity_variance_n, grad_Z_wrt_cavity_mean_n):
        if math.isnan(grad_Z_wrt_cavity_mean_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_mean is nan (got {})".format(
                grad_Z_wrt_cavity_mean_n))
        if math.isnan(grad_Z_wrt_cavity_variance_n):
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "grad_Z_wrt_cavity_variance is nan (got {})".format(
                    grad_Z_wrt_cavity_variance_n))
        if nu_n <= 0:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError("nu_n must be positive (got {})".format(nu_n))
        if nu_n > 1.0 / variance + self.EPS:
            print(
                "cavity_mean_n={} \n"
                "cavity_variance_n={} \n"
                "target={} \n"
                "z1 = {} z2 = {} \n"
                "Z_n = {} \n"
                "norm_pdf_z1 = {} \n"
                "norm_pdf_z2 = {} \n"
                "beta = {} alpha = {}".format(
                    cavity_mean_n, cavity_variance_n, target, z1, z2, Z_n,
                    norm_pdf_z1, norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                    grad_Z_wrt_cavity_mean_n))
            raise ValueError(
                "nu_n must be less than 1.0 / (cavity_variance_n + "
                "noise_variance) = {}, got {}".format(
                    1.0 / variance, nu_n))
        return 0

    def _include(
            self, target, cavity_mean_n, cavity_variance_n,
            cutpoints_t, cutpoints_tplus1, noise_variance,
            numerically_stable=False):
        """
        Update the approximate posterior by incorporating the message
        p(t_i|m_i) into Q^{\i}(\bm{f}).
        Wei Chu, Zoubin Ghahramani 2005 page 20, Eq. (23)
        This includes one true-observation likelihood, and 'tilts' the
        approximation towards the true posterior. It updates the approximation
        to the true posterior by minimising a moment-matching KL divergence
        between the tilted distribution and the posterior distribution. This
        gives us an approximate posterior in the approximating family. The
        update to posterior_cov is a rank-1 update (see the outer product of
        two 1d vectors), and so it essentially constructs a piecewise low rank
        approximation to the GP posterior covariance matrix, until convergence
        (by which point it will no longer be low rank).
        :arg int target: The ordinal class index of the current site
            (the class of the datapoint that is "left out").
        :arg float cavity_mean_n: The cavity mean of the current site.
        :arg float cavity_variance_n: The cavity variance of the current site.
        :arg float cutpoints_t: The upper cutpoint parameters.
        :arg float cutpoints_tplus1: The lower cutpoint parameter.
        :arg float noise_variance: Initialisation of noise variance. If
            `None` then initialised to one, default `None`.
        :arg bool numerically_stable: Boolean variable for assert valid
            numerical values. Default `False'.
        :returns: A (10,) tuple containing cavity mean and variance, and old
            site states.
        """
        variance = cavity_variance_n + noise_variance
        std_dev = np.sqrt(variance)
        # Compute Z
        norm_cdf_z2 = 0.0
        norm_cdf_z1 = 1.0
        norm_pdf_z1 = 0.0
        norm_pdf_z2 = 0.0
        z1 = 0.0
        z2 = 0.0
        if target == 0:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z1_abs = np.abs(z1)
            if z1_abs > self.upper_bound:
                z1 = np.sign(z1) * self.upper_bound
            Z_n = norm_cdf(z1) - norm_cdf_z2
            norm_pdf_z1 = norm_z_pdf(z1)
        elif target == self.J - 1:
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            z2_abs = np.abs(z2)
            if z2_abs > self.upper_bound:
                z2 = np.sign(z2) * self.upper_bound
            Z_n = norm_cdf_z1 - norm_cdf(z2)
            norm_pdf_z2 = norm_z_pdf(z2)
        else:
            z1 = (cutpoints_tplus1 - cavity_mean_n) / std_dev
            z2 = (cutpoints_t - cavity_mean_n) / std_dev
            Z_n = norm_cdf(z1) - norm_cdf(z2)
            norm_pdf_z1 = norm_z_pdf(z1)
            norm_pdf_z2 = norm_z_pdf(z2)
        if Z_n < self.EPS:
            if np.abs(np.exp(-0.5*z1**2 + 0.5*z2**2) - 1.0) > self.EPS**2:
                grad_Z_wrt_cavity_mean_n = (z1 * np.exp(
                        -0.5*z1**2 + 0.5*z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5 * z1 ** 2) + 0.5 * z2 ** 2) - 1.0)
                        * variance
                )
                grad_Z_wrt_cavity_variance_n = (
                    -1.0 + (z1**2 + 0.5 * z2**2) - z2**2) / (
                    (
                        (np.exp(-0.5*z1**2 + 0.5 * z2**2) - 1.0)
                        * 2.0 * variance)
                )
                grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
                nu_n = (
                    grad_Z_wrt_cavity_mean_n_2
                    - 2.0 * grad_Z_wrt_cavity_variance_n)
            else:
                grad_Z_wrt_cavity_mean_n = 0.0
                grad_Z_wrt_cavity_mean_n_2 = 0.0
                grad_Z_wrt_cavity_variance_n = -(
                    1.0 - self.EPS)/(2.0 * variance)
                nu_n = (1.0 - self.EPS) / variance
                warnings.warn(
                    "Z_n must be greater than tolerance={} (got {}): "
                    "SETTING to Z_n to approximate value\n"
                    "z1={}, z2={}".format(
                        self.EPS, Z_n, z1, z2))
            if nu_n >= 1.0 / variance:
                nu_n = (1.0 - self.EPS) / variance
            if nu_n <= 0.0:
                nu_n = self.EPS * variance
        else:
            grad_Z_wrt_cavity_variance_n = (
                - z1 * norm_pdf_z1 + z2 * norm_pdf_z2) / (
                    2.0 * variance * Z_n)  # beta
            grad_Z_wrt_cavity_mean_n = (
                - norm_pdf_z1 + norm_pdf_z2) / (
                    std_dev * Z_n)  # alpha/gamma
            grad_Z_wrt_cavity_mean_n_2 = grad_Z_wrt_cavity_mean_n**2
            nu_n = (grad_Z_wrt_cavity_mean_n_2
                - 2.0 * grad_Z_wrt_cavity_variance_n)
        # Update alphas
        if numerically_stable:
            self._assert_valid_values(
                nu_n, variance, cavity_mean_n, cavity_variance_n, target,
                z1, z2, Z_n, norm_pdf_z1,
                norm_pdf_z2, grad_Z_wrt_cavity_variance_n,
                grad_Z_wrt_cavity_mean_n)
        # hnew = loomean + loovar * alpha;
        posterior_mean_n_new = (
            cavity_mean_n + cavity_variance_n * grad_Z_wrt_cavity_mean_n)
        # cnew = loovar - loovar * nu * loovar;
        posterior_covariance_n_new = (
            cavity_variance_n - cavity_variance_n**2 * nu_n)
        # pnew = nu / (1.0 - loovar * nu);
        precision_EP_n = nu_n / (1.0 - cavity_variance_n * nu_n)
        # print("posterior_mean_n_new", posterior_mean_n_new)
        # print("nu_n", nu_n)
        # print("precision_EP_n", precision_EP_n)
        # mnew = loomean + alpha / nu;
        mean_EP_n = cavity_mean_n + grad_Z_wrt_cavity_mean_n / nu_n
        # snew = Zi * sqrt(loovar * pnew + 1.0)*exp(0.5 * alpha * alpha / nu);
        amplitude_EP_n = Z_n * np.sqrt(
            cavity_variance_n * precision_EP_n + 1.0) * np.exp(
                0.5 * grad_Z_wrt_cavity_mean_n_2 / nu_n)
        return (
            mean_EP_n, precision_EP_n, amplitude_EP_n, Z_n,
            grad_Z_wrt_cavity_mean_n,
            posterior_mean_n_new, posterior_covariance_n_new, z1, z2, nu_n)

    def _update(
        self, index, mean_EP_n_old, posterior_cov,
        posterior_mean_n, posterior_variance_n,
        precision_EP_n_old,
        grad_Z_wrt_cavity_mean_n, posterior_mean_n_new, posterior_mean,
        posterior_covariance_n_new, diff, numerically_stable=False):
        """
        Update the posterior mean and covariance.

        Projects the tilted distribution on to an approximating family.
        The update for the t_n is a rank-1 update. Constructs a low rank
        approximation to the GP posterior covariance matrix.

        :arg int index: The index of the current likelihood (the index of the
            datapoint that is "left out").
        :arg float mean_EP_n_old: The state of the individual (site) mean (N,).
        :arg posterior_cov: The current approximate posterior covariance
            (N, N).
        :type posterior_cov: :class:`numpy.ndarray`
        :arg float posterior_variance_n: The current approximate posterior
            site variance.
        :arg float posterior_mean_n: The current site approximate posterior
            mean.
        :arg float precision_EP_n_old: The state of the individual (site)
            variance (N,).
        :arg float grad_Z_wrt_cavity_mean_n: The gradient of the log
            normalising constant with respect to the site cavity mean
            (The EP "weight").
        :arg float posterior_mean_n_new: The state of the site approximate
            posterior mean.
        :arg float posterior_covariance_n_new: The state of the site
            approximate posterior variance.
        :arg float diff: The differance between precision_EP_n and
            precision_EP_n_old.
        :returns: The updated approximate posterior mean and covariance.
        :rtype: tuple (`numpy.ndarray`, `numpy.ndarray`)
        """
        rho = diff / (1 + diff * posterior_variance_n)
        eta = (
            grad_Z_wrt_cavity_mean_n
            + precision_EP_n_old * (posterior_mean_n - mean_EP_n_old)) / (
                1.0 - posterior_variance_n * precision_EP_n_old)
        a_n = posterior_cov[:, index]  # The index'th column of posterior_cov
        posterior_cov = posterior_cov - rho * np.outer(a_n, a_n)
        posterior_mean += eta * a_n
        if numerically_stable is True:
            # TODO is this inequality meant to be the other way around?
            # TODO is hnew meant to be the EP weights, grad_Z_wrt_cavity_mean_n
            # assert(fabs((settings->alpha+index)->postcov[index]-alpha->cnew)<EPS)
            if np.abs(
                    posterior_covariance_n_new
                    - posterior_cov[index, index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_covariance_n_new - posterior_cov[index, "
                    "index]) must be less than some tolerance. Got (posterior_"
                    "covariance_n_new={}, posterior_cov_index_index={}, diff="
                    "{})".format(
                    posterior_covariance_n_new, posterior_cov[index, index],
                    posterior_covariance_n_new - posterior_cov[index, index]))
            # assert(fabs((settings->alpha+index)->pair->postmean-alpha->hnew)<EPS)
            if np.abs(posterior_mean_n_new - posterior_mean[index]) > self.EPS:
                raise ValueError(
                    "np.abs(posterior_mean_n_new - posterior_mean[index]) must"
                    " be less than some tolerance. Got (posterior_mean_n_new="
                    "{}, posterior_mean_index={}, diff={})".format(
                        posterior_mean_n_new, posterior_mean[index],
                        posterior_mean_n_new - posterior_mean[index]))
        return posterior_cov, posterior_mean

    def _approximate_log_marginal_likelihood(
            self, posterior_cov, precision_EP, amplitude_EP, mean_EP,
            numerical_stability):
        """
        Calculate the approximate log marginal likelihood.
        TODO: need to finish this. Probably not useful if using EP.

        :arg posterior_cov: The approximate posterior covariance.
        :type posterior_cov:
        :arg precision_EP: The state of the individual (site) variance (N,).
        :type precision_EP:
        :arg amplitude_EP: The state of the individual (site) amplitudes (N,).
        :type amplitude EP:
        :arg mean_EP: The state of the individual (site) mean (N,).
        :type mean_EP:
        :arg bool numerical_stability: If the calculation is made in a
            numerically stable manner.
        """
        precision_matrix = np.diag(precision_EP)
        inverse_precision_matrix = 1. / precision_matrix  # Since it is a diagonal, this is the inverse.
        log_amplitude_EP = np.log(amplitude_EP)
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp\
                - temp.T @ mean_EP
        if numerical_stability is True:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.trace(
                    np.log(inverse_precision_matrix)))
            approximate_marginal_likelihood = np.add(
                    approximate_marginal_likelihood, B/2)
            approximate_marginal_likelihood = np.subtract(
                approximate_marginal_likelihood, 0.5 * np.trace(
                    np.log(self.K + inverse_precision_matrix)))
            return np.sum(approximate_marginal_likelihood)
        else:
            approximate_marginal_likelihood = np.add(
                log_amplitude_EP, 0.5 * np.log(np.linalg.det(
                    inverse_precision_matrix)))  # TODO: use log det C trick
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, B/2
            )
            approximate_marginal_likelihood = np.add(
                approximate_marginal_likelihood, 0.5 * np.log(
                    np.linalg.det(self.K + inverse_precision_matrix))
            )  # TODO: use log det C trick
            return np.sum(approximate_marginal_likelihood)

    def grid_over_hyperparameters(
            self, domain, res,
            indices=None,
            posterior_mean_0=None, posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None, amplitude_EP_0=None,
            first_step=1, write=False, verbose=False):
        """
        Return meshgrid values of fx and gx over hyperparameter space.

        The particular hyperparameter space is inferred from the user inputs,
        indices.
        """
        steps = self.N  # TODO: let user specify this
        (x1s, x2s,
        xlabel, ylabel,
        xscale, yscale,
        xx, yy,
        thetas, fxs,
        gxs, gx_0, intervals,
        indices_where) = self._grid_over_hyperparameters_initiate(
            res, domain, indices, self.cutpoints)
        for i, phi in enumerate(thetas):
            self._grid_over_hyperparameters_update(
                phi, indices, self.cutpoints)
            if verbose:
                print(
                    "cutpoints_0 = {}, varphi_0 = {}, noise_variance_0 = {}, "
                    "variance_0 = {}".format(
                        self.cutpoints, self.kernel.varphi, self.noise_variance,
                        self.kernel.variance))
            # Reset parameters
            iteration = 0
            error = np.inf
            posterior_mean = posterior_mean_0
            posterior_cov = posterior_cov_0
            mean_EP = mean_EP_0
            precision_EP = precision_EP_0
            amplitude_EP = amplitude_EP_0
            while error / steps > self.EPS**2:
                iteration += 1
                (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov, mean_EP,
                 precision_EP, amplitude_EP, containers) = self.approximate(
                    steps, posterior_mean_0=posterior_mean, posterior_cov_0=posterior_cov,
                    mean_EP_0=mean_EP, precision_EP_0=precision_EP,
                    amplitude_EP_0=amplitude_EP,
                    first_step=first_step, write=False)
                if verbose:
                    print("({}), error={}".format(iteration, error))
            print("{}/{}".format(i + 1, len(thetas)))
            weight, precision_EP, L_cov, cov = self.compute_weights(
                precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
            t1, t2, t3, t4, t5 = self.compute_integrals_vector(
                np.diag(posterior_cov), posterior_mean, self.noise_variance)
            fx = self.objective(
                precision_EP, posterior_mean,
                t1, L_cov, cov, weight)
            fxs[i] = fx
            gx = self.objective_gradient(
                gx_0.copy(), intervals, self.kernel.varphi,
                self.noise_variance,
                t2, t3, t4, t5, cov, weight, indices)
            gxs[i] = gx[indices_where]
            if verbose:
                print("function call {}, gradient vector {}".format(fx, gx))
                print("varphi={}, noise_variance={}, fx={}".format(
                    self.kernel.varphi, self.noise_variance, fx))
        if x2s is not None:
            return (fxs.reshape((len(x1s), len(x2s))), gxs, xx, yy,
                xlabel, ylabel, xscale, yscale)
        else:
            return (fxs, gxs, x1s, None, xlabel, ylabel, xscale, yscale)

    def approximate_posterior(
            self, phi, indices, steps=None,
            posterior_mean_0=None, return_reparameterised=False,
            posterior_cov_0=None, mean_EP_0=None,
            precision_EP_0=None,
            amplitude_EP_0=None, first_step=1, verbose=True):
        """
        Optimisation routine for hyperparameters.

        :arg phi: (log-)hyperparameters to be optimised.
        :type phi:
        :arg indices:
        :type indices:
        :arg steps:
        :type steps:
        :arg posterior_mean_0:
        :type posterior_mean_0:
        :arg return_reparameterised:
        :type return_reparameterised:
        :arg posterior_cov_0:
        :type posterior_cov_0:
        :arg mean_EP_0:
        :type mean_EP_0:
        :arg precision_EP_0:
        :type precision_EP_0:
        :arg amplitude_EP_0:
        :type amplitude_EP_0:
        :arg int first_step:
        :arg bool write:
        :arg bool verbose:
        :return: fx the objective and gx the objective gradient
        """
        # Update prior covariance and get hyperparameters from phi
        (intervals, steps, error, iteration, indices_where,
        gx) = self._hyperparameter_training_step_initialise(
            phi, indices, steps)
        posterior_mean = posterior_mean_0
        posterior_cov = posterior_cov_0
        mean_EP = mean_EP_0
        precision_EP = precision_EP_0
        amplitude_EP = amplitude_EP_0
        while error / steps > self.EPS**2:
            iteration += 1
            (error, grad_Z_wrt_cavity_mean, posterior_mean, posterior_cov,
            mean_EP, precision_EP, amplitude_EP,
            containers) = self.approximate(
                steps, posterior_mean_0=posterior_mean,
                posterior_cov_0=posterior_cov, mean_EP_0=mean_EP,
                precision_EP_0=precision_EP,
                amplitude_EP_0=amplitude_EP,
                first_step=first_step, write=False)
        # TODO: this part requires an inverse, could it be sparsified
        # by putting q(f) = p(f_m) R(f_n). This is probably how FITC works
        (weight, precision_EP, L_cov, cov) = self.compute_weights(
            precision_EP, mean_EP, grad_Z_wrt_cavity_mean)
        # Try optimisation routine
        t1, t2, t3, t4, t5 = self.compute_integrals_vector(
            np.diag(posterior_cov), posterior_mean, self.noise_variance)
        fx = self.objective(precision_EP, posterior_mean, t1,
            L_cov, cov, weight)
        gx = np.zeros(1 + self.J - 1 + 1 + 1)
        gx = self.objective_gradient(
            gx, intervals, self.kernel.varphi, self.noise_variance,
            t2, t3, t4, t5, cov, weight, indices)
        gx = gx[np.where(indices != 0)]
        if verbose:
            print(
                "\ncutpoints={}, noise_variance={}, "
                "varphi={}\nfunction_eval={}".format(
                    self.cutpoints, self.noise_variance,
                    self.kernel.varphi, fx))
        if return_reparameterised is True:
            return fx, gx, weight, (cov, True)
        elif return_reparameterised is False:
            return fx, gx, posterior_mean, (posterior_cov, False)
        elif return_reparameterised is None:
            return fx, gx

    def compute_integrals_vector(
            self, posterior_variance, posterior_mean, noise_variance):
        """
        Compute the integrals required for the gradient evaluation.
        """
        noise_std = np.sqrt(noise_variance)
        mean_ts = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_ts) / (
                noise_variance + posterior_variance)
        mean_tplus1s = (posterior_mean * noise_variance
            + posterior_variance * self.cutpoints_tplus1s) / (
                noise_variance + posterior_variance)
        sigma = np.sqrt(
            (noise_variance * posterior_variance) / (
            noise_variance + posterior_variance))
        a_ts = mean_ts - 5.0 * sigma
        b_ts = mean_ts + 5.0 * sigma
        h_ts = b_ts - a_ts
        a_tplus1s = mean_tplus1s - 5.0 * sigma
        b_tplus1s = mean_tplus1s + 5.0 * sigma
        h_tplus1s = b_tplus1s - a_tplus1s
        y_0 = np.zeros((20, self.N))
        t1 = fromb_t1_vector(
                y_0.copy(), posterior_mean, posterior_variance,
                self.cutpoints_ts, self.cutpoints_tplus1s,
                noise_std, self.EPS, self.EPS_2, self.N)
        t2 = fromb_t2_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t2[self.indices_where_0] = 0.0
        t3 = fromb_t3_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t3[self.indices_where_J_1] = 0.0
        t4 = fromb_t4_vector(
                y_0.copy(), mean_tplus1s, sigma,
                a_tplus1s, b_tplus1s,
                h_tplus1s, posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t4[self.indices_where_J_1] = 0.0
        t5 = fromb_t5_vector(
                y_0.copy(), mean_ts, sigma,
                a_ts, b_ts, h_ts,
                posterior_mean,
                posterior_variance,
                self.cutpoints_ts,
                self.cutpoints_tplus1s,
                noise_variance, noise_std, self.EPS, self.EPS_2, self.N)
        t5[self.indices_where_0] = 0.0
        return t1, t2, t3, t4, t5

    def objective(
            self, precision_EP, posterior_mean, t1, L_cov, cov,
            weights):
        """
        Calculate fx, the variational lower bound of the log marginal
        likelihood at the EP equilibrium.

        .. math::
                \mathcal{F(\theta)} =,

            where :math:`F(\theta)` is the variational lower bound of the log
            marginal likelihood at the EP equilibrium,
            :math:`h`, :math:`\Pi`, :math:`K`. #TODO

        :arg precision_EP:
        :type precision_EP:
        :arg posterior_mean:
        :type posterior_mean:
        :arg t1:
        :type t1:
        :arg L_cov:
        :type L_cov:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :returns: fx
        :rtype: float
        """
        # Fill possible zeros in with machine precision
        precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        fx = -np.sum(np.log(np.diag(L_cov)))  # log det cov
        fx -= 0.5 * posterior_mean.T @ weights
        fx -= 0.5 * np.sum(np.log(precision_EP))
        # cov = L^{-1} L^{-T}  # requires a backsolve with the identity
        # TODO: check if there is a simpler calculation that can be done
        fx -= 0.5 * np.sum(np.divide(np.diag(cov), precision_EP))
        fx += np.sum(t1)
        # Regularisation - penalise large varphi (overfitting)
        # fx -= 0.1 * self.kernel.varphi
        return -fx

    def objective_gradient(
            self, gx, intervals, varphi, noise_variance,
            t2, t3, t4, t5, cov, weights, indices):
        """
        Calculate gx, the jacobian of the variational lower bound of the
        log marginal likelihood at the EP equilibrium.

        .. math::
                \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

            where :math:`F(\theta)` is the variational lower bound of the 
            log marginal likelihood at the EP equilibrium,
            :math:`\theta` is the set of hyperparameters,
            :math:`h`, :math:`\Pi`, :math:`K`.  #TODO

        :arg intervals:
        :type intervals:
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg varphi: The kernel hyper-parameters.
        :type varphi: :class:`numpy.ndarray` or float.
        :arg t2:
        :type t2:
        :arg t3:
        :type t3:
        :arg t4:
        :type t4:
        :arg t5:
        :type t5:
        :arg cov:
        :type cov:
        :arg weights:
        :type weights:
        :return: gx
        :rtype: float
        """
        if indices is not None:
            # Update gx
            if indices[0]:
                # For gx[0] -- ln\sigma
                gx[0] = np.sum(t5 - t4)
                # gx[0] *= -0.5 * noise_variance  # This is a typo in the Chu code
                gx[0] *= np.sqrt(noise_variance)
            # For gx[1] -- \b_1
            if indices[1]:
                gx[1] = np.sum(t3 - t2)
            # For gx[2] -- ln\Delta^r
            for j in range(2, self.J):
                if indices[j]:
                    targets = self.t_train[self.grid]
                    gx[j] = np.sum(t3[targets == j - 1])
                    gx[j] -= np.sum(t2[targets == self.J - 1])
                    # TODO: check this, since it may be an `else` condition!!!!
                    gx[j] += np.sum(t3[targets > j - 1] - t2[targets > j - 1])
                    gx[j] *= intervals[j - 2]
            # For gx[self.J] -- variance
            if indices[self.J]:
                # For gx[self.J] -- s
                # TODO: Need to check this is correct: is it directly analogous to
                # gradient wrt log varphi?
                partial_K_s = self.kernel.kernel_partial_derivative_s(
                    self.X_train, self.X_train)
                # VC * VC * a' * partial_K_varphi * a / 2
                gx[self.J] = varphi * 0.5 * weights.T @ partial_K_s @ weights  # That's wrong. not the same calculation.
                # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                gx[self.J] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_s))
                # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                # gx[self.J] -= 0.1 * varphi
                gx[self.J] *= 2.0  # since varphi = kappa / 2
            # For gx[self.J + 1] -- varphi
            if indices[self.J + 1]:
                partial_K_varphi = self.kernel.kernel_partial_derivative_varphi(
                    self.X_train, self.X_train)
                # elif 1:
                #     gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights
                # TODO: This needs fixing/ checking vs original code
                if 0:
                    for l in range(self.kernel.L):
                        K = self.kernel.num_hyperparameters[l]
                        KK = 0
                        for k in range(K):
                            gx[self.J + KK + k] = varphi[l] * 0.5 * weights.T @ partial_K_varphi[l][k] @ weights
                        KK += K
                else:
                    # VC * VC * a' * partial_K_varphi * a / 2
                    gx[self.J + 1] = varphi * 0.5 * weights.T @ partial_K_varphi @ weights  # That's wrong. not the same calculation.
                    # equivalent to -= varphi * 0.5 * np.trace(cov @ partial_K_varphi)
                    gx[self.J + 1] -= varphi * 0.5 * np.sum(np.multiply(cov, partial_K_varphi))
                    # ad-hoc Regularisation term - penalise large varphi, but Occam's term should do this already
                    # gx[self.J] -= 0.1 * varphi
        return -gx

    def approximate_evidence(self, mean_EP, precision_EP, amplitude_EP, posterior_cov):
        """
        TODO: check and return line could be at risk of overflow
        Compute the approximate evidence at the EP solution.

        :return:
        """
        temp = np.multiply(mean_EP, precision_EP)
        B = temp.T @ posterior_cov @ temp - np.multiply(
            temp, mean_EP)
        Pi_inv = np.diag(1. / precision_EP)
        return (
            np.prod(
                amplitude_EP) * np.sqrt(np.linalg.det(Pi_inv)) * np.exp(B / 2)
                / np.sqrt(np.linalg.det(np.add(Pi_inv, self.K))))

    def compute_weights(
        self, precision_EP, mean_EP, grad_Z_wrt_cavity_mean,
        L_cov=None, cov=None, numerically_stable=False):
        """
        TODO: There may be an issue, where grad_Z_wrt_cavity_mean is updated
        when it shouldn't be, on line 2045.

        Compute the regression weights required for the gradient evaluation,
        and check that they are in equilibrium with
        the gradients of Z wrt cavity means.

        A matrix inverse is always required to evaluate fx.

        :arg precision_EP:
        :arg mean_EP:
        :arg grad_Z_wrt_cavity_mean:
        :arg L_cov: . Default `None`.
        :arg cov: . Default `None`.
        """
        if np.any(precision_EP == 0.0):
            # TODO: Only check for equilibrium if it has been updated in this swipe
            warnings.warn("Some sample(s) have not been updated.\n")
            precision_EP[precision_EP == 0.0] = self.EPS * self.EPS
        Pi_inv = np.diag(1. / precision_EP)
        if L_cov is None or cov is None:
            (L_cov, lower) = cho_factor(
                Pi_inv + self.K)
            L_covT_inv = solve_triangular(
                L_cov.T, np.eye(self.N), lower=True)
            # TODO It is necessary to do this triangular solve to get
            # diag(cov) for the lower bound on the marginal likelihood
            # calculation. Note no tf implementation for diag(A^{-1}) yet.
            cov = solve_triangular(L_cov, L_covT_inv, lower=False)
        if numerically_stable:
            # This is 3-4 times slower on CPU,
            # what about with jit compiled CPU or GPU?
            # Is this ever more stable than a matmul by the inverse?
            g = cho_solve((L_cov, False), mean_EP)
            weight = cho_solve((L_cov.T, True), g)
        else:
            weight = cov @ mean_EP
        if np.any(
            np.abs(weight - grad_Z_wrt_cavity_mean) > np.sqrt(self.EPS)):
            warnings.warn("Fatal error: the weights are not in equilibrium wit"
                "h the gradients".format(
                    weight, grad_Z_wrt_cavity_mean))
        return weight, precision_EP, L_cov, cov
