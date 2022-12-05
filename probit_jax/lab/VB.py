import lab as B
from math import inf
from scipy.stats import expon
from probit.lab.utilities import (
    truncated_norm_normalising_constant, matrix_inverse, log_over_sqrt_2_pi)


def noise_variance_weight(cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound, upper_bound2):
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = truncated_norm_normalising_constant(
            cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    return (noise_std * (norm_pdf_z1s - norm_pdf_z2s) / Z,
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s)


def objective_VB(
        N, posterior_mean, weight, trace_cov,
        trace_posterior_cov_div_var, Z, noise_variance,
        log_det_cov):
    """
    Calculate fx, the variational lower bound of the log marginal
    likelihood.

    .. math::
            \mathcal{F()} =,

        where
    """
    trace_K_inv_posterior_cov = noise_variance * trace_cov
    return (0.5 * trace_posterior_cov_div_var
        + 0.5 * trace_K_inv_posterior_cov
        + 0.5 * posterior_mean.T @ weight
        - 0.5 * N * B.log(noise_variance)
        - 0.5 * log_det_cov
        - 0.5 * N
        - B.sum(B.log(Z)))


def objective_gradient_VB(
        gx, intervals, cutpoints_ts, cutpoints_tplus1s, theta, variance,
        noise_std, y_train, posterior_mean, weight, cov,
        trace_posterior_cov_div_var, partial_K_theta, partial_K_variance,
        J, D, ARD,
        upper_bound, upper_bound2, trainables,
        numerical_stability=True, verbose=False):
    """
    Calculate gx, the jacobian of the variational lower bound of the log
    marginal likelihood at the VB equilibrium,

    .. math::
            \mathcal{\frac{\partial F(\theta)}{\partial \theta}}

        where :math:`F(\theta)` is the variational lower bound of the log
        marginal likelihood at the EP equilibrium,
        :math:`\theta` is the set of hyperparameters, :math:`h`,
        :math:`\Pi`, :math:`K`. #TODO

    :arg intervals: The vector of the first cutpoint and the intervals
        between cutpoints for unconstrained optimisation of the cutpoint
        parameters.
    :type intervals: :class:`numpy.ndarray`
    :arg theta: The kernel hyper-parameters.
    :type theta: :class:`numpy.ndarray` or float.
    :arg float noise_variance: The noise variance.
    :arg float noise_std:
    :arg m: The posterior mean.
    :type m: :class:`numpy.ndarray`
    :arg cov: An intermediate matrix in calculating the posterior
        covariance, posterior_cov.
    :type cov: :class:`numpy.ndarray`
    :arg posterior_cov: The posterior covariance.
    :type posterior_cov: :class:`numpy.ndarray`
    :arg K_inv: The inverse of the prior covariance.
    :type K_inv: :class:`numpy.ndarray`
    :arg Z: The array of normalising constants.
    :type Z: :class:`numpy.ndarray`
    :arg bool numerical_stability: If the function is evaluated in a
        numerically stable way, default `True`.
    :return: fx
    :rtype: float
    """
    (Z,
    norm_pdf_z1s, norm_pdf_z2s,
    z1s, z2s, *_) = truncated_norm_normalising_constant(
        cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound=upper_bound, upper_bound2=upper_bound2)

    if trainables is not None:
        # For gx[0] -- ln\sigma
        if trainables[0]:
            z1s = B.where(z1s == -inf, 0.0, z1s)
            z1s = B.where(z1s == inf, 0.0, z1s)
            z2s = B.where(z2s == -inf, 0.0, z2s)
            z2s = B.where(z2s == inf, 0.0, z2s)
            gx[0] = -trace_posterior_cov_div_var - B.sum(
                (z1s * norm_pdf_z1s - z2s * norm_pdf_z2s) / Z)
            if verbose: print("gx_sigma = ", gx[0])
        if trainables[1:J]:
            w1 = norm_pdf_z1s / Z
            w2 = norm_pdf_z2s / Z
        # For gx[1] -- \b_1
        if trainables[1]:
            # For gx[1], \phi_b^1
            gx[1] = B.sum(w1 - w2)
            gx[1] = gx[1] / noise_std
        # For gx[2] -- ln\Delta^r
        for j in range(2, J):
            # Prepare D f / D delta_l
            if trainables[j]:
                gx[j] -= B.sum(B.where(y_train == j - 1, w2, 0))
                gx[j] -= B.sum(B.where(y_train > j - 1, w2 - w1, 0))
                gx[j] = gx[j] * intervals[j - 2] / noise_std
        if trainables[J]:
            gx[J] = (- variance * weight.T @ partial_K_variance @ weight
                + variance * B.sum(B.multiply(partial_K_variance, cov)))
        # For kernel parameters
        if ARD:
            for d in range(D):
                if trainables[J + 1][d]:
                    if numerical_stability is True:
                        # Update gx[-1], the partial derivative of th
                        # lower bound wrt the lengthscale
                        gx[J + 1 + d] = (
                    - theta[d] / 2 * weight.T @ partial_K_theta[d] @ weight
                    + theta[d] / 2 * B.einsum(
                        'ij, ji ->', partial_K_theta[d], cov))
                        if verbose: print("gx = {}".format(gx[J + 1 + d]))
        else:
            if trainables[J + 1]:
                if numerical_stability is True:
                    # Update gx[-1], the partial derivative of the lower bound
                    # wrt the lengthscale
                    gx[J + 1] = (
                        - theta / 2 * weight.T @ partial_K_theta @ weight
                        + theta / 2 * B.einsum(
                            'ij, ji ->', partial_K_theta, cov))
                    if verbose: print("gx = {}".format(gx[J + 1]))
    return gx


def update_posterior_mean_VB(noise_std, posterior_mean, cov,
        cutpoints_ts, cutpoints_tplus1s, K,
        upper_bound, upper_bound2=None):
    """Update VB approximation posterior covariance."""
    noise_var_w, Z, *_ = noise_variance_weight(
            cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound, upper_bound2)
    w = cov @ (posterior_mean + noise_var_w)
    return K @ w, w, Z


def update_posterior_covariance_VB(noise_variance, N, K):
    """Update posterior covariances.
    # TODO: rename to Jacobian?
    """
    cov, L_cov = matrix_inverse(noise_variance * B.eye(N) + K, N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    return L_cov, cov, log_det_cov, trace_cov, trace_posterior_cov_div_var


def update_hyperparameter_posterior_VB(
    posterior_mean, theta, theta_hyperparameters):
    theta = _theta(
        posterior_mean, theta_hyperparameters,
        n_samples=10)
    theta_hyperparameters = _theta_hyperparameters(theta)


def _theta_hyperparameters(
        theta, theta_hyperhyperparameters):
    """
    Return the approximate posterior mean of the kernel
    theta hyperparameters.

    Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
    Probit Regression with Gaussian Process Priors," in Neural Computation,
    vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
    doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(10).

    :arg theta: Posterior mean approximate of theta.
    :type theta: :class:`numpy.ndarray`
    :return: The approximate posterior mean of the hyperhyperparameters psi
        Girolami and Rogers Page 9 Eq.(10).
    """
    return B.divide(
        B.add(1, theta_hyperhyperparameters[0]),
        B.add(theta_hyperhyperparameters[1], theta))


def _theta(
        self, posterior_mean, weight, theta_hyperparameters, N, n_samples=10,
        vectorised=False):
    """
    Return the weights of the importance sampler for theta.

    Reference: M. Girolami and S. Rogers, "Variational Bayesian Multinomial
    Probit Regression with Gaussian Process Priors," in Neural Computation,
    vol. 18, no. 8, pp. 1790-1817, Aug. 2006,
    doi: 10.1162/neco.2006.18.8.1790.2005 Page 9 Eq.(9).

    :arg posterior_mean: approximate posterior mean.
    :arg theta_hyperparameters: approximate posterior mean of the kernel
        hyperparameters.
    :arg int n_samples: The number of samples for the importance sampling
        estimate, 500 is used in 2005 Page 13.
    """
    thetas = sample_thetas(  # (n_samples, N, N) for ISO case, depends on shape of theta hyper-hyperparameter
        theta_hyperparameters, n_samples)  # (n_samples, )
    log_thetas = B.log(thetas)
    Ks_samples = self.kernel.kernel_matrices(
        self.X_train, self.X_train, thetas)  # (n_samples, N, N)
    Ks_samples = B.add(Ks_samples, self.epsilon * B.eye(N))
    if vectorised:
        raise ValueError("TODO")
    else:
        log_ws = B.empty((n_samples,))
        # Scalar version
        for i in range(n_samples):
            L_K = B.cholesky(Ks_samples[i])
            half_log_det_K = B.sum(B.log(B.diag(L_K)))  # correct sign?
            # TODO 2021 - something not quite right - converges to zero
            # TODO: 28/07/2022 this may have been fixed by a fixing bug in the
            # code that did not update theta? Test
            # TODO: 28/07/2022 also may have made this more stable by using
            # K_inv @ posterior_mean = weight
            log_ws[i] = log_over_sqrt_2_pi - half_log_det_K\
                - 0.5 * posterior_mean.T @ weight
    # Normalise the w vectors
    max_log_ws = B.max(log_ws)
    log_normalising_constant = max_log_ws + B.log(
        B.sum(B.exp(log_ws - max_log_ws), axis=0))
    log_ws = B.subtract(log_ws, log_normalising_constant)
    print(B.sum(B.exp(log_ws)))
    element_prod = B.add(log_thetas, log_ws)
    element_prod = B.exp(element_prod)
    return B.sum(element_prod, axis=0)


def sample_thetas(theta_hyperparameter, n_samples):
    """
    Take n_samples of theta, given the hyperparameter of theta.

    theta_hyperparameter is a rate parameter since, with an uninformative
    prior (sigma=tau=0), then the posterior mean of Q(psi) is
    psi_tilde = 1. / theta_tilde. Therefore, by taking the expected value of
    the prior on theta ~ Exp(psi_tilde),
    we expect to obtain theta_tilde = 1. / psi_tilde. We get this if
    psi_tilde is a rate.

    :arg psi: float (Array) of hyper-hyperparameter(s)
    :type psi: :class:`np.ndarray`
    :arg int n_samples: The number of samples for the importance sample.
    """
    # scale = theta_hyperparameter
    scale = 1. / theta_hyperparameter
    shape = B.shape(theta_hyperparameter)
    if shape == ():
        size = (n_samples,)
    else:
        size = (n_samples, shape[0], shape[1])
    return expon.rvs(scale=scale, size=size)
