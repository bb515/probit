import lab as B
from probit.lab.utilities import (
    truncated_norm_normalising_constant, matrix_inverse)


def noise_variance_weight(cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
        upper_bound, upper_bound2):
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = truncated_norm_normalising_constant(
            cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    return (noise_std * (norm_pdf_z1s - norm_pdf_z2s) / Z,
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s)


def f_VB(posterior_mean, model, data, N, upper_bound=None, upper_bound2=None):
    kernel, cutpoints, noise_variance = model
    X, y = data
    kernel, cutpoints, noise_variance = model
    K = B.dense(kernel(X))
    noise_std = B.sqrt(noise_variance)
    cutpoints_ts = cutpoints[y]
    cutpoints_tplus1s = cutpoints[y + 1]
    cov, L_cov = matrix_inverse(
        noise_std**2 * B.eye(N) + K, N)
    noise_variance_w, *_ = noise_variance_weight(cutpoints_ts, cutpoints_tplus1s, noise_std,
        posterior_mean, upper_bound, upper_bound2)
    w = cov @ (posterior_mean + noise_variance_w)
    return K @ w


def jacobian_VB(model, data, N, K):
    """Update posterior covariances."""
    X, _ = data
    kernel, _, noise_variance = model
    K = B.dense(kernel(X))
    cov, L_cov = matrix_inverse(noise_variance * B.eye(N) + K, N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    return L_cov, cov, log_det_cov, trace_cov, trace_posterior_cov_div_var


def objective_VB(
        posterior_mean, model, data, N, upper_bound=None, upper_bound2=None):
    """
    Calculate fx, the variational lower bound of the log marginal
    likelihood.

    .. math::
            \mathcal{F()} =,

        where
    """
    kernel, cutpoints, noise_variance = model
    X, y = data
    kernel, cutpoints, noise_variance = model
    K = B.dense(kernel(X))
    noise_std = B.sqrt(noise_variance)
    cutpoints_ts = cutpoints[y]
    cutpoints_tplus1s = cutpoints[y + 1]
    cov, L_cov = matrix_inverse(noise_std**2 * B.eye(N) + K, N)
    noise_variance_w, Z, *_ = noise_variance_weight(
        cutpoints_ts, cutpoints_tplus1s, noise_std,posterior_mean,
        upper_bound, upper_bound2)
    w = cov @ (posterior_mean + noise_variance_w)
    cov, L_cov = matrix_inverse(noise_std**2 * B.eye(N) + K, N)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    trace_K_inv_posterior_cov = noise_std**2 * trace_cov
    return (0.5 * trace_posterior_cov_div_var
        + 0.5 * trace_K_inv_posterior_cov
        + 0.5 * posterior_mean.T @ w
        - N * B.log(noise_std)
        - 0.5 * log_det_cov
        - 0.5 * N
        - B.sum(B.log(Z)))
