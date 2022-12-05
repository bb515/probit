import lab as B
from probit_jax.implicit.utilities import linear_solve, matrix_inverse


def f_VB(prior_parameters, likelihood_parameters, prior, grad_log_likelihood, posterior_mean, data):
    K = B.dense(prior(prior_parameters)(data[0]))
    N = B.shape(data[0])[0]
    return K @ linear_solve(likelihood_parameters[0]**2 * B.eye(N) + K,
        posterior_mean + likelihood_parameters[0] * grad_log_likelihood(posterior_mean, data[1], likelihood_parameters))


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


# def objective_VBSS(
#         posterior_mean, model, data, N, upper_bound=None, upper_bound2=None):
#     """
#     Calculate fx, the variational lower bound of the log marginal
#     likelihood.

#     .. math::
#             \mathcal{F()} =,

#         where
#     """
#     kernel, cutpoints, noise_variance = model
#     X, y = data
#     kernel, cutpoints, noise_variance = model
#     K = B.dense(kernel(X))
#     noise_std = B.sqrt(noise_variance)
#     cutpoints_ts = cutpoints[y]
#     cutpoints_tplus1s = cutpoints[y + 1]
#     cov, L_cov = matrix_inverse(noise_std**2 * B.eye(N) + K, N)
#     noise_variance_w = noise_std * grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)
#     w = cov @ (posterior_mean + noise_variance_w)
#     cov, L_cov = matrix_inverse(noise_std**2 * B.eye(N) + K, N)
#     log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
#     trace_cov = B.sum(B.diag(cov))
#     trace_posterior_cov_div_var = B.einsum(
#         'ij, ij -> ', K, cov)
#     trace_K_inv_posterior_cov = noise_std**2 * trace_cov
#     return (0.5 * trace_posterior_cov_div_var
#         + 0.5 * trace_K_inv_posterior_cov
#         + 0.5 * posterior_mean.T @ w
#         - N * B.log(noise_std)
#         - 0.5 * log_det_cov
#         - 0.5 * N
#         - B.sum(B.log(Z)))


def objective_VBSSS(prior_parameters, likelihood_parameters, prior,
        log_likelihood, grad_log_likelihood, posterior_mean, data):
    """TODO: this is meant to match analytic version, but struggles."""
    K = B.dense(prior(prior_parameters)(data[0]))
    N = B.shape(data[0])[0]
    L_cov = B.cholesky(likelihood_parameters[0]**2 * B.eye(N) + K)
    w = B.cholesky_solve(L_cov, posterior_mean + grad_log_likelihood(posterior_mean, data[1], likelihood_parameters))
    L_covT_inv = B.triangular_solve(L_cov, B.eye(N), lower_a=True)
    cov = B.triangular_solve(L_cov.T, L_covT_inv, lower_a=False)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    trace_K_inv_posterior_cov = likelihood_parameters[0]**2 * trace_cov
    return (0.5 * trace_posterior_cov_div_var
        + 0.5 * trace_K_inv_posterior_cov
        + 0.5 * posterior_mean.T @ w
        - N * B.log(likelihood_parameters[0])
        - 0.5 * log_det_cov
        - 0.5 * N
        - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters)))


def objective_VB(prior_parameters, likelihood_parameters, prior,
        log_likelihood, grad_log_likelihood, posterior_mean, data):
    K = B.dense(prior(prior_parameters)(data[0]))
    N = B.shape(data[0])[0]
    L_cov = B.cholesky(likelihood_parameters[0]**2 * B.eye(N) + K)
    L_covT_inv = B.triangular_solve(L_cov, B.eye(N), lower_a=True)
    cov = B.triangular_solve(L_cov.T, L_covT_inv, lower_a=False)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum(
        'ij, ij -> ', K, cov)
    trace_K_inv_posterior_cov = likelihood_parameters[0]**2 * trace_cov
    return (0.5 * trace_posterior_cov_div_var
        + 0.5 * trace_K_inv_posterior_cov
        + 0.5 * posterior_mean.T @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)
        - N * B.log(likelihood_parameters[0])
        - 0.5 * log_det_cov
        - 0.5 * N
        - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters)))
