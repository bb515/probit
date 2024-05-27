import lab.jax as B


def f_VB(
    prior_parameters, likelihood_parameters, prior, grad_log_likelihood, weight, data
):
    K = B.dense(prior(prior_parameters)(data[0]))
    N = B.shape(data[0])[0]
    posterior_mean = K @ weight
    L_cov = B.cholesky(likelihood_parameters[0] ** 2 * B.eye(N) + K)
    return B.cholesky_solve(
        L_cov,
        posterior_mean
        + likelihood_parameters[0]
        * grad_log_likelihood(posterior_mean, data[1], likelihood_parameters),
    )


def objective_VB(
    prior_parameters, likelihood_parameters, prior, log_likelihood, weight, data
):
    K = B.dense(prior(prior_parameters)(data[0]))
    posterior_mean = K @ weight
    N = B.shape(data[0])[0]
    L_cov = B.cholesky(likelihood_parameters[0] ** 2 * B.eye(N) + K)
    L_covT_inv = B.triangular_solve(L_cov, B.eye(N), lower_a=True)
    cov = B.triangular_solve(L_cov.T, L_covT_inv, lower_a=False)
    log_det_cov = -2 * B.sum(B.log(B.diag(L_cov)))
    trace_cov = B.sum(B.diag(cov))
    trace_posterior_cov_div_var = B.einsum("ij, ij -> ", K, cov)
    trace_K_inv_posterior_cov = likelihood_parameters[0] ** 2 * trace_cov
    return (
        0.5 * trace_posterior_cov_div_var
        + 0.5 * trace_K_inv_posterior_cov
        + 0.5 * posterior_mean.T @ weight
        - N * B.log(likelihood_parameters[0])
        - 0.5 * log_det_cov
        - 0.5 * N
        - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters))
    )
