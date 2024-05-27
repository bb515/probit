import lab.jax as B


def f_LA(
    prior_parameters, likelihood_parameters, prior, grad_log_likelihood, weight, data
):
    K = B.dense(prior(prior_parameters)(data[0]))
    posterior_mean = K @ weight
    return grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)


def objective_LA(
    prior_parameters,
    likelihood_parameters,
    prior,
    log_likelihood,
    hessian_log_likelihood,
    weight,
    data,
):
    K = B.dense(prior(prior_parameters)(data[0]))
    posterior_mean = K @ weight
    precision = -hessian_log_likelihood(posterior_mean, data[1], likelihood_parameters)
    L_cov = B.cholesky(prior(prior_parameters)(data[0]) + B.diag(1.0 / precision))
    return (
        -B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters))
        + 0.5 * posterior_mean.T @ weight
        + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * B.sum(B.log(precision))
    )
