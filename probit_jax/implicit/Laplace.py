from cmath import inf
import lab.jax as B
from math import inf
from probit_jax.implicit.utilities import (
    probit_likelihood, matrix_inverse)  # TODO: temp


# TODO delete weight
def weight(cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
        upper_bound, upper_bound2):
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = probit_likelihood(
            cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    return ((norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std,
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s)


def f_LA(prior_parameters, likelihood_parameters, prior, grad_log_likelihood, hessian_log_likelihood, posterior_mean, data):
    K = B.dense(prior(prior_parameters)(data[0]))
    return K @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)


# TODO work out what for
# Can either define a fixed_point method fwd_solve using this jacobian or
# define it implicitly by using f_LA and Newton's method
def jacobian_LA(posterior_mean, noise_std, cutpoints_ts, cutpoints_tplus1s,
        K, N, upper_bound=None, upper_bound2=None):
    w, Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s = weight(
        cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
        upper_bound, upper_bound2)
    # This is not for numerical stability, it is mathematically correct
    z1s = B.where(z1s == -inf, 0.0, z1s)
    z1s = B.where(z1s == inf, 0.0, z1s)
    z2s = B.where(z2s == -inf, 0.0, z2s)
    z2s = B.where(z2s == inf, 0.0, z2s)
    precision  = w**2 + (
        z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
        ) / Z / noise_std**2
    cov, L_cov = matrix_inverse(K + B.diag(1. / precision), N)
    return cov, L_cov


def objective_LA(prior_parameters, likelihood_parameters, prior,
        log_likelihood, grad_log_likelihood, hessian_log_likelihood, posterior_mean, data):
    precision = -hessian_log_likelihood(posterior_mean, data[1], likelihood_parameters) + 1e-8
    L_cov = B.cholesky(prior(prior_parameters)(data[0]) + B.diag(1./ precision))

    return (
        - B.sum(log_likelihood(posterior_mean, data[1], likelihood_parameters))
        + 0.5 * posterior_mean.T @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)  # TODO minus before grad?
        + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * B.sum(B.log(precision))
    )
