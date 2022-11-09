from cmath import inf
import lab.jax as B
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from math import inf
from probit_jax.lab.utilities import (
    probit_likelihood, matrix_inverse)  # TODO: temp
from probit_jax.implicit.utilities import linear_solve
from probit_jax.implicit.utilities import norm_cdf, norm_pdf
from jax.experimental.host_callback import id_print


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

# # This cannot be correct
# def f_LA(prior_parameters, likelihood_parameters, prior, grad_log_likelihood, hessian_log_likelihood, posterior_mean, data):
#     K = B.dense(prior(prior_parameters)(data[0]))
#     return K @ linear_solve(B.diag(-hessian_log_likelihood(
#         posterior_mean, data[1], likelihood_parameters)) + K,
#         posterior_mean + grad_log_likelihood(posterior_mean, data[1], likelihood_parameters))


# def f_LA(prior_parameters, likelihood_parameters, prior, grad_log_likelihood,
#         hessian_log_likelihood, posterior_mean, data):
#     # w is grad negative log likelihood. Is it possible to use vmap and still parameterize it,
#     # and take gradients wrt to f. Would it be best to do this via grad or vjp?
#     # TODO: test how scales with data when jitted, less memory intensive?
#     # When wanting to differentiate through this, what to keep as static args?
#     # Maybe put things as static args and let JAX recompile when it wants.
#     #jitted-functions cannot have functions as arguments
#     # so dlikelihood_df and K must be static
#     K = B.dense(prior(prior_parameters)(data[0]))
#     return - K @ grad_log_likelihood(posterior_mean, data[1], likelihood_parameters)


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
    # TODO: check that precision is the same as the hessian
    precision  = w**2 + (
        z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
        ) / Z / noise_std**2
    cov, L_cov = matrix_inverse(K + B.diag(1. / precision), N)
    return cov, L_cov


# def objective_LA(
#         prior_parameters, likelihood_parameters, prior,
#         log_likelihood, grad_log_likelihood, hessian_log_likelihood,
#         posterior_mean, data):
#     (X, y) = data
#     cutpoints_tplus1 = likelihood_parameters[1][y + 1]
#     cutpoints_t = likelihood_parameters[1][y]
#     noise_std = likelihood_parameters[0]
#     z2s = (cutpoints_tplus1 - posterior_mean) / noise_std
#     z1s = (cutpoints_t - posterior_mean) / noise_std
#     Z  = norm_cdf(z2s) - norm_cdf(z1s)
#     norm_pdf_z1s = norm_pdf(z1s)
#     norm_pdf_z2s = norm_pdf(z2s)
#     w = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
 
#     z1s = B.where(z1s == -inf, 0.0, z1s)
#     z1s = B.where(z1s == inf, 0.0, z1s)
#     z2s = B.where(z2s == -inf, 0.0, z2s)
#     z2s = B.where(z2s == inf, 0.0, z2s)
#     precision  = w**2 + (
#         z2s * norm_pdf_z2s - z1s * norm_pdf_z1s
#         ) / Z / noise_std**2
#     L_cov = B.cholesky(prior(prior_parameters)(X) + B.diag(1./ precision))

#     # id_print(-B.sum(B.log(Z)))
#     # id_print(0.5 * posterior_mean.T @ w)
#     # id_print(B.sum(B.log(B.diag(L_cov))))
#     # id_print(0.5 * B.sum(B.log(precision)))

#     return (-B.sum(B.log(Z))
#         + 0.5 * posterior_mean.T @ w
#         + B.sum(B.log(B.diag(L_cov)))
#         + 0.5 * B.sum(B.log(precision)))


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
