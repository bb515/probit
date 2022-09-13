from cmath import inf
import lab.jax as B
import jax
from jax import grad, jit, vmap
from math import inf
from probit.lab.utilities import (
    probit_likelihood, matrix_inverse)

# # It could be really useful to store cutpoints_ts, but cleaner not to.

# probit_likelihood(y, cutpoints, noise_std, f, upper_bound=None, upper_bound2=None, tolerance=None)

# # Really should pass around scalars and use vmap, but difficult since scalar doesn't have the ordering information of the data.

# weight = jit(grad(lambda f: truncated_norm_normalising_constant(
#     data, likelihood_parameters, f,
#     upper_bound=None, upper_bound2=None, tolerance=None)))

# prior = model()

# weights, cov = lambda prior_parameters, likelihood_parameters: approximator(prior_parameters, likelihood_parameters, data)  # is this accessible via defining a loss_and_grad function?
# loss_and_grad = lambda prior_parameters, likelihood_parameters: approximator.take_grad(prior_parameters, likelihood_parameters, data)

# # Then can revaluate for any given prior and likelihood parameters.

# approximator = Approximator(prior, likelihood)  # jit compiles some functions? as long as the prior and likelihood take in some parameters.


def weight(cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
        upper_bound, upper_bound2):
    (Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s,
        _, _) = probit_likelihood(
            cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
            upper_bound=upper_bound, upper_bound2=upper_bound2)
    return ((norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std,
        Z, norm_pdf_z1s, norm_pdf_z2s, z1s, z2s)


def f_LA(prior_parameters, likelihood_parameters, prior, grad_likelihood, posterior_mean, data):
    # TODO: test how scales with data when jitted, less memory intensive?
    # When wanting to differentiate through this, what to keep as static args?
    # Maybe put things as static args and let JAX recompile when it wants.
    #jitted-functions cannot have functions as arguments
    # so dlikelihood_df and K must be static
    return prior(prior_parameters)(data[0]) @ grad_likelihood(likelihood_parameters, posterior_mean, data[1])


def f_LASS(posterior_mean, noise_std, cutpoints_ts, cutpoints_tplus1s, K,
        upper_bound=None, upper_bound2=None):
    w, *_ = weight(cutpoints_ts, cutpoints_tplus1s, noise_std, posterior_mean,
        upper_bound, upper_bound2)
    return K @ w


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


def objective_LASS(
        posterior_mean, noise_std, cutpoints_ts, cutpoints_tplus1s,
        K, upper_bound=None, upper_bound2=None):
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
    L_cov = B.cholesky(K + B.diag(1. / precision))
    return (-B.sum(B.log(Z))
        + 0.5 * posterior_mean.T @ weight
        + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * B.sum(B.log(precision)))


def objective_LA(
        prior_parameters, likelihood_parameters, prior, grad_likelihood,
        hessian_likelihood, posterior_mean, data):
    likelihood = likelihood(likelihood_parameters, posterior_mean, data[1])
    weight = grad_likelihood(likelihood_parameters, posterior_mean, data[1])
    precision = hessian_likelihood(likelihood_parameters, posterior_mean, data[1])
    L_cov = B.cholesky(prior(prior_parameters)(data[0]) + B.diag(1./ precision))
    return (-B.sum(B.log(likelihood))
        + 0.5 * posterior_mean.T @ weight
        + B.sum(B.log(B.diag(L_cov)))
        + 0.5 * B.sum(B.log(precision)))
