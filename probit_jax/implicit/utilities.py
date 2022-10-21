"""Utility functions for probit."""
import lab as B
import jax
import jax.numpy as jnp
from jax import (vmap, grad, jit)
import warnings
from math import inf
from functools import partial
from jax.experimental.host_callback import id_print

over_sqrt_2_pi = 1. / B.sqrt(2 * B.pi)
log_over_sqrt_2_pi = -0.5 * B.log(2 * B.pi)
sqrt_2 = B.sqrt(2)


def ndtr(z):
    return 0.5 * (1 + jax.lax.erf(z/sqrt_2))


@partial(jax.jit, static_argnames=['N'])  # TODO: keep this here?
def matrix_inverse(matrix, N):
    L_cov = B.cholesky(matrix)
    L_covT_inv = B.triangular_solve(L_cov, B.eye(N), lower_a=True)
    cov = B.triangular_solve(L_cov.T, L_covT_inv, lower_a=False)
    return cov, L_cov


@partial(jax.jit)
def linear_solve(A, b):
    L_cov = B.cholesky(A)
    return B.cholesky_solve(L_cov, b)


def return_prob_vector(b, cutpoints_t, cutpoints_tplus1, noise_std):
    return ndtr((cutpoints_tplus1 - b) / noise_std) - ndtr(
        (cutpoints_t - b) / noise_std)


def posterior_covariance(K, cov, precision):
    return K @ cov @ B.diag(1./precision)


def norm_z_pdf(z):
    # TODO: mathematically justified
    z = jnp.clip(z, -1e10, 1e10)
    return over_sqrt_2_pi * jnp.exp(- 0.5 * z**2)


def norm_z_logpdf(x):
    return log_over_sqrt_2_pi - x**2 / 2.0


def norm_pdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_pdf(z) / scale


def norm_logpdf(x, loc=0.0, scale=1.0):
    z = (x - loc) / scale
    return norm_z_logpdf(z) - B.log(scale)


def norm_cdf(x):
    return ndtr(x)


def log_multivariate_normal_pdf(
        x, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        x = x - mean
    # log likelihood
    return -0.5 * B.log(2 * B.pi) - half_log_det_cov - 0.5 * x.T @ cov_inv @ x 


def log_multivariate_normal_pdf_vectorised(
        xs, cov_inv, half_log_det_cov, mean=None):
    """Get the pdf of the multivariate normal distribution."""
    if mean is not None:
        xs = xs - mean
    return -0.5 * B.log(2 * B.pi) - half_log_det_cov - 0.5 * B.einsum(
        'kj, kj -> k', B.einsum('ij, ki -> kj', cov_inv, xs), xs)


def h(x):
    """
    Polynomial part of a series expansion for log survival function for a
    normal random variable. With the third term, for x>4, this is accurate
    to three decimal places. The third term becomes significant when sigma
    is large. 
    """
    # jax.debug.print("x={}", x)
    return -1 * x**-2 + 5 / 2 * x**-4 - 37 / 3 * x**-6


def probit_likelihood(
        f, y, likelihood_parameters):
    return probit(
        likelihood_parameters[0],
        likelihood_parameters[1][y], likelihood_parameters[1][y + 1],
        f)


def log_probit_likelihood(
        f, y, likelihood_parameters):
    # jax.debug.print("probit_ll={}", probit_likelihood(f, y, likelihood_parameters))
    return jnp.log(probit_likelihood(f, y, likelihood_parameters) + 1e-10)

def _Z_tails(z1, z2):
    """
    Series expansion at infinity.

    Even for z1, z2 >= 4 this is accurate to three decimal places.
    """
    #TODO: mathematically justified
    z1 = jnp.clip(z1, -1e10, 1e10)
    z2 = jnp.clip(z2, -1e10, 1e10)

    return over_sqrt_2_pi * (
    1 / z1 * jnp.exp(-0.5 * z1**2 + h(z1)) - 1 / z2 * jnp.exp(
        -0.5 * z2**2 + h(z2)))


def _Z_far_tails(z):
    """Prevents overflow at large z."""
    z = jnp.clip(z, -1e10, 1e10)
    return over_sqrt_2_pi / z * jnp.exp(-0.5 * z**2 + h(z))


def _safe_Z(f, y, likelihood_parameters):
    """Calculate the difference in CDFs between two z-scores, where z2 >= z1.
    Use approximations to avoid catastrophic cancellation at extreme values.
    
    Nans are tracked through gradients. This function ensures that the functions
    are not evaluated at possible nan values."""

    cutpoints_tplus1 = likelihood_parameters[1][y + 1]
    cutpoints_t = likelihood_parameters[1][y]
    noise_std = likelihood_parameters[0]
    z2s = (cutpoints_tplus1 - f) / noise_std
    z1s = (cutpoints_t - f) / noise_std

    tol = 1e-16
    upper_bound = 3
    upper_bound2 = 6

    Z  = norm_cdf(z2s) - norm_cdf(z1s)

    if upper_bound is not None:
        # Remove any zero-values of z1s and z2s to avoid divide-by-zero
        # these values aren't used - only to avoid nans (https://github.com/google/jax/issues/1052 and 8247)
        _z1s = jnp.where(jnp.abs(z1s) < tol, tol, z1s)
        _z2s = jnp.where(jnp.abs(z2s) < tol, tol, z2s)

        # Using series expansion approximations
        Z = jnp.where(z1s > upper_bound, _Z_tails(jnp.where(_z1s < upper_bound, 1, _z1s), z2s), Z)
        Z = jnp.where(z2s < -upper_bound, _Z_tails(z1s, jnp.where(_z2s > -upper_bound, 1, _z2s)), Z)
        if upper_bound2 is not None:
            # Using one sided series expansion approximations
            Z = jnp.where(z1s > upper_bound2, _Z_far_tails(jnp.where(_z1s < upper_bound2, 1, _z1s)), Z)
            Z = jnp.where(z2s < -upper_bound2, _Z_far_tails(-jnp.where(_z2s > -upper_bound2, 1, _z2s)), Z)
    # Avoid divide-by-zero errors
    Z = jnp.where(Z < tol, tol, Z)
    # jax.debug.print("any_nans={}", jnp.any(jnp.isnan(1/z1s)))

    return Z, z1s, z2s




def grad_log_probit_likelihood(
        f, y, likelihood_parameters):
    noise_std = likelihood_parameters[0]
    Z, z1s, z2s = _safe_Z(f, y, likelihood_parameters)
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)
        
    return (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std



def hessian_log_probit_likelihood(
        f, y, likelihood_parameters):
    jax.debug.print("f={}\ny={}\nlp={}", f, y, likelihood_parameters)
    noise_std = likelihood_parameters[0]

    Z, z1s, z2s = _safe_Z(f, y, likelihood_parameters)
    norm_pdf_z1s = norm_pdf(z1s)
    norm_pdf_z2s = norm_pdf(z2s)
    w = (norm_pdf_z1s - norm_pdf_z2s) / Z / noise_std
    z1s = jnp.where(z1s == -inf, 0.0, z1s)
    z1s = jnp.where(z1s == inf, 0.0, z1s)
    z2s = jnp.where(z2s == -inf, 0.0, z2s)
    z2s = jnp.where(z2s == inf, 0.0, z2s)

    return -w**2 + (
        z1s * norm_pdf_z1s - z2s * norm_pdf_z2s
        ) / Z / noise_std**2


def probit(
        noise_std, cutpoints_y, cutpoints_yplus1, f):
    """
    Return the normalising constants for the truncated normal distribution
    in a numerically stable manner.

    :arg float noise_std: The noise standard deviation.
    :arg cutpoints_y: cutpoints[y_train] (N, ) array of cutpoints
    :type cutpoints_y: :class:`numpy.ndarray`
    :arg cutpoints_yplus1: cutpoints[y_train + 1] (N, ) array of cutpoints
    :type cutpoints_y: :class:`numpy.ndarray`
    :arg f: The mean vector.
    :type f: :class:`numpy.ndarray`
    :arg y: data
    :type y: :class:`numpy.ndarray`
    :arg float upper_bound: The threshold of the normal z value for which
        the pdf is close enough to zero.
    :arg float upper_bound2: The threshold of the normal z value for which
        the pdf is close enough to zero. 
    :arg float tolerance: The tolerated absolute error.
    :returns: Z
    :rtype: :class:`numpy.ndarray`
    """
    # Otherwise
    safe_z1s = jnp.where(cutpoints_y == -jnp.inf, 0.0, (cutpoints_y - f))
    safe_z2s = jnp.where(cutpoints_yplus1 == jnp.inf, 0.0, (cutpoints_yplus1 - f))
    norm_cdf_z1s = jnp.where(cutpoints_y == -jnp.inf, 0.0, norm_cdf(safe_z1s / noise_std))
    norm_cdf_z2s = jnp.where(cutpoints_yplus1 == jnp.inf, 1.0, norm_cdf(safe_z2s / noise_std))
    Z = norm_cdf_z2s - norm_cdf_z1s
    return Z


def ordinal_predictive_distributions(
        posterior_pred_mean, posterior_pred_std, N_test,
        cutpoints):
    """
    Return predictive distributions for the ordinal likelihood.
    """
    J = B.size(cutpoints) - 1
    predictive_distributions = B.ones(N_test, J)
    for j in range(J):
        Z, *_ = probit(
                posterior_pred_std,
                cutpoints[j], cutpoints[j + 1],
                posterior_pred_mean,
                )
        predictive_distributions[:, j] = Z
    return predictive_distributions


def predict_reparameterised(
        Kss, Kfs, cov, weight, cutpoints, noise_variance):
    """
    Make posterior prediction over ordinal classes of X_test.

    :arg X_test: The new data points, array like (N_test, D).
    :arg cov: A covariance matrix used in calculation of posterior
        predictions. (\sigma^2I + K)^{-1} Array like (N, N).
    :type cov: :class:`numpy.ndarray`
    :arg weight: The approximate inverse-covariance-posterior-mean.
        .. math::
            \nu = (\mathbf{K} + \sigma^{2}\mathbf{I})^{-1} \mathbf{y}
            = \mathbf{K}^{-1} \mathbf{f}
        Array like (N,).
    :type weight: :class:`numpy.ndarray`
    :arg cutpoints: (J + 1, ) array of the cutpoints.
    :type cutpoints: :class:`numpy.ndarray`.
    :arg float noise_variance: The noise variance.
    :arg bool numerically_stable: Use matmul or triangular solve.
        Default `False`. 
    :return: A Monte Carlo estimate of the class probabilities.
    :rtype tuple: ((N_test, J), (N_test,), (N_test,))
    """
    N_test = B.shape(Kss)[0]
    temp = cov @ Kfs
    posterior_variance = Kss - B.einsum(
        'ij, ij -> j', Kfs, temp)
    posterior_std = B.sqrt(posterior_variance)
    posterior_pred_mean = Kfs.T @ weight
    posterior_pred_variance = posterior_variance + noise_variance
    posterior_pred_std = B.sqrt(posterior_pred_variance)
    return (
        ordinal_predictive_distributions(
                posterior_pred_mean, posterior_pred_std, N_test, cutpoints,
                ),
            posterior_pred_mean, posterior_std)


def sample_g(g, f, y_train, cutpoints, noise_std, N):
    """TODO: Seems like this has to be done in numpy or numba."""
    for i in range(N):
        # Target class index
        j_true = y_train[i]
        g_i = -inf  # this is a trick for the next line
        # Sample from the truncated Gaussian
        while g_i > cutpoints[j_true + 1] or g_i <= cutpoints[j_true]:
            # sample y
            # TODO: test if this works
            g_i = f[i] + (f[i] - B.randn(1)) / noise_std
        # Add sample to the Y vector
        g[i] = g_i
    return g


class CutpointValueError(Exception):
    """
    An invalid cutpoint argument was used to construct the classifier model.
    """

    def __init__(self, cutpoint):
        """
        Construct the exception.

        :arg cutpoint: The cutpoint parameters array.
        :type cutpoint: :class:`numpy.array` or list

        :rtype: :class:`CutpointValueError`
        """
        message = (
                "The cutpoint list or array "
                "must be in ascending order, "
                f" {cutpoint} was given."
                )

        super().__init__(message)
